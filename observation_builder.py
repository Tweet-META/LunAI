from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BulletState:
    x: float
    y: float
    radius: float
    vx: float
    vy: float


@dataclass(frozen=True)
class PlayerState:
    x: float
    y: float
    radius: float
    previous_action: int = 0


@dataclass(frozen=True)
class ObservationConfig:
    playfield_width: int = 600
    playfield_height: int = 700
    blue_grid: tuple[int, int] = (8, 8)
    yellow_size: tuple[int, int] = (320, 320)
    yellow_grid: tuple[int, int] = (16, 16)
    red_size: tuple[int, int] = (128, 128)
    red_map: tuple[int, int] = (64, 64)
    pccm_prediction_frames: int = 5
    pccm_halo_width: float = 32.0
    pccm_wall_margin: float = 0.12
    pccm_upper_field_threshold: float = 0.70
    pccm_upper_field_cost: float = 0.30
    pccm_soft_cap: float = 0.8
    pccm_implementation: str = "reference"


# Return a player-centered window that may extend outside the field.
def centered_window(center_x: float, center_y: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(round(center_x - width / 2))
    y1 = int(round(center_y - height / 2))
    return x1, y1, x1 + width, y1 + height


# Rasterize circular bullet hitboxes into a binary map.
def make_occupancy_map(width: int, height: int, bullets: list[BulletState]) -> np.ndarray:
    occupancy = np.zeros((height, width), dtype=np.float32)
    for bullet in bullets:
        r = max(1, int(np.ceil(bullet.radius)))
        cx = int(round(bullet.x))
        cy = int(round(bullet.y))
        x1 = max(0, cx - r)
        x2 = min(width, cx + r + 1)
        y1 = max(0, cy - r)
        y2 = min(height, cy + r + 1)
        if x1 >= x2 or y1 >= y2:
            continue

        yy, xx = np.ogrid[y1:y2, x1:x2]
        mask = (xx - bullet.x) ** 2 + (yy - bullet.y) ** 2 <= bullet.radius ** 2
        occupancy[y1:y2, x1:x2][mask] = 1.0
    return occupancy


# Build a padded summed-area table for fast rectangle sums.
def make_integral_image(binary_map: np.ndarray) -> np.ndarray:
    integral = binary_map.cumsum(axis=0).cumsum(axis=1)
    return np.pad(integral, ((1, 0), (1, 0)), mode="constant")


# Convert a possibly padded window into a fixed-size hitbox density grid.
def density_grid(integral: np.ndarray, window: tuple[int, int, int, int], grid_shape: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = window
    rows, cols = grid_shape
    field_h = integral.shape[0] - 1
    field_w = integral.shape[1] - 1
    xs = np.linspace(x1, x2, cols + 1).round().astype(int)
    ys = np.linspace(y1, y2, rows + 1).round().astype(int)
    clipped_xs = np.clip(xs, 0, field_w)
    clipped_ys = np.clip(ys, 0, field_h)
    cell_x1 = clipped_xs[:-1][None, :]
    cell_x2 = clipped_xs[1:][None, :]
    cell_y1 = clipped_ys[:-1][:, None]
    cell_y2 = clipped_ys[1:][:, None]
    occupied = (
        integral[cell_y2, cell_x2]
        - integral[cell_y1, cell_x2]
        - integral[cell_y2, cell_x1]
        + integral[cell_y1, cell_x1]
    )
    area = (cell_x2 - cell_x1) * (cell_y2 - cell_y1)
    out = np.divide(
        occupied,
        area,
        out=np.zeros((rows, cols), dtype=np.float32),
        where=area > 0,
    )
    return np.clip(out, 0.0, 1.0)


# Build a map that marks the playable fraction of every local cell.
def valid_area_grid(
    window: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    field_w: int,
    field_h: int,
) -> np.ndarray:
    x1, y1, x2, y2 = window
    rows, cols = grid_shape
    xs = np.linspace(x1, x2, cols + 1).round().astype(int)
    ys = np.linspace(y1, y2, rows + 1).round().astype(int)
    cell_widths = np.diff(xs)
    cell_heights = np.diff(ys)
    cell_area = np.maximum(1, np.outer(cell_heights, cell_widths))
    clipped_xs = np.clip(xs, 0, field_w)
    clipped_ys = np.clip(ys, 0, field_h)
    playable_widths = np.maximum(0, np.diff(clipped_xs))
    playable_heights = np.maximum(0, np.diff(clipped_ys))
    playable_area = np.outer(playable_heights, playable_widths)
    return (playable_area / cell_area).astype(np.float32)


# Return world-space center coordinates for every cell in one map.
def grid_cell_centers(
    window: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = window
    rows, cols = grid_shape
    xs = x1 + (np.arange(cols, dtype=np.float32) + 0.5) * (x2 - x1) / cols
    ys = y1 + (np.arange(rows, dtype=np.float32) + 0.5) * (y2 - y1) / rows
    return np.meshgrid(xs, ys)


# Combine independent soft costs without allowing a simple sum to exceed one.
def combine_soft_cost(old_cost: np.ndarray, new_cost: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - old_cost) * (1.0 - new_cost)


# Pool high-risk samples without letting one isolated sample dominate a large cell.
def top_fraction_pool(
    values: np.ndarray,
    output_shape: tuple[int, int],
    fraction: float = 0.25,
) -> np.ndarray:
    out_rows, out_cols = output_shape
    rows, cols = values.shape
    if rows % out_rows != 0 or cols % out_cols != 0:
        raise ValueError(f"Cannot pool shape {values.shape} into {output_shape}.")
    block_rows = rows // out_rows
    block_cols = cols // out_cols
    blocks = values.reshape(out_rows, block_rows, out_cols, block_cols)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(out_rows, out_cols, -1)
    count = max(1, int(np.ceil(blocks.shape[-1] * fraction)))
    partition_index = blocks.shape[-1] - count
    top_values = np.partition(blocks, partition_index, axis=-1)[..., partition_index:]
    return np.mean(top_values, axis=-1, dtype=np.float32)


# Average dense wall samples when projecting them into a coarser map.
def average_pool(values: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    out_rows, out_cols = output_shape
    rows, cols = values.shape
    if rows % out_rows != 0 or cols % out_cols != 0:
        raise ValueError(f"Cannot pool shape {values.shape} into {output_shape}.")
    block_rows = rows // out_rows
    block_cols = cols // out_cols
    blocks = values.reshape(out_rows, block_rows, out_cols, block_cols)
    return np.mean(blocks, axis=(1, 3), dtype=np.float32)


# Resize one smooth cost map with bilinear interpolation.
def bilinear_resize(values: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    out_rows, out_cols = output_shape
    in_rows, in_cols = values.shape
    y = np.clip((np.arange(out_rows) + 0.5) * in_rows / out_rows - 0.5, 0.0, in_rows - 1.0)
    x = np.clip((np.arange(out_cols) + 0.5) * in_cols / out_cols - 0.5, 0.0, in_cols - 1.0)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(in_rows - 1, y0 + 1)
    x1 = np.minimum(in_cols - 1, x0 + 1)
    wy = (y - y0).astype(np.float32)[:, None]
    wx = (x - x0).astype(np.float32)[None, :]
    top = (1.0 - wx) * values[y0[:, None], x0[None, :]] + wx * values[y0[:, None], x1[None, :]]
    bottom = (1.0 - wx) * values[y1[:, None], x0[None, :]] + wx * values[y1[:, None], x1[None, :]]
    return ((1.0 - wy) * top + wy * bottom).astype(np.float32)


# Build soft costs for walls and the less-preferred upper playfield.
def environment_pccm_cost(
    xx: np.ndarray,
    yy: np.ndarray,
    inside: np.ndarray,
    field_w: int,
    field_h: int,
    wall_margin: float,
    upper_field_threshold: float,
    upper_field_cost: float,
) -> np.ndarray:
    horizontal_margin = max(1.0, field_w * wall_margin)
    vertical_margin = max(1.0, field_h * wall_margin)
    environment_cost = np.zeros(xx.shape, dtype=np.float32)
    wall_distances = (
        (xx, horizontal_margin),
        (field_w - xx, horizontal_margin),
        (yy, vertical_margin),
        (field_h - yy, vertical_margin),
    )
    for distance, margin in wall_distances:
        contribution = np.where(
            inside,
            0.5 * np.clip(1.0 - distance / margin, 0.0, 1.0),
            0.0,
        ).astype(np.float32)
        environment_cost = combine_soft_cost(environment_cost, contribution)

    upper_boundary = max(1.0, field_h * upper_field_threshold)
    upper_contribution = np.where(
        inside,
        upper_field_cost * np.clip(1.0 - yy / upper_boundary, 0.0, 1.0),
        0.0,
    ).astype(np.float32)
    return combine_soft_cost(environment_cost, upper_contribution)


# Build PCCM samples with full-grid NumPy broadcasting as the reference.
def pccm_sample_components_reference(
    bullets: list[BulletState],
    player_radius: float,
    window: tuple[int, int, int, int],
    sample_shape: tuple[int, int],
    field_w: int,
    field_h: int,
    prediction_frames: int,
    halo_width: float,
    wall_margin: float,
    fps: float = 60.0,
    upper_field_threshold: float = 0.70,
    upper_field_cost: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xx, yy = grid_cell_centers(window, sample_shape)
    inside = (xx >= 0.0) & (xx < field_w) & (yy >= 0.0) & (yy < field_h)
    current_cost = np.zeros(sample_shape, dtype=np.float32)
    prediction_cost = np.zeros(sample_shape, dtype=np.float32)
    hard_collision = np.zeros(sample_shape, dtype=np.float32)

    if bullets:
        x1, y1, x2, y2 = window
        horizon_seconds = prediction_frames / fps
        relevant_bullets = []
        for bullet in bullets:
            future_x = bullet.x + bullet.vx * horizon_seconds
            future_y = bullet.y + bullet.vy * horizon_seconds
            padding = bullet.radius + player_radius + halo_width
            if (
                max(bullet.x, future_x) + padding >= x1
                and min(bullet.x, future_x) - padding < x2
                and max(bullet.y, future_y) + padding >= y1
                and min(bullet.y, future_y) - padding < y2
            ):
                relevant_bullets.append(bullet)

        if relevant_bullets:
            bullet_x = np.asarray([bullet.x for bullet in relevant_bullets], dtype=np.float32)
            bullet_y = np.asarray([bullet.y for bullet in relevant_bullets], dtype=np.float32)
            velocity_x = np.asarray([bullet.vx for bullet in relevant_bullets], dtype=np.float32)
            velocity_y = np.asarray([bullet.vy for bullet in relevant_bullets], dtype=np.float32)
            radii = np.asarray(
                [max(1.0, bullet.radius + player_radius) for bullet in relevant_bullets],
                dtype=np.float32,
            )
            times = np.arange(prediction_frames + 1, dtype=np.float32) / fps
            future_x = bullet_x[:, None] + velocity_x[:, None] * times[None, :]
            future_y = bullet_y[:, None] + velocity_y[:, None] * times[None, :]
            dx = xx[None, None, :, :] - future_x[:, :, None, None]
            dy = yy[None, None, :, :] - future_y[:, :, None, None]
            distances = np.sqrt(dx * dx + dy * dy)
            falloff = np.clip(
                1.0 - np.maximum(0.0, distances - radii[:, None, None, None]) / halo_width,
                0.0,
                1.0,
            )
            time_weights = 1.0 - np.arange(prediction_frames + 1, dtype=np.float32) / (prediction_frames + 1.0)
            contributions = 0.5 * falloff * time_weights[None, :, None, None]
            contributions *= inside[None, None, :, :]
            current_cost = 1.0 - np.prod(1.0 - contributions[:, 0], axis=0)
            prediction_cost = 1.0 - np.prod(1.0 - contributions[:, 1:], axis=(0, 1))
            hard_collision = np.any(
                distances[:, 0] <= radii[:, None, None],
                axis=0,
            ).astype(np.float32)

    wall_cost = environment_pccm_cost(
        xx,
        yy,
        inside,
        field_w,
        field_h,
        wall_margin,
        upper_field_threshold,
        upper_field_cost,
    )

    hard_collision[~inside] = 0.0
    return current_cost, prediction_cost, wall_cost, hard_collision


# Convert one world-space support box into conservative sample-grid bounds.
def pccm_roi_bounds_from_aabb(
    minimum_x: float,
    minimum_y: float,
    maximum_x: float,
    maximum_y: float,
    window: tuple[int, int, int, int],
    sample_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = window
    rows, cols = sample_shape
    cell_width = (x2 - x1) / cols
    cell_height = (y2 - y1) / rows

    # The extra outward sample is harmless because the exact falloff becomes zero.
    col1 = max(0, int(np.floor((minimum_x - x1) / cell_width - 0.5)))
    col2 = min(cols, int(np.ceil((maximum_x - x1) / cell_width - 0.5)) + 1)
    row1 = max(0, int(np.floor((minimum_y - y1) / cell_height - 0.5)))
    row2 = min(rows, int(np.ceil((maximum_y - y1) / cell_height - 0.5)) + 1)
    if col1 >= col2 or row1 >= row2:
        return None
    return row1, row2, col1, col2


# Convert one circular support area into conservative sample-grid bounds.
def pccm_roi_bounds(
    center_x: float,
    center_y: float,
    support_radius: float,
    window: tuple[int, int, int, int],
    sample_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    return pccm_roi_bounds_from_aabb(
        center_x - support_radius,
        center_y - support_radius,
        center_x + support_radius,
        center_y + support_radius,
        window,
        sample_shape,
    )


# Build PCCM samples by evaluating each bullet only inside its exact support ROI.
def pccm_sample_components_roi(
    bullets: list[BulletState],
    player_radius: float,
    window: tuple[int, int, int, int],
    sample_shape: tuple[int, int],
    field_w: int,
    field_h: int,
    prediction_frames: int,
    halo_width: float,
    wall_margin: float,
    fps: float = 60.0,
    upper_field_threshold: float = 0.70,
    upper_field_cost: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xx, yy = grid_cell_centers(window, sample_shape)
    inside = (xx >= 0.0) & (xx < field_w) & (yy >= 0.0) & (yy < field_h)
    current_cost = np.zeros(sample_shape, dtype=np.float32)
    prediction_cost = np.zeros(sample_shape, dtype=np.float32)
    hard_collision = np.zeros(sample_shape, dtype=np.float32)
    times = np.arange(prediction_frames + 1, dtype=np.float32) / np.float32(fps)
    time_weights = 1.0 - np.arange(
        prediction_frames + 1,
        dtype=np.float32,
    ) / np.float32(prediction_frames + 1.0)

    x1, y1, x2, y2 = window
    horizon_seconds = prediction_frames / fps
    relevant_bullets = []
    for bullet in bullets:
        future_x = bullet.x + bullet.vx * horizon_seconds
        future_y = bullet.y + bullet.vy * horizon_seconds
        padding = bullet.radius + player_radius + halo_width
        if (
            max(bullet.x, future_x) + padding >= x1
            and min(bullet.x, future_x) - padding < x2
            and max(bullet.y, future_y) + padding >= y1
            and min(bullet.y, future_y) - padding < y2
        ):
            relevant_bullets.append(bullet)

    for bullet in relevant_bullets:
        bullet_x = np.float32(bullet.x)
        bullet_y = np.float32(bullet.y)
        velocity_x = np.float32(bullet.vx)
        velocity_y = np.float32(bullet.vy)
        collision_radius = np.float32(max(1.0, bullet.radius + player_radius))
        support_radius = float(collision_radius + np.float32(halo_width))
        future_x = bullet_x + velocity_x * times
        future_y = bullet_y + velocity_y * times

        current_bounds = pccm_roi_bounds(
            float(future_x[0]),
            float(future_y[0]),
            support_radius,
            window,
            sample_shape,
        )
        if current_bounds is not None:
            row1, row2, col1, col2 = current_bounds
            local_x = xx[row1:row2, col1:col2]
            local_y = yy[row1:row2, col1:col2]
            local_inside = inside[row1:row2, col1:col2]
            dx = local_x - future_x[0]
            dy = local_y - future_y[0]
            distances = np.sqrt(dx * dx + dy * dy)
            falloff = np.clip(
                1.0 - np.maximum(0.0, distances - collision_radius) / halo_width,
                0.0,
                1.0,
            )
            contribution = (np.float32(0.5) * falloff * local_inside).astype(np.float32)
            target = current_cost[row1:row2, col1:col2]
            target[:] = combine_soft_cost(target, contribution)
            hard_target = hard_collision[row1:row2, col1:col2]
            hard_target[(distances <= collision_radius) & local_inside] = 1.0

        if prediction_frames > 0:
            prediction_bounds = pccm_roi_bounds_from_aabb(
                float(np.min(future_x[1:])) - support_radius,
                float(np.min(future_y[1:])) - support_radius,
                float(np.max(future_x[1:])) + support_radius,
                float(np.max(future_y[1:])) + support_radius,
                window,
                sample_shape,
            )
            if prediction_bounds is not None:
                row1, row2, col1, col2 = prediction_bounds
                local_x = xx[row1:row2, col1:col2]
                local_y = yy[row1:row2, col1:col2]
                local_inside = inside[row1:row2, col1:col2]
                dx = local_x[None, :, :] - future_x[1:, None, None]
                dy = local_y[None, :, :] - future_y[1:, None, None]
                distances = np.sqrt(dx * dx + dy * dy)
                falloff = np.clip(
                    1.0 - np.maximum(0.0, distances - collision_radius) / halo_width,
                    0.0,
                    1.0,
                )
                contributions = (
                    np.float32(0.5)
                    * falloff
                    * time_weights[1:, None, None]
                    * local_inside[None, :, :]
                ).astype(np.float32)
                combined_contribution = 1.0 - np.prod(1.0 - contributions, axis=0)
                target = prediction_cost[row1:row2, col1:col2]
                target[:] = combine_soft_cost(target, combined_contribution)

    wall_cost = environment_pccm_cost(
        xx,
        yy,
        inside,
        field_w,
        field_h,
        wall_margin,
        upper_field_threshold,
        upper_field_cost,
    )

    hard_collision[~inside] = 0.0
    return current_cost, prediction_cost, wall_cost, hard_collision


# Select the PCCM sampler while keeping both implementations directly testable.
def pccm_sample_components(
    bullets: list[BulletState],
    player_radius: float,
    window: tuple[int, int, int, int],
    sample_shape: tuple[int, int],
    field_w: int,
    field_h: int,
    prediction_frames: int,
    halo_width: float,
    wall_margin: float,
    fps: float = 60.0,
    implementation: str = "reference",
    upper_field_threshold: float = 0.70,
    upper_field_cost: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if implementation == "auto":
        # Broadcasting wins on the tiny blue grid; ROI wins on 32x32 samples.
        implementation = "roi" if sample_shape[0] * sample_shape[1] >= 1024 else "reference"
    implementations = {
        "reference": pccm_sample_components_reference,
        "roi": pccm_sample_components_roi,
    }
    sampler = implementations.get(implementation)
    if sampler is None:
        raise ValueError(f"Unknown PCCM implementation: {implementation}.")
    return sampler(
        bullets,
        player_radius,
        window,
        sample_shape,
        field_w,
        field_h,
        prediction_frames,
        halo_width,
        wall_margin,
        fps,
        upper_field_threshold,
        upper_field_cost,
    )


# Project one continuous PCCM rule directly into a target observation grid.
def projected_pccm(
    bullets: list[BulletState],
    player_radius: float,
    window: tuple[int, int, int, int],
    output_shape: tuple[int, int],
    sample_shape: tuple[int, int],
    field_w: int,
    field_h: int,
    prediction_frames: int,
    halo_width: float,
    wall_margin: float,
    soft_cap: float,
    implementation: str = "reference",
    upper_field_threshold: float = 0.70,
    upper_field_cost: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current, prediction, wall, hard = pccm_sample_components(
        bullets,
        player_radius,
        window,
        sample_shape,
        field_w,
        field_h,
        prediction_frames,
        halo_width,
        wall_margin,
        implementation=implementation,
        upper_field_threshold=upper_field_threshold,
        upper_field_cost=upper_field_cost,
    )
    if sample_shape[0] > output_shape[0] or sample_shape[1] > output_shape[1]:
        current = top_fraction_pool(current, output_shape)
        prediction = top_fraction_pool(prediction, output_shape)
        wall = average_pool(wall, output_shape)
        hard = np.max(
            hard.reshape(
                output_shape[0],
                sample_shape[0] // output_shape[0],
                output_shape[1],
                sample_shape[1] // output_shape[1],
            ),
            axis=(1, 3),
        )
    elif sample_shape != output_shape:
        current = bilinear_resize(current, output_shape)
        prediction = bilinear_resize(prediction, output_shape)
        wall = bilinear_resize(wall, output_shape)
        hard = np.zeros(output_shape, dtype=np.float32)

    soft = combine_soft_cost(combine_soft_cost(current, prediction), wall)
    final = np.clip(soft, 0.0, soft_cap).astype(np.float32)
    final[hard > 0.0] = 1.0
    return current.astype(np.float32), prediction.astype(np.float32), wall.astype(np.float32), final


def red_occupancy_map(
    bullets: list[BulletState],
    window: tuple[int, int, int, int],
    map_shape: tuple[int, int],
) -> np.ndarray:
    # Build the local red-zone collision occupancy map.
    x1, y1, x2, y2 = window
    rows, cols = map_shape
    occupancy = np.zeros((rows, cols), dtype=np.float32)
    win_w = max(1, x2 - x1)
    win_h = max(1, y2 - y1)

    for bullet in bullets:
        bx1 = bullet.x - bullet.radius
        bx2 = bullet.x + bullet.radius
        by1 = bullet.y - bullet.radius
        by2 = bullet.y + bullet.radius
        if bx2 < x1 or bx1 >= x2 or by2 < y1 or by1 >= y2:
            continue

        col1 = max(0, int(np.floor((bx1 - x1) / win_w * cols)))
        col2 = min(cols, int(np.ceil((bx2 - x1) / win_w * cols)))
        row1 = max(0, int(np.floor((by1 - y1) / win_h * rows)))
        row2 = min(rows, int(np.ceil((by2 - y1) / win_h * rows)))
        if col1 >= col2 or row1 >= row2:
            continue

        yy, xx = np.ogrid[row1:row2, col1:col2]
        cell_x = x1 + (xx + 0.5) * win_w / cols
        cell_y = y1 + (yy + 0.5) * win_h / rows
        mask = (cell_x - bullet.x) ** 2 + (cell_y - bullet.y) ** 2 <= bullet.radius ** 2
        occupancy[row1:row2, col1:col2][mask] = 1.0

    return occupancy


class ObservationBuilder:
    # Store observation settings for later builds.
    def __init__(self, config: ObservationConfig | None = None):
        self.config = config or ObservationConfig()
        if self.config.pccm_prediction_frames < 1:
            raise ValueError("PCCM prediction frames must be positive.")
        if self.config.pccm_halo_width <= 0.0:
            raise ValueError("PCCM halo width must be positive.")
        if not 0.0 < self.config.pccm_wall_margin <= 0.5:
            raise ValueError("PCCM wall margin must be in (0, 0.5].")
        if not 0.0 < self.config.pccm_upper_field_threshold <= 1.0:
            raise ValueError("PCCM upper-field threshold must be in (0, 1].")
        if not 0.0 <= self.config.pccm_upper_field_cost < self.config.pccm_soft_cap:
            raise ValueError("PCCM upper-field cost must be in [0, soft cap).")
        if not 0.0 < self.config.pccm_soft_cap < 1.0:
            raise ValueError("PCCM soft cap must be in (0, 1).")
        if self.config.pccm_implementation not in {"auto", "reference", "roi"}:
            raise ValueError(f"Unknown PCCM implementation: {self.config.pccm_implementation}.")

    # Build the full fixed-size observation dictionary.
    def build(self, bullets: list[BulletState], player: PlayerState) -> dict[str, np.ndarray]:
        cfg = self.config
        full_window = (0, 0, cfg.playfield_width, cfg.playfield_height)
        yellow_window = centered_window(player.x, player.y, cfg.yellow_size[0], cfg.yellow_size[1])
        red_window = centered_window(player.x, player.y, cfg.red_size[0], cfg.red_size[1])

        collision_bullets = [
            BulletState(
                x=bullet.x,
                y=bullet.y,
                radius=bullet.radius + player.radius,
                vx=bullet.vx,
                vy=bullet.vy,
            )
            for bullet in bullets
        ]
        occupancy = make_occupancy_map(cfg.playfield_width, cfg.playfield_height, collision_bullets)
        integral = make_integral_image(occupancy)
        yellow_valid = valid_area_grid(yellow_window, cfg.yellow_grid, cfg.playfield_width, cfg.playfield_height)
        red_valid = valid_area_grid(red_window, cfg.red_map, cfg.playfield_width, cfg.playfield_height)
        player_x = np.clip(player.x / cfg.playfield_width, 0.0, 1.0)
        player_y = np.clip(player.y / cfg.playfield_height, 0.0, 1.0)
        left_margin = player_x
        right_margin = 1.0 - player_x
        top_margin = player_y
        bottom_margin = 1.0 - player_y

        observation = {
            "blue_density": density_grid(integral, full_window, cfg.blue_grid),
            "blue_valid": np.ones(cfg.blue_grid, dtype=np.float32),
            "yellow_density": density_grid(integral, yellow_window, cfg.yellow_grid),
            "yellow_valid": yellow_valid,
            "red_valid": red_valid,
            "player_features": np.array(
                [
                    player_x,
                    player_y,
                    player.radius / max(cfg.playfield_width, cfg.playfield_height),
                    player.previous_action / 8.0,
                    left_margin,
                    right_margin,
                    top_margin,
                    bottom_margin,
                ],
                dtype=np.float32,
            ),
            "_occupancy_map": occupancy,
            "_blue_window": np.array(full_window, dtype=np.int32),
            "_yellow_window": np.array(yellow_window, dtype=np.int32),
            "_red_window": np.array(red_window, dtype=np.int32),
        }

        red_occ = red_occupancy_map(
            collision_bullets,
            red_window,
            cfg.red_map,
        )
        blue_components = projected_pccm(
            bullets,
            player.radius,
            full_window,
            cfg.blue_grid,
            (16, 16),
            cfg.playfield_width,
            cfg.playfield_height,
            cfg.pccm_prediction_frames,
            cfg.pccm_halo_width,
            cfg.pccm_wall_margin,
            cfg.pccm_soft_cap,
            implementation=cfg.pccm_implementation,
            upper_field_threshold=cfg.pccm_upper_field_threshold,
            upper_field_cost=cfg.pccm_upper_field_cost,
        )
        yellow_components = projected_pccm(
            bullets,
            player.radius,
            yellow_window,
            cfg.yellow_grid,
            (32, 32),
            cfg.playfield_width,
            cfg.playfield_height,
            cfg.pccm_prediction_frames,
            cfg.pccm_halo_width,
            cfg.pccm_wall_margin,
            cfg.pccm_soft_cap,
            implementation=cfg.pccm_implementation,
            upper_field_threshold=cfg.pccm_upper_field_threshold,
            upper_field_cost=cfg.pccm_upper_field_cost,
        )
        red_components = projected_pccm(
            bullets,
            player.radius,
            red_window,
            cfg.red_map,
            (32, 32),
            cfg.playfield_width,
            cfg.playfield_height,
            cfg.pccm_prediction_frames,
            cfg.pccm_halo_width,
            cfg.pccm_wall_margin,
            cfg.pccm_soft_cap,
            implementation=cfg.pccm_implementation,
            upper_field_threshold=cfg.pccm_upper_field_threshold,
            upper_field_cost=cfg.pccm_upper_field_cost,
        )
        observation.update(
            {
                "blue_pccm": blue_components[3],
                "yellow_pccm": yellow_components[3],
                "red_occupancy": red_occ,
                "red_pccm": red_components[3],
            }
        )
        observation["red_pccm"][red_occ > 0.0] = 1.0
        return observation
