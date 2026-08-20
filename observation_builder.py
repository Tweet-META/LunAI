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
    half_width: float = 0.0
    half_height: float = 0.0


@dataclass(frozen=True)
class PlayerState:
    x: float
    y: float
    radius: float
    previous_action: int = 0
    half_width: float = 0.0
    half_height: float = 0.0


@dataclass(frozen=True)
class ObservationConfig:
    playfield_width: int = 600
    playfield_height: int = 700
    playable_bounds: tuple[float, float, float, float] | None = None
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


# Return whether one hazard uses an axis-aligned rectangle.
def is_aabb_hazard(hazard: BulletState) -> bool:
    return hazard.half_width > 0.0 and hazard.half_height > 0.0


# Expand one hazard by the player hitbox for point-based collision checks.
def expand_hazard_for_player(hazard: BulletState, player: PlayerState) -> BulletState:
    if is_aabb_hazard(hazard):
        player_half_width = player.half_width if player.half_width > 0.0 else player.radius
        player_half_height = player.half_height if player.half_height > 0.0 else player.radius
        half_width = hazard.half_width + player_half_width
        half_height = hazard.half_height + player_half_height
        return BulletState(
            x=hazard.x,
            y=hazard.y,
            radius=max(half_width, half_height),
            vx=hazard.vx,
            vy=hazard.vy,
            half_width=half_width,
            half_height=half_height,
        )
    return BulletState(
        x=hazard.x,
        y=hazard.y,
        radius=hazard.radius + player.radius,
        vx=hazard.vx,
        vy=hazard.vy,
    )


# Return a player-centered window that may extend outside the field.
def centered_window(center_x: float, center_y: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(round(center_x - width / 2))
    y1 = int(round(center_y - height / 2))
    return x1, y1, x1 + width, y1 + height


# Rasterize circular and rectangular collision hitboxes into a binary map.
def make_occupancy_map(width: int, height: int, bullets: list[BulletState]) -> np.ndarray:
    occupancy = np.zeros((height, width), dtype=np.float32)
    for bullet in bullets:
        half_width = bullet.half_width if is_aabb_hazard(bullet) else bullet.radius
        half_height = bullet.half_height if is_aabb_hazard(bullet) else bullet.radius
        x1 = max(0, int(np.floor(bullet.x - half_width)))
        x2 = min(width, int(np.ceil(bullet.x + half_width)) + 1)
        y1 = max(0, int(np.floor(bullet.y - half_height)))
        y2 = min(height, int(np.ceil(bullet.y + half_height)) + 1)
        if x1 >= x2 or y1 >= y2:
            continue

        yy, xx = np.ogrid[y1:y2, x1:x2]
        if is_aabb_hazard(bullet):
            mask = (np.abs(xx - bullet.x) <= bullet.half_width) & (
                np.abs(yy - bullet.y) <= bullet.half_height
            )
        else:
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
    playable_bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    x1, y1, x2, y2 = window
    rows, cols = grid_shape
    xs = np.linspace(x1, x2, cols + 1).round().astype(int)
    ys = np.linspace(y1, y2, rows + 1).round().astype(int)
    cell_widths = np.diff(xs)
    cell_heights = np.diff(ys)
    cell_area = np.maximum(1, np.outer(cell_heights, cell_widths))
    left, top, right, bottom = playable_bounds or (0.0, 0.0, float(field_w), float(field_h))
    clipped_xs = np.clip(xs, left, right)
    clipped_ys = np.clip(ys, top, bottom)
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
    playable_bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    left, top, right, bottom = playable_bounds or (0.0, 0.0, float(field_w), float(field_h))
    playable_width = right - left
    playable_height = bottom - top
    horizontal_margin = max(1.0, playable_width * wall_margin)
    vertical_margin = max(1.0, playable_height * wall_margin)
    environment_cost = np.zeros(xx.shape, dtype=np.float32)
    wall_distances = (
        (xx - left, horizontal_margin),
        (right - xx, horizontal_margin),
        (yy - top, vertical_margin),
        (bottom - yy, vertical_margin),
    )
    for distance, margin in wall_distances:
        contribution = np.where(
            inside,
            0.5 * np.clip(1.0 - distance / margin, 0.0, 1.0),
            0.0,
        ).astype(np.float32)
        environment_cost = combine_soft_cost(environment_cost, contribution)

    upper_boundary = top + max(1.0, playable_height * upper_field_threshold)
    upper_contribution = np.where(
        inside,
        upper_field_cost * np.clip(1.0 - (yy - top) / max(1.0, upper_boundary - top), 0.0, 1.0),
        0.0,
    ).astype(np.float32)
    return combine_soft_cost(environment_cost, upper_contribution)


# Build PCCM samples with full-grid NumPy broadcasting as the reference.
def pccm_sample_components(
    bullets: list[BulletState],
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
    playable_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xx, yy = grid_cell_centers(window, sample_shape)
    left, top, right, bottom = playable_bounds or (0.0, 0.0, float(field_w), float(field_h))
    inside = (xx >= left) & (xx <= right) & (yy >= top) & (yy <= bottom)
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
            half_width = bullet.half_width if is_aabb_hazard(bullet) else bullet.radius
            half_height = bullet.half_height if is_aabb_hazard(bullet) else bullet.radius
            padding_x = half_width + halo_width
            padding_y = half_height + halo_width
            if (
                max(bullet.x, future_x) + padding_x >= x1
                and min(bullet.x, future_x) - padding_x < x2
                and max(bullet.y, future_y) + padding_y >= y1
                and min(bullet.y, future_y) - padding_y < y2
            ):
                relevant_bullets.append(bullet)

        if relevant_bullets:
            bullet_x = np.asarray([bullet.x for bullet in relevant_bullets], dtype=np.float32)
            bullet_y = np.asarray([bullet.y for bullet in relevant_bullets], dtype=np.float32)
            velocity_x = np.asarray([bullet.vx for bullet in relevant_bullets], dtype=np.float32)
            velocity_y = np.asarray([bullet.vy for bullet in relevant_bullets], dtype=np.float32)
            radii = np.asarray([max(0.1, bullet.radius) for bullet in relevant_bullets], dtype=np.float32)
            half_widths = np.asarray(
                [bullet.half_width if is_aabb_hazard(bullet) else 0.0 for bullet in relevant_bullets],
                dtype=np.float32,
            )
            half_heights = np.asarray(
                [bullet.half_height if is_aabb_hazard(bullet) else 0.0 for bullet in relevant_bullets],
                dtype=np.float32,
            )
            aabb_mask = np.asarray([is_aabb_hazard(bullet) for bullet in relevant_bullets], dtype=bool)
            times = np.arange(prediction_frames + 1, dtype=np.float32) / fps
            future_x = bullet_x[:, None] + velocity_x[:, None] * times[None, :]
            future_y = bullet_y[:, None] + velocity_y[:, None] * times[None, :]
            dx = xx[None, None, :, :] - future_x[:, :, None, None]
            dy = yy[None, None, :, :] - future_y[:, :, None, None]
            distances = np.sqrt(dx * dx + dy * dy)
            circle_outside = np.maximum(0.0, distances - radii[:, None, None, None])
            box_dx = np.maximum(0.0, np.abs(dx) - half_widths[:, None, None, None])
            box_dy = np.maximum(0.0, np.abs(dy) - half_heights[:, None, None, None])
            box_outside = np.sqrt(box_dx * box_dx + box_dy * box_dy)
            outside_distance = np.where(
                aabb_mask[:, None, None, None],
                box_outside,
                circle_outside,
            )
            falloff = np.clip(
                1.0 - outside_distance / halo_width,
                0.0,
                1.0,
            )
            time_weights = 1.0 - np.arange(prediction_frames + 1, dtype=np.float32) / (prediction_frames + 1.0)
            contributions = 0.5 * falloff * time_weights[None, :, None, None]
            contributions *= inside[None, None, :, :]
            current_cost = 1.0 - np.prod(1.0 - contributions[:, 0], axis=0)
            prediction_cost = 1.0 - np.prod(1.0 - contributions[:, 1:], axis=(0, 1))
            circle_hard = distances[:, 0] <= radii[:, None, None]
            box_hard = (np.abs(dx[:, 0]) <= half_widths[:, None, None]) & (
                np.abs(dy[:, 0]) <= half_heights[:, None, None]
            )
            hard_collision = np.any(
                np.where(aabb_mask[:, None, None], box_hard, circle_hard),
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
        playable_bounds,
    )

    hard_collision[~inside] = 0.0
    return current_cost, prediction_cost, wall_cost, hard_collision


# Project one continuous PCCM rule directly into a target observation grid.
def projected_pccm(
    bullets: list[BulletState],
    window: tuple[int, int, int, int],
    output_shape: tuple[int, int],
    sample_shape: tuple[int, int],
    field_w: int,
    field_h: int,
    prediction_frames: int,
    halo_width: float,
    wall_margin: float,
    soft_cap: float,
    upper_field_threshold: float = 0.70,
    upper_field_cost: float = 0.30,
    playable_bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    current, prediction, wall, hard = pccm_sample_components(
        bullets,
        window,
        sample_shape,
        field_w,
        field_h,
        prediction_frames,
        halo_width,
        wall_margin,
        upper_field_threshold=upper_field_threshold,
        upper_field_cost=upper_field_cost,
        playable_bounds=playable_bounds,
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
    return final


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
        half_width = bullet.half_width if is_aabb_hazard(bullet) else bullet.radius
        half_height = bullet.half_height if is_aabb_hazard(bullet) else bullet.radius
        bx1 = bullet.x - half_width
        bx2 = bullet.x + half_width
        by1 = bullet.y - half_height
        by2 = bullet.y + half_height
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
        if is_aabb_hazard(bullet):
            mask = (np.abs(cell_x - bullet.x) <= bullet.half_width) & (
                np.abs(cell_y - bullet.y) <= bullet.half_height
            )
        else:
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
        if self.config.playable_bounds is not None:
            left, top, right, bottom = self.config.playable_bounds
            if not (
                0.0 <= left < right <= self.config.playfield_width
                and 0.0 <= top < bottom <= self.config.playfield_height
            ):
                raise ValueError("Playable bounds must stay inside the playfield.")

    # Build the full fixed-size observation dictionary.
    def build(self, bullets: list[BulletState], player: PlayerState) -> dict[str, np.ndarray]:
        cfg = self.config
        full_window = (0, 0, cfg.playfield_width, cfg.playfield_height)
        playable_bounds = cfg.playable_bounds or (
            0.0,
            0.0,
            float(cfg.playfield_width),
            float(cfg.playfield_height),
        )
        yellow_window = centered_window(player.x, player.y, cfg.yellow_size[0], cfg.yellow_size[1])
        red_window = centered_window(player.x, player.y, cfg.red_size[0], cfg.red_size[1])

        collision_bullets = [expand_hazard_for_player(bullet, player) for bullet in bullets]
        occupancy = make_occupancy_map(cfg.playfield_width, cfg.playfield_height, collision_bullets)
        integral = make_integral_image(occupancy)
        blue_valid = valid_area_grid(
            full_window,
            cfg.blue_grid,
            cfg.playfield_width,
            cfg.playfield_height,
            playable_bounds,
        )
        yellow_valid = valid_area_grid(
            yellow_window,
            cfg.yellow_grid,
            cfg.playfield_width,
            cfg.playfield_height,
            playable_bounds,
        )
        red_valid = valid_area_grid(
            red_window,
            cfg.red_map,
            cfg.playfield_width,
            cfg.playfield_height,
            playable_bounds,
        )
        player_x = np.clip(player.x / cfg.playfield_width, 0.0, 1.0)
        player_y = np.clip(player.y / cfg.playfield_height, 0.0, 1.0)
        left, top, right, bottom = playable_bounds
        left_margin = np.clip((player.x - left) / (right - left), 0.0, 1.0)
        right_margin = np.clip((right - player.x) / (right - left), 0.0, 1.0)
        top_margin = np.clip((player.y - top) / (bottom - top), 0.0, 1.0)
        bottom_margin = np.clip((bottom - player.y) / (bottom - top), 0.0, 1.0)

        observation = {
            "blue_density": density_grid(integral, full_window, cfg.blue_grid),
            "blue_valid": blue_valid,
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
        blue_pccm = projected_pccm(
            collision_bullets,
            full_window,
            cfg.blue_grid,
            (16, 16),
            cfg.playfield_width,
            cfg.playfield_height,
            cfg.pccm_prediction_frames,
            cfg.pccm_halo_width,
            cfg.pccm_wall_margin,
            cfg.pccm_soft_cap,
            upper_field_threshold=cfg.pccm_upper_field_threshold,
            upper_field_cost=cfg.pccm_upper_field_cost,
            playable_bounds=playable_bounds,
        )
        yellow_pccm = projected_pccm(
            collision_bullets,
            yellow_window,
            cfg.yellow_grid,
            (32, 32),
            cfg.playfield_width,
            cfg.playfield_height,
            cfg.pccm_prediction_frames,
            cfg.pccm_halo_width,
            cfg.pccm_wall_margin,
            cfg.pccm_soft_cap,
            upper_field_threshold=cfg.pccm_upper_field_threshold,
            upper_field_cost=cfg.pccm_upper_field_cost,
            playable_bounds=playable_bounds,
        )
        red_pccm = projected_pccm(
            collision_bullets,
            red_window,
            cfg.red_map,
            (32, 32),
            cfg.playfield_width,
            cfg.playfield_height,
            cfg.pccm_prediction_frames,
            cfg.pccm_halo_width,
            cfg.pccm_wall_margin,
            cfg.pccm_soft_cap,
            upper_field_threshold=cfg.pccm_upper_field_threshold,
            upper_field_cost=cfg.pccm_upper_field_cost,
            playable_bounds=playable_bounds,
        )
        red_pccm[red_valid <= 0.0] = 0.0
        red_pccm[(red_occ > 0.0) & (red_valid > 0.0)] = 1.0

        observation.update(
            {
                "blue_pccm": blue_pccm,
                "yellow_pccm": yellow_pccm,
                "red_occupancy": red_occ,
                "red_pccm": red_pccm,
                "_reward_red_pccm": red_pccm.copy(),
            }
        )
        return observation
