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
    speed: float


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
    blue_grid: tuple[int, int] = (6, 6)
    yellow_size: tuple[int, int] = (320, 320)
    yellow_grid: tuple[int, int] = (16, 16)
    red_size: tuple[int, int] = (128, 128)
    red_map: tuple[int, int] = (64, 64)
    max_speed: float = 500.0


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


# Read the occupied area inside one rectangle from an integral image.
def rectangle_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    return float(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])


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


def average_speed_grid(
    bullets: list[BulletState],
    window: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    max_speed: float,
) -> np.ndarray:
    # Compute average bullet speed for each grid cell.
    x1, y1, x2, y2 = window
    rows, cols = grid_shape
    speed_sum = np.zeros((rows, cols), dtype=np.float32)
    counts = np.zeros((rows, cols), dtype=np.float32)
    win_w = max(1, x2 - x1)
    win_h = max(1, y2 - y1)

    for bullet in bullets:
        if not (x1 <= bullet.x < x2 and y1 <= bullet.y < y2):
            continue
        col = min(cols - 1, int((bullet.x - x1) / win_w * cols))
        row = min(rows - 1, int((bullet.y - y1) / win_h * rows))
        speed_sum[row, col] += bullet.speed
        counts[row, col] += 1.0

    out = np.divide(speed_sum, counts, out=np.zeros_like(speed_sum), where=counts > 0)
    return np.clip(out / max_speed, 0.0, 1.0)


def red_local_maps(
    bullets: list[BulletState],
    window: tuple[int, int, int, int],
    map_shape: tuple[int, int],
    max_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Build local red-zone occupancy and motion maps.
    x1, y1, x2, y2 = window
    rows, cols = map_shape
    occupancy = np.zeros((rows, cols), dtype=np.float32)
    vx_map = np.zeros((rows, cols), dtype=np.float32)
    vy_map = np.zeros((rows, cols), dtype=np.float32)
    speed_map = np.zeros((rows, cols), dtype=np.float32)
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
        target = speed_map[row1:row2, col1:col2]
        update = mask & ((bullet.speed / max_speed) >= target)

        occupancy[row1:row2, col1:col2][mask] = 1.0
        target[update] = min(1.0, bullet.speed / max_speed)
        vx_map[row1:row2, col1:col2][update] = np.clip(bullet.vx / max_speed, -1.0, 1.0)
        vy_map[row1:row2, col1:col2][update] = np.clip(bullet.vy / max_speed, -1.0, 1.0)

    return occupancy, vx_map, vy_map, speed_map


class ObservationBuilder:
    # Store observation settings for later builds.
    def __init__(self, config: ObservationConfig | None = None):
        self.config = config or ObservationConfig()

    # Build the full fixed-size observation dictionary.
    def build(self, bullets: list[BulletState], player: PlayerState) -> dict[str, np.ndarray]:
        cfg = self.config
        full_window = (0, 0, cfg.playfield_width, cfg.playfield_height)
        yellow_window = centered_window(player.x, player.y, cfg.yellow_size[0], cfg.yellow_size[1])
        red_window = centered_window(player.x, player.y, cfg.red_size[0], cfg.red_size[1])

        occupancy = make_occupancy_map(cfg.playfield_width, cfg.playfield_height, bullets)
        integral = make_integral_image(occupancy)
        red_occ, red_vx, red_vy, red_speed = red_local_maps(bullets, red_window, cfg.red_map, cfg.max_speed)
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
            "blue_speed": average_speed_grid(bullets, full_window, cfg.blue_grid, cfg.max_speed),
            "yellow_density": density_grid(integral, yellow_window, cfg.yellow_grid),
            "yellow_speed": average_speed_grid(bullets, yellow_window, cfg.yellow_grid, cfg.max_speed),
            "yellow_valid": yellow_valid,
            "red_occupancy": red_occ,
            "red_vx": red_vx,
            "red_vy": red_vy,
            "red_speed": red_speed,
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
        return observation
