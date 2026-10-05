"""Single-threaded rasterizers preserving the reference circular occupancy."""

from __future__ import annotations

import numpy as np
from numba import njit


def _pack_circles(bullets):
    # The NumPy reference evaluates circle geometry in float64. Keep both that
    # precision and Python's radius**2 result at the exact hitbox boundary.
    return np.asarray([
        (bullet.x, bullet.y, bullet.radius, bullet.radius ** 2)
        for bullet in bullets
    ], dtype=np.float64).reshape(-1, 4)


@njit(cache=True, nogil=True, fastmath=False)
def _rasterize_world(width, height, circles):
    occupancy = np.zeros((height, width), dtype=np.float32)
    for index in range(circles.shape[0]):
        bx, by, radius, radius_squared = circles[index]
        extent = max(1, int(np.ceil(radius)))
        cx, cy = int(round(bx)), int(round(by))
        x1, x2 = max(0, cx - extent), min(width, cx + extent + 1)
        y1, y2 = max(0, cy - extent), min(height, cy + extent + 1)
        for row in range(y1, y2):
            dy = row - by
            dy_squared = dy * dy
            for col in range(x1, x2):
                dx = col - bx
                if dx * dx + dy_squared <= radius_squared:
                    occupancy[row, col] = np.float32(1.0)
    return occupancy


@njit(cache=True, nogil=True, fastmath=False)
def _rasterize_red(window, shape, circles):
    x1, y1, x2, y2 = window
    rows, cols = shape
    occupancy = np.zeros((rows, cols), dtype=np.float32)
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    for index in range(circles.shape[0]):
        bx, by, radius, radius_squared = circles[index]
        bx1, bx2 = bx - radius, bx + radius
        by1, by2 = by - radius, by + radius
        if bx2 < x1 or bx1 >= x2 or by2 < y1 or by1 >= y2:
            continue
        col1 = max(0, int(np.floor((bx1 - x1) / width * cols)))
        col2 = min(cols, int(np.ceil((bx2 - x1) / width * cols)))
        row1 = max(0, int(np.floor((by1 - y1) / height * rows)))
        row2 = min(rows, int(np.ceil((by2 - y1) / height * rows)))
        for row in range(row1, row2):
            cell_y = y1 + (row + 0.5) * height / rows
            dy = cell_y - by
            dy_squared = dy * dy
            for col in range(col1, col2):
                cell_x = x1 + (col + 0.5) * width / cols
                dx = cell_x - bx
                if dx * dx + dy_squared <= radius_squared:
                    occupancy[row, col] = np.float32(1.0)
    return occupancy


def make_occupancy_map_numba(width, height, bullets):
    return _rasterize_world(width, height, _pack_circles(bullets))


def red_occupancy_map_numba(bullets, window, map_shape):
    return _rasterize_red(window, map_shape, _pack_circles(bullets))
