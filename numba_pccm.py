"""Optional single-threaded, fused CPU kernel for reference PCCM samples."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError as exc:
    raise ImportError(
        "The numba PCCM backend needs its optional dependencies: "
        "python -m pip install -r requirements-numba.txt"
    ) from exc


@njit(cache=True, nogil=True, fastmath=False)
def _sample_bullets(xx, yy, inside, future_x, future_y, radii, weights, halo):
    """Accumulate float32 products without bullet/time/grid temporary arrays."""
    current = np.zeros(xx.shape, dtype=np.float32)
    prediction = np.zeros(xx.shape, dtype=np.float32)
    hard = np.zeros(xx.shape, dtype=np.float32)
    one = np.float32(1.0)
    half = np.float32(0.5)
    zero = np.float32(0.0)
    for row in range(xx.shape[0]):
        for col in range(xx.shape[1]):
            if not inside[row, col]:
                continue
            current_product = one
            prediction_product = one
            for bullet in range(radii.size):
                radius = radii[bullet]
                support = radius + halo
                support_squared = support * support
                for frame in range(weights.size):
                    dx = xx[row, col] - future_x[bullet, frame]
                    dy = yy[row, col] - future_y[bullet, frame]
                    distance_squared = dx * dx + dy * dy
                    # Outside the halo the contribution is zero. Keep the full
                    # grid traversal; no per-bullet ROI slices or allocations.
                    if distance_squared > support_squared:
                        continue
                    distance = np.sqrt(distance_squared)
                    if frame == 0 and distance <= radius:
                        hard[row, col] = one
                    falloff = min(one, max(zero, one - max(zero, distance - radius) / halo))
                    contribution = half * falloff * weights[frame]
                    if frame == 0:
                        current_product *= one - contribution
                    else:
                        prediction_product *= one - contribution
            current[row, col] = one - current_product
            prediction[row, col] = one - prediction_product
    return current, prediction, hard


def pccm_sample_components_numba(
    bullets,
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
    from observation_builder import environment_pccm_cost, grid_cell_centers

    xx, yy = grid_cell_centers(window, sample_shape)
    inside = (xx >= 0.0) & (xx < field_w) & (yy >= 0.0) & (yy < field_h)
    x1, y1, x2, y2 = window
    horizon_seconds = prediction_frames / fps
    relevant = []
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
            relevant.append((
                bullet.x, bullet.y, bullet.vx, bullet.vy,
                max(1.0, bullet.radius + player_radius),
            ))

    if relevant:
        state = np.asarray(relevant, dtype=np.float32)
        # Match reference float32 prediction/weight rounding, using only small
        # bullet-by-time arrays shared by every sample point.
        times = np.arange(prediction_frames + 1, dtype=np.float32) / fps
        future_x = state[:, 0, None] + state[:, 2, None] * times[None, :]
        future_y = state[:, 1, None] + state[:, 3, None] * times[None, :]
        weights = 1.0 - np.arange(prediction_frames + 1, dtype=np.float32) / (prediction_frames + 1.0)
        current, prediction, hard = _sample_bullets(
            xx, yy, inside, future_x, future_y,
            np.ascontiguousarray(state[:, 4]), weights, np.float32(halo_width),
        )
    else:
        current = np.zeros(sample_shape, dtype=np.float32)
        prediction = np.zeros_like(current)
        hard = np.zeros_like(current)

    wall = environment_pccm_cost(
        xx, yy, inside, field_w, field_h, wall_margin,
        upper_field_threshold, upper_field_cost,
    )
    return current, prediction, wall, hard
