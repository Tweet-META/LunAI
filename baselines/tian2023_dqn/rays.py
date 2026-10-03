from __future__ import annotations

import numpy as np


def cast_rays(bullets, player, width: float, height: float, ray_count: int = 36) -> np.ndarray:
    """Nearest circle intersected by each ray, not nearest angular sector.

    Start upwards and proceed clockwise in screen coordinates. Rays end at the
    playfield boundary; walls are not detections. Features are center dx/W,
    dy/H, vx/W, vy/H (velocity in pixels/second), detected. Empty rays are zero.
    """
    if width <= 0 or height <= 0 or ray_count < 1:
        raise ValueError("Positive field dimensions and ray count required.")
    result = np.zeros((ray_count, 5), dtype=np.float32)
    if not bullets:
        return result.ravel()
    circles = np.asarray([(b.x, b.y, b.radius, b.vx, b.vy) for b in bullets], dtype=np.float64)
    if not np.isfinite(circles).all() or np.any(circles[:, 2] < 0):
        raise ValueError("Circle data must be finite with nonnegative radii.")
    centers = circles[:, :2] - (player.x, player.y)
    angles = np.arange(ray_count) * (2 * np.pi / ray_count)
    directions = np.column_stack((np.sin(angles), -np.cos(angles)))
    along = directions @ centers.T
    perpendicular_sq = np.maximum(0, np.sum(centers**2, axis=1)[None, :] - along**2)
    discriminant = circles[None, :, 2] ** 2 - perpendicular_sq
    half_chord = np.sqrt(np.maximum(0, discriminant))
    entry = np.maximum(0, along - half_chord)
    exit_distance = along + half_chord
    # End at the first boundary along this direction (no off-screen information).
    bounds = np.full((ray_count, 2), np.inf)
    for axis, (position, size) in enumerate(((player.x, width), (player.y, height))):
        component = directions[:, axis]
        positive, negative = component > 1e-12, component < -1e-12
        bounds[positive, axis] = (size - position) / component[positive]
        bounds[negative, axis] = -position / component[negative]
    maximum = np.maximum(0, bounds.min(axis=1))
    hit = (discriminant >= -1e-9) & (exit_distance >= 0) & (entry <= maximum[:, None])
    distances = np.where(hit, entry, np.inf)
    nearest = distances.argmin(axis=1)
    detected = np.isfinite(distances[np.arange(ray_count), nearest])
    chosen = nearest[detected]
    result[detected, :2] = centers[chosen] / (width, height)
    result[detected, 2:4] = circles[chosen, 3:5] / (width, height)
    result[detected, 4] = 1
    return result.ravel()
