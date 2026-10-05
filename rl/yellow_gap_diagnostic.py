"""Geometry for a sequence of single-row walls with nearby random gaps."""

import random


FIELD_WIDTH = 384
WALL_Y = 230.0
WALL_SPEED = 162.0  # pixels/second, or 2.7 pixels/game frame
WAVE_INTERVAL_FRAMES = 60
WAVE_COUNT = 30
BULLET_SPACING = 8
GAP_HALF_WIDTH = 20.0

# Keep every opening in the middle of the playfield. Consecutive openings
# move enough to prevent staying in one lane, but remain in Yellow's view.
GAP_CENTERS = tuple(float(x) for x in range(112, 273, 8))
MIN_GAP_SHIFT = 32.0
MAX_GAP_SHIFT = 80.0


def choose_gap_centers(count: int) -> tuple[float, ...]:
    """Choose reproducible openings with bounded changes between waves."""
    if count < 0:
        raise ValueError("Wave count cannot be negative.")
    if count == 0:
        return ()

    centers = [random.choice(GAP_CENTERS)]
    for _ in range(1, count):
        previous = centers[-1]
        nearby = [
            center for center in GAP_CENTERS
            if MIN_GAP_SHIFT <= abs(center - previous) <= MAX_GAP_SHIFT
        ]
        centers.append(random.choice(nearby))
    return tuple(centers)


def wall_points(gap_center: float, wall_y: float = WALL_Y) -> tuple[tuple[float, float], ...]:
    """Return one descending row with a single passable opening."""
    if gap_center not in GAP_CENTERS:
        raise ValueError(f"Gap center must be one of {GAP_CENTERS}: {gap_center}")
    return tuple(
        (float(x), float(wall_y))
        for x in range(BULLET_SPACING // 2, FIELD_WIDTH, BULLET_SPACING)
        if abs(x - gap_center) > GAP_HALF_WIDTH
    )
