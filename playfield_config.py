"""Spatial settings for the TH06-sized Pygame experiment."""

LEGACY_PLAYFIELD_SIZE = (600, 700)
PLAYFIELD_SIZE = (384, 448)
WORLD_SCALE = PLAYFIELD_SIZE[0] / LEGACY_PLAYFIELD_SIZE[0]

YELLOW_SIZE = (204, 204)
RED_SIZE = (64, 64)
PCCM_HALO_WIDTH = 20.0


def scale_distance(value: float) -> float:
    return value * WORLD_SCALE
