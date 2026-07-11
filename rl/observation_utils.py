from __future__ import annotations

import numpy as np


OBSERVATION_KEYS = (
    "blue_density",
    "blue_speed",
    "yellow_density",
    "yellow_speed",
    "yellow_valid",
    "red_occupancy",
    "red_vx",
    "red_vy",
    "red_speed",
    "red_valid",
    "player_features",
)


# Convert an observation dictionary into one flat float vector.
def flatten_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    parts = [np.asarray(observation[key], dtype=np.float32).ravel() for key in OBSERVATION_KEYS]
    return np.concatenate(parts).astype(np.float32)


# Return the flat vector size for one observation dictionary.
def observation_dim(observation: dict[str, np.ndarray]) -> int:
    return int(flatten_observation(observation).shape[0])


# Check that a flat observation vector is finite.
def validate_flat_observation(vector: np.ndarray) -> None:
    if vector.ndim != 1:
        raise ValueError(f"Observation vector must be 1D, got shape {vector.shape}.")
    if not np.isfinite(vector).all():
        raise ValueError("Observation vector contains NaN or inf.")
