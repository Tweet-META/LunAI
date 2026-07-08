from __future__ import annotations

from typing import TypedDict

import numpy as np


class CNNObservation(TypedDict):
    red: np.ndarray
    yellow: np.ndarray
    blue: np.ndarray
    player: np.ndarray


# Convert one observation dictionary into CNN-friendly arrays.
def cnn_observation(observation: dict[str, np.ndarray]) -> CNNObservation:
    red = np.stack(
        [
            np.asarray(observation["red_occupancy"], dtype=np.float32),
            np.asarray(observation["red_vx"], dtype=np.float32),
            np.asarray(observation["red_vy"], dtype=np.float32),
            np.asarray(observation["red_speed"], dtype=np.float32),
        ],
        axis=0,
    )
    yellow = np.stack(
        [
            np.asarray(observation["yellow_density"], dtype=np.float32),
            np.asarray(observation["yellow_speed"], dtype=np.float32),
        ],
        axis=0,
    )
    blue = np.stack(
        [
            np.asarray(observation["blue_density"], dtype=np.float32),
            np.asarray(observation["blue_speed"], dtype=np.float32),
        ],
        axis=0,
    )
    player = np.asarray(observation["player_features"], dtype=np.float32)
    state: CNNObservation = {
        "red": red,
        "yellow": yellow,
        "blue": blue,
        "player": player,
    }
    validate_cnn_observation(state)
    return state


# Stack a list of CNN observations into one batch.
def stack_cnn_observations(states: list[CNNObservation]) -> CNNObservation:
    if not states:
        raise ValueError("Cannot stack an empty state list.")
    batch: CNNObservation = {
        "red": np.stack([state["red"] for state in states]).astype(np.float32),
        "yellow": np.stack([state["yellow"] for state in states]).astype(np.float32),
        "blue": np.stack([state["blue"] for state in states]).astype(np.float32),
        "player": np.stack([state["player"] for state in states]).astype(np.float32),
    }
    validate_cnn_observation(batch, batched=True)
    return batch


# Return the CNN input shapes from one observation.
def cnn_observation_shapes(observation: dict[str, np.ndarray]) -> dict[str, tuple[int, ...]]:
    state = cnn_observation(observation)
    return {key: tuple(value.shape) for key, value in state.items()}


# Check that CNN inputs have the expected rank and finite values.
def validate_cnn_observation(state: CNNObservation, batched: bool = False) -> None:
    expected_dims = 4 if batched else 3
    if state["red"].ndim != expected_dims:
        raise ValueError(f"Red state must have {expected_dims} dims, got {state['red'].shape}.")
    if state["yellow"].ndim != expected_dims:
        raise ValueError(f"Yellow state must have {expected_dims} dims, got {state['yellow'].shape}.")
    if state["blue"].ndim != expected_dims:
        raise ValueError(f"Blue state must have {expected_dims} dims, got {state['blue'].shape}.")

    player_dims = 2 if batched else 1
    if state["player"].ndim != player_dims:
        raise ValueError(f"Player state must have {player_dims} dims, got {state['player'].shape}.")

    for key, value in state.items():
        if not np.isfinite(value).all():
            raise ValueError(f"CNN observation contains NaN or inf in {key}.")
