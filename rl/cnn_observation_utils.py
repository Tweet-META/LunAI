from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np


class CNNObservation(TypedDict):
    red: np.ndarray
    yellow: np.ndarray
    blue: np.ndarray
    player: np.ndarray


MAP_HISTORY_KEYS = {
    "red": ("red_occupancy", "red_pccm", "red_valid"),
    "yellow": ("yellow_density", "yellow_pccm", "yellow_valid"),
    "blue": ("blue_density", "blue_pccm", "blue_valid"),
}


# Return the PCCM map keys stored in frame history.
def cnn_map_keys() -> tuple[str, ...]:
    return tuple(
        key
        for map_name in ("blue", "yellow", "red")
        for key in MAP_HISTORY_KEYS[map_name]
    )


# Stack one map type from oldest frame to current frame along its channel axis.
def stack_map_history(
    map_history: Sequence[dict[str, np.ndarray]],
    map_name: str,
) -> np.ndarray:
    if not map_history:
        raise ValueError("Map history cannot be empty.")
    keys = MAP_HISTORY_KEYS[map_name]
    layers = [
        np.asarray(frame[key], dtype=np.float32)
        for frame in map_history
        for key in keys
    ]
    return np.stack(layers, axis=0)


# Convert one current observation and its map history into CNN-friendly arrays.
def cnn_observation(
    observation: dict[str, np.ndarray],
    map_history: Sequence[dict[str, np.ndarray]] | None = None,
) -> CNNObservation:
    if map_history is None:
        map_history = (observation,)

    red = stack_map_history(map_history, "red")
    yellow = stack_map_history(map_history, "yellow")
    blue = stack_map_history(map_history, "blue")
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


# Return the CNN input shapes from one observation and its map history.
def cnn_observation_shapes(
    observation: dict[str, np.ndarray],
    map_history: Sequence[dict[str, np.ndarray]] | None = None,
) -> dict[str, tuple[int, ...]]:
    state = cnn_observation(observation, map_history)
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
