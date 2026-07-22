from __future__ import annotations

import numpy as np


SURVIVAL_REWARD = 0.1
COLLISION_PENALTY = 30.0
ACTION_CHANGE_PENALTY = 0.0
PCCM_STATE_PENALTY_WEIGHT = 0.1
BLOCKED_MOVEMENT_PENALTY_WEIGHT = 0.05
WALL_PROXIMITY_MARGIN = 0.12


# Convert the normalized player position into red-map cell coordinates.
def player_red_map_position(observation: dict[str, np.ndarray]) -> tuple[float, float]:
    red_map = observation["red_occupancy"]
    rows, cols = red_map.shape
    player_x = float(observation["player_features"][0])
    player_y = float(observation["player_features"][1])
    red_window = observation.get("_red_window")
    blue_window = observation.get("_blue_window")

    if red_window is None or blue_window is None:
        return (cols - 1) / 2.0, (rows - 1) / 2.0

    field_w = max(1.0, float(blue_window[2] - blue_window[0]))
    field_h = max(1.0, float(blue_window[3] - blue_window[1]))
    world_x = player_x * field_w
    world_y = player_y * field_h
    win_w = max(1.0, float(red_window[2] - red_window[0]))
    win_h = max(1.0, float(red_window[3] - red_window[1]))
    map_x = (world_x - float(red_window[0])) / win_w * cols
    map_y = (world_y - float(red_window[1])) / win_h * rows
    return float(np.clip(map_x, 0.0, cols - 1.0)), float(np.clip(map_y, 0.0, rows - 1.0))


# Sample the red PCCM at the player's continuous map position.
def local_pccm_cost(observation: dict[str, np.ndarray]) -> float:
    pccm = observation.get("_reward_red_pccm")
    if pccm is None:
        pccm = observation.get("red_pccm")
    if pccm is None:
        return 0.0
    x, y = player_red_map_position(observation)
    rows, cols = pccm.shape
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(cols - 1, x0 + 1)
    y1 = min(rows - 1, y0 + 1)
    tx = x - x0
    ty = y - y0
    top = (1.0 - tx) * pccm[y0, x0] + tx * pccm[y0, x1]
    bottom = (1.0 - tx) * pccm[y1, x0] + tx * pccm[y1, x1]
    value = (1.0 - ty) * top + ty * bottom
    return float(np.clip(value, 0.0, 1.0))


# Measure linear proximity to the four playfield walls.
def wall_proximity(observation: dict[str, np.ndarray], margin: float = WALL_PROXIMITY_MARGIN) -> float:
    if not 0.0 < margin <= 0.5:
        raise ValueError(f"Wall margin must be in (0, 0.5], got {margin}.")
    player_x = float(np.clip(observation["player_features"][0], 0.0, 1.0))
    player_y = float(np.clip(observation["player_features"][1], 0.0, 1.0))
    horizontal = max(0.0, 1.0 - min(player_x, 1.0 - player_x) / margin)
    vertical = max(0.0, 1.0 - min(player_y, 1.0 - player_y) / margin)
    return horizontal + vertical


# Measure the requested movement that was blocked by the playfield boundary.
def blocked_movement_ratio(
    requested_dx: float,
    requested_dy: float,
    actual_dx: float,
    actual_dy: float,
) -> float:
    requested_length = float(np.hypot(requested_dx, requested_dy))
    if requested_length <= 1e-8:
        return 0.0
    blocked_dx = requested_dx - actual_dx
    blocked_dy = requested_dy - actual_dy
    blocked_length = float(np.hypot(blocked_dx, blocked_dy))
    return float(np.clip(blocked_length / requested_length, 0.0, 1.0))


# Compute the total reward for one real game frame.
def compute_frame_reward(
    observation: dict[str, np.ndarray],
    action: int,
    previous_action: int,
    collided: bool,
    blocked_ratio: float = 0.0,
) -> float:
    if collided:
        return -COLLISION_PENALTY

    pccm_penalty = PCCM_STATE_PENALTY_WEIGHT * local_pccm_cost(observation)
    blocked_penalty = BLOCKED_MOVEMENT_PENALTY_WEIGHT * float(np.clip(blocked_ratio, 0.0, 1.0))
    # action_change_penalty = ACTION_CHANGE_PENALTY if action != previous_action else 0.0

    return (
        SURVIVAL_REWARD
        - pccm_penalty 
        - blocked_penalty 
        # - action_change_penalty
    )
