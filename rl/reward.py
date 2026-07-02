from __future__ import annotations

import numpy as np


NEAR_DANGER_SIGMA = 3.0
DISTANCE_SAFETY_SCALE = 0.01


# Convert an action id into simple x and y movement signs.
def action_components(action: int) -> tuple[int, int]:
    return {
        0: (0, 0),
        1: (0, -1),
        2: (0, 1),
        3: (-1, 0),
        4: (1, 0),
        5: (-1, -1),
        6: (1, -1),
        7: (-1, 1),
        8: (1, 1),
    }[action]


# Check whether the current action reverses the previous movement.
def is_reversal(action: int, previous_action: int) -> bool:
    action_x, action_y = action_components(action)
    previous_x, previous_y = action_components(previous_action)
    reverse_x = action_x != 0 and previous_x != 0 and action_x != previous_x
    reverse_y = action_y != 0 and previous_y != 0 and action_y != previous_y
    return reverse_x or reverse_y


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


# Estimate how dangerous the red map is near the player.
def nearby_bullet_danger(observation: dict[str, np.ndarray]) -> float:
    red_occupancy = observation["red_occupancy"]
    if float(np.max(red_occupancy)) <= 0.0:
        return 0.0

    red_speed = observation.get("red_speed", np.zeros_like(red_occupancy))
    player_x, player_y = player_red_map_position(observation)
    rows, cols = red_occupancy.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    dist_sq = (xx - player_x) ** 2 + (yy - player_y) ** 2
    weights = np.exp(-dist_sq / (2.0 * NEAR_DANGER_SIGMA ** 2))
    occupancy_danger = float(np.max(red_occupancy * weights))
    speed_danger = float(np.max(red_occupancy * red_speed * weights))
    return float(np.clip(0.8 * occupancy_danger + 0.2 * speed_danger, 0.0, 1.0))


# Give a small reward when bullets are not close to the player.
def distance_safety_reward(observation: dict[str, np.ndarray]) -> float:
    red_occupancy = observation["red_occupancy"]
    if float(np.max(red_occupancy)) <= 0.0:
        return DISTANCE_SAFETY_SCALE * 0.5

    danger = nearby_bullet_danger(observation)
    return DISTANCE_SAFETY_SCALE * (1.0 - danger)


# Compute a minimal baseline reward for one step.
def compute_reward(
    observation: dict[str, np.ndarray],
    action: int,
    previous_action: int,
    collided: bool,
) -> float:
    survival_reward = 0.03
    safety_reward = distance_safety_reward(observation)
    collision_penalty = 10.0 if collided else 0.0
    action_change_penalty = 0.01 if action != previous_action else 0.0
    reversal_penalty = 0.03 if is_reversal(action, previous_action) else 0.0
    return survival_reward + safety_reward - collision_penalty - action_change_penalty - reversal_penalty
