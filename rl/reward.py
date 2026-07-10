from __future__ import annotations

import numpy as np


NEAR_DANGER_PENALTY_SCALE = 0.03
EDGE_MARGIN = 0.14
SIDE_EDGE_PENALTY_SCALE = 0.025
VERTICAL_EDGE_PENALTY_SCALE = 0.02
CORNER_PENALTY_SCALE = 0.055
SAFE_STAY_REWARD = 0.006


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
    distances = np.sqrt((xx - player_x) ** 2 + (yy - player_y) ** 2)
    max_distance = max(
        float(np.hypot(player_x, player_y)),
        float(np.hypot(cols - 1.0 - player_x, player_y)),
        float(np.hypot(player_x, rows - 1.0 - player_y)),
        float(np.hypot(cols - 1.0 - player_x, rows - 1.0 - player_y)),
        1.0,
    )
    weights = np.clip(1.0 - distances / max_distance, 0.0, 1.0)
    occupancy_danger = float(np.max(red_occupancy * weights))
    speed_danger = float(np.max(red_occupancy * red_speed * weights))
    return float(np.clip(0.8 * occupancy_danger + 0.2 * speed_danger, 0.0, 1.0))


# Penalize bullets that are too close to the player.
def near_danger_penalty(observation: dict[str, np.ndarray]) -> float:
    danger = nearby_bullet_danger(observation)
    return NEAR_DANGER_PENALTY_SCALE * danger


# Check whether the red local map has any bullet near the player.
def red_zone_has_bullets(observation: dict[str, np.ndarray]) -> bool:
    red_occupancy = observation["red_occupancy"]
    return float(np.max(red_occupancy)) > 0.0


# Penalize camping near walls and corners.
def boundary_penalty(observation: dict[str, np.ndarray]) -> float:
    player_features = observation["player_features"]
    if player_features.shape[0] >= 8:
        left_margin = float(player_features[4])
        right_margin = float(player_features[5])
        top_margin = float(player_features[6])
        bottom_margin = float(player_features[7])
    else:
        player_x = float(player_features[0])
        player_y = float(player_features[1])
        left_margin = player_x
        right_margin = 1.0 - player_x
        top_margin = player_y
        bottom_margin = 1.0 - player_y

    left = max(0.0, EDGE_MARGIN - left_margin) / EDGE_MARGIN
    right = max(0.0, EDGE_MARGIN - right_margin) / EDGE_MARGIN
    top = max(0.0, EDGE_MARGIN - top_margin) / EDGE_MARGIN
    bottom = max(0.0, EDGE_MARGIN - bottom_margin) / EDGE_MARGIN
    side_pressure = max(left, right)
    vertical_pressure = max(top, bottom)
    corner_pressure = min(side_pressure, vertical_pressure)
    side_penalty = SIDE_EDGE_PENALTY_SCALE * side_pressure
    vertical_penalty = VERTICAL_EDGE_PENALTY_SCALE * vertical_pressure
    corner_penalty = CORNER_PENALTY_SCALE * corner_pressure
    return side_penalty + vertical_penalty + corner_penalty


# Reward staying still only when the red local map is empty.
def safe_stay_reward(observation: dict[str, np.ndarray], action: int) -> float:
    if action != 0:
        return 0.0
    if red_zone_has_bullets(observation):
        return 0.0
    return SAFE_STAY_REWARD


# Compute a minimal baseline reward for one step.
def compute_reward(
    observation: dict[str, np.ndarray],
    action: int,
    previous_action: int,
    collided: bool,
) -> float:
    survival_reward = 0.1
    stay_reward = safe_stay_reward(observation, action)
    danger_penalty = near_danger_penalty(observation)
    wall_penalty = boundary_penalty(observation)
    collision_penalty = 50.0 if collided else 0.0
    action_change_penalty = 0.005 if action != previous_action else 0.0
    reversal_penalty = 0.02 if is_reversal(action, previous_action) else 0.0
    return (
        survival_reward
        + stay_reward
        - danger_penalty
        - wall_penalty
        - collision_penalty
        - action_change_penalty
        - reversal_penalty
    )
