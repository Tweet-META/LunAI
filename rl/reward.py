from __future__ import annotations

import numpy as np


SURVIVAL_REWARD = 0.1
COLLISION_PENALTY = 30.0
ACTION_CHANGE_PENALTY = 0.0
DANGER_POTENTIAL_BETA = 0.2
WALL_PROXIMITY_MARGIN = 0.12
LOWER_FIELD_THRESHOLD = 0.70


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


# Convert local danger into a state potential.
def danger_potential(observation: dict[str, np.ndarray]) -> float:
    return -DANGER_POTENTIAL_BETA * nearby_bullet_danger(observation)


# Reward a transition according to its change in danger potential.
def danger_potential_shaping(
    previous_observation: dict[str, np.ndarray],
    observation: dict[str, np.ndarray],
    terminal: bool,
    gamma: float,
) -> float:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"Reward gamma must be in [0, 1], got {gamma}.")
    previous_phi = danger_potential(previous_observation)
    current_phi = 0.0 if terminal else danger_potential(observation)
    return gamma * current_phi - previous_phi


# Measure linear proximity to the four playfield walls.
def wall_proximity(observation: dict[str, np.ndarray], margin: float = WALL_PROXIMITY_MARGIN) -> float:
    if not 0.0 < margin <= 0.5:
        raise ValueError(f"Wall margin must be in (0, 0.5], got {margin}.")
    player_x = float(np.clip(observation["player_features"][0], 0.0, 1.0))
    player_y = float(np.clip(observation["player_features"][1], 0.0, 1.0))
    horizontal = max(0.0, 1.0 - min(player_x, 1.0 - player_x) / margin)
    vertical = max(0.0, 1.0 - min(player_y, 1.0 - player_y) / margin)
    return horizontal + vertical


# Reward movement away from nearby walls without penalizing parallel movement.
def wall_proximity_shaping(
    previous_observation: dict[str, np.ndarray],
    observation: dict[str, np.ndarray],
    weight: float,
) -> float:
    if weight < 0.0:
        raise ValueError(f"Wall shaping weight must be non-negative, got {weight}.")
    previous_proximity = wall_proximity(previous_observation)
    current_proximity = wall_proximity(observation)
    return float(weight) * (previous_proximity - current_proximity)


# Penalize every frame spent close to a wall.
def wall_state_penalty(observation: dict[str, np.ndarray], weight: float) -> float:
    if weight < 0.0:
        raise ValueError(f"Wall state penalty weight must be non-negative, got {weight}.")
    return float(weight) * wall_proximity(observation)


# Measure how far the player has moved above the preferred lower field.
def upper_field_proximity(
    observation: dict[str, np.ndarray],
    threshold: float = LOWER_FIELD_THRESHOLD,
) -> float:
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"Lower field threshold must be in (0, 1], got {threshold}.")
    player_y = float(np.clip(observation["player_features"][1], 0.0, 1.0))
    return max(0.0, (threshold - player_y) / threshold)


# Penalize every frame spent above the preferred lower field.
def upper_field_state_penalty(
    observation: dict[str, np.ndarray],
    weight: float,
    threshold: float = LOWER_FIELD_THRESHOLD,
) -> float:
    if weight < 0.0:
        raise ValueError(f"Upper field penalty weight must be non-negative, got {weight}.")
    return float(weight) * upper_field_proximity(observation, threshold)


# Compute the base reward for one real game frame.
def compute_frame_reward(action: int, previous_action: int, collided: bool) -> float:
    collision_penalty = COLLISION_PENALTY if collided else 0.0
    action_change_penalty = ACTION_CHANGE_PENALTY if action != previous_action else 0.0
    return SURVIVAL_REWARD - collision_penalty - action_change_penalty
