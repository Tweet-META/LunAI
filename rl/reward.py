from __future__ import annotations

import numpy as np


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


# Compute a minimal baseline reward for one step.
def compute_reward(
    observation: dict[str, np.ndarray],
    action: int,
    previous_action: int,
    collided: bool,
) -> float:
    survival_reward = 0.03
    collision_penalty = 10.0 if collided else 0.0
    action_change_penalty = 0.01 if action != previous_action else 0.0
    reversal_penalty = 0.03 if is_reversal(action, previous_action) else 0.0
    return survival_reward - collision_penalty - action_change_penalty - reversal_penalty
