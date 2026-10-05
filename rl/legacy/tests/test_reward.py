import unittest

import numpy as np

from rl.reward import (
    ACTION_CHANGE_PENALTY,
    BLOCKED_MOVEMENT_PENALTY_WEIGHT,
    COLLISION_PENALTY,
    PCCM_STATE_PENALTY_WEIGHT,
    SURVIVAL_REWARD,
    blocked_movement_ratio,
    compute_frame_reward,
)


# Build one observation with a uniform local PCCM cost.
def make_pccm_observation(cost: float) -> dict[str, np.ndarray]:
    occupancy = np.zeros((5, 5), dtype=np.float32)
    return {
        "red_occupancy": occupancy,
        "red_pccm": np.full((5, 5), cost, dtype=np.float32),
        "player_features": np.array([0.5, 0.5, 0.01, 0.0], dtype=np.float32),
    }


class RewardTests(unittest.TestCase):
    # Check that action changes have one small cost without reversal cost.
    def test_action_change_penalty(self) -> None:
        observation = make_pccm_observation(0.0)
        same_reward = compute_frame_reward(observation, action=3, previous_action=3, collided=False)
        reverse_reward = compute_frame_reward(observation, action=3, previous_action=4, collided=False)
        other_change_reward = compute_frame_reward(observation, action=3, previous_action=1, collided=False)
        self.assertAlmostEqual(same_reward, SURVIVAL_REWARD)
        self.assertAlmostEqual(reverse_reward, SURVIVAL_REWARD - ACTION_CHANGE_PENALTY)
        self.assertAlmostEqual(reverse_reward, other_change_reward)

    # Check that collision remains the dominant penalty.
    def test_collision_penalty(self) -> None:
        reward = compute_frame_reward(make_pccm_observation(1.0), action=0, previous_action=0, collided=True)
        self.assertAlmostEqual(reward, -COLLISION_PENALTY)

    # Check that PCCM danger reduces the final reward directly.
    def test_pccm_reward(self) -> None:
        reward = compute_frame_reward(make_pccm_observation(0.8), action=0, previous_action=0, collided=False)
        self.assertAlmostEqual(reward, SURVIVAL_REWARD - 0.8 * PCCM_STATE_PENALTY_WEIGHT)

    # Check that reward uses the hidden full PCCM instead of the visible ablation map.
    def test_hidden_reward_pccm(self) -> None:
        observation = make_pccm_observation(0.0)
        observation["_reward_red_pccm"] = np.full((5, 5), 0.8, dtype=np.float32)
        reward = compute_frame_reward(observation, action=0, previous_action=0, collided=False)
        self.assertAlmostEqual(reward, SURVIVAL_REWARD - 0.8 * PCCM_STATE_PENALTY_WEIGHT)

    # Check that collision frames do not double-count PCCM danger.
    def test_collision_skips_pccm_penalty(self) -> None:
        reward = compute_frame_reward(make_pccm_observation(1.0), action=0, previous_action=0, collided=True)
        self.assertAlmostEqual(reward, -COLLISION_PENALTY)

    # Check that fully blocked movement has the maximum ratio.
    def test_fully_blocked_movement(self) -> None:
        ratio = blocked_movement_ratio(5.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(ratio, 1.0)

    # Check that a diagonal wall hit only counts the blocked axis.
    def test_partly_blocked_diagonal_movement(self) -> None:
        ratio = blocked_movement_ratio(1.0, 1.0, 0.0, 1.0)
        self.assertAlmostEqual(ratio, 1.0 / np.sqrt(2.0))

    # Check movements that do not push into a wall.
    def test_unblocked_movement(self) -> None:
        self.assertAlmostEqual(blocked_movement_ratio(0.0, 5.0, 0.0, 5.0), 0.0)
        self.assertAlmostEqual(blocked_movement_ratio(-5.0, 0.0, -5.0, 0.0), 0.0)
        self.assertAlmostEqual(blocked_movement_ratio(0.0, 0.0, 0.0, 0.0), 0.0)

    # Check that the blocked ratio reduces one frame reward.
    def test_blocked_movement_penalty(self) -> None:
        reward = compute_frame_reward(
            make_pccm_observation(0.0),
            action=3,
            previous_action=3,
            collided=False,
            blocked_ratio=1.0,
        )
        self.assertAlmostEqual(reward, SURVIVAL_REWARD - BLOCKED_MOVEMENT_PENALTY_WEIGHT)


if __name__ == "__main__":
    unittest.main()
