import unittest

import numpy as np

from rl.reward import SURVIVAL_REWARD, compute_frame_reward


class RewardObjectiveTests(unittest.TestCase):
    @staticmethod
    def observation(pccm_value: float, player_x: float = 0.5, player_y: float = 0.5) -> dict[str, np.ndarray]:
        return {
            "red_occupancy": np.zeros((64, 64), dtype=np.float32),
            "_reward_red_pccm": np.full((64, 64), pccm_value, dtype=np.float32),
            "player_features": np.asarray([player_x, player_y], dtype=np.float32),
        }

    def test_collision_frame_has_zero_reward(self):
        reward = compute_frame_reward(self.observation(1.0), 0, 0, collided=True, blocked_ratio=1.0)
        self.assertEqual(reward, 0.0)

    def test_edge_and_corner_frames_have_zero_reward_without_penalty(self):
        edge = compute_frame_reward(
            self.observation(1.0, player_x=0.0), 0, 0, collided=False, blocked_ratio=1.0
        )
        corner = compute_frame_reward(
            self.observation(1.0, player_x=0.0, player_y=0.0), 0, 0, collided=False, blocked_ratio=1.0
        )
        self.assertEqual(edge, 0.0)
        self.assertEqual(corner, 0.0)

    def test_safe_frame_receives_full_survival_reward(self):
        reward = compute_frame_reward(self.observation(0.0), 0, 0, collided=False, blocked_ratio=0.0)
        self.assertAlmostEqual(reward, SURVIVAL_REWARD)

    def test_pccm_and_blocked_movement_do_not_change_reward(self):
        safe = compute_frame_reward(self.observation(0.0), 0, 0, collided=False, blocked_ratio=0.0)
        dangerous = compute_frame_reward(self.observation(1.0), 0, 0, collided=False, blocked_ratio=1.0)
        self.assertAlmostEqual(dangerous, safe)

    def test_configured_pccm_penalty_reduces_reward(self):
        reward = compute_frame_reward(
            self.observation(0.4),
            0,
            0,
            collided=False,
            pccm_reward_weight=0.1,
        )
        self.assertAlmostEqual(reward, 0.06, places=6)

    def test_configured_pccm_penalty_cannot_make_reward_negative(self):
        reward = compute_frame_reward(
            self.observation(1.0),
            0,
            0,
            collided=False,
            pccm_reward_weight=1.0,
        )
        self.assertEqual(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
