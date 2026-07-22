import unittest

import numpy as np

from observation_builder import BulletState, ObservationBuilder, ObservationConfig, PlayerState
from rl.reward import local_pccm_cost


class PCCMObservationModeTests(unittest.TestCase):
    # Build one observation mode from the same moving bullet state.
    def build_observation(self, mode: str) -> dict[str, np.ndarray]:
        builder = ObservationBuilder(ObservationConfig(pccm_observation_mode=mode))
        bullets = [BulletState(x=250.0, y=500.0, radius=8.0, vx=600.0, vy=0.0)]
        player = PlayerState(x=300.0, y=500.0, radius=3.0)
        return builder.build(bullets, player)

    # Check that occupancy-only keeps the PCCM channels present but zero.
    def test_occupancy_only_zeros_visible_pccm(self) -> None:
        observation = self.build_observation("occupancy_only")
        for key in ("blue_pccm", "yellow_pccm", "red_pccm"):
            self.assertTrue(np.all(observation[key] == 0.0), key)

    # Check that static PCCM excludes the future trajectory contribution.
    def test_static_differs_from_trajectory(self) -> None:
        static = self.build_observation("static")
        trajectory = self.build_observation("trajectory")
        self.assertFalse(np.allclose(static["red_pccm"], trajectory["red_pccm"]))

    # Check that all visible modes keep the same hidden reward PCCM.
    def test_reward_pccm_is_mode_independent(self) -> None:
        observations = [
            self.build_observation(mode)
            for mode in ("occupancy_only", "static", "trajectory")
        ]
        expected = observations[0]["_reward_red_pccm"]
        for observation in observations[1:]:
            np.testing.assert_allclose(observation["_reward_red_pccm"], expected)
        costs = [local_pccm_cost(observation) for observation in observations]
        np.testing.assert_allclose(costs, np.repeat(costs[0], len(costs)))

    # Check that unknown observation modes fail immediately.
    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ObservationBuilder(ObservationConfig(pccm_observation_mode="unknown"))


if __name__ == "__main__":
    unittest.main()
