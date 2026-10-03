import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from observation_builder import BulletState, PlayerState
from baselines.tian2023_dqn.common import BASE, epsilon_at, load_config, output_path
from baselines.tian2023_dqn.rays import cast_rays


class RayTests(unittest.TestCase):
    def setUp(self):
        self.player = PlayerState(50, 50, 2)

    def cast(self, bullets):
        return cast_rays(bullets, self.player, 100, 100).reshape(36, 5)

    def test_empty_and_nearest_intersection(self):
        np.testing.assert_array_equal(self.cast([]), np.zeros((36, 5)))
        # The larger farther-center circle has the closer surface (y=42 vs y=41).
        bullets = [BulletState(50, 40, 1, 0, 10), BulletState(50, 35, 7, 20, -30)]
        rays = self.cast(bullets)
        np.testing.assert_allclose(rays[0], [0, -.15, .2, -.3, 1])
        np.testing.assert_array_equal(rays[18], np.zeros(5))

    def test_ray_circle_not_angular_bin(self):
        # A circle between neighboring rays is invisible if neither intersects it.
        angle = np.deg2rad(5)
        b = BulletState(50 + 30 * np.sin(angle), 50 - 30 * np.cos(angle), .1, 0, 0)
        self.assertEqual(self.cast([b])[:, 4].sum(), 0)
        # Tangency counts, but off-screen circles beyond the field boundary do not.
        b = BulletState(52, 30, 2, 0, 0)
        self.assertEqual(self.cast([b])[0, 4], 1)
        b = BulletState(50, -20, 2, 0, 0)
        self.assertEqual(self.cast([b])[0, 4], 0)

    def test_origin_inside_circle_and_axis_order(self):
        np.testing.assert_array_equal(self.cast([BulletState(50, 50, 3, 0, 0)])[:, 4], np.ones(36))
        rays = self.cast([BulletState(70, 50, 1, 10, 20)])
        np.testing.assert_allclose(rays[9], [.2, 0, .1, .2, 1])

    def test_output_confinement_and_schedule(self):
        with self.assertRaises(ValueError):
            output_path("../../checkpoints/should_not_exist.pt")
        config = load_config()
        self.assertAlmostEqual(epsilon_at(0, config), 1)
        self.assertAlmostEqual(epsilon_at(500000, config), .05)
        self.assertAlmostEqual(epsilon_at(2000000, config), .05)


class EnvironmentTests(unittest.TestCase):
    def test_shared_physics_rewards_and_no_pccm_build(self):
        from baselines.tian2023_dqn.environment import RayTouhouEnv
        from rl.touhou_rl_env import TouhouRLEnv
        for level in load_config()["levels"]:
            traces = []
            for ray_mode in (False, True):
                env = (RayTouhouEnv(levels=[level], max_frames=260) if ray_mode else
                       TouhouRLEnv(level_file=level, action_repeat=1, max_steps=260, pccm_reward_weight=0))
                try:
                    if ray_mode:
                        env.builder.build = lambda *a, **kw: self.fail("PCCM builder called")
                    obs = env.reset(seed=10001)
                    trace = []
                    for frame in range(260):
                        # Reaches the wall, exercising the shared reward attenuation.
                        action = 3 if frame < 160 else 0
                        obs, reward, done, info = env.step(action)
                        if ray_mode:
                            self.assertEqual(set(obs), {"rays", "player_features"})
                            self.assertEqual(obs["rays"].shape, (180,))
                            self.assertTrue(np.isfinite(obs["rays"]).all())
                        trace.append((reward, done, info["collided"], info["frame_steps"],
                                      info["bullets"], *obs["player_features"][:2]))
                        if done:
                            break
                    traces.append(trace)
                finally:
                    env.close()
            np.testing.assert_array_equal(traces[0], traces[1], err_msg=level)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DQNTests(unittest.TestCase):
    def test_targets_stop_at_terminal_and_use_target_max(self):
        from baselines.tian2023_dqn.agent import dqn_targets
        out = dqn_targets(torch.tensor([1., 2.]), torch.tensor([1., 0.]),
                          torch.tensor([[100., 200.], [3., 4.]]), .5)
        torch.testing.assert_close(out, torch.tensor([1., 4.]))

    def test_update_target_sync_and_checkpoint_roundtrip(self):
        from baselines.tian2023_dqn.agent import DQNAgent, ReplayBuffer
        from baselines.tian2023_dqn.train import save_model
        from baselines.tian2023_dqn.common import FORMAT
        config = load_config()
        config.update(batch_size=8, replay_capacity=16, learning_starts=8)
        agent = DQNAgent(config, "cpu")
        rng = np.random.default_rng(42)
        replay = ReplayBuffer(16)
        for i in range(20):
            replay.add(rng.normal(size=180), i % 9, .1, rng.normal(size=180), i % 3 == 0)
        self.assertEqual(replay.size, 16)
        before = {k: v.clone() for k, v in agent.online.state_dict().items()}
        loss = agent.update(replay, rng)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in agent.online.state_dict().items()))
        for k, v in agent.target.state_dict().items():
            torch.testing.assert_close(v, before[k])
        agent.sync_target()
        for k, v in agent.target.state_dict().items():
            torch.testing.assert_close(v, agent.online.state_dict()[k])
        scratch = BASE / ".test_outputs"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            path = Path(folder) / "model.pt"
            save_model(path, agent, config, 0, 20, 1)
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint["format"], FORMAT)
            other = DQNAgent(config, "cpu")
            other.online.load_state_dict(checkpoint["model"])
            states = torch.tensor(rng.normal(size=(2, 180)), dtype=torch.float32)
            torch.testing.assert_close(agent.online(states), other.online(states))


if __name__ == "__main__":
    unittest.main()
