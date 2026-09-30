"""Numerical regression checks for the optional compiled CPU PCCM."""

import importlib.util
import unittest

import numpy as np

from observation_builder import (
    BulletState, ObservationBuilder, ObservationConfig, PlayerState,
    centered_window, grid_cell_centers, pccm_sample_components,
)


@unittest.skipUnless(importlib.util.find_spec("numba"), "Numba is not installed")
class NumbaPCCMTests(unittest.TestCase):
    def compare_components(self, bullets, window, shape, frames=5, halo=20.0, radius=1.92):
        args = (bullets, radius, window, shape, 384, 448, frames, halo, 0.12)
        expected = pccm_sample_components(*args, implementation="reference")
        actual = pccm_sample_components(*args, implementation="numba")
        for index in range(4):
            self.assertEqual(actual[index].dtype, np.dtype("float32"))
            self.assertTrue(np.isfinite(actual[index]).all())
            if index == 3:
                np.testing.assert_array_equal(expected[index], actual[index])
            else:
                np.testing.assert_allclose(expected[index], actual[index], rtol=0, atol=3e-6)

    def test_sparse_dense_empty_and_offscreen_components(self):
        rng = np.random.default_rng(20261001)
        bullets = [
            BulletState(
                x=float(rng.uniform(-40, 420)), y=float(rng.uniform(-40, 490)),
                radius=float(rng.uniform(0.05, 10)),
                vx=float(rng.uniform(-700, 700)), vy=float(rng.uniform(-700, 700)),
            ) for _ in range(256)
        ]
        cases = [
            ((0, 0, 384, 448), (16, 16)),
            (centered_window(192, 360, 204, 204), (32, 32)),
            (centered_window(192, 360, 64, 64), (32, 32)),
            (centered_window(0, 448, 204, 204), (32, 32)),
            ((-300, -300, -96, -96), (32, 32)),
        ]
        for count in (0, 1, 256):
            for window, shape in cases:
                for frames, halo in ((1, 1.0), (5, 20.0), (12, 35.5)):
                    with self.subTest(count=count, window=window, frames=frames, halo=halo):
                        self.compare_components(bullets[:count], window, shape, frames, halo)

    def test_hitbox_and_halo_boundaries(self):
        window, shape = (0, 0, 64, 64), (32, 32)
        xx, yy = grid_cell_centers(window, shape)
        x, y = float(xx[8, 8]), float(yy[8, 8])
        for boundary in (x + 4.0, x + 24.0):
            for bx in (np.nextafter(np.float32(boundary), np.float32(-np.inf)),
                       boundary, np.nextafter(np.float32(boundary), np.float32(np.inf))):
                with self.subTest(bx=bx):
                    self.compare_components(
                        [BulletState(float(bx), y, 2.0, -90.0, 0.0)],
                        window, shape, radius=2.0,
                    )

    def test_complete_observations_and_reward_cost(self):
        rng = np.random.default_rng(913)
        bullets = [BulletState(
            float(rng.uniform(0, 384)), float(rng.uniform(0, 448)),
            float(rng.uniform(1, 8)), float(rng.uniform(-180, 180)),
            float(rng.uniform(-180, 180)),
        ) for _ in range(160)]
        for mode in ("occupancy_only", "static", "trajectory"):
            builders = [ObservationBuilder(ObservationConfig(
                pccm_implementation=backend, pccm_observation_mode=mode,
            )) for backend in ("reference", "numba")]
            for x, y in ((192, 360), (3, 3), (381, 445)):
                player = PlayerState(x, y, 1.92, 4)
                expected, actual = [builder.build(bullets, player) for builder in builders]
                self.assertEqual(expected.keys(), actual.keys())
                for key in expected:
                    with self.subTest(mode=mode, player=(x, y), key=key):
                        np.testing.assert_allclose(expected[key], actual[key], rtol=0, atol=3e-6)
                        if key in ("red_occupancy", "_occupancy_map"):
                            np.testing.assert_array_equal(expected[key], actual[key])


if __name__ == "__main__":
    unittest.main()
