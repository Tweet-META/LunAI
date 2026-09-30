"""Exact binary-mask parity, including subpixel positions and circle edges."""

import importlib.util
import unittest

import numpy as np

from observation_builder import BulletState, make_occupancy_map, red_occupancy_map


@unittest.skipUnless(importlib.util.find_spec("numba"), "Numba is not installed")
class NumbaOccupancyTests(unittest.TestCase):
    def assert_maps_match(self, bullets):
        args = (384, 448, bullets)
        expected = make_occupancy_map(*args)
        actual = make_occupancy_map(*args, implementation="numba")
        self.assertEqual(actual.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(expected, actual)
        for window, shape in (
            ((0, 0, 64, 64), (64, 64)),
            ((160, 328, 224, 392), (64, 64)),
            ((-32, 416, 32, 480), (64, 64)),
            ((10, -20, 82, 71), (25, 39)),
            ((-600, -500, -300, -200), (16, 16)),
        ):
            with self.subTest(window=window, shape=shape):
                expected = red_occupancy_map(bullets, window, shape)
                actual = red_occupancy_map(bullets, window, shape, implementation="numba")
                self.assertEqual(actual.dtype, np.dtype("float32"))
                np.testing.assert_array_equal(expected, actual)

    def test_empty_sparse_dense_and_overlapping(self):
        rng = np.random.default_rng(20261001)
        for count in (0, 1, 500):
            bullets = [BulletState(
                float(rng.uniform(-50, 430)), float(rng.uniform(-50, 500)),
                float(rng.uniform(0.01, 35)), 0, 0,
            ) for _ in range(count)]
            with self.subTest(count=count):
                self.assert_maps_match(bullets)
        self.assert_maps_match([BulletState(192.125, 360.5, 12.75, 0, 0)] * 100)

    def test_rounding_clipping_and_large_radius(self):
        for x, y, radius in (
            (-0.5, 0.5, 1), (1.5, 2.5, 2.01), (383.5, 447.5, 2.0),
            (192.5, 360.5, 0), (192.5, 360.5, 0.001),
            (-20.25, 12.125, 30), (200.25, 200.5, 500),
            (-1000, -1000, 1),
        ):
            with self.subTest(x=x, y=y, radius=radius):
                self.assert_maps_match([BulletState(x, y, radius, 0, 0)])

    def test_exact_circle_boundary_and_adjacent_floats(self):
        # Radius 5 gives exact integer and half-pixel sample-boundary cases.
        for x, y in ((20, 20), (20.5, 20.5), (192.5, 360.5)):
            for radius in (np.nextafter(5.0, 0.0), 5.0, np.nextafter(5.0, np.inf)):
                with self.subTest(x=x, y=y, radius=radius):
                    self.assert_maps_match([BulletState(x, y, float(radius), 0, 0)])


if __name__ == "__main__":
    unittest.main()
