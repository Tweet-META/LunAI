"""Check optional PyTorch PCCM against the established NumPy reference."""

import importlib.util
import unittest

import numpy as np

from observation_builder import BulletState, centered_window, projected_pccm


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchPCCMTests(unittest.TestCase):
    def test_cpu_torch_matches_reference_across_scales(self):
        import torch

        rng = np.random.default_rng(20260928)
        bullets = [
            BulletState(
                x=float(rng.uniform(-16.0, 400.0)),
                y=float(rng.uniform(-16.0, 464.0)),
                radius=float(rng.uniform(2.0, 8.0)),
                vx=float(rng.uniform(-160.0, 160.0)),
                vy=float(rng.uniform(-160.0, 280.0)),
            )
            for _ in range(48)
        ]
        cases = (
            ((0, 0, 384, 448), (8, 8), (16, 16)),
            (centered_window(192.0, 384.0, 204, 204), (16, 16), (32, 32)),
            (centered_window(192.0, 384.0, 64, 64), (64, 64), (32, 32)),
            (centered_window(10.0, 438.0, 204, 204), (16, 16), (32, 32)),
        )
        for sample_bullets in ([], bullets):
            for window, output_shape, sample_shape in cases:
                args = (
                    sample_bullets, 1.92, window, output_shape, sample_shape,
                    384, 448, 5, 20.0, 0.12, 0.8,
                )
                reference = projected_pccm(*args, implementation="reference")
                implementations = ("torch_cpu", "torch_cuda") if torch.cuda.is_available() else ("torch_cpu",)
                for implementation in implementations:
                    actual = projected_pccm(*args, implementation=implementation)
                    for expected_map, actual_map in zip(reference, actual):
                        np.testing.assert_allclose(expected_map, actual_map, rtol=0, atol=5e-4)


if __name__ == "__main__":
    unittest.main()
