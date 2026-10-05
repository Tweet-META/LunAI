"""Check CUDA PCCM against the reference, then time complete observations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pygame
import torch

from tools.benchmark_pccm_level import benchmark_pass, collect_scene_snapshots, create_builder


PCCM_KEYS = ("blue_pccm", "yellow_pccm", "red_pccm", "_reward_red_pccm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level-file", default="level_th06_stage3_spell2.json")
    parser.add_argument("--warmup-frames", type=int, default=600)
    parser.add_argument("--sample-frames", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU visible to PyTorch.")
    if args.sample_frames < 1 or args.repeats < 1:
        raise ValueError("Sample frames and repeats must be positive.")

    pygame.init()
    pygame.display.set_mode((1, 1))
    try:
        snapshots = collect_scene_snapshots(
            args.level_file, args.warmup_frames, args.sample_frames, args.seed
        )
        builders = {
            "reference": create_builder("reference"),
            "torch_cuda": create_builder("torch_cuda"),
        }
        max_error = 0.0
        for bullets, player in snapshots[: min(20, len(snapshots))]:
            expected = builders["reference"].build(bullets, player)
            actual = builders["torch_cuda"].build(bullets, player)
            for key in PCCM_KEYS:
                max_error = max(max_error, float(np.max(np.abs(expected[key] - actual[key]))))
        print(f"device={torch.cuda.get_device_name(0)}")
        print(f"level={args.level_file} snapshots={len(snapshots)}")
        print(f"max_pccm_abs_error={max_error:.8g}")
        if max_error > 5e-4:
            raise AssertionError("CUDA PCCM differs from the CPU reference; do not train with it.")

        for builder in builders.values():
            for bullets, player in snapshots[: min(3, len(snapshots))]:
                builder.build(bullets, player)
        results: dict[str, list[tuple[float, float]]] = {name: [] for name in builders}
        for repeat in range(args.repeats):
            order = tuple(builders) if repeat % 2 == 0 else tuple(reversed(builders))
            for name in order:
                results[name].append(benchmark_pass(builders[name], snapshots))
        for name, values in results.items():
            print(
                f"{name}: observation_fps={median(value[0] for value in values):.2f} "
                f"median_build_ms={median(value[1] for value in values):.3f}"
            )
        cpu_fps = median(value[0] for value in results["reference"])
        cuda_fps = median(value[0] for value in results["torch_cuda"])
        print(f"observation_speedup={cuda_fps / cpu_fps:.3f}x")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
