from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pygame  # noqa: E402

from assets.scripts.math_and_data.enviroment import FPS, GAME_ZONE  # noqa: E402
from observation_builder import ObservationBuilder, ObservationConfig  # noqa: E402
from observation_sources import scene_to_observation_state  # noqa: E402


# Create one builder with a selectable PCCM implementation.
def create_builder(implementation: str) -> ObservationBuilder:
    return ObservationBuilder(
        ObservationConfig(
            playfield_width=GAME_ZONE[2],
            playfield_height=GAME_ZONE[3],
            blue_grid=(8, 8),
            yellow_size=(320, 320),
            yellow_grid=(16, 16),
            red_size=(128, 128),
            red_map=(64, 64),
            pccm_prediction_frames=5,
            pccm_halo_width=24.0,
            pccm_wall_margin=0.12,
            pccm_soft_cap=0.8,
            pccm_implementation=implementation,
        )
    )


# Advance a real GameScene and retain immutable observation inputs.
def collect_scene_snapshots(
    level_file: str,
    warmup_frames: int,
    sample_frames: int,
    seed: int,
) -> list[tuple[list, object]]:
    from assets.scripts.scenes.GameScene import GameScene

    random.seed(seed)
    np.random.seed(seed)
    scene = GameScene(level_file=level_file)
    scene.player.training_invincible = True
    previous_enemy_positions = {}
    snapshots = []
    delta_time = 1.0 / FPS

    for frame in range(warmup_frames + sample_frames):
        scene.update(delta_time)
        bullets, player, previous_enemy_positions = scene_to_observation_state(
            scene,
            GAME_ZONE,
            0,
            previous_enemy_positions,
            delta_time,
        )
        if frame >= warmup_frames:
            snapshots.append((bullets, player))
    return snapshots


# Measure one pass over the same retained scene snapshots.
def benchmark_pass(
    builder: ObservationBuilder,
    snapshots: list[tuple[list, object]],
) -> tuple[float, float]:
    frame_times = []
    pass_start = perf_counter()
    for bullets, player in snapshots:
        frame_start = perf_counter()
        builder.build(bullets, player)
        frame_times.append(perf_counter() - frame_start)
    elapsed = perf_counter() - pass_start
    fps = len(snapshots) / elapsed
    median_ms = median(frame_times) * 1000.0
    return fps, median_ms


# Compare reference and hybrid paths with alternating measurement order.
def run_benchmark(
    snapshots: list[tuple[list, object]],
    repeats: int,
) -> dict[str, dict[str, float]]:
    builders = {
        "reference": create_builder("reference"),
        "auto": create_builder("auto"),
    }
    for builder in builders.values():
        for bullets, player in snapshots[: min(5, len(snapshots))]:
            builder.build(bullets, player)

    measurements = {
        "reference": {"fps": [], "median_ms": []},
        "auto": {"fps": [], "median_ms": []},
    }
    for repeat_index in range(repeats):
        order = ("reference", "auto") if repeat_index % 2 == 0 else ("auto", "reference")
        for implementation in order:
            fps, median_ms = benchmark_pass(builders[implementation], snapshots)
            measurements[implementation]["fps"].append(fps)
            measurements[implementation]["median_ms"].append(median_ms)

    return {
        implementation: {
            "fps": median(values["fps"]),
            "median_ms": median(values["median_ms"]),
        }
        for implementation, values in measurements.items()
    }


# Run the dedicated real-level PCCM performance comparison.
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PCCM implementations on a real game level.")
    parser.add_argument("--level-file", default="level_benchmark_pccm.json")
    parser.add_argument("--warmup-frames", type=int, default=120)
    parser.add_argument("--sample-frames", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    if args.warmup_frames < 0 or args.sample_frames < 1 or args.repeats < 1:
        raise ValueError("Warmup must be non-negative; sample frames and repeats must be positive.")

    pygame.init()
    pygame.display.set_mode((1, 1))
    snapshots = collect_scene_snapshots(
        args.level_file,
        args.warmup_frames,
        args.sample_frames,
        args.seed,
    )
    bullet_counts = [len(bullets) for bullets, _ in snapshots]
    results = run_benchmark(snapshots, args.repeats)
    reference_fps = results["reference"]["fps"]
    auto_fps = results["auto"]["fps"]
    speedup = auto_fps / reference_fps

    print("Real-level PCCM benchmark")
    print(f"level={args.level_file}")
    print(
        f"snapshots={len(snapshots)}, bullets_mean={mean(bullet_counts):.2f}, "
        f"bullets_min={min(bullet_counts)}, bullets_max={max(bullet_counts)}"
    )
    for implementation in ("reference", "auto"):
        values = results[implementation]
        print(
            f"{implementation}: observation_fps={values['fps']:.2f}, "
            f"median_build_ms={values['median_ms']:.3f}"
        )
    print(f"auto_speedup={speedup:.3f}x ({(speedup - 1.0) * 100.0:+.1f}%)")
    pygame.quit()


if __name__ == "__main__":
    main()
