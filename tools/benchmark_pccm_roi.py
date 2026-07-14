from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from observation_builder import (  # noqa: E402
    BulletState,
    centered_window,
    pccm_sample_components_reference,
    pccm_sample_components_roi,
    projected_pccm,
)


SCALE_CASES = (
    ("blue", (0, 0, 600, 700), (8, 8), (16, 16)),
    ("yellow", centered_window(300.0, 560.0, 320, 320), (16, 16), (32, 32)),
    ("red", centered_window(300.0, 560.0, 128, 128), (64, 64), (32, 32)),
)


# Create deterministic bullets spread across and around the playfield.
def make_bullets(seed: int, count: int) -> list[BulletState]:
    rng = np.random.default_rng(seed)
    bullets = []
    for _ in range(count):
        x = float(rng.uniform(-40.0, 640.0))
        y = float(rng.uniform(-40.0, 740.0))
        radius = float(rng.uniform(2.0, 14.0))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        speed = float(rng.uniform(40.0, 520.0))
        bullets.append(
            BulletState(
                x=x,
                y=y,
                radius=radius,
                vx=float(np.cos(angle) * speed),
                vy=float(np.sin(angle) * speed),
            )
        )
    return bullets


# Return aggregate consistency metrics across random players and all scales.
def consistency_metrics(
    bullets: list[BulletState],
    seed: int,
    random_player_count: int = 12,
) -> tuple[float, float, int]:
    max_abs_error = 0.0
    absolute_error_sum = 0.0
    value_count = 0
    hard_collision_mismatch_count = 0
    rng = np.random.default_rng(seed)
    player_positions = [
        (2.0, 2.0),
        (598.0, 2.0),
        (2.0, 698.0),
        (598.0, 698.0),
    ]
    player_positions.extend(
        zip(
            rng.uniform(0.0, 600.0, random_player_count),
            rng.uniform(0.0, 700.0, random_player_count),
        )
    )

    for player_x, player_y in player_positions:
        scale_cases = (
            ((0, 0, 600, 700), (16, 16)),
            (centered_window(float(player_x), float(player_y), 320, 320), (32, 32)),
            (centered_window(float(player_x), float(player_y), 128, 128), (32, 32)),
        )
        for window, sample_shape in scale_cases:
            reference = pccm_sample_components_reference(
                bullets, 3.0, window, sample_shape, 600, 700, 5, 24.0, 0.12
            )
            optimized = pccm_sample_components_roi(
                bullets, 3.0, window, sample_shape, 600, 700, 5, 24.0, 0.12
            )
            for reference_map, optimized_map in zip(reference[:3], optimized[:3]):
                difference = np.abs(reference_map - optimized_map)
                max_abs_error = max(max_abs_error, float(np.max(difference)))
                absolute_error_sum += float(np.sum(difference, dtype=np.float64))
                value_count += difference.size
            hard_collision_mismatch_count += int(np.count_nonzero(reference[3] != optimized[3]))
    mean_abs_error = absolute_error_sum / max(1, value_count)
    return max_abs_error, mean_abs_error, hard_collision_mismatch_count


# Time one complete three-scale PCCM build and each individual scale.
def benchmark_implementation(
    bullets: list[BulletState],
    implementation: str,
    repeats: int,
) -> tuple[float, dict[str, float]]:
    scale_samples: dict[str, list[float]] = {name: [] for name, *_ in SCALE_CASES}
    total_samples = []

    # Warm up imports, allocations, and NumPy dispatch before measuring.
    for _, window, output_shape, sample_shape in SCALE_CASES:
        projected_pccm(
            bullets,
            3.0,
            window,
            output_shape,
            sample_shape,
            600,
            700,
            5,
            24.0,
            0.12,
            0.8,
            implementation=implementation,
        )

    for _ in range(repeats):
        total_start = perf_counter()
        for scale_name, window, output_shape, sample_shape in SCALE_CASES:
            scale_start = perf_counter()
            projected_pccm(
                bullets,
                3.0,
                window,
                output_shape,
                sample_shape,
                600,
                700,
                5,
                24.0,
                0.12,
                0.8,
                implementation=implementation,
            )
            scale_samples[scale_name].append((perf_counter() - scale_start) * 1000.0)
        total_samples.append((perf_counter() - total_start) * 1000.0)

    scale_medians = {name: median(samples) for name, samples in scale_samples.items()}
    return median(total_samples), scale_medians


# Print consistency checks and performance at the requested bullet counts.
def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and benchmark ROI PCCM sampling.")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("Benchmark repeats must be positive.")

    validation_bullets = make_bullets(args.seed, 500)
    max_error, mean_error, hard_mismatch = consistency_metrics(validation_bullets, args.seed + 1)
    print("PCCM consistency")
    print(f"max_abs_error={max_error:.9g}")
    print(f"mean_abs_error={mean_error:.9g}")
    print(f"hard_collision_mismatch_count={hard_mismatch}")
    if max_error > 1e-5 or mean_error > 1e-6 or hard_mismatch != 0:
        raise AssertionError("ROI PCCM output does not match the reference implementation.")
    print()
    print("Median PCCM build time (ms)")
    print("bullets  implementation      blue     yellow        red      total")

    for count in (80, 200, 500):
        bullets = make_bullets(args.seed + count, count)
        for implementation in ("reference", "roi", "auto"):
            total, scales = benchmark_implementation(bullets, implementation, args.repeats)
            print(
                f"{count:7d}  {implementation:14s}"
                f" {scales['blue']:9.3f} {scales['yellow']:10.3f}"
                f" {scales['red']:10.3f} {total:10.3f}"
            )


if __name__ == "__main__":
    main()
