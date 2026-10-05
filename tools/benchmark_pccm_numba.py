"""Validate Numba PCCM, then time complete observations on identical snapshots."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack, nullcontext
from functools import wraps
import json
import platform
import sys
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import through the backend to give an actionable error if Numba is missing.
import numba_pccm  # noqa: F401, E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
import observation_builder as maps  # noqa: E402
from tools.benchmark_pccm_level import benchmark_pass, collect_scene_snapshots, create_builder  # noqa: E402
import pygame  # noqa: E402

PCCM_KEYS = ("blue_pccm", "yellow_pccm", "red_pccm", "_reward_red_pccm")
PARITY_ATOL = 3e-6


@contextmanager
def reference_occupancy():
    """Reproduce the first Numba version for a matched before/after comparison."""
    def force_reference(function):
        @wraps(function)
        def build(*args, **kwargs):
            kwargs["implementation"] = "reference"
            return function(*args, **kwargs)
        return build

    with ExitStack() as context:
        for name in ("make_occupancy_map", "red_occupancy_map"):
            context.enter_context(patch.object(maps, name, force_reference(getattr(maps, name))))
        yield


def benchmark_level(level, args):
    snapshots = collect_scene_snapshots(level, args.warmup_frames, args.sample_frames, args.seed)
    counts = [len(bullets) for bullets, _ in snapshots]
    builders = {name: create_builder(name) for name in ("reference", "numba")}
    if args.compare_occupancy:
        builders = {
            "reference": builders["reference"],
            "numba_pccm_only": create_builder("numba"),
            "numba": builders["numba"],
        }
    started = perf_counter()
    builders["numba"].build(*snapshots[0])
    first_build_seconds = perf_counter() - started
    print(f"level={level} snapshots={len(snapshots)}", flush=True)
    print(f"bullets_mean={mean(counts):.2f} bullets_min={min(counts)} bullets_max={max(counts)}")
    print(f"first_numba_build_seconds={first_build_seconds:.3f} (compilation/cache loading excluded below)")

    max_error = 0.0
    for bullets, player in snapshots:
        expected = builders["reference"].build(bullets, player)
        actual = builders["numba"].build(bullets, player)
        for key in PCCM_KEYS:
            if not np.isfinite(actual[key]).all():
                raise AssertionError(f"Nonfinite Numba PCCM: {key}")
            error = float(np.max(np.abs(expected[key] - actual[key])))
            max_error = max(max_error, error)
            if error > PARITY_ATOL:
                raise AssertionError(f"Numba differs from reference: {key}, error={error}; do not train with it.")
        for key in expected.keys() - set(PCCM_KEYS):
            np.testing.assert_array_equal(expected[key], actual[key], err_msg=key)
    print(f"max_pccm_abs_error={max_error:.8g} checked_snapshots={len(snapshots)}", flush=True)
    print("occupancy_density_and_other_inputs=exact_match", flush=True)

    measurements = {name: [] for name in builders}
    for repeat in range(args.repeats):
        names = tuple(builders)
        offset = repeat % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            context = reference_occupancy() if name == "numba_pccm_only" else nullcontext()
            with context:
                measurements[name].append(benchmark_pass(builders[name], snapshots))
    timings = {}
    for name, values in measurements.items():
        timings[name] = {
            "observation_fps": median(value[0] for value in values),
            "median_build_ms": median(value[1] for value in values),
        }
        print(f"{name}: observation_fps={timings[name]['observation_fps']:.2f} "
              f"median_build_ms={timings[name]['median_build_ms']:.3f}")
    speedup = timings["numba"]["observation_fps"] / timings["reference"]["observation_fps"]
    print(f"observation_speedup={speedup:.3f}x", flush=True)
    occupancy_speedup = None
    if args.compare_occupancy:
        occupancy_speedup = timings["numba"]["observation_fps"] / timings["numba_pccm_only"]["observation_fps"]
        print(f"occupancy_incremental_speedup={occupancy_speedup:.3f}x", flush=True)
    return {
        "level": level, "snapshots": len(snapshots),
        "bullets_mean": mean(counts), "bullets_min": min(counts), "bullets_max": max(counts),
        "max_pccm_abs_error": max_error, "first_numba_build_seconds": first_build_seconds,
        "timings": timings, "observation_speedup": speedup,
        "occupancy_incremental_speedup": occupancy_speedup,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    levels = parser.add_mutually_exclusive_group()
    levels.add_argument("--level-file", default=None)
    levels.add_argument("--level-files", nargs="+", default=None)
    parser.add_argument("--warmup-frames", type=int, default=600)
    parser.add_argument("--sample-frames", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--json-path", default="")
    parser.add_argument("--compare-occupancy", action="store_true",
                        help="Also measure the previous PCCM-only Numba implementation.")
    args = parser.parse_args()
    if args.warmup_frames < 0 or args.sample_frames < 1 or args.repeats < 1:
        parser.error("Warmup must be non-negative; sample frames and repeats must be positive.")
    selected_levels = ([args.level_file] if args.level_file else args.level_files) or [
        f"level_th06_stage3_spell{spell}.json" for spell in (1, 2, 3)
    ]
    metadata = {
        "python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
        "warmup_frames": args.warmup_frames, "sample_frames": args.sample_frames,
        "repeats": args.repeats, "seed": args.seed, "parity_atol": PARITY_ATOL,
        "compare_occupancy": args.compare_occupancy,
    }
    print(json.dumps(metadata), flush=True)
    pygame.init()
    pygame.display.set_mode((1, 1))
    try:
        results = [benchmark_level(level, args) for level in selected_levels]
    finally:
        pygame.quit()
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
