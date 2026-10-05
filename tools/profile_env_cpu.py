"""Compare CPU environment cycles and separately attribute their hot paths.

Uses an invincible, stationary player and continues after collision signals to
sample the same dense section. These timings are not policy evaluation or PPO
throughput: model inference, optimization and process IPC are excluded.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
import json
import os
from pathlib import Path
import platform
from statistics import mean, median
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numba
import numpy as np
import numba_pccm
import observation_builder as maps
import observation_sources as sources
from rl import cnn_observation_utils as cnn
from rl.touhou_rl_env import TouhouRLEnv


class Timers:
    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)

    def wrap(self, function, label):
        @wraps(function)
        def timed(*args, **kwargs):
            name = label(*args, **kwargs) if callable(label) else label
            start = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.seconds[name] += perf_counter() - start
                self.calls[name] += 1
        return timed

    def install(self, context):
        from assets.scripts.scenes.GameScene import GameScene

        targets = [
            (GameScene, "update", "game_update"),
            (sources, "scene_to_observation_state", "extract_state"),
            (maps.ObservationBuilder, "build", "observation_total"),
            (maps, "make_occupancy_map", "occupancy_map"),
            (maps, "make_integral_image", "integral_image"),
            (maps, "density_grid", "density_grids"),
            (maps, "valid_area_grid", "valid_grids"),
            (maps, "red_occupancy_map", "red_occupancy"),
            (maps, "pccm_sample_components", "pccm_samples_nested"),
            (maps, "environment_pccm_cost", "wall_cost_nested"),
            (maps, "top_fraction_pool", "pooling_nested"),
            (maps, "bilinear_resize", "resize_nested"),
            (numba_pccm, "_sample_bullets", "numba_kernel_nested"),
            (TouhouRLEnv, "_append_map_snapshot", "history_copy"),
            (cnn, "cnn_observation", "cnn_input"),
        ]
        for owner, attribute, label in targets:
            context.enter_context(patch.object(owner, attribute, self.wrap(getattr(owner, attribute), label)))

        def scale_label(*args, **kwargs):
            shape = args[3] if len(args) > 3 else kwargs["output_shape"]
            return {(8, 8): "blue_pccm", (16, 16): "yellow_pccm", (64, 64): "red_pccm"}[tuple(shape)]

        context.enter_context(patch.object(maps, "projected_pccm", self.wrap(maps.projected_pccm, scale_label)))


def prepare(level, backend, args):
    env = TouhouRLEnv(
        level_file=level, pccm_implementation=backend,
        action_repeat=1, frame_stack=4, max_steps=args.warmup_frames + args.sample_frames + 1,
        pccm_reward_weight=0.1,
    )
    env.reset(seed=args.seed)
    env.scene.player.training_invincible = True
    for frame in range(args.warmup_frames):
        env.scene.update(1 / env.FPS)
        # Maintain enemy velocity history and seed the four-frame input at the
        # end of warmup without spending observation time on all earlier frames.
        if frame >= args.warmup_frames - 5:
            observation = env.get_observation()
            env.last_observation = observation
            env._append_map_snapshot(observation)
    env.frame_steps = args.warmup_frames
    return env


def sample(env, count):
    times, counts = [], []
    for _ in range(count):
        start = perf_counter()
        observation, reward, done, info = env.step(0)
        cnn.cnn_observation(observation, env.get_map_history())
        times.append(perf_counter() - start)
        counts.append(info["bullets"])
    return {
        "cycle_mean_ms": mean(times) * 1000,
        "cycle_median_ms": median(times) * 1000,
        "cycle_p95_ms": float(np.percentile(times, 95)) * 1000,
        "environment_fps": count / sum(times),
        "bullets_mean": mean(counts), "bullet_counts": counts,
    }


def attribute(level, backend, args):
    env = prepare(level, backend, args)
    timers = Timers()
    try:
        with ExitStack() as context:
            timers.install(context)
            measurement = sample(env, args.sample_frames)
    finally:
        env.close()
    ms = {name: seconds * 1000 / args.sample_frames for name, seconds in timers.seconds.items()}
    parts = ("occupancy_map", "integral_image", "density_grids", "valid_grids",
             "red_occupancy", "blue_pccm", "yellow_pccm", "red_pccm")
    ms["observation_other"] = ms["observation_total"] - sum(ms.get(key, 0) for key in parts)
    ms["environment_other"] = measurement["cycle_mean_ms"] - sum(
        ms.get(key, 0) for key in ("game_update", "extract_state", "observation_total", "history_copy", "cnn_input")
    )
    return {"instrumented_cycle_mean_ms": measurement["cycle_mean_ms"], "component_mean_ms": ms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level-files", nargs="+", default=[f"level_th06_stage3_spell{i}.json" for i in (1, 2, 3)])
    parser.add_argument("--warmup-frames", type=int, default=600)
    parser.add_argument("--sample-frames", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--json-path", default="evaluation_logs/cpu_environment_profile.json")
    args = parser.parse_args()
    if args.warmup_frames < 5 or args.sample_frames < 1 or args.repeats < 1:
        parser.error("Warmup must be >=5; samples and repeats must be positive.")
    report = {
        "metadata": {
            "python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__,
            "processor": platform.processor(), "settings": vars(args),
            "workload": "stationary invincible player; collisions do not stop benchmark; frame_stack=4; action_repeat=1",
            "excluded": "reset/warmup/JIT, inference, PPO updates, multiprocessing IPC",
        }, "results": [],
    }
    print(json.dumps(report["metadata"]), flush=True)
    for level in args.level_files:
        with (ROOT / "assets" / "levels" / level).open(encoding="utf-8") as file:
            if (args.warmup_frames + args.sample_frames) / 60 >= json.load(file)["length"]:
                parser.error(f"Sample interval reaches the end of {level}.")
        print(f"Measuring {level}", flush=True)
        runs = {backend: [] for backend in ("reference", "numba")}
        for repeat in range(args.repeats):
            order = ("reference", "numba") if repeat % 2 == 0 else ("numba", "reference")
            for backend in order:
                env = prepare(level, backend, args)
                try:
                    runs[backend].append(sample(env, args.sample_frames))
                finally:
                    env.close()
                print(f"  {backend} repeat={repeat + 1} mean_ms={runs[backend][-1]['cycle_mean_ms']:.3f}", flush=True)
        counts = runs["reference"][0]["bullet_counts"]
        assert all(run["bullet_counts"] == counts for values in runs.values() for run in values)
        timing = {
            backend: {key: median(run[key] for run in values)
                      for key in values[0] if key != "bullet_counts"}
            for backend, values in runs.items()
        }
        breakdown = {backend: attribute(level, backend, args) for backend in runs}
        speedup = timing["numba"]["environment_fps"] / timing["reference"]["environment_fps"]
        result = {"level": level, "timing": timing, "speedup": speedup, "breakdown": breakdown}
        report["results"].append(result)
        print(json.dumps(result, indent=2), flush=True)
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
