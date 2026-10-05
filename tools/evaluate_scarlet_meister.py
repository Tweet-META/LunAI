"""Evaluate all formal variants before/after Scarlet Meister fine-tuning."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from finetune_scarlet_meister import (
    ROOT, LEVEL, VARIANTS, SOURCE_FRAMES, TARGET_FRAMES, make_plan, check_checkpoint,
)


def evaluation_plan(phase, output, episodes=50, device="auto"):
    training = make_plan([0, 1, 2], list(VARIANTS), device)
    jobs = []
    # Schedule adapted models first; all jobs share the same evaluation seeds.
    phases = ["after", "before"] if phase == "both" else [phase]
    for item in phases:
        for run in training:
            model = run["model"] if item == "after" else run["source"]
            folder = "after_300k" if item == "after" else "before_0k"
            log = output / folder / f"seed_{run['seed']}" / (Path(model).stem + ".csv")
            jobs.append(dict(
                seed=run["seed"], variant=run["variant"], phase=item, model=model,
                frames=TARGET_FRAMES if item == "after" else SOURCE_FRAMES,
                log=str(log), episodes=episodes,
                arguments=["-u", "-m", "rl.evaluate_ppo_cnn", "--model-path", model,
                           "--level-file", LEVEL, "--episodes", str(episodes),
                           "--max-steps", "1800", "--action-repeat", "1",
                           "--seed", "10000", "--device", device,
                           "--pccm-implementation", "numba", "--log-path", str(log)],
            ))
    return jobs


def inspect_result(job):
    with Path(job["log"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    n = job["episodes"]
    if len(rows) != n or [int(r["episode"]) for r in rows] != list(range(1, n + 1)):
        raise ValueError(f"Expected {n} complete, consecutive episodes")
    if [int(r["evaluation_seed"]) for r in rows] != list(range(10001, 10001 + n)):
        raise ValueError("Unexpected evaluation seeds")
    for r in rows:
        frames, collisions = int(r["frame_steps"]), int(r["collisions"])
        expected_model = (ROOT / job["model"]).resolve()
        if (ROOT / r["model_path"]).resolve() != expected_model:
            raise ValueError("Unexpected model in evaluation CSV")
        if r["policy_mode"] != "greedy" or r["level_file"] != LEVEL:
            raise ValueError("Unexpected policy/level")
        if not 0 < frames <= 1800 or int(r["decision_steps"]) != frames:
            raise ValueError("Unexpected episode length")
        if collisions not in (0, 1) or int(r["completed"]) != int(collisions == 0 and frames == 1800):
            raise ValueError("Inconsistent completion/collision flags")
        if sum(int(r[f"action_{i}"]) for i in range(9)) != frames:
            raise ValueError("Action counts do not sum to episode length")
        if not math.isfinite(float(r["episode_reward"])):
            raise ValueError("Nonfinite episode reward")
    return dict(mean_frames=statistics.mean(int(r["frame_steps"]) for r in rows),
                clears=sum(int(r["completed"]) for r in rows), episodes=n)


def evaluate_one(job, environment):
    log = Path(job["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    result = {k: job[k] for k in ("seed", "variant", "phase", "model", "frames", "log")}
    try:
        with log.with_suffix(".stdout.txt").open("x", encoding="utf-8") as stdout, \
             log.with_suffix(".stderr.txt").open("x", encoding="utf-8") as stderr:
            process = subprocess.run(
                [sys.executable, *job["arguments"]], cwd=ROOT, env=environment,
                stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        result["exit_code"] = process.returncode
        if process.returncode:
            raise RuntimeError(f"Exit code {process.returncode}; see {log.with_suffix('.stderr.txt')}")
        result.update(inspect_result(job), status="complete")
    except Exception as error:
        result.update(status="failed", error=str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("both", "before", "after"), default="both")
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    if options.workers < 1 or options.episodes < 1:
        parser.error("workers and episodes must be positive")
    output = ROOT / "eval" / ("scarlet_meister_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    jobs = evaluation_plan(options.phase, output, options.episodes, options.device)
    if options.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    missing = [job["model"] for job in jobs if not (ROOT / job["model"]).is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    for module in ("torch", "numpy", "pygame", "dotenv", "numba"):
        importlib.import_module(module)
    if not (ROOT / "assets/levels" / LEVEL).is_file():
        raise FileNotFoundError(LEVEL)
    for job in jobs:
        check_checkpoint(ROOT / job["model"], job, job["frames"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        environment[name] = "1"
    results = []
    print(f"Evaluating {len(jobs)} checkpoints, at most {options.workers} concurrent.\nOutput: {output}", flush=True)
    with ThreadPoolExecutor(max_workers=options.workers) as pool:
        futures = [pool.submit(evaluate_one, job, environment) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            label = f"{result['phase']} / seed {result['seed']} / {result['variant']}"
            detail = (f"mean={result['mean_frames']:.2f}, clears={result['clears']}/{result['episodes']}"
                      if result["status"] == "complete" else result["error"])
            print(f"[{len(results)}/{len(jobs)}] {label}: {result['status']} {detail}", flush=True)
    if any(result["status"] != "complete" for result in results):
        raise SystemExit(f"Some evaluations failed. Completed CSVs are retained. See {output / 'summary.json'}")
    print(f"All {len(jobs)} evaluations complete: {output}", flush=True)


if __name__ == "__main__":
    main()
