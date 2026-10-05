"""Sequentially fine-tune the five formal PPO variants, preserving their ablations."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FRAMES = 2_000_000
TARGET_FRAMES = 2_300_000
LEVEL = "level_th06_stage6_scarlet_meister_hard.json"
# Name, observation scales, visible PCCM mode, PCC reward weight.
VARIANTS = {
    "full": ("lunai_v9.1", "full", "trajectory", "0"),
    "full_pcc": ("lunai_v9.1_pcc", "full", "trajectory", "0.1"),
    "noyellow": ("lunai_v9.1_noyellow", "red_blue", "trajectory", "0"),
    "noyellow_pcc": ("lunai_v9.1_noyellow_pcc", "red_blue", "trajectory", "0.1"),
    "nopccm": ("lunai_v9.1_nopccm", "full", "occupancy_only", "0"),
}


def make_plan(seeds, variants, device="auto", root=ROOT):
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Provide distinct, nonnegative seeds.")
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("Provide distinct variants.")
    plan = []
    for seed in seeds:
        for variant in variants:
            name, scales, mode, weight = VARIANTS[variant]
            source = Path(f"checkpoints/formal_1800_2m/seed_{seed}/{name}.pt")
            if seed == 0 and variant != "nopccm":
                legacy = Path(f"checkpoints/formal_1800_2m/{name}.pt")
                if (root / source).exists() and (root / legacy).exists():
                    raise ValueError(f"Two seed-0 sources exist; keep one unambiguous source: {source}, {legacy}")
                if not (root / source).exists():
                    source = legacy
            model = Path(f"checkpoints/scarlet_meister_300k/seed_{seed}/{name}.pt")
            log = Path(f"training_logs/scarlet_meister_300k/seed_{seed}/{name}.csv")
            args = [
                "-u", "-m", "rl.train_ppo_cnn",
                "--episodes", "1000000", "--max-steps", "1800",
                "--num-envs", "8", "--action-repeat", "1",
                "--frame-stack", "4", "--frame-stack-interval", "1",
                "--level-files", LEVEL, "--level-spawn-time-jitter", "0",
                "--player-start-margin", "51.2", "--seed", str(seed),
                "--rollout-steps", "1024", "--minibatch-size", "256", "--update-epochs", "4",
                "--gamma", "0.99", "--gae-lambda", "0.95",
                "--learning-rate", "0.00004", "--learning-rate-final", "0.00004",
                "--entropy-coef", "0.0015", "--entropy-coef-final", "0.0015",
                "--clip-range", "0.2", "--value-coef", "0.5",
                "--max-grad-norm", "0.5", "--target-kl", "0.03",
                "--hidden-dim", "128", "--architecture-version", "2",
                "--pccm-implementation", "numba", "--pccm-observation-mode", mode,
                "--observation-scales", scales, "--pccm-reward-weight", weight,
                "--pccm-prediction-frames", "5", "--pccm-halo-width", "20",
                "--pccm-wall-margin", "0.12", "--pccm-upper-field-threshold", "0.7",
                "--pccm-upper-field-cost", "0.3", "--save-interval", "5",
                "--max-total-frame-steps", str(TARGET_FRAMES), "--device", device,
                "--load-path", source.as_posix(), "--model-path", model.as_posix(),
                "--log-path", log.as_posix(),
            ]
            plan.append(dict(seed=seed, variant=variant, source=source.as_posix(),
                             model=model.as_posix(), log=log.as_posix(), arguments=args))
    return plan


def check_checkpoint(path, run, frames):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actual = checkpoint.get("training_state", {}).get("total_frame_steps")
    if actual != frames:
        raise ValueError(f"{path}: expected {frames} cumulative frames, got {actual}.")
    _, scales, mode, _ = VARIANTS[run["variant"]]
    expected = dict(observation_scales=scales, pccm_observation_mode=mode, frame_stack=4,
                    frame_stack_interval=1, action_repeat=1, architecture_version=2,
                    hidden_dim=128, pccm_prediction_frames=5, pccm_halo_width=20,
                    pccm_wall_margin=.12, pccm_upper_field_threshold=.7, pccm_upper_field_cost=.3,
                    gamma=.99)
    config = checkpoint.get("config", {})
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{path}: {key}={config.get(key)!r}, expected {value!r}.")
    if "model" not in checkpoint or "optimizer" not in checkpoint:
        raise ValueError(f"{path}: missing model/optimizer state.")


def preflight(plan, root=ROOT):
    missing = [str(root / run["source"]) for run in plan if not (root / run["source"]).is_file()]
    existing = [str(root / run[key]) for run in plan for key in ("model", "log") if (root / run[key]).exists()]
    if missing or existing:
        raise ValueError("Preflight failed.\n" + "\n".join(
            [f"Missing source: {p}" for p in missing] + [f"Output already exists: {p}" for p in existing]))
    for name in ("torch", "numpy", "pygame", "dotenv", "numba"):
        importlib.import_module(name)
    for relative in (f"assets/levels/{LEVEL}", "rl/scarlet_meister.py",
                     "assets/scripts/classes/game_logic/ScarletMeisterEnemy.py"):
        if not (root / relative).is_file():
            raise FileNotFoundError(f"Missing Scarlet Meister file: {relative}")
    for run in plan:
        check_checkpoint(root / run["source"], run, SOURCE_FRAMES)
        print(f"Checked seed {run['seed']} / {run['variant']}: 2M source", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without dependencies, checkpoints or writes")
    parser.add_argument("--check-only", action="store_true", help="Validate every source/output/dependency without training")
    options = parser.parse_args()
    plan = make_plan(options.seeds, options.variants, options.device)
    if options.dry_run:
        print(json.dumps(plan, indent=2))
        return
    preflight(plan)
    if options.check_only:
        print(f"All {len(plan)} sources passed; no training started.")
        return
    for index, run in enumerate(plan, 1):
        print(f"\n[{index}/{len(plan)}] seed {run['seed']} / {run['variant']}: +300000 frames", flush=True)
        log = ROOT / run["log"]
        log.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = log.with_suffix(".transfer.json")
        record = dict(run, source_frames=SOURCE_FRAMES, target_frames=TARGET_FRAMES,
                      additional_frames=TARGET_FRAMES - SOURCE_FRAMES, level=LEVEL,
                      source_sha256=hashlib.sha256((ROOT / run["source"]).read_bytes()).hexdigest(),
                      status="running")
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        try:
            # An argument list avoids PowerShell quoting and asynchronous ExitCode pitfalls.
            subprocess.run([sys.executable, *run["arguments"]], cwd=ROOT, check=True)
            check_checkpoint(ROOT / run["model"], run, TARGET_FRAMES)
            record["status"] = "complete"
        except BaseException:
            record["status"] = "interrupted_or_failed"
            raise
        finally:
            metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Finished {len(plan)} models; added {len(plan) * 300000:,} environment frames.")


if __name__ == "__main__":
    main()
