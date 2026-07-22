from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np


# Create the parent directory for an output path.
def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# Create a CSV log file with optional run metadata and the PPO training header.
def write_log_header(
    path: Path,
    run_config: dict[str, object] | None = None,
    include_env_id: bool = False,
    extra_columns: Sequence[str] = (),
) -> None:
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        if run_config is not None:
            f.write(f"# run_config: {json.dumps(run_config, sort_keys=True)}\n")
        writer = csv.writer(f)
        header = [
            "episode",
            "update",
            "global_step",
            "total_frame_steps",
            "decision_steps",
            "frame_steps",
            "episode_reward",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "hp",
            "collisions",
        ]
        header.extend(extra_columns)
        if include_env_id:
            header.append("env_id")
        writer.writerow(header)


# Append one row to a CSV training log.
def append_log(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# Load one JSON configuration file for a training run.
def load_run_config(path_text: str) -> dict[str, object]:
    path = Path(path_text)
    try:
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except OSError as error:
        raise ValueError(f"Could not read config file: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Config file is not valid JSON: {path}") from error

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain one JSON object: {path}")
    return config


# Parse arguments after applying optional JSON defaults.
def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="")
    preliminary_args, _ = config_parser.parse_known_args()

    if preliminary_args.config:
        config = load_run_config(preliminary_args.config)
        valid_keys = {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}
        unknown_keys = sorted(set(config) - valid_keys)
        if unknown_keys:
            unknown_text = ", ".join(unknown_keys)
            raise ValueError(f"Unknown config keys: {unknown_text}")
        parser.set_defaults(**config)

    return parser.parse_args()


# Compute GAE advantages and value targets for one rollout.
def compute_gae(
    rewards: list[float],
    dones: list[bool],
    values: list[float],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_advantage = 0.0
    next_value = float(last_value)

    for index in reversed(range(len(rewards))):
        next_nonterminal = 1.0 - float(dones[index])
        delta = rewards[index] + gamma * next_value * next_nonterminal - values[index]
        last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
        advantages[index] = last_advantage
        next_value = values[index]

    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns.astype(np.float32)


# Return a value that moves linearly from start to end.
def linear_schedule(start: float, end: float, progress: float) -> float:
    clipped_progress = float(np.clip(progress, 0.0, 1.0))
    return start + (end - start) * clipped_progress


# Return normalized progress for episode- or step-limited training.
def training_progress(args: argparse.Namespace, counters: dict[str, int]) -> float:
    if args.max_total_frame_steps > 0:
        return counters["total_frame_steps"] / args.max_total_frame_steps
    if args.max_total_decision_steps > 0:
        return counters["global_step"] / args.max_total_decision_steps
    if args.episodes <= 1:
        return 1.0
    return (counters["episode"] - 1) / (args.episodes - 1)


# Check whether a global training budget has been reached.
def training_limit_reached(args: argparse.Namespace, counters: dict[str, int]) -> bool:
    frame_limit_reached = args.max_total_frame_steps > 0 and counters["total_frame_steps"] >= args.max_total_frame_steps
    decision_limit_reached = args.max_total_decision_steps > 0 and counters["global_step"] >= args.max_total_decision_steps
    return frame_limit_reached or decision_limit_reached


# Describe which global training budget stopped the run.
def training_stop_reason(args: argparse.Namespace, counters: dict[str, int]) -> str:
    if args.max_total_frame_steps > 0 and counters["total_frame_steps"] >= args.max_total_frame_steps:
        return f"max_total_frame_steps={args.max_total_frame_steps}"
    if args.max_total_decision_steps > 0 and counters["global_step"] >= args.max_total_decision_steps:
        return f"max_total_decision_steps={args.max_total_decision_steps}"
    return "episode_limit"
