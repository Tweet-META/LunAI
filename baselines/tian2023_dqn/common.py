from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]
FORMAT = "lunai_tian_ray_dqn_v1"


def output_path(value: str | Path) -> Path:
    """Keep baseline artifacts in its own folder, even after env changes cwd."""
    path = Path(value)
    path = (BASE / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(BASE):
        raise ValueError(f"Baseline output must be inside {BASE}: {path}")
    return path


def load_config(path: Path | None = None) -> dict:
    config = json.loads((path or BASE / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(c: dict) -> None:
    for key in ("ray_count", "replay_capacity", "batch_size", "learning_starts", "train_every",
                "target_update_frames", "epsilon_decay_frames", "max_episode_frames",
                "total_frames", "checkpoint_every", "torch_threads"):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if c["hidden_dims"] != [128, 64] or c["ray_count"] != 36:
        raise ValueError("This baseline fixes the paper's 36 rays and hidden layers 128, 64.")
    if c["batch_size"] > min(c["replay_capacity"], c["learning_starts"]):
        raise ValueError("Replay capacity and learning_starts must cover a batch.")
    if not 0 <= c["epsilon_end"] <= c["epsilon_start"] <= 1 or not 0 <= c["gamma"] <= 1:
        raise ValueError("Invalid epsilon/gamma range")
    if any(not math.isfinite(c[k]) or c[k] <= 0 for k in ("learning_rate", "max_grad_norm")):
        raise ValueError("Invalid optimizer setting")
    if not c["levels"] or any(not (PROJECT / "assets/levels" / level).is_file() for level in c["levels"]):
        raise ValueError("Every level must exist in the shared assets/levels directory.")


def epsilon_at(frame: int, c: dict) -> float:
    fraction = min(1.0, max(0.0, frame / c["epsilon_decay_frames"]))
    return c["epsilon_start"] + fraction * (c["epsilon_end"] - c["epsilon_start"])


def provenance(c: dict) -> dict:
    paths = list(BASE.glob("*.py")) + [BASE / "config.json"]
    paths += list((PROJECT / "assets/scripts").rglob("*.py"))
    paths += [PROJECT / name for name in (
        "rl/touhou_rl_env.py", "rl/reward.py", "observation_sources.py", "playfield_config.py", ".env")]
    paths += [PROJECT / "assets/levels" / level for level in c["levels"]]
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "sha256": {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(set(paths)) if p.is_file()},
    }
