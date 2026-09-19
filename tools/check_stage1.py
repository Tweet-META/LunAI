"""Check completion and summarize learning trends for all stage-one runs."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import torch


ROOT = Path(__file__).resolve().parent.parent
TARGET_FRAMES = 300_000
VARIANTS = ("full", "red_blue", "red_only")
SEEDS = (0, 100, 200)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        lines = (line for line in stream if not line.startswith("#"))
        return list(csv.DictReader(lines))


def summarize(rows: list[dict[str, str]]) -> tuple[int, float, float, float, float]:
    window = min(50, max(1, len(rows) // 5))
    early = rows[:window]
    late = rows[-window:]
    early_survival = mean(float(row["frame_steps"]) for row in early)
    late_survival = mean(float(row["frame_steps"]) for row in late)
    early_collision = mean(float(row["collisions"]) > 0 for row in early)
    late_collision = mean(float(row["collisions"]) > 0 for row in late)
    return len(rows), early_survival, late_survival, early_collision, late_collision


def main() -> int:
    print(
        "run                             frames   episodes   survival early->late   collision early->late   status"
    )
    all_complete = True
    for variant in VARIANTS:
        for seed in SEEDS:
            name = f"stage1_{variant}_seed{seed}"
            checkpoint_path = ROOT / "checkpoints" / f"{name}.pt"
            log_path = ROOT / "training_logs" / f"{name}.csv"
            console_path = ROOT / "training_logs" / f"{name}.console.txt"

            problems = []
            frames = 0
            rows: list[dict[str, str]] = []
            if not checkpoint_path.exists():
                problems.append("no checkpoint")
            else:
                try:
                    checkpoint = load_checkpoint(checkpoint_path)
                    frames = int(checkpoint.get("training_state", {}).get("total_frame_steps", 0))
                    if frames < TARGET_FRAMES:
                        problems.append("partial checkpoint")
                except Exception as error:
                    problems.append(f"bad checkpoint: {type(error).__name__}")

            if not log_path.exists():
                problems.append("no csv")
            else:
                try:
                    rows = load_rows(log_path)
                    if not rows:
                        problems.append("empty csv")
                except Exception as error:
                    problems.append(f"bad csv: {type(error).__name__}")

            if not console_path.exists():
                problems.append("no console log")
            else:
                console = console_path.read_text(encoding="utf-8", errors="replace")
                if "Training finished" not in console:
                    problems.append("no finish marker")
                if "Traceback (most recent call last)" in console:
                    problems.append("traceback")

            status = "COMPLETE" if not problems else "; ".join(problems)
            all_complete &= not problems
            if rows:
                count, early_s, late_s, early_c, late_c = summarize(rows)
                trend = f"{early_s:7.1f}->{late_s:7.1f}"
                collision = f"{early_c:5.1%}->{late_c:5.1%}"
            else:
                count, trend, collision = 0, "      n/a      ", "    n/a    "
            print(f"{name:<31} {frames:>7} {count:>10}   {trend}         {collision}   {status}")

    print()
    if all_complete:
        print("PASS: all nine runs reached 300,000 frames and produced readable outputs.")
        print("The trends above are diagnostic only; use fixed-seed evaluation for paper comparisons.")
        return 0
    print("INCOMPLETE: rerun tools/run_stage1_all.ps1 to resume unfinished checkpoints.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
