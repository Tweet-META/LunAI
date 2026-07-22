from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rl.touhou_rl_env import TouhouRLEnv


# Create a CSV file for per-episode random-policy results.
def initialize_log(path: Path, action_names: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_path",
        "policy_mode",
        "episode",
        "evaluation_seed",
        "level_file",
        "decision_steps",
        "frame_steps",
        "episode_reward",
        "completed",
        "collisions",
        "hp",
        "final_player_x",
        "final_player_y",
        *(f"action_{name}" for name in action_names),
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fieldnames).writeheader()


# Append one completed episode to the evaluation CSV file.
def append_log(path: Path, row: dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writerow(row)


# Evaluate uniformly random actions in the current environment.
def evaluate(args: argparse.Namespace) -> None:
    env = TouhouRLEnv(
        max_steps=args.max_steps,
        action_repeat=args.action_repeat,
        level_file=args.level_file,
        level_files=args.level_files,
        random_player_start=args.random_player_start,
        player_start_margin=args.player_start_margin,
        pccm_observation_mode="trajectory",
    )
    log_path = Path(args.log_path)
    action_names = tuple(str(name) for name in env.ACTIONS)
    initialize_log(log_path, action_names)
    episode_frames: list[int] = []
    episode_rewards: list[float] = []
    episode_completions: list[int] = []

    try:
        for episode in range(1, args.episodes + 1):
            evaluation_seed = args.seed + episode
            rng = np.random.default_rng(evaluation_seed)
            observation = env.reset(seed=evaluation_seed)
            done = False
            total_reward = 0.0
            total_collisions = 0
            action_counts = np.zeros(len(env.ACTIONS), dtype=np.int32)
            last_info: dict[str, object] = {}

            while not done:
                action = int(rng.integers(0, len(env.ACTIONS)))
                action_counts[action] += 1
                observation, reward, done, info = env.step(action)
                total_reward += float(reward)
                total_collisions += int(info.get("collided", False))
                last_info = info

            frames = int(last_info.get("frame_steps", 0))
            decisions = int(last_info.get("decision_steps", 0))
            completed = int(total_collisions == 0)
            player_features = observation["player_features"]
            row: dict[str, object] = {
                "model_path": "random",
                "policy_mode": "random",
                "episode": episode,
                "evaluation_seed": evaluation_seed,
                "level_file": env.current_level_file,
                "decision_steps": decisions,
                "frame_steps": frames,
                "episode_reward": total_reward,
                "completed": completed,
                "collisions": total_collisions,
                "hp": last_info.get("hp", 0),
                "final_player_x": float(player_features[0]),
                "final_player_y": float(player_features[1]),
            }
            row.update({f"action_{i}": int(count) for i, count in enumerate(action_counts)})
            append_log(log_path, row)
            episode_frames.append(frames)
            episode_rewards.append(total_reward)
            episode_completions.append(completed)
            print(
                f"episode={episode}, frame_steps={frames}, reward={total_reward:.3f}, "
                f"completed={completed}, level={env.current_level_file}"
            )

        print("-" * 64)
        print(f"episodes={args.episodes}")
        print(f"mean_frame_steps={np.mean(episode_frames):.2f}")
        print(f"completion_rate={100.0 * np.mean(episode_completions):.2f}%")
        print(f"mean_reward={np.mean(episode_rewards):.3f}")
        print(f"log_path={log_path}")
    finally:
        env.close()


# Build command-line arguments for random-policy evaluation.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--level-file", type=str, default="level_6.json")
    parser.add_argument("--level-files", nargs="*", default=[])
    parser.add_argument("--random-player-start", action="store_true")
    parser.add_argument("--player-start-margin", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--log-path", type=str, required=True)
    return parser


# Parse arguments and start evaluation.
def main() -> None:
    evaluate(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
