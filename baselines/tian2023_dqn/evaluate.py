from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .agent import DQNAgent
from .common import BASE, FORMAT, output_path, validate_config
from .environment import RayTouhouEnv


def evaluate(args):
    model_path = Path(args.model)
    model_path = model_path.resolve() if model_path.is_absolute() else (BASE / model_path).resolve()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a Tian ray-DQN checkpoint.")
    config = checkpoint["config"]
    validate_config(config)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    torch.set_num_threads(config["torch_threads"])
    agent = DQNAgent(config, args.device)
    agent.online.load_state_dict(checkpoint["model"])
    agent.online.eval()
    log = output_path(args.log or f"eval/seed_{checkpoint['seed']}_{checkpoint['total_frame_steps']}_{Path(args.level).stem}.csv")
    if log.exists():
        raise FileExistsError(f"Evaluation log exists: {log}")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = RayTouhouEnv(levels=[args.level], max_frames=config["max_episode_frames"], render=args.render)
    fields = ["model_path", "policy_mode", "episode", "evaluation_seed", "level_file", "decision_steps",
              "frame_steps", "episode_reward", "completed", "collisions", "hp", "final_player_x", "final_player_y"]
    fields += [f"action_{i}" for i in range(9)]
    frames, clears = [], []
    try:
        with log.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for episode in range(1, args.episodes + 1):
                evaluation_seed = args.seed + episode
                state = env.reset(seed=evaluation_seed)["rays"]
                rng = np.random.default_rng(evaluation_seed)
                total_reward, collisions, done = 0.0, 0, False
                while not done:
                    action = agent.act(state, 0.0, rng)
                    obs, reward, done, info = env.step(action)
                    state = obs["rays"]
                    total_reward += reward
                    collisions += int(info["collided"])
                completed = int(collisions == 0 and info["frame_steps"] >= config["max_episode_frames"])
                row = dict(zip(fields[:13], [str(model_path), "greedy", episode, evaluation_seed,
                           args.level, info["decision_steps"], info["frame_steps"], total_reward,
                           completed, collisions, info["hp"], *obs["player_features"]]))
                row.update({f"action_{i}": count for i, count in enumerate(info["action_counts"])})
                writer.writerow(row)
                stream.flush()
                frames.append(info["frame_steps"])
                clears.append(completed)
                print(f"episode={episode} frames={frames[-1]} completed={completed}", flush=True)
    finally:
        env.close()
    print(json.dumps({"episodes": len(frames), "mean_frames": float(np.mean(frames)),
                      "clear_rate": float(np.mean(clears)), "log": str(log)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Absolute path or relative to baseline folder")
    parser.add_argument("--level", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log", help="Relative to baseline folder; existing logs are not overwritten")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
