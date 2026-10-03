from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .agent import DQNAgent, ReplayBuffer
from .common import FORMAT, epsilon_at, load_config, output_path, provenance, validate_config
from .environment import RayTouhouEnv


def save_model(path, agent, config, seed, frames, episodes):
    temporary = path.with_suffix(".tmp")
    torch.save({
        "format": FORMAT, "config": config, "seed": seed,
        "total_frame_steps": frames, "completed_episodes": episodes,
        "updates": agent.updates, "model": agent.online.state_dict(),
        "checkpoint_kind": "evaluation_only",
    }, temporary)
    temporary.replace(path)


def train(args):
    config = load_config(args.config.resolve() if args.config else None)
    if args.total_frames is not None:
        config["total_frames"] = args.total_frames
    validate_config(config)
    if args.seed < 0 or args.seed > 4292:
        raise ValueError("Seed must be between 0 and 4292.")
    run = output_path(args.output_dir or f"runs/seed_{args.seed}")
    run.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # Independent generators are not reset by the environment's global seeding.
    action_rng = np.random.default_rng(args.seed)
    replay_rng = np.random.default_rng(args.seed + 100000)
    agent = DQNAgent(config, args.device)
    replay = ReplayBuffer(config["replay_capacity"])
    metadata = {"format": FORMAT, "config": config, "seed": args.seed,
                "device": str(agent.device), "torch": torch.__version__, "numpy": np.__version__,
                "provenance": provenance(config), "status": "running"}
    (run / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    env = RayTouhouEnv(levels=config["levels"], max_frames=config["max_episode_frames"],
                       ray_count=config["ray_count"])
    columns = ["episode", "total_frame_steps", "level_file", "frame_steps", "episode_reward",
               "completed", "collisions", "epsilon", "loss", "updates", "wall_time_ratio", "elapsed_seconds"]
    frames, episodes, episode_return, loss, done = 0, 0, 0.0, None, False
    started = time.perf_counter()
    try:
        state = env.reset(seed=args.seed * 1000000 + 1)["rays"]
        with (run / "training.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            while frames < config["total_frames"]:
                epsilon = epsilon_at(frames, config)
                action = agent.act(state, epsilon, action_rng)
                observation, reward, done, info = env.step(action)
                next_state = observation["rays"]
                # The 1800-frame task is a finite episode; both collision and clear end it.
                replay.add(state, action, reward, next_state, done)
                state = next_state
                frames += 1
                episode_return += reward
                if frames >= config["learning_starts"] and frames % config["train_every"] == 0:
                    loss = agent.update(replay, replay_rng)
                if frames % config["target_update_frames"] == 0:
                    agent.sync_target()
                if done:
                    episodes += 1
                    writer.writerow(dict(zip(columns, [
                        episodes, frames, info["level_file"], info["frame_steps"], episode_return,
                        int(not info["collided"] and info["frame_steps"] >= config["max_episode_frames"]),
                        int(info["collided"]), epsilon, loss, agent.updates,
                        info["wall_time_ratio"], time.perf_counter() - started,
                    ])))
                    stream.flush()
                    print(f"episode={episodes} frames={frames} survival={info['frame_steps']} "
                          f"reward={episode_return:.3f} epsilon={epsilon:.4f} loss={loss}", flush=True)
                    episode_return = 0.0
                    if frames < config["total_frames"]:
                        state = env.reset(seed=(args.seed * 1000000 + episodes + 1) % (2**32))["rays"]
                if frames % config["checkpoint_every"] == 0 or frames == config["total_frames"]:
                    path = run / f"tian_dqn_{frames}.pt"
                    save_model(path, agent, config, args.seed, frames, episodes)
                    print(f"Saved {path}", flush=True)
        metadata["status"] = "complete"
    except BaseException:
        metadata["status"] = "interrupted_or_failed"
        if frames:
            save_model(run / "interrupted.pt", agent, config, args.seed, frames, episodes)
        raise
    finally:
        metadata.update(total_frame_steps=frames, completed_episodes=episodes,
                        partial_episode_frames=env.frame_steps if not done else 0,
                        elapsed_seconds=time.perf_counter() - started)
        (run / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Tian-style DQN reimplementation (shared environment).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--total-frames", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", help="Relative to this baseline folder; refuses existing directories.")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
