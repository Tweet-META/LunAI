from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rl.observation_utils import flatten_observation, observation_dim, validate_flat_observation
from rl.ppo_agent import PPOAgent, PPOConfig
from rl.touhou_rl_env import TouhouRLEnv


# Create the parent directory for an output path.
def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# Create a CSV log file with the PPO training header.
def write_log_header(path: Path) -> None:
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "update",
                "global_step",
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
        )


# Append one row to the PPO training log.
def append_log(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


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


# Collect one on-policy rollout from the environment.
def collect_rollout(
    env: TouhouRLEnv,
    agent: PPOAgent,
    state: np.ndarray,
    args: argparse.Namespace,
    counters: dict[str, int],
    log_path: Path,
    latest_metrics: dict[str, float],
) -> tuple[dict[str, list], np.ndarray]:
    rollout = {
        "states": [],
        "actions": [],
        "log_probs": [],
        "rewards": [],
        "dones": [],
        "values": [],
    }
    episode_reward = counters.get("episode_reward", 0.0)
    episode_collisions = counters.get("episode_collisions", 0)
    last_info = {}

    while len(rollout["states"]) < args.rollout_steps and counters["episode"] <= args.episodes:
        action, log_prob, value = agent.select_action(state)
        next_observation, reward, done, info = env.step(action)
        next_state = flatten_observation(next_observation)
        validate_flat_observation(next_state)

        rollout["states"].append(state)
        rollout["actions"].append(action)
        rollout["log_probs"].append(log_prob)
        rollout["rewards"].append(float(reward))
        rollout["dones"].append(bool(done))
        rollout["values"].append(value)

        state = next_state
        episode_reward += float(reward)
        episode_collisions += int(info.get("collided", False))
        counters["global_step"] += 1
        last_info = info

        if done:
            append_log(
                log_path,
                [
                    counters["episode"],
                    counters["update"],
                    counters["global_step"],
                    info.get("decision_steps", 0),
                    info.get("frame_steps", 0),
                    f"{episode_reward:.6f}",
                    f"{latest_metrics.get('policy_loss', 0.0):.6f}",
                    f"{latest_metrics.get('value_loss', 0.0):.6f}",
                    f"{latest_metrics.get('entropy', 0.0):.6f}",
                    f"{latest_metrics.get('approx_kl', 0.0):.6f}",
                    info.get("hp", 0),
                    episode_collisions,
                ],
            )
            print(
                f"episode={counters['episode']}, update={counters['update']}, decision_steps={info.get('decision_steps', 0)}, "
                f"frame_steps={info.get('frame_steps', 0)}, reward={episode_reward:.3f}, "
                f"policy_loss={latest_metrics.get('policy_loss', 0.0):.5f}, value_loss={latest_metrics.get('value_loss', 0.0):.5f}, "
                f"entropy={latest_metrics.get('entropy', 0.0):.5f}, kl={latest_metrics.get('approx_kl', 0.0):.5f}, "
                f"hp={info.get('hp', 0)}, collisions={episode_collisions}"
            )
            counters["episode"] += 1
            episode_reward = 0.0
            episode_collisions = 0
            observation = env.reset(seed=args.seed + counters["episode"])
            state = flatten_observation(observation)
            validate_flat_observation(state)

    counters["episode_reward"] = episode_reward
    counters["episode_collisions"] = episode_collisions
    counters["last_done"] = bool(last_info and last_info.get("collided", False))
    return rollout, state


# Train a PPO agent on the Touhou RL environment.
def train(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = TouhouRLEnv(render_mode="human" if args.render else None, max_steps=args.max_steps, action_repeat=args.action_repeat)
    first_observation = env.reset(seed=args.seed)
    state_dim = observation_dim(first_observation)
    agent = PPOAgent(
        PPOConfig(
            state_dim=state_dim,
            action_dim=9,
            hidden_dim_1=args.hidden_dim_1,
            hidden_dim_2=args.hidden_dim_2,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            target_kl=args.target_kl,
            device=args.device,
        )
    )

    log_path = Path(args.log_path)
    model_path = Path(args.model_path)
    write_log_header(log_path)
    ensure_parent_dir(model_path)

    state = flatten_observation(first_observation)
    validate_flat_observation(state)
    counters = {
        "episode": 1,
        "update": 0,
        "global_step": 0,
        "episode_reward": 0.0,
        "episode_collisions": 0,
        "last_done": False,
    }
    latest_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

    try:
        while counters["episode"] <= args.episodes:
            rollout, state = collect_rollout(env, agent, state, args, counters, log_path, latest_metrics)
            if not rollout["states"]:
                break

            last_value = 0.0 if rollout["dones"][-1] else agent.state_value(state)
            advantages, returns = compute_gae(
                rollout["rewards"],
                rollout["dones"],
                rollout["values"],
                last_value,
                args.gamma,
                args.gae_lambda,
            )
            latest_metrics = agent.update(
                np.asarray(rollout["states"], dtype=np.float32),
                np.asarray(rollout["actions"], dtype=np.int64),
                np.asarray(rollout["log_probs"], dtype=np.float32),
                returns,
                advantages,
            )
            counters["update"] += 1

            if counters["update"] % args.save_interval == 0:
                agent.save(str(model_path))

        agent.save(str(model_path))
    finally:
        env.close()


# Build the command line parser for PPO training.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--hidden-dim-1", type=int, default=64)
    parser.add_argument("--hidden-dim-2", type=int, default=32)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--model-path", type=str, default="checkpoints/ppo_baseline.pt")
    parser.add_argument("--log-path", type=str, default="training_logs/ppo_log.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render", action="store_true")
    return parser


# Parse arguments and start PPO training.
def main() -> None:
    parser = build_arg_parser()
    train(parser.parse_args())


if __name__ == "__main__":
    main()
