from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rl.cnn_observation_utils import CNNObservation, cnn_observation, cnn_observation_shapes, stack_cnn_observations
from rl.ppo_cnn_agent import CNNPPOAgent, CNNPPOConfig, load_cnn_ppo_config
from rl.touhou_rl_env import TouhouRLEnv
from rl.train_ppo import (
    append_log,
    compute_gae,
    ensure_parent_dir,
    linear_schedule,
    training_limit_reached,
    training_progress,
    training_stop_reason,
    write_log_header,
)


# Apply scheduled PPO hyperparameters before one update.
def update_scheduled_hyperparams(agent: CNNPPOAgent, args: argparse.Namespace, counters: dict[str, int]) -> None:
    progress = training_progress(args, counters)

    entropy_coef_final = args.entropy_coef if args.entropy_coef_final < 0.0 else args.entropy_coef_final
    learning_rate_final = args.learning_rate if args.learning_rate_final < 0.0 else args.learning_rate_final
    agent.config.entropy_coef = linear_schedule(args.entropy_coef, entropy_coef_final, progress)

    scheduled_lr = linear_schedule(args.learning_rate, learning_rate_final, progress)
    for group in agent.optimizer.param_groups:
        group["lr"] = scheduled_lr


# Check that a loaded checkpoint matches the current observation shapes.
def validate_checkpoint_shapes(
    config: CNNPPOConfig,
    shapes: dict[str, tuple[int, ...]],
    frame_stack: int,
) -> None:
    if config.frame_stack != frame_stack:
        raise ValueError(
            f"Checkpoint frame_stack={config.frame_stack}, but --frame-stack={frame_stack}."
        )
    if config.red_shape != shapes["red"]:
        raise ValueError(f"Checkpoint red_shape={config.red_shape}, but environment red_shape={shapes['red']}.")
    if config.yellow_shape != shapes["yellow"]:
        raise ValueError(f"Checkpoint yellow_shape={config.yellow_shape}, but environment yellow_shape={shapes['yellow']}.")
    if config.blue_shape != shapes["blue"]:
        raise ValueError(f"Checkpoint blue_shape={config.blue_shape}, but environment blue_shape={shapes['blue']}.")
    if config.player_dim != shapes["player"][0]:
        raise ValueError(f"Checkpoint player_dim={config.player_dim}, but environment player_dim={shapes['player'][0]}.")


# Collect one on-policy rollout from the environment.
def collect_rollout(
    env: TouhouRLEnv,
    agent: CNNPPOAgent,
    state: CNNObservation,
    args: argparse.Namespace,
    counters: dict[str, int],
    log_path: Path,
    latest_metrics: dict[str, float],
) -> tuple[dict[str, list], CNNObservation]:
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

    while (
        len(rollout["states"]) < args.rollout_steps
        and counters["episode"] <= args.episodes
        and not training_limit_reached(args, counters)
    ):
        action, log_prob, value = agent.select_action(state)
        next_observation, reward, done, info = env.step(action)
        next_state = cnn_observation(next_observation, env.get_map_history())

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
        current_frame_steps = int(info.get("frame_steps", 0))
        frame_delta = max(0, current_frame_steps - counters["episode_frame_steps"])
        counters["episode_frame_steps"] = current_frame_steps
        counters["total_frame_steps"] += frame_delta
        last_info = info

        if done:
            append_log(
                log_path,
                [
                    counters["episode"],
                    counters["update"],
                    counters["global_step"],
                    counters["total_frame_steps"],
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
                f"frame_steps={info.get('frame_steps', 0)}, total_frame_steps={counters['total_frame_steps']}, reward={episode_reward:.3f}, "
                f"policy_loss={latest_metrics.get('policy_loss', 0.0):.5f}, value_loss={latest_metrics.get('value_loss', 0.0):.5f}, "
                f"entropy={latest_metrics.get('entropy', 0.0):.5f}, kl={latest_metrics.get('approx_kl', 0.0):.5f}, "
                f"hp={info.get('hp', 0)}, collisions={episode_collisions}"
            )
            counters["episode"] += 1
            episode_reward = 0.0
            episode_collisions = 0
            counters["episode_frame_steps"] = 0
            if counters["episode"] <= args.episodes and not training_limit_reached(args, counters):
                observation = env.reset(seed=args.seed + counters["episode"])
                state = cnn_observation(observation, env.get_map_history())

    counters["episode_reward"] = episode_reward
    counters["episode_collisions"] = episode_collisions
    counters["last_done"] = bool(last_info and last_info.get("collided", False))
    return rollout, state


# Train a CNN PPO agent on the Touhou RL environment.
def train(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = TouhouRLEnv(
        render_mode="human" if args.render else None,
        max_steps=args.max_steps,
        action_repeat=args.action_repeat,
        level_file=args.level_file,
        random_player_start=args.random_player_start,
        player_start_margin=args.player_start_margin,
        frame_stack=args.frame_stack,
    )
    first_observation = env.reset(seed=args.seed)
    shapes = cnn_observation_shapes(first_observation, env.get_map_history())

    if args.load_path:
        load_path = Path(args.load_path)
        config = load_cnn_ppo_config(str(load_path), device=args.device)
        validate_checkpoint_shapes(config, shapes, args.frame_stack)
        config.gamma = args.gamma
        config.gae_lambda = args.gae_lambda
        config.learning_rate = args.learning_rate
        config.clip_range = args.clip_range
        config.entropy_coef = args.entropy_coef
        config.value_coef = args.value_coef
        config.max_grad_norm = args.max_grad_norm
        config.update_epochs = args.update_epochs
        config.minibatch_size = args.minibatch_size
        config.target_kl = args.target_kl
        agent = CNNPPOAgent(config)
        agent.load(str(load_path))
        for group in agent.optimizer.param_groups:
            group["lr"] = args.learning_rate
        print(f"Loaded CNN PPO checkpoint from {load_path}")
    else:
        agent = CNNPPOAgent(
            CNNPPOConfig(
                red_shape=shapes["red"],
                yellow_shape=shapes["yellow"],
                blue_shape=shapes["blue"],
                player_dim=shapes["player"][0],
                frame_stack=args.frame_stack,
                action_dim=9,
                hidden_dim=args.hidden_dim,
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

    state = cnn_observation(first_observation, env.get_map_history())
    counters = {
        "episode": 1,
        "update": 0,
        "global_step": 0,
        "total_frame_steps": 0,
        "episode_frame_steps": 0,
        "episode_reward": 0.0,
        "episode_collisions": 0,
        "last_done": False,
    }
    latest_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

    try:
        while counters["episode"] <= args.episodes and not training_limit_reached(args, counters):
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
            update_scheduled_hyperparams(agent, args, counters)
            latest_metrics = agent.update(
                stack_cnn_observations(rollout["states"]),
                np.asarray(rollout["actions"], dtype=np.int64),
                np.asarray(rollout["log_probs"], dtype=np.float32),
                returns,
                advantages,
            )
            counters["update"] += 1

            if counters["update"] % args.save_interval == 0:
                agent.save(str(model_path))

        agent.save(str(model_path))
        print(
            f"Training finished: reason={training_stop_reason(args, counters)}, "
            f"episodes_completed={counters['episode'] - 1}, decision_steps={counters['global_step']}, "
            f"total_frame_steps={counters['total_frame_steps']}"
        )
    finally:
        env.close()


# Build the command line parser for CNN PPO training.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--max-total-frame-steps", type=int, default=0)
    parser.add_argument("--max-total-decision-steps", type=int, default=0)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--frame-stack", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--level-file", type=str, default="level_1.json")
    parser.add_argument("--random-player-start", action="store_true")
    parser.add_argument("--player-start-margin", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-rate-final", type=float, default=-1.0)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--entropy-coef-final", type=float, default=-1.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--model-path", type=str, default="checkpoints/ppo_cnn_baseline.pt")
    parser.add_argument("--log-path", type=str, default="training_logs/ppo_cnn_log.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render", action="store_true")
    return parser


# Parse arguments and start CNN PPO training.
def main() -> None:
    parser = build_arg_parser()
    train(parser.parse_args())


if __name__ == "__main__":
    main()
