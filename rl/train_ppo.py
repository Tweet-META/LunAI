from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rl.observation_utils import flatten_observation, observation_dim, validate_flat_observation
from rl.ppo_agent import PPOAgent, PPOConfig, load_ppo_config
from rl.touhou_rl_env import TouhouRLEnv


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


# Append one row to the PPO training log.
def append_log(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


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


# Apply scheduled PPO hyperparameters before one update.
def update_scheduled_hyperparams(agent: PPOAgent, args: argparse.Namespace, counters: dict[str, int]) -> None:
    progress = training_progress(args, counters)

    entropy_coef_final = args.entropy_coef if args.entropy_coef_final < 0.0 else args.entropy_coef_final
    learning_rate_final = args.learning_rate if args.learning_rate_final < 0.0 else args.learning_rate_final
    agent.config.entropy_coef = linear_schedule(args.entropy_coef, entropy_coef_final, progress)

    scheduled_lr = linear_schedule(args.learning_rate, learning_rate_final, progress)
    for group in agent.optimizer.param_groups:
        group["lr"] = scheduled_lr


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

    while (
        len(rollout["states"]) < args.rollout_steps
        and counters["episode"] <= args.episodes
        and not training_limit_reached(args, counters)
    ):
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

    env = TouhouRLEnv(
        render_mode="human" if args.render or args.render_debug else None,
        max_steps=args.max_steps,
        action_repeat=args.action_repeat,
        level_file=args.level_file,
        level_files=args.level_files,
        level_spawn_time_jitter=args.level_spawn_time_jitter,
        random_player_start=args.random_player_start,
        player_start_margin=args.player_start_margin,
        reward_gamma=args.gamma,
        danger_shaping_enabled=args.danger_shaping_enabled,
        wall_shaping_weight=args.wall_shaping_weight,
        wall_state_penalty_weight=args.wall_state_penalty_weight,
        upper_field_penalty_weight=args.upper_field_penalty_weight,
        lower_field_threshold=args.lower_field_threshold,
        render_debug=args.render_debug,
    )
    first_observation = env.reset(seed=args.seed)
    state_dim = observation_dim(first_observation)
    if args.load_path:
        load_path = Path(args.load_path)
        config = load_ppo_config(str(load_path), device=args.device)
        if config.state_dim != state_dim:
            raise ValueError(f"Checkpoint state_dim={config.state_dim}, but environment state_dim={state_dim}.")
        if config.action_dim != 9:
            raise ValueError(f"Checkpoint action_dim={config.action_dim}, but environment action_dim=9.")
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
        agent = PPOAgent(config)
        agent.load(str(load_path))
        for group in agent.optimizer.param_groups:
            group["lr"] = args.learning_rate
        print(f"Loaded PPO checkpoint from {load_path}")
    else:
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
    write_log_header(log_path, vars(args))
    ensure_parent_dir(model_path)

    state = flatten_observation(first_observation)
    validate_flat_observation(state)
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
        print(
            f"Training finished: reason={training_stop_reason(args, counters)}, "
            f"episodes_completed={counters['episode'] - 1}, decision_steps={counters['global_step']}, "
            f"total_frame_steps={counters['total_frame_steps']}"
        )
    finally:
        env.close()


# Build the command line parser for PPO training.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--max-total-frame-steps", type=int, default=0)
    parser.add_argument("--max-total-decision-steps", type=int, default=0)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--level-file", type=str, default="level_1.json")
    parser.add_argument("--level-files", nargs="*", default=[])
    parser.add_argument("--level-spawn-time-jitter", type=float, default=0.0)
    parser.add_argument("--random-player-start", action="store_true")
    parser.add_argument("--player-start-margin", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--danger-shaping", action=argparse.BooleanOptionalAction, default=True, dest="danger_shaping_enabled")
    parser.add_argument("--wall-shaping-weight", type=float, default=0.01)
    parser.add_argument("--wall-state-penalty-weight", type=float, default=0.0)
    parser.add_argument("--upper-field-penalty-weight", type=float, default=0.0)
    parser.add_argument("--lower-field-threshold", type=float, default=0.70)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-rate-final", type=float, default=-1.0)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--entropy-coef-final", type=float, default=-1.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--hidden-dim-1", type=int, default=64)
    parser.add_argument("--hidden-dim-2", type=int, default=32)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--model-path", type=str, default="checkpoints/ppo_baseline.pt")
    parser.add_argument("--log-path", type=str, default="training_logs/ppo_log.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-debug", action="store_true")
    return parser


# Parse arguments and start PPO training.
def main() -> None:
    parser = build_arg_parser()
    train(parse_args_with_config(parser))


if __name__ == "__main__":
    main()
