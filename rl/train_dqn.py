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

from rl.dqn_agent import DQNAgent, DQNConfig
from rl.observation_utils import flatten_observation, observation_dim, validate_flat_observation
from rl.replay_buffer import ReplayBuffer
from rl.touhou_rl_env import TouhouRLEnv


# Compute a linearly decayed epsilon value.
def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    ratio = min(1.0, step / decay_steps)
    return start + ratio * (end - start)


# Create the parent directory for an output path.
def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# Create a CSV log file with the training header.
def write_log_header(path: Path) -> None:
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "global_step", "decision_steps", "frame_steps", "episode_reward", "epsilon", "mean_loss", "hp", "collisions"])


# Append one row to the training CSV log.
def append_log(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# Train a baseline DQN agent on the Touhou RL environment.
def train(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = TouhouRLEnv(render_mode="human" if args.render else None, max_steps=args.max_steps, action_repeat=args.action_repeat)
    first_observation = env.reset(seed=args.seed)
    state_dim = observation_dim(first_observation)
    agent = DQNAgent(
        DQNConfig(
            state_dim=state_dim,
            action_dim=9,
            hidden_dim_1=args.hidden_dim_1,
            hidden_dim_2=args.hidden_dim_2,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            target_update_interval=args.target_update_interval,
            device=args.device,
        )
    )
    replay_buffer = ReplayBuffer(args.buffer_size, state_dim)
    log_path = Path(args.log_path)
    model_path = Path(args.model_path)
    write_log_header(log_path)
    ensure_parent_dir(model_path)

    global_step = 0
    try:
        for episode in range(1, args.episodes + 1):
            observation = env.reset(seed=args.seed + episode)
            state = flatten_observation(observation)
            validate_flat_observation(state)
            done = False
            episode_reward = 0.0
            decision_steps = 0
            collisions = 0
            losses = []
            info = {}

            while not done:
                epsilon = linear_epsilon(global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
                action = agent.select_action(state, epsilon)
                next_observation, reward, done, info = env.step(action)
                next_state = flatten_observation(next_observation)
                validate_flat_observation(next_state)
                replay_buffer.add(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                decision_steps += 1
                global_step += 1
                collisions += int(info.get("collided", False))

                if len(replay_buffer) >= args.learning_starts and len(replay_buffer) >= args.batch_size:
                    batch = replay_buffer.sample(args.batch_size)
                    losses.append(agent.update(batch))

            mean_loss = float(np.mean(losses)) if losses else 0.0
            epsilon = linear_epsilon(global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
            append_log(
                log_path,
                [
                    episode,
                    global_step,
                    decision_steps,
                    info.get("frame_steps", 0),
                    f"{episode_reward:.6f}",
                    f"{epsilon:.6f}",
                    f"{mean_loss:.6f}",
                    info.get("hp", 0),
                    collisions,
                ],
            )
            print(
                f"episode={episode}, decision_steps={decision_steps}, frame_steps={info.get('frame_steps', 0)}, reward={episode_reward:.3f}, "
                f"epsilon={epsilon:.3f}, loss={mean_loss:.5f}, hp={info.get('hp', 0)}, collisions={collisions}"
            )

            if episode % args.save_interval == 0:
                agent.save(str(model_path))

        agent.save(str(model_path))
    finally:
        env.close()


# Build the command line parser for DQN training.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim-1", type=int, default=64)
    parser.add_argument("--hidden-dim-2", type=int, default=32)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50000)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--model-path", type=str, default="checkpoints/dqn_baseline.pt")
    parser.add_argument("--log-path", type=str, default="training_logs/dqn_log.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render", action="store_true")
    return parser


# Parse arguments and start DQN training.
def main() -> None:
    parser = build_arg_parser()
    train(parser.parse_args())


if __name__ == "__main__":
    main()
