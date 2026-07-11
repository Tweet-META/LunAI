from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rl.cnn_observation_utils import cnn_observation, cnn_observation_shapes
from rl.ppo_cnn_agent import CNNPPOAgent, load_cnn_ppo_config
from rl.touhou_rl_env import TouhouRLEnv
from rl.train_ppo_cnn import validate_checkpoint_shapes


# Print policy probabilities with action names from highest to lowest.
def print_action_probs(env: TouhouRLEnv, probs: np.ndarray) -> None:
    order = np.argsort(probs)[::-1]
    summary = ", ".join(f"{env.ACTIONS[int(i)]}={probs[int(i)]:.3f}" for i in order)
    print(f"action_probs: {summary}")


# Evaluate a saved CNN PPO model.
def evaluate(args: argparse.Namespace) -> None:
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
    config = load_cnn_ppo_config(str(Path(args.model_path)), device=args.device)
    validate_checkpoint_shapes(config, shapes, args.frame_stack)

    agent = CNNPPOAgent(config)
    agent.load(str(Path(args.model_path)))
    episode_rewards = []
    episode_frames = []
    episode_decisions = []
    collisions = []

    try:
        for episode in range(1, args.episodes + 1):
            observation = env.reset(seed=args.seed + episode)
            state = cnn_observation(observation, env.get_map_history())
            if args.print_action_probs:
                print_action_probs(env, agent.action_probs(state))

            done = False
            total_reward = 0.0
            total_collisions = 0
            last_info = {}
            action_counts = np.zeros(9, dtype=np.int32)

            while not done:
                if args.stochastic:
                    action, _, _ = agent.select_action(state)
                else:
                    action = agent.select_greedy_action(state)
                action_counts[action] += 1
                observation, reward, done, info = env.step(action)
                state = cnn_observation(observation, env.get_map_history())
                total_reward += reward
                total_collisions += int(info.get("collided", False))
                last_info = info

            frames = int(last_info.get("frame_steps", 0))
            decisions = int(last_info.get("decision_steps", 0))
            episode_rewards.append(total_reward)
            episode_frames.append(frames)
            episode_decisions.append(decisions)
            collisions.append(total_collisions)
            print(
                f"episode={episode}, decision_steps={decisions}, frame_steps={frames}, reward={total_reward:.3f}, "
                f"hp={last_info.get('hp', 0)}, collisions={total_collisions}"
            )
            if args.print_actions:
                action_summary = ", ".join(f"{env.ACTIONS[i]}={count}" for i, count in enumerate(action_counts) if count > 0)
                player_features = observation["player_features"]
                print(f"actions: {action_summary}")
                print(f"final_player_xy=({player_features[0]:.3f}, {player_features[1]:.3f})")

        print("-" * 64)
        print(
            f"mean_decision_steps={np.mean(episode_decisions):.2f}, mean_frame_steps={np.mean(episode_frames):.2f}, "
            f"mean_reward={np.mean(episode_rewards):.3f}, mean_collisions={np.mean(collisions):.2f}"
        )
    finally:
        env.close()


# Build the command line parser for CNN PPO evaluation.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/ppo_cnn_baseline.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--frame-stack", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--level-file", type=str, default="level_1.json")
    parser.add_argument("--random-player-start", action="store_true")
    parser.add_argument("--player-start-margin", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--print-actions", action="store_true")
    parser.add_argument("--print-action-probs", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser


# Parse arguments and run CNN PPO evaluation.
def main() -> None:
    parser = build_arg_parser()
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
