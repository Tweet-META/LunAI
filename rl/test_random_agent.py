from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rl.touhou_rl_env import TouhouRLEnv


OBS_KEYS = (
    "blue_density",
    "blue_speed",
    "yellow_density",
    "yellow_speed",
    "red_occupancy",
    "red_vx",
    "red_vy",
    "red_speed",
    "player_features",
)


# Check observation shapes, finite values, and valid ranges.
def validate_observation(observation: dict[str, np.ndarray]) -> None:
    expected_shapes = {
        "blue_density": (6, 6),
        "blue_speed": (6, 6),
        "yellow_density": (8, 8),
        "yellow_speed": (8, 8),
        "red_occupancy": (32, 32),
        "red_vx": (32, 32),
        "red_vy": (32, 32),
        "red_speed": (32, 32),
        "player_features": (4,),
    }
    for key, shape in expected_shapes.items():
        value = observation[key]
        if value.shape != shape:
            raise AssertionError(f"{key} shape expected {shape}, got {value.shape}")
        if not np.isfinite(value).all():
            raise AssertionError(f"{key} contains NaN or inf")

    for key in ("blue_density", "blue_speed", "yellow_density", "yellow_speed", "red_occupancy", "red_speed"):
        value = observation[key]
        if value.min() < -1e-6 or value.max() > 1.0 + 1e-6:
            raise AssertionError(f"{key} should be in [0, 1], got min={value.min()}, max={value.max()}")

    for key in ("red_vx", "red_vy"):
        value = observation[key]
        if value.min() < -1.0 - 1e-6 or value.max() > 1.0 + 1e-6:
            raise AssertionError(f"{key} should be in [-1, 1], got min={value.min()}, max={value.max()}")


# Print shape and value range for each public observation key.
def print_ranges(observation: dict[str, np.ndarray]) -> None:
    for key in OBS_KEYS:
        value = observation[key]
        print(f"{key}: {value.shape}, min={value.min():.3f}, max={value.max():.3f}")


# Run random actions to smoke-test the environment.
def run_random_agent(episodes: int, max_steps: int, action_repeat: int, render: bool) -> None:
    env = TouhouRLEnv(render_mode="human" if render else None, max_steps=max_steps, action_repeat=action_repeat)
    episode_rewards = []
    episode_frame_steps = []
    episode_decision_steps = []
    episode_collisions = []
    try:
        for episode in range(episodes):
            observation = env.reset(seed=episode)
            validate_observation(observation)
            total_reward = 0.0
            total_collisions = 0
            done = False
            last_info = {}

            while not done:
                action = np.random.randint(0, 9)
                observation, reward, done, info = env.step(action)
                validate_observation(observation)
                total_reward += reward
                total_collisions += int(info.get("collided", False))
                last_info = info
            episode_rewards.append(total_reward)
            episode_frame_steps.append(int(last_info.get("frame_steps", 0)))
            episode_decision_steps.append(int(last_info.get("decision_steps", 0)))
            episode_collisions.append(total_collisions)
            print(
                f"episode={episode + 1}, decision_steps={last_info.get('decision_steps', 0)}, frame_steps={last_info.get('frame_steps', 0)}, "
                f"hp={last_info.get('hp', 0)}, reward={total_reward:.3f}, "
                f"bullets={last_info.get('bullets', 0)}, collisions={total_collisions}, collided={last_info.get('collided', False)}"
            )
            print_ranges(observation)
            print("-" * 64)

        print("Random agent summary")
        print("-" * 64)
        print(f"episodes={episodes}, max_steps={max_steps}, action_repeat={action_repeat}")
        print(f"mean_decision_steps={np.mean(episode_decision_steps):.2f}")
        print(f"mean_frame_steps={np.mean(episode_frame_steps):.2f}")
        print(f"min_frame_steps={np.min(episode_frame_steps)}")
        print(f"max_frame_steps={np.max(episode_frame_steps)}")
        print(f"mean_reward={np.mean(episode_rewards):.3f}")
        print(f"mean_collisions={np.mean(episode_collisions):.2f}")
    finally:
        env.close()


# Parse command line arguments and run the random test.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    run_random_agent(args.episodes, args.max_steps, args.action_repeat, args.render)


if __name__ == "__main__":
    main()
