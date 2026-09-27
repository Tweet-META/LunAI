"""Render the Yellow-gap diagnostic before training an agent on it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pygame

from rl.touhou_rl_env import TouhouRLEnv
from rl.yellow_gap_diagnostic import FIELD_WIDTH, WAVE_INTERVAL_FRAMES


def manual_action() -> int:
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    if up and left:
        return 5
    if up and right:
        return 6
    if down and left:
        return 7
    if down and right:
        return 8
    if up:
        return 1
    if down:
        return 2
    if left:
        return 3
    if right:
        return 4
    return 0


def oracle_action(env: TouhouRLEnv, player_x: float) -> int:
    """Follow the known gap; this checks reachability, not learning."""
    centers = env.scene.diagnostic_gap_centers
    if not centers:
        return 0
    wave = current_wave(env.frame_steps, len(centers))
    target = centers[wave]
    if player_x < target - 2.0:
        return 4
    if player_x > target + 2.0:
        return 3
    return 0


def current_wave(frame_steps: int, count: int) -> int:
    # The first wall fires on the first update; later walls fire at the configured interval.
    return min(max(0, (frame_steps - 1) // WAVE_INTERVAL_FRAMES), count - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--policy", choices=("oracle", "manual", "stay"), default="oracle")
    parser.add_argument("--debug", action="store_true", help="Show the PCCM observation panels.")
    args = parser.parse_args()

    env = TouhouRLEnv(
        render_mode="human",
        render_debug=args.debug,
        max_steps=1800,
        action_repeat=1,
        frame_stack=4,
        frame_stack_interval=1,
        level_file="level_yellow_gap_diagnostic.json",
    )
    try:
        for episode in range(args.episodes):
            observation = env.reset(seed=args.seed + episode)
            done = False
            while not done:
                player_x = float(observation["player_features"][0]) * FIELD_WIDTH
                if args.policy == "manual":
                    action = manual_action()
                elif args.policy == "oracle":
                    action = oracle_action(env, player_x)
                else:
                    action = 0
                observation, _, done, info = env.step(action)
            print(
                f"episode={episode + 1} seed={args.seed + episode} "
                f"frames={info['frame_steps']} collided={info['collided']} "
                f"gap_centers={','.join(str(int(x)) for x in env.scene.diagnostic_gap_centers)}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
