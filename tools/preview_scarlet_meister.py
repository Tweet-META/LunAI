"""Preview Hard Scarlet Meister without a trained checkpoint (arrows/WASD)."""

import argparse
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pygame
from rl.touhou_rl_env import TouhouRLEnv
from tools.preview_yellow_gap import manual_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=10001)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--policy', choices=('manual', 'stay'), default='manual')
    parser.add_argument('--invincible', action='store_true', help='Preview only: continue after collisions.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--pccm-implementation', choices=('reference', 'numba'), default='numba')
    args = parser.parse_args()
    env = TouhouRLEnv(
        render_mode='human',
        max_steps=1800, action_repeat=1, frame_stack=4,
        level_file='level_th06_stage6_scarlet_meister_hard.json',
        render_debug=args.debug, pccm_implementation=args.pccm_implementation,
    )
    env.render_mode = None
    # Handle quit before stepping; draw explicitly to avoid the environment's
    # automatic render consuming the quit event and closing mid-episode.
    try:
        for episode in range(args.episodes):
            env.reset(seed=args.seed + episode)
            env.scene.player.training_invincible = args.invincible
            pygame.display.set_caption('Scarlet Meister / Hard' + (' / INVINCIBLE PREVIEW' if args.invincible else ''))
            contacts = 0
            for _ in range(1800):
                if any(event.type == pygame.QUIT for event in pygame.event.get()):
                    return
                if pygame.key.get_pressed()[pygame.K_ESCAPE]:
                    return
                action = manual_action() if args.policy == 'manual' else 0
                _, _, done, info = env.step(action)
                contacts += int(info['collided'])
                env.scene.render(env.screen, env.clock)
                if args.debug:
                    from tools.visualization_debug import draw_observation_panels
                    draw_observation_panels(env.screen, env.last_observation)
                env._draw_reward_panel()
                pygame.display.flip()
                env.clock.tick(env.FPS)
                if done and not args.invincible:
                    break
            print(f"episode={episode+1} frames={env.frame_steps} contact_frames={contacts} preview_invincible={args.invincible}")
    finally:
        env.close()


if __name__ == '__main__':
    main()
