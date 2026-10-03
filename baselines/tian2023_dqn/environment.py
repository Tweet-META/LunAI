from __future__ import annotations

import numpy as np
import pygame

from observation_sources import scene_to_observation_state
from rl.touhou_rl_env import TouhouRLEnv
from .rays import cast_rays


class RayTouhouEnv(TouhouRLEnv):
    """Shared game physics/reward, independent ray observation adapter.

    Only observation['rays'] reaches DQN. player_features serves the existing
    wall reward/logger; it is NOT concatenated to the network input.
    """

    def __init__(self, *, levels, max_frames=1800, ray_count=36, render=False):
        self.ray_count = ray_count
        super().__init__(
            level_files=levels, max_steps=max_frames, action_repeat=1,
            frame_stack=1, pccm_reward_weight=0.0,
            pccm_observation_mode="occupancy_only", pccm_implementation="reference",
            render_mode="human" if render else None,
        )
        self.MAP_KEYS = ()  # No map construction or frame stacking in this adapter.

    def get_observation(self):
        bullets, player, self.previous_enemy_positions = scene_to_observation_state(
            self.scene, self.GAME_ZONE, self.previous_action,
            self.previous_enemy_positions, 1 / self.FPS,
        )
        width, height = self.GAME_ZONE[2:]
        return {
            "rays": cast_rays(bullets, player, width, height, self.ray_count),
            "player_features": np.clip(np.asarray([player.x / width, player.y / height], dtype=np.float32), 0, 1),
        }

    def render(self):
        if any(event.type == pygame.QUIT or
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
               for event in pygame.event.get()):
            raise KeyboardInterrupt
        super().render()
