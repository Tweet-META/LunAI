from __future__ import annotations

import os
import random
import sys
from collections.abc import Sequence
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from rl.reward import (
    compute_frame_reward,
    danger_potential_shaping,
    local_pccm_cost,
    pccm_transition_shaping,
    upper_field_state_penalty,
    wall_proximity,
    wall_proximity_shaping,
    wall_state_penalty,
)
from rl.cnn_observation_utils import cnn_map_keys


PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


class TouhouRLEnv:
    ACTIONS = {
        0: "stay",
        1: "up",
        2: "down",
        3: "left",
        4: "right",
        5: "up_left",
        6: "up_right",
        7: "down_left",
        8: "down_right",
    }

    # Create the RL wrapper around the existing GameScene.
    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int | None = None,
        action_repeat: int = 3,
        level_file: str = "level_1.json",
        level_files: Sequence[str] | None = None,
        level_spawn_time_jitter: float = 0.0,
        random_player_start: bool = False,
        player_start_margin: float = 80.0,
        frame_stack: int = 1,
        frame_stack_interval: int = 1,
        training_invincible: bool = False,
        reward_gamma: float = 0.99,
        danger_shaping_enabled: bool = True,
        wall_shaping_weight: float = 0.01,
        wall_state_penalty_weight: float = 0.0,
        upper_field_penalty_weight: float = 0.0,
        lower_field_threshold: float = 0.70,
        observation_schema: str = "motion",
        pccm_shaping_weight: float = 0.0,
        pccm_prediction_frames: int = 5,
        pccm_halo_width: float = 24.0,
        pccm_wall_margin: float = 0.12,
        render_debug: bool = False,
    ):
        if not 1 <= int(frame_stack) <= 5:
            raise ValueError(f"frame_stack must be in 1..5, got {frame_stack}.")
        if not 1 <= int(frame_stack_interval) <= 5:
            raise ValueError(f"frame_stack_interval must be in 1..5, got {frame_stack_interval}.")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_repeat = max(1, int(action_repeat))
        configured_levels = tuple(str(path) for path in (level_files or (level_file,)))
        if not configured_levels or any(not path for path in configured_levels):
            raise ValueError("At least one non-empty level file is required.")
        if float(level_spawn_time_jitter) < 0.0:
            raise ValueError(f"level_spawn_time_jitter must be non-negative, got {level_spawn_time_jitter}.")
        self.level_file = level_file
        self.level_files = configured_levels
        self.level_spawn_time_jitter = float(level_spawn_time_jitter)
        self.current_level_file = configured_levels[0]
        self.random_player_start = random_player_start
        self.player_start_margin = float(player_start_margin)
        self.frame_stack = int(frame_stack)
        self.frame_stack_interval = int(frame_stack_interval)
        self.map_history_size = 1 + (self.frame_stack - 1) * self.frame_stack_interval
        self.training_invincible = bool(training_invincible)
        if not 0.0 <= float(reward_gamma) <= 1.0:
            raise ValueError(f"reward_gamma must be in [0, 1], got {reward_gamma}.")
        self.reward_gamma = float(reward_gamma)
        self.danger_shaping_enabled = bool(danger_shaping_enabled)
        if float(wall_shaping_weight) < 0.0:
            raise ValueError(f"wall_shaping_weight must be non-negative, got {wall_shaping_weight}.")
        self.wall_shaping_weight = float(wall_shaping_weight)
        if float(wall_state_penalty_weight) < 0.0:
            raise ValueError(f"wall_state_penalty_weight must be non-negative, got {wall_state_penalty_weight}.")
        self.wall_state_penalty_weight = float(wall_state_penalty_weight)
        if float(upper_field_penalty_weight) < 0.0:
            raise ValueError(f"upper_field_penalty_weight must be non-negative, got {upper_field_penalty_weight}.")
        if not 0.0 < float(lower_field_threshold) <= 1.0:
            raise ValueError(f"lower_field_threshold must be in (0, 1], got {lower_field_threshold}.")
        self.upper_field_penalty_weight = float(upper_field_penalty_weight)
        self.lower_field_threshold = float(lower_field_threshold)
        if observation_schema not in {"motion", "pccm"}:
            raise ValueError(f"Unknown observation schema: {observation_schema}.")
        if float(pccm_shaping_weight) < 0.0:
            raise ValueError(f"PCCM shaping weight must be non-negative, got {pccm_shaping_weight}.")
        self.observation_schema = observation_schema
        self.MAP_KEYS = cnn_map_keys(observation_schema)
        self.pccm_shaping_weight = float(pccm_shaping_weight)
        self.pccm_prediction_frames = int(pccm_prediction_frames)
        self.pccm_halo_width = float(pccm_halo_width)
        self.pccm_wall_margin = float(pccm_wall_margin)
        self.render_debug = bool(render_debug)
        self._configure_pygame()

        from assets.scripts.math_and_data.enviroment import FPS, GAME_ZONE, SIZE, db_module
        from assets.scripts.math_and_data.Vector2 import Vector2
        from observation_builder import ObservationBuilder, ObservationConfig

        self.FPS = FPS
        self.GAME_ZONE = GAME_ZONE
        self.SIZE = SIZE
        self.Vector2 = Vector2
        self.db_module = db_module
        self.builder = ObservationBuilder(
            ObservationConfig(
                playfield_width=GAME_ZONE[2],
                playfield_height=GAME_ZONE[3],
                blue_grid=(8, 8),
                yellow_size=(320, 320),
                yellow_grid=(16, 16),
                red_size=(128, 128),
                red_map=(64, 64),
                max_speed=500.0,
                observation_schema=self.observation_schema,
                pccm_prediction_frames=self.pccm_prediction_frames,
                pccm_halo_width=self.pccm_halo_width,
                pccm_wall_margin=self.pccm_wall_margin,
                pccm_debug=False,
            )
        )

        self.scene = None
        self.screen = None
        self.clock = pygame.time.Clock()
        self.steps = 0
        self.frame_steps = 0
        self.previous_action = 0
        self.previous_enemy_positions: dict[int, tuple[float, float]] = {}
        self.map_history: deque[dict[str, np.ndarray]] = deque(maxlen=self.map_history_size)
        self.last_hp = 0
        self.last_observation = None
        self.last_reward = 0.0
        self.episode_reward = 0.0
        self.last_collided = False
        self.last_pccm_cost = 0.0
        self.last_pccm_shaping_reward = 0.0
        self.episode_pccm_cost_sum = 0.0
        self.episode_pccm_shaping_reward = 0.0
        self.episode_wall_frames = 0
        self.episode_action_counts = np.zeros(len(self.ACTIONS), dtype=np.int64)

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(SIZE, pygame.DOUBLEBUF, 16)
            pygame.display.set_caption("Touhou RL Env")

    # Initialize pygame for either human rendering or headless training.
    def _configure_pygame(self) -> None:
        if self.render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        pygame.init()
        try:
            pygame.mixer.pre_init(48000, -16, 2, 4096)
            pygame.mixer.init()
        except pygame.error:
            pass
        if self.render_mode != "human":
            pygame.display.set_mode((1, 1))

    # Start a new episode and return the first observation.
    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        from assets.scripts.scenes.GameScene import GameScene

        self.current_level_file = random.choice(self.level_files)
        self.scene = GameScene(level_file=self.current_level_file)
        if self.level_spawn_time_jitter > 0.0:
            for enemy_data in self.scene.level_enemies:
                enemy_data["time"] = float(enemy_data["time"]) + random.uniform(0.0, self.level_spawn_time_jitter)
            self.scene.level_enemies.sort(key=lambda enemy: enemy["time"])
        self.scene.player.training_invincible = self.training_invincible
        if self.random_player_start:
            self._randomize_player_start()
        self.steps = 0
        self.frame_steps = 0
        self.previous_action = 0
        self.previous_enemy_positions = {}
        self.last_hp = self.scene.player.hp
        self.last_observation = self.get_observation()
        self._reset_map_history(self.last_observation)
        self.last_reward = 0.0
        self.episode_reward = 0.0
        self.last_collided = False
        self.last_pccm_cost = local_pccm_cost(self.last_observation)
        self.last_pccm_shaping_reward = 0.0
        self.episode_pccm_cost_sum = 0.0
        self.episode_pccm_shaping_reward = 0.0
        self.episode_wall_frames = 0
        self.episode_action_counts.fill(0)
        return self.last_observation

    # Apply one action, advance one frame, and return the RL step data.
    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if self.scene is None:
            self.reset()
        action = int(action)
        if action not in self.ACTIONS:
            raise ValueError(f"Action must be in 0..8, got {action}.")

        direction = self._action_to_vector(action)
        total_reward = 0.0
        contact_frames = 0
        collided = False
        done = False
        observation = None
        decision_start_observation = self.last_observation
        if decision_start_observation is None:
            raise RuntimeError("The environment has no observation before stepping.")
        previous_action_for_reward = self.previous_action
        self.previous_action = action
        self.steps += 1
        self.episode_action_counts[action] += 1
        previous_frame_observation = decision_start_observation
        action_pccm_shaping_reward = 0.0

        for repeat_index in range(self.action_repeat):
            self.scene.player.slow = False
            self.scene.player.move(direction)
            self.scene.update(1 / self.FPS)

            self.frame_steps += 1
            observation = self.get_observation()
            self._append_map_snapshot(observation)
            collided = self.scene.player.collided_this_frame
            contact_frames += int(collided)
            done = self._is_done(collided)
            frame_reward = compute_frame_reward(action, previous_action_for_reward, collided)
            frame_reward -= wall_state_penalty(observation, self.wall_state_penalty_weight)
            frame_reward -= upper_field_state_penalty(
                observation,
                self.upper_field_penalty_weight,
                self.lower_field_threshold,
            )
            pccm_reward = pccm_transition_shaping(
                previous_frame_observation,
                observation,
                collided,
                self.pccm_shaping_weight,
            )
            frame_reward += pccm_reward
            action_pccm_shaping_reward += pccm_reward
            if done or repeat_index == self.action_repeat - 1:
                if self.danger_shaping_enabled:
                    frame_reward += danger_potential_shaping(
                        decision_start_observation,
                        observation,
                        done,
                        self.reward_gamma,
                    )
                frame_reward += wall_proximity_shaping(
                    decision_start_observation,
                    observation,
                    self.wall_shaping_weight,
                )
            total_reward += frame_reward
            self.last_hp = self.scene.player.hp
            self.last_observation = observation
            self.last_reward = frame_reward
            self.episode_reward += frame_reward
            self.last_collided = collided
            self.last_pccm_cost = local_pccm_cost(observation)
            self.last_pccm_shaping_reward = pccm_reward
            self.episode_pccm_cost_sum += self.last_pccm_cost
            self.episode_pccm_shaping_reward += pccm_reward
            self.episode_wall_frames += int(wall_proximity(observation) > 0.0)
            previous_action_for_reward = action
            previous_frame_observation = observation
            if self.render_mode == "human":
                self.render()
            if done:
                break

        info = {
            "time": self.scene.time,
            "hp": self.scene.player.hp,
            "collided": collided,
            "contact_frames": contact_frames,
            "steps": self.frame_steps,
            "decision_steps": self.steps,
            "frame_steps": self.frame_steps,
            "bullets": len(self.scene.enemy_bullets) + len(self.scene.enemies),
            "action_repeat": self.action_repeat,
            "level_file": self.current_level_file,
            "local_pccm_cost": self.last_pccm_cost,
            "mean_local_pccm": self.episode_pccm_cost_sum / max(1, self.frame_steps),
            "pccm_shaping_reward": action_pccm_shaping_reward,
            "episode_pccm_shaping_reward": self.episode_pccm_shaping_reward,
            "wall_time_ratio": self.episode_wall_frames / max(1, self.frame_steps),
            "action_counts": self.episode_action_counts.tolist(),
        }
        self.last_hp = self.scene.player.hp
        self.last_observation = observation
        self.last_reward = total_reward
        self.last_collided = collided
        return observation, total_reward, done, info

    # Convert the current scene state into the observation dictionary.
    def get_observation(self) -> dict[str, np.ndarray]:
        from observation_sources import scene_to_observation_state

        bullets, player, self.previous_enemy_positions = scene_to_observation_state(
            self.scene,
            self.GAME_ZONE,
            self.previous_action,
            self.previous_enemy_positions,
            1 / self.FPS,
        )
        return self.builder.build(bullets, player)

    # Return evenly spaced map snapshots from the oldest selected frame to the current frame.
    def get_map_history(self) -> tuple[dict[str, np.ndarray], ...]:
        if len(self.map_history) != self.map_history_size:
            raise RuntimeError("Map history is not initialized. Call reset() before requesting it.")
        selected_frames = tuple(self.map_history)[::self.frame_stack_interval]
        if len(selected_frames) != self.frame_stack:
            raise RuntimeError("Map history does not match the requested frame stack.")
        return selected_frames

    # Reset the map history by repeating the initial game-frame map.
    def _reset_map_history(self, observation: dict[str, np.ndarray]) -> None:
        self.map_history.clear()
        for _ in range(self.map_history_size):
            self._append_map_snapshot(observation)

    # Save the current frame's bullet maps without storing player features.
    def _append_map_snapshot(self, observation: dict[str, np.ndarray]) -> None:
        snapshot = {
            key: np.asarray(observation[key], dtype=np.float32).copy()
            for key in self.MAP_KEYS
        }
        self.map_history.append(snapshot)

    # Render the game and optional observation panels in human mode.
    def render(self) -> None:
        if self.render_mode != "human":
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.scene.render(self.screen, self.clock)
        if self.render_debug and self.last_observation is not None:
            from tools.visualization_debug import draw_observation_panels, draw_realtime_overlay

            draw_realtime_overlay(self.screen, self.last_observation, (self.GAME_ZONE[0], self.GAME_ZONE[1]), show_grids=True)
            draw_observation_panels(self.screen, self.last_observation)
        self._draw_reward_panel()
        pygame.display.flip()
        self.clock.tick(self.FPS)

    # Release pygame and database resources.
    def close(self) -> None:
        try:
            self.db_module.close()
        except Exception:
            pass
        pygame.quit()

    # Convert a discrete action id into a movement vector.
    def _action_to_vector(self, action: int):
        v = self.Vector2
        return {
            0: v.zero(),
            1: v.up(),
            2: v.down(),
            3: v.left(),
            4: v.right(),
            5: v.up() + v.left(),
            6: v.up() + v.right(),
            7: v.down() + v.left(),
            8: v.down() + v.right(),
        }[action]

    # Move the player to a random lower-field start position for training.
    def _randomize_player_start(self) -> None:
        margin = max(0.0, self.player_start_margin)
        left = self.GAME_ZONE[0] + margin
        right = self.GAME_ZONE[0] + self.GAME_ZONE[2] - margin
        top = self.GAME_ZONE[1] + self.GAME_ZONE[3] * 0.55
        bottom = self.GAME_ZONE[1] + self.GAME_ZONE[3] - margin
        if right <= left or bottom <= top:
            return

        x = float(np.random.uniform(left, right))
        y = float(np.random.uniform(top, bottom))
        self.scene.player.position = self.Vector2(x, y)
        self.scene.player.collider.position = self.scene.player.position

    # Draw live reward information on the game screen.
    def _draw_reward_panel(self) -> None:
        font = pygame.font.Font(None, 24 if self.render_debug else 28)
        lines = [
            f"step reward: {self.last_reward:+.4f}",
            f"total reward: {self.episode_reward:+.2f}",
            f"decision_steps: {self.steps}",
            f"frame_steps: {self.frame_steps}",
            f"hp: {self.scene.player.hp}",
            f"collided: {self.last_collided}",
        ]
        x = self.GAME_ZONE[0] + self.GAME_ZONE[2] + (330 if self.render_debug else 50)
        y = 560
        for index, line in enumerate(lines):
            label = font.render(line, True, (255, 255, 255)).convert_alpha()
            self.screen.blit(label, (x, y + index * 26))

    # Check whether the current episode should stop.
    def _is_done(self, collided: bool) -> bool:
        if collided and not self.training_invincible:
            return True
        if self.max_steps is not None and self.frame_steps >= self.max_steps:
            return True
        if self.scene.player.hp < 0:
            return True
        if self.scene.next is not self.scene:
            return True
        return False


if __name__ == "__main__":
    env = TouhouRLEnv(render_mode="human")
    obs = env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(np.random.randint(0, 9))
    env.close()
