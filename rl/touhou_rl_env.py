from __future__ import annotations

import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from rl.reward import compute_reward


PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


class TouhouRLEnv:
    MAP_KEYS = (
        "blue_density",
        "blue_speed",
        "yellow_density",
        "yellow_speed",
        "yellow_valid",
        "red_occupancy",
        "red_vx",
        "red_vy",
        "red_valid",
    )
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
        random_player_start: bool = False,
        player_start_margin: float = 80.0,
        frame_stack: int = 1,
        frame_stack_interval: int = 1,
        training_invincible: bool = False,
    ):
        if not 1 <= int(frame_stack) <= 5:
            raise ValueError(f"frame_stack must be in 1..5, got {frame_stack}.")
        if not 1 <= int(frame_stack_interval) <= 5:
            raise ValueError(f"frame_stack_interval must be in 1..5, got {frame_stack_interval}.")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_repeat = max(1, int(action_repeat))
        self.level_file = level_file
        self.random_player_start = random_player_start
        self.player_start_margin = float(player_start_margin)
        self.frame_stack = int(frame_stack)
        self.frame_stack_interval = int(frame_stack_interval)
        self.map_history_size = 1 + (self.frame_stack - 1) * self.frame_stack_interval
        self.training_invincible = bool(training_invincible)
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
                blue_grid=(6, 6),
                yellow_size=(320, 320),
                yellow_grid=(16, 16),
                red_size=(128, 128),
                red_map=(64, 64),
                max_speed=500.0,
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

        self.scene = GameScene(level_file=self.level_file)
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
        previous_action_for_reward = self.previous_action
        self.previous_action = action
        self.steps += 1

        for _ in range(self.action_repeat):
            self.scene.player.slow = False
            self.scene.player.move(direction)
            self.scene.update(1 / self.FPS)

            self.frame_steps += 1
            observation = self.get_observation()
            self._append_map_snapshot(observation)
            collided = self.scene.player.collided_this_frame
            contact_frames += int(collided)
            done = self._is_done(collided)
            frame_reward = compute_reward(
                observation,
                action,
                previous_action_for_reward,
                collided
            )
            total_reward += frame_reward
            self.last_hp = self.scene.player.hp
            self.last_observation = observation
            self.last_reward = frame_reward
            self.episode_reward += frame_reward
            self.last_collided = collided
            previous_action_for_reward = action
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

    # Render the game and debug observation panels in human mode.
    def render(self) -> None:
        if self.render_mode != "human":
            return
        from tools.visualization_debug import draw_observation_panels, draw_realtime_overlay

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.scene.render(self.screen, self.clock)
        if self.last_observation is not None:
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
        font = pygame.font.Font(None, 28)
        lines = [
            f"step reward: {self.last_reward:+.4f}",
            f"total reward: {self.episode_reward:+.2f}",
            f"decision_steps: {self.steps}",
            f"frame_steps: {self.frame_steps}",
            f"hp: {self.scene.player.hp}",
            f"collided: {self.last_collided}",
        ]
        x = self.GAME_ZONE[0] + self.GAME_ZONE[2] + 50
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
