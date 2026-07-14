from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pygame
from pygame.locals import DOUBLEBUF, KEYDOWN, QUIT


PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from assets.scripts.math_and_data.enviroment import FPS, GAME_ZONE, SIZE, db_module  # noqa: E402
from assets.scripts.math_and_data.Vector2 import Vector2  # noqa: E402
from observation_builder import ObservationBuilder, ObservationConfig  # noqa: E402
from observation_sources import scene_to_observation_state  # noqa: E402
from tools.visualization_debug import draw_observation_panels, draw_realtime_overlay  # noqa: E402


if TYPE_CHECKING:
    from assets.scripts.scenes.GameScene import GameScene


ACTION_TO_VECTOR = {
    0: Vector2.zero(),
    1: Vector2.up(),
    2: Vector2.down(),
    3: Vector2.left(),
    4: Vector2.right(),
    5: Vector2.up() + Vector2.left(),
    6: Vector2.up() + Vector2.right(),
    7: Vector2.down() + Vector2.left(),
    8: Vector2.down() + Vector2.right(),
}


# Convert a keyboard direction vector into one action id.
def vector_to_action(direction: Vector2) -> int:
    x = int(direction.x())
    y = int(direction.y())
    if x == 0 and y == 0:
        return 0
    if x == 0 and y < 0:
        return 1
    if x == 0 and y > 0:
        return 2
    if x < 0 and y == 0:
        return 3
    if x > 0 and y == 0:
        return 4
    if x < 0 and y < 0:
        return 5
    if x > 0 and y < 0:
        return 6
    if x < 0 and y > 0:
        return 7
    return 8


# Read arrow keys and return the current movement direction.
def current_direction_from_keyboard() -> Vector2:
    keys = pygame.key.get_pressed()
    direction = Vector2.zero()
    if keys[pygame.K_UP]:
        direction += Vector2.up()
    if keys[pygame.K_DOWN]:
        direction += Vector2.down()
    if keys[pygame.K_LEFT]:
        direction += Vector2.left()
    if keys[pygame.K_RIGHT]:
        direction += Vector2.right()
    return direction


# Print compact shape and range information for debugging.
def print_observation_summary(observation: dict[str, object], bullet_count: int) -> None:
    print(f"bullets: {bullet_count}")
    for key in (
        "blue_density",
        "blue_pccm",
        "blue_valid",
        "yellow_density",
        "yellow_pccm",
        "yellow_valid",
        "red_occupancy",
        "red_pccm",
        "red_valid",
        "player_features",
    ):
        value = observation[key]
        print(f"{key}: {value.shape}, min={value.min():.3f}, max={value.max():.3f}")
    print("-" * 48)


# Draw a small pause label at the top of the screen.
def draw_pause_label(screen: pygame.Surface, font: pygame.font.Font) -> None:
    label = font.render("PAUSED", True, (255, 255, 255)).convert_alpha()
    padding = 16
    rect = label.get_rect(center=(SIZE[0] // 2, 40))
    background = pygame.Rect(
        rect.left - padding,
        rect.top - padding // 2,
        rect.width + padding * 2,
        rect.height + padding,
    )
    overlay = pygame.Surface(background.size, pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 170))
    screen.blit(overlay, background.topleft)
    pygame.draw.rect(screen, (0, 0, 0), background, 2)
    screen.blit(label, rect)


# Run the live game with observation overlays.
def main() -> None:
    parser = argparse.ArgumentParser(description="Show one live multi-scale observation map.")
    parser.add_argument("--level-file", default="level_1.json")
    args = parser.parse_args()

    pygame.init()
    pygame.mixer.pre_init(48000, -16, 2, 4096)
    pygame.mixer.init()

    screen = pygame.display.set_mode(SIZE, DOUBLEBUF, 16)
    pygame.display.set_caption("Touhou observation map")
    clock = pygame.time.Clock()

    from assets.scripts.scenes.GameScene import GameScene

    scene = GameScene(level_file=args.level_file)
    pause_font = pygame.font.Font(None, 48)

    builder = ObservationBuilder(
        ObservationConfig(
            playfield_width=GAME_ZONE[2],
            playfield_height=GAME_ZONE[3],
            blue_grid=(8, 8),
            yellow_size=(320, 320),
            yellow_grid=(16, 16),
            red_size=(128, 128),
            red_map=(64, 64),
            max_speed=500.0,
            observation_schema="pccm",
            pccm_debug=False,
        )
    )

    delta_time = 1 / FPS
    previous_action = 0
    previous_enemy_positions = {}
    frame = 0
    paused = False
    running = True
    while running and scene is not None:
        events = pygame.event.get()
        for event in events:
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == pygame.K_ESCAPE:
                paused = not paused

        if not isinstance(scene, GameScene):
            scene = GameScene(level_file=args.level_file)
            delta_time = 1 / FPS
            previous_action = 0
            previous_enemy_positions = {}
            frame = 0
            paused = False
            continue

        direction = current_direction_from_keyboard()
        previous_action = vector_to_action(direction)

        if not paused:
            scene.process_input(events)
            scene.update(delta_time)

        bullets, player, previous_enemy_positions = scene_to_observation_state(
            scene,
            GAME_ZONE,
            previous_action,
            previous_enemy_positions,
            delta_time,
        )
        observation = builder.build(bullets, player)

        scene.render(screen, clock)
        draw_realtime_overlay(screen, observation, (GAME_ZONE[0], GAME_ZONE[1]), show_grids=True)
        draw_observation_panels(screen, observation)
        if paused:
            draw_pause_label(screen, pause_font)

        pygame.display.flip()
        if not paused:
            scene = scene.next
        delta_time = clock.tick(FPS) / 1000
        if not paused:
            frame += 1

        if not paused and frame % FPS == 0:
            print_observation_summary(observation, len(bullets))

    db_module.close()
    pygame.quit()


if __name__ == "__main__":
    main()
