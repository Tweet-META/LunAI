from __future__ import annotations

import pygame
import numpy as np


BLUE_FILL = (70, 130, 255, 32)
YELLOW_FILL = (255, 220, 70, 56)
RED_FILL = (255, 70, 70, 76)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


# Convert a playfield window into a screen-space rectangle.
def _screen_rect(window: np.ndarray | tuple[int, int, int, int], game_origin: tuple[int, int]) -> pygame.Rect:
    x1, y1, x2, y2 = [int(v) for v in window]
    return pygame.Rect(game_origin[0] + x1, game_origin[1] + y1, x2 - x1, y2 - y1)


# Draw blue, yellow, and red observation zones over the game.
def draw_zone_overlay(screen: pygame.Surface, observation: dict[str, np.ndarray], game_origin: tuple[int, int]) -> None:
    draw_colored_rect(screen, _screen_rect(observation["_blue_window"], game_origin), BLUE_FILL, 2)
    draw_colored_rect(screen, _screen_rect(observation["_yellow_window"], game_origin), YELLOW_FILL, 2)
    draw_colored_rect(screen, _screen_rect(observation["_red_window"], game_origin), RED_FILL, 3)


# Draw one translucent filled rectangle with a black border.
def draw_colored_rect(screen: pygame.Surface, rect: pygame.Rect, color: tuple[int, int, int, int], border_width: int) -> None:
    overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
    overlay.fill(color)
    screen.blit(overlay, rect.topleft)
    pygame.draw.rect(screen, BLACK, rect, border_width)


# Draw black grid lines for one observation window.
def draw_grid_lines(screen: pygame.Surface, window: np.ndarray, game_origin: tuple[int, int], grid_shape: tuple[int, int]) -> None:
    rect = _screen_rect(window, game_origin)
    rows, cols = grid_shape
    for col in range(1, cols):
        x = rect.left + round(rect.width * col / cols)
        pygame.draw.line(screen, BLACK, (x, rect.top), (x, rect.bottom), 1)
    for row in range(1, rows):
        y = rect.top + round(rect.height * row / rows)
        pygame.draw.line(screen, BLACK, (rect.left, y), (rect.right, y), 1)


def draw_realtime_overlay(
    screen: pygame.Surface,
    observation: dict[str, np.ndarray],
    game_origin: tuple[int, int],
    show_grids: bool = True,
) -> None:
    # Draw live zone overlays and optional grid lines.
    draw_zone_overlay(screen, observation, game_origin)
    if show_grids:
        draw_grid_lines(screen, observation["_blue_window"], game_origin, observation["blue_density"].shape)
        draw_grid_lines(screen, observation["_yellow_window"], game_origin, observation["yellow_density"].shape)
        draw_grid_lines(screen, observation["_red_window"], game_origin, observation["red_occupancy"].shape)


# Map a normalized value to a tinted heat color.
def heat_color(value: float, tint: tuple[int, int, int]) -> tuple[int, int, int]:
    value = float(np.clip(value, 0.0, 1.0))
    base = int(30 + 210 * value)
    return (
        min(255, int(tint[0] * value + base * (1.0 - value))),
        min(255, int(tint[1] * value + base * (1.0 - value))),
        min(255, int(tint[2] * value + base * (1.0 - value))),
    )


def draw_heatmap_panel(
    screen: pygame.Surface,
    values: np.ndarray,
    rect: pygame.Rect,
    tint: tuple[int, int, int],
    border_color: tuple[int, int, int] = BLACK,
) -> None:
    # Draw one fixed-size heatmap panel.
    rows, cols = values.shape
    cell_w = rect.width / cols
    cell_h = rect.height / rows
    pygame.draw.rect(screen, (18, 18, 18), rect)
    for row in range(rows):
        for col in range(cols):
            cell = pygame.Rect(
                round(rect.left + col * cell_w),
                round(rect.top + row * cell_h),
                max(1, round(cell_w)),
                max(1, round(cell_h)),
            )
            pygame.draw.rect(screen, heat_color(values[row, col], tint), cell)
    pygame.draw.rect(screen, border_color, rect, 2)


# Draw the player hitbox marker inside a red-zone panel.
def draw_player_hitbox_marker(screen: pygame.Surface, rect: pygame.Rect, observation: dict[str, np.ndarray]) -> None:
    player_features = observation["player_features"]
    red_window = observation["_red_window"]
    play_x = float(player_features[0]) * int(observation["_blue_window"][2])
    play_y = float(player_features[1]) * int(observation["_blue_window"][3])
    red_x1, red_y1, red_x2, red_y2 = [int(v) for v in red_window]
    red_w = max(1, red_x2 - red_x1)
    red_h = max(1, red_y2 - red_y1)
    marker_x = rect.left + round((play_x - red_x1) / red_w * rect.width)
    marker_y = rect.top + round((play_y - red_y1) / red_h * rect.height)
    pygame.draw.circle(screen, WHITE, (marker_x, marker_y), 5)
    pygame.draw.circle(screen, BLACK, (marker_x, marker_y), 6, 2)


# Draw the six debug panels for density, speed, and red maps.
def draw_observation_panels(screen: pygame.Surface, observation: dict[str, np.ndarray], origin: tuple[int, int] = (690, 420)) -> None:
    x, y = origin
    draw_heatmap_panel(screen, observation["blue_density"], pygame.Rect(x, y, 120, 120), (70, 130, 255))
    draw_heatmap_panel(screen, observation["blue_speed"], pygame.Rect(x + 140, y, 120, 120), (80, 210, 255))
    draw_heatmap_panel(screen, observation["yellow_density"], pygame.Rect(x, y + 145, 120, 120), (255, 220, 70))
    draw_heatmap_panel(screen, observation["yellow_speed"], pygame.Rect(x + 140, y + 145, 120, 120), (255, 170, 60))
    red_occupancy_rect = pygame.Rect(x, y + 290, 120, 120)
    red_speed_rect = pygame.Rect(x + 140, y + 290, 120, 120)
    draw_heatmap_panel(screen, observation["red_occupancy"], red_occupancy_rect, (255, 70, 70))
    draw_heatmap_panel(screen, observation["red_speed"], red_speed_rect, (255, 120, 120))
    draw_player_hitbox_marker(screen, red_occupancy_rect, observation)
    draw_player_hitbox_marker(screen, red_speed_rect, observation)


# Save the main observation maps as a debug image.
def save_debug_image(path: str, observation: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    panels = [
        ("blue_density", observation["blue_density"]),
        ("blue_speed", observation["blue_speed"]),
        ("yellow_density", observation["yellow_density"]),
        ("yellow_speed", observation["yellow_speed"]),
        ("red_occupancy", observation["red_occupancy"]),
        ("red_speed", observation["red_speed"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for ax, (title, values) in zip(axes.flat, panels):
        ax.imshow(values, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
