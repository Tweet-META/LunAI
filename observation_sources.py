from __future__ import annotations

from observation_builder import BulletState, PlayerState


# Convert live enemies, bullets, and player state into observation data.
def scene_to_observation_state(
    scene,
    game_zone: tuple[int, int, int, int],
    previous_action: int,
    previous_enemy_positions: dict[int, tuple[float, float]],
    delta_time: float,
) -> tuple[list[BulletState], PlayerState, dict[int, tuple[float, float]]]:
    origin_x, origin_y, _, _ = game_zone
    bullets = []
    for bullet in scene.enemy_bullets:
        velocity = bullet.velocity()
        hitbox_position = bullet.collider.position
        bullets.append(
            BulletState(
                x=hitbox_position.x() - origin_x,
                y=hitbox_position.y() - origin_y,
                radius=bullet.collider.radius,
                vx=velocity.x(),
                vy=velocity.y(),
                speed=bullet.speed,
            )
        )

    next_enemy_positions = {}
    dt = max(delta_time, 1e-6)
    for enemy in scene.enemies:
        hitbox_position = enemy.collider.position
        current_position = (hitbox_position.x(), hitbox_position.y())
        previous_position = previous_enemy_positions.get(id(enemy), current_position)
        vx = (current_position[0] - previous_position[0]) / dt
        vy = (current_position[1] - previous_position[1]) / dt
        speed = (vx * vx + vy * vy) ** 0.5
        next_enemy_positions[id(enemy)] = current_position
        bullets.append(
            BulletState(
                x=hitbox_position.x() - origin_x,
                y=hitbox_position.y() - origin_y,
                radius=enemy.collider.radius,
                vx=vx,
                vy=vy,
                speed=speed,
            )
        )

    player = scene.player
    player_state = PlayerState(
        x=player.collider.position.x() - origin_x,
        y=player.collider.position.y() - origin_y,
        radius=player.collider.radius,
        previous_action=previous_action,
    )
    return bullets, player_state, next_enemy_positions
