from copy import copy

import pygame.transform

from assets.scripts.classes.game_logic.BulletData import BulletData
from assets.scripts.math_and_data.Vector2 import Vector2
from assets.scripts.math_and_data.enviroment import *
from playfield_config import scale_distance


class Bullet:
    def __init__(self, bullet_data: BulletData, position: Vector2, angle: float, speed: float, angular_speed=0,
                 motion: dict | None = None):

        self.sprite_sheet = bullet_data.sprite_sheet

        sprite_sheet = []
        for i in range(len(self.sprite_sheet)):
            sprite = self.sprite_sheet[i]
            if bullet_data.sprite_scale != 1.0:
                width = max(1, round(sprite.get_width() * bullet_data.sprite_scale))
                height = max(1, round(sprite.get_height() * bullet_data.sprite_scale))
                sprite = pygame.transform.scale(sprite, (width, height))
            new_sprite = pygame.sprite.Sprite()
            new_sprite.image = pygame.transform.rotate(sprite, angle)
            new_sprite.rect = new_sprite.image.get_rect()
            sprite_sheet.append(new_sprite)

        self.sprite_sheet = sprite_sheet

        self.position: Vector2 = position
        self.collider = copy(bullet_data.collider)

        self.angle: float = angle
        self.speed: float = speed
        self.base_speed: float = speed
        self.angular_speed: float = angular_speed
        self.motion = dict(bullet_data.motion)
        self.motion.update(motion or {})
        self.bounce_mode = self.motion.get("bounce_mode")
        self.bounces_remaining = int(self.motion.get("bounce_count", 0))
        acceleration = self.motion.get("acceleration")
        self.acceleration = Vector2(*acceleration) if acceleration is not None else None
        self.acceleration_delay = float(self.motion.get("acceleration_delay", 0.0))
        self.initial_speed_boost = float(self.motion.get("initial_speed_boost", 0.0))
        self.speed_boost_duration = float(self.motion.get("speed_boost_duration", 0.0))
        self.acceleration_started = False
        self.age = 0.0
        self.linear_velocity = (Vector2.up() * self.speed).rotate(self.angle)

        self.current_sprite = 0
        self.change_sprite_timer = 0
        self.animation_speed = bullet_data.animation_speed

    def velocity(self) -> Vector2:
        if self.acceleration is not None:
            return self.linear_velocity
        return (Vector2.up() * self.speed).rotate(self.angle)

    def move(self, delta_time) -> bool:
        if self.acceleration is not None:
            if self.speed_boost_duration > 0.0 and self.age < self.speed_boost_duration:
                remaining = 1.0 - self.age / self.speed_boost_duration
                boosted_speed = self.base_speed + self.initial_speed_boost * remaining
                self.linear_velocity = (Vector2.up() * boosted_speed).rotate(self.angle)
            else:
                if not self.acceleration_started:
                    self.linear_velocity = (Vector2.up() * self.base_speed).rotate(self.angle)
                    self.acceleration_started = True
                if self.age >= self.acceleration_delay:
                    self.linear_velocity += self.acceleration * delta_time
                    velocity_length = self.linear_velocity.length()
                    if velocity_length > 1e-8:
                        self.speed = float(velocity_length)

        self.position += self.velocity() * delta_time
        self.age += delta_time

        if self.bounces_remaining > 0:
            self._bounce_at_playfield_edge()

        self.collider.position = self.position + self.collider.offset.rotate(self.angle)

        self.angle += self.angular_speed * delta_time
        sprite = self.get_sprite()
        margin = scale_distance(50)
        if (self.position.x() - sprite.rect.w // 2 < GAME_ZONE[0] - margin or
            self.position.y() - sprite.rect.h // 2 < GAME_ZONE[1] - margin) or \
                (self.position.x() + sprite.rect.w // 2 > GAME_ZONE[0] + GAME_ZONE[2] + margin or
                 self.position.y() + sprite.rect.h // 2 > GAME_ZONE[1] + GAME_ZONE[3] + margin):
            del self
            return False
        return True

    def _bounce_at_playfield_edge(self) -> None:
        """Apply TH06's 0x800 behavior: reflect at the sides and top, but exit at the bottom."""
        sprite = self.get_sprite()
        half_width = sprite.rect.w / 2
        half_height = sprite.rect.h / 2
        left, top, width, height = GAME_ZONE
        bounced = False

        if self.position.x() + half_width < left or self.position.x() - half_width > left + width:
            self.angle = (-self.angle) % 360
            bounced = True
        if self.bounce_mode == "sides_top" and self.position.y() + half_height < top:
            self.angle = (180 - self.angle) % 360
            bounced = True
        elif self.bounce_mode == "all_edges" and (
                self.position.y() + half_height < top or
                self.position.y() - half_height > top + height):
            self.angle = (180 - self.angle) % 360
            bounced = True

        if bounced:
            self.bounces_remaining -= 1

    def next_sprite(self) -> None:
        self.change_sprite_timer += 1
        if self.change_sprite_timer == FPS - self.animation_speed:
            self.current_sprite = (self.current_sprite + 1) % self.sprite_sheet.length
            self.change_sprite_timer = 0

    def get_sprite(self) -> pygame.sprite.Sprite:
        sprite = self.sprite_sheet[self.current_sprite]
        sprite.rect = sprite.image.get_rect()
        sprite.rect.center = self.position.to_tuple()

        return sprite
