import random
import math

import numpy as np

from assets.scripts.classes.game_logic.Bullet import Bullet
from assets.scripts.classes.game_logic.BulletData import BulletData
from assets.scripts.classes.game_logic.Enemy import Enemy
from assets.scripts.classes.game_logic.Player import Player
from assets.scripts.math_and_data.Vector2 import Vector2
from playfield_config import scale_distance


class AttackFunctions:
    delta_angle = 0

    # Return the bullet angle that points from origin to target.
    @staticmethod
    def aimed_angle(origin: Vector2, target: Vector2) -> float:
        direction = target - origin
        if direction.length() <= 1e-8:
            return 0.0
        return float(np.rad2deg(np.arctan2(-direction.x(), -direction.y())))

    @staticmethod
    def ring(center: Vector2, number_of_bullets: int, bullet_data: BulletData, speed: float, angular_speed: float = 0,
             delta_angle: float = 0):
        bullets = [
            Bullet(
                bullet_data,
                center,
                (360 * n / number_of_bullets + delta_angle) % 360,
                speed,
                angular_speed
            )
            for n in range(number_of_bullets)
        ]

        return bullets

    @staticmethod
    def random(center: Vector2, number_of_bullets: int, bullet_data: BulletData, speed: float,
               angular_speed: float = 0):
        # Create bullets with random outgoing angles.
        bullets = [
            Bullet(
                bullet_data,
                center,
                random.randint(0, 360),
                speed,
                angular_speed
            )
            for _ in range(number_of_bullets)
        ]

        return bullets

    @staticmethod
    def th06_aimed_circle(center: Vector2, number_of_bullets: int, number_of_layers: int,
                          bullet_data: BulletData, speed1: float, speed2: float,
                          player: Player, enemy: Enemy):
        """Reproduce TH06 CIRCLE_AIMED, including its original layer speed interpolation."""
        aimed_angle = AttackFunctions.aimed_angle(enemy.position, player.position)
        return [
            Bullet(
                bullet_data,
                center,
                aimed_angle + bullet_index * 360 / number_of_bullets,
                speed1 - (speed1 - speed2) * layer_index / number_of_layers,
            )
            for layer_index in range(number_of_layers)
            for bullet_index in range(number_of_bullets)
        ]

    @staticmethod
    def th06_aimed_fan(center: Vector2, number_of_bullets: int, number_of_layers: int,
                       bullet_data: BulletData, speed1: float, speed2: float,
                       angle_step: float, player: Player, enemy: Enemy):
        """Reproduce TH06 FAN_AIMED with centered angles and layered speeds."""
        aimed_angle = AttackFunctions.aimed_angle(enemy.position, player.position)
        half = (number_of_bullets - 1) / 2
        return [
            Bullet(
                bullet_data,
                center,
                aimed_angle + (bullet_index - half) * angle_step,
                speed1 - (speed1 - speed2) * layer_index / number_of_layers,
            )
            for layer_index in range(number_of_layers)
            for bullet_index in range(number_of_bullets)
        ]

    # Build one TH06-style ring from several interleaved bullet groups.
    @staticmethod
    def th06_multiring(center: Vector2, group_counts: list[int], bullet_data: BulletData,
                       speed: float, base_angle: float, group_angle_step: float):
        bullets = []
        for group_index, count in enumerate(group_counts):
            group_angle = base_angle + group_index * group_angle_step
            bullets.extend(
                Bullet(
                    bullet_data,
                    center,
                    group_angle + bullet_index * 360.0 / count,
                    speed,
                )
                for bullet_index in range(count)
            )
        return bullets

    # Expand TH06 Stage 3 Spell 1's seven-frame rotating ring loop.
    @staticmethod
    def th06_stage3_spell1(bullet_data: BulletData, start_time: float, duration: float):
        interval = 7.0 / 60.0
        group_counts = [2, 3, 4, 2, 4, 3, 2]
        group_step = -math.degrees(0.1134464)
        sweep_step = math.degrees(0.1308997)
        attacks = []
        volley_count = int(duration / interval)
        for volley in range(volley_count):
            sweep_index = volley % 96
            sweep = sweep_index if sweep_index < 48 else 96 - sweep_index
            th06_angle = 50.0 + sweep * sweep_step
            base_angle = -th06_angle - 90.0
            attacks.append(
                (
                    AttackFunctions.th06_multiring,
                    round(start_time + volley * interval, 6),
                    [Vector2.zero(), group_counts, bullet_data, 2.6 * 60.0, base_angle, group_step],
                )
            )
        return attacks

    # Emit bullets with TH06's random initial velocity and fixed world-space acceleration.
    @staticmethod
    def th06_random_acceleration(center: Vector2, groups: list[list[float]], bullet_data: BulletData):
        bullets = []
        for count, acceleration_per_frame, acceleration_angle_degrees in groups:
            acceleration = float(acceleration_per_frame) * 60.0 * 60.0
            angle_radians = math.radians(float(acceleration_angle_degrees))
            motion = {
                "acceleration": [
                    math.cos(angle_radians) * acceleration,
                    math.sin(angle_radians) * acceleration,
                ],
                "acceleration_delay": 17.0 / 60.0,
                "initial_speed_boost": 5.0 * 60.0,
                "speed_boost_duration": 16.0 / 60.0,
            }
            for _ in range(int(count)):
                th06_angle = random.uniform(-math.pi, math.pi)
                speed = random.uniform(0.3, 1.0) * 60.0
                bullets.append(
                    Bullet(
                        bullet_data,
                        center,
                        -math.degrees(th06_angle) - 90.0,
                        speed,
                        motion=motion,
                    )
                )
        return bullets

    # Expand TH06 Stage 3 Spell 2's alternating acceleration petals.
    @staticmethod
    def th06_stage3_spell2(bullet_data: BulletData, start_time: float, duration: float):
        attacks = []
        cycle_frames = 220
        cycle = 0
        while cycle * cycle_frames / 60.0 < duration:
            cycle_start = start_time + cycle * cycle_frames / 60.0
            for volley in range(20):
                attacks.append(
                    (
                        AttackFunctions.th06_random_acceleration,
                        round(cycle_start + volley * 4.0 / 60.0, 6),
                        [Vector2.zero(), [[4, 0.027, 90.0]], bullet_data],
                    )
                )

            second_phase = cycle_start + 160.0 / 60.0
            for volley in range(20):
                groups = [
                    [1, 0.024, 180.0],
                    [1, 0.024, 0.0],
                    [1, 0.024, 135.0],
                    [1, 0.024, 45.0],
                ]
                attacks.append(
                    (
                        AttackFunctions.th06_random_acceleration,
                        round(second_phase + volley * 3.0 / 60.0, 6),
                        [Vector2.zero(), groups, bullet_data],
                    )
                )
            cycle += 1
        return [attack for attack in attacks if attack[1] < start_time + duration]

    # Expand TH06 Stage 3 Spell 3's two counter-rotating acceleration arms.
    @staticmethod
    def th06_stage3_spell3(bullet_data: BulletData, start_time: float, duration: float):
        attacks = []
        cycle_interval = 6.0 / 60.0
        cycle_count = int(duration / cycle_interval)
        for cycle in range(cycle_count):
            rotation = cycle * 6.0
            main_count = 3 if cycle * cycle_interval >= 20.0 else 2
            events = [
                (0, [[main_count, 0.016, -90.0 - rotation]]),
                (1, [[1, 0.018, -rotation]]),
                (2, [[1, 0.018, -180.0 - rotation]]),
                (3, [[main_count, 0.016, 90.0 + rotation]]),
                (4, [[1, 0.018, 180.0 + rotation]]),
                (5, [[1, 0.016, rotation]]),
            ]
            for frame_offset, groups in events:
                attacks.append(
                    (
                        AttackFunctions.th06_random_acceleration,
                        round(start_time + cycle * cycle_interval + frame_offset / 60.0, 6),
                        [Vector2.zero(), groups, bullet_data],
                    )
                )
        return attacks

    @staticmethod
    def random_cone(center: Vector2, number_of_bullets: int, bullet_data: BulletData, angle: float,
                    spread: float, speed: float, angular_speed: float = 0):
        # Create random bullets inside a limited angular cone.
        bullets = [
            Bullet(
                bullet_data,
                center,
                random.uniform(angle - spread, angle + spread),
                speed,
                angular_speed
            )
            for _ in range(number_of_bullets)
        ]

        return bullets

    @staticmethod
    def rectangle_wall(center: Vector2, columns: int, rows: int, bullet_data: BulletData,
                       width: float, height: float, angle: float, speed: float,
                       angular_speed: float = 0):
        # Create a solid rectangular wall of bullets.
        x_offsets = np.linspace(-width * 0.5, width * 0.5, columns)
        y_offsets = np.linspace(0.0, height, rows)
        return [
            Bullet(
                bullet_data,
                center + Vector2(float(x_offset), float(y_offset)),
                angle,
                speed,
                angular_speed,
            )
            for y_offset in y_offsets
            for x_offset in x_offsets
        ]

    @staticmethod
    def cone(center: Vector2, angle, number_of_bullets: int, bullet_data: BulletData, speed: float, delta_angle: int, angular_speed=0, player: Player=None, enemy: Enemy=None):
        aimed_angle = angle if angle != "player" else AttackFunctions.aimed_angle(enemy.position, player.position)
        bullets = [
            Bullet(
                bullet_data,
                center,
                aimed_angle + i * delta_angle * 0.5,
                speed,
                angular_speed
            )

            for i in range(-number_of_bullets + 1, number_of_bullets, 2)
        ]

        return bullets

    @staticmethod
    def wide_cone(number_of_bullets: int, number_of_cones: int, bullet_data: BulletData, angle: int, speed: float, delta_angle: int,
                  start_time: float, delay: float, angular_speed=0, player: Player=None, enemy: Enemy=None):
        center = Vector2.zero()

        attacks = [
            (
                AttackFunctions.cone,
                round(start_time + delay * n, 3),
                [center, angle, number_of_bullets, bullet_data, speed, delta_angle, angular_speed, player, enemy]
            )
            for n in range(number_of_cones)
        ]

        return attacks

    @staticmethod
    def wide_ring(number_of_bullets: int, number_of_rings: int, bullet_data: BulletData, speed: float,
                  start_time: float, delay: float, angular_speed: float = 0, delta_angle: float = 0, rand_center=False):

        attacks = [
            (
                AttackFunctions.ring,
                round(start_time + delay * n, 3),
                [Vector2.zero() if not rand_center else\
            Vector2.one().rotate(random.randint(0, 360)) * scale_distance(25), number_of_bullets, bullet_data, speed, angular_speed, n * delta_angle]
            )
            for n in range(number_of_rings)
        ]

        return attacks

    @staticmethod
    def long_random(number_of_bullets: int, number_of_randoms: int, bullet_data: BulletData, speed: float,
                    start_time: float, delay: float, angular_speed: float = 0, rand_center=False):
        # Schedule repeated random bullet bursts.
        attacks = [
            (
                AttackFunctions.random,
                round(start_time + delay * n, 3),
                [Vector2.zero() if not rand_center else\
            Vector2.one().rotate(random.randint(0, 360)) * scale_distance(25), number_of_bullets, bullet_data, speed, angular_speed]
            )
            for n in range(number_of_randoms)
        ]

        return attacks

    @staticmethod
    def long_random_cone(number_of_bullets: int, number_of_randoms: int, bullet_data: BulletData,
                         angle: float, spread: float, speed: float, start_time: float, delay: float,
                         angular_speed: float = 0, rand_center=False):
        # Schedule repeated random bursts inside a limited cone.
        attacks = [
            (
                AttackFunctions.random_cone,
                round(start_time + delay * n, 3),
                [
                    Vector2.zero() if not rand_center else
                    Vector2.one().rotate(random.randint(0, 360)) * scale_distance(25),
                    number_of_bullets,
                    bullet_data,
                    angle,
                    spread,
                    speed,
                    angular_speed,
                ]
            )
            for n in range(number_of_randoms)
        ]

        return attacks
