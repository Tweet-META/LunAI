"""Simulator adapter for the isolated Hard Scarlet Meister spell."""

import math
from os.path import join as path_join

from assets.scripts.classes.game_logic.Bullet import Bullet
from assets.scripts.classes.game_logic.BulletData import BulletData
from assets.scripts.classes.game_logic.Collider import Collider
from assets.scripts.classes.game_logic.Enemy import Enemy
from assets.scripts.classes.hud_and_rendering.SpriteSheet import SpriteSheet
from assets.scripts.math_and_data.Vector2 import Vector2
from assets.scripts.math_and_data.enviroment import GAME_ZONE
from rl.scarlet_meister import ScarletMeisterPattern


class ScarletMeisterEnemy(Enemy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        origin = self.position - Vector2(GAME_ZONE[0], GAME_ZONE[1])
        self.pattern = ScarletMeisterPattern(origin.to_tuple())
        self.frame_credit = 0.0
        sheet = SpriteSheet(path_join('assets', 'sprites', 'projectiles_and_items', 'bullet_0.png')).crop((16, 16))
        self.bullet_types = {
            radius: BulletData(sheet, Collider(radius), sprite_scale=scale)
            for radius, scale in ((3.0, 1.0), (8.0, 2.0), (16.0, 4.0))
        }

    def update(self):
        super().update()
        self.frame_credit += self.scene.delta_time * 60.0
        while self.frame_credit >= 1.0 - 1e-9:
            self.frame_credit -= 1.0
            shots = self.pattern.step(
                self.target.position.x() - GAME_ZONE[0],
                self.target.position.y() - GAME_ZONE[1],
                len(self.bullets),
            )
            self.position = Vector2(self.pattern.x + GAME_ZONE[0], self.pattern.y + GAME_ZONE[1])
            self.bullets.extend(
                Bullet(
                    self.bullet_types[shot.radius],
                    Vector2(shot.x + GAME_ZONE[0], shot.y + GAME_ZONE[1]),
                    math.degrees(shot.angle) + 90.0,
                    shot.speed * 60.0,
                )
                for shot in shots
            )
        self.t += self.scene.delta_time

    def move(self):
        # Pattern movement replaces the spline only for this enemy class.
        self.collider.position = self.position + self.collider.offset
