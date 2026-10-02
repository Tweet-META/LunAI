"""TH06 Hard Scarlet Meister: ECL sub 39 (sequence) and sub 37 (volleys).

Coordinates are native 384x448 playfield pixels; angles use screen-space atan2.
The Python random stream is seeded by TouhouRLEnv, not the original TH06 RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


FIRST_VOLLEY_FRAME = 184
CYCLE_FRAMES = 368
SPELL_FRAMES = 1800
SWEEP_STEP = math.pi / 8.0
# count, speed low/high (pixels/frame), angle half-width, radius, active delay
# The last five 1..2 px/frame bullets in sub 37 are Lunatic-only and excluded.
VOLLEY_GROUPS = (
    (1, 6.2, 6.2, 0.0, 16.0, 23),
    (3, 4.0, 6.0, math.pi / 32.0, 8.0, 31),
    (5, 3.0, 4.0, math.pi / 20.0, 8.0, 31),
    (10, 2.0, 3.0, math.pi / 8.0, 3.0, 0),
)


@dataclass(frozen=True)
class Shot:
    x: float
    y: float
    angle: float
    speed: float
    radius: float
    spawn_frame: int
    active_frame: int


class ScarletMeisterPattern:
    """Advance the original attack schedule one 60 Hz frame at a time."""

    def __init__(self, start=(192.0, 64.0), rng=None):
        self.rng = random if rng is None else rng
        self.frame = 0
        self.x, self.y = map(float, start)
        self.pending: list[Shot] = []
        self.move_start = 0
        self.move_duration = 120
        self.move_origin = (self.x, self.y)
        self.move_delta = (192.0 - self.x, 112.0 - self.y)
        self.sweep_angle = 0.0
        self.volleys_fired = 0

    def _move(self):
        elapsed = min(self.move_duration, max(0, self.frame - self.move_start))
        fraction = 1.0 - (1.0 - elapsed / self.move_duration) ** 2
        x = self.move_origin[0] + fraction * self.move_delta[0]
        y = self.move_origin[1] + fraction * self.move_delta[1]
        self.x = min(352.0, max(32.0, x))
        self.y = min(120.0, max(48.0, y))

    def _start_random_move(self):
        # ECL MOVERANDINBOUND reflects headings near the movement bounds.
        angle = self.rng.uniform(-math.pi, math.pi)
        if self.x < 128.0:
            if angle > math.pi / 2:
                angle = math.pi - angle
            elif angle < -math.pi / 2:
                angle = -math.pi - angle
        if self.x > 256.0:
            if 0.0 <= angle < math.pi / 2:
                angle = math.pi - angle
            elif -math.pi / 2 < angle <= 0.0:
                angle = -math.pi - angle
        if self.y < 96.0 and angle < 0.0:
            angle = -angle
        if self.y > 72.0 and angle > 0.0:
            angle = -angle
        self.move_start = self.frame
        self.move_duration = 90
        self.move_origin = (self.x, self.y)
        distance = 2.5 * 90 / 2
        self.move_delta = (math.cos(angle) * distance, math.sin(angle) * distance)

    def _volley(self, angle: float, capacity: int):
        self.volleys_fired += 1
        for count, low, high, spread, radius, delay in VOLLEY_GROUPS:
            for _ in range(count):
                if len(self.pending) >= capacity:
                    return
                # The native random-angle formula samples from upper to lower.
                direction = angle if spread == 0 else angle + spread * (1 - 2 * self.rng.random())
                speed = low if low == high else low + (high - low) * self.rng.random()
                # Spawn flags 2: half-speed motion while non-collidable; the
                # activation frame also receives the usual full-speed move.
                offset = (delay + 1) * speed / 2 if delay else 0.0
                self.pending.append(Shot(
                    self.x + math.cos(direction) * offset,
                    self.y + math.sin(direction) * offset,
                    direction, speed, radius, self.frame, self.frame + delay,
                ))

    def step(self, player_x: float, player_y: float, active_bullets: int = 0) -> list[Shot]:
        self.frame += 1
        self._move()
        if FIRST_VOLLEY_FRAME <= self.frame < SPELL_FRAMES:
            phase = (self.frame - FIRST_VOLLEY_FRAME) % CYCLE_FRAMES
            aimed = phase in (0, 6, 12, 18, 24, 161, 167)
            forward = 30 <= phase <= 78 and (phase - 30) % 3 == 0
            backward = 173 <= phase <= 221 and (phase - 173) % 3 == 0
            if phase in (30, 173):
                self.sweep_angle = math.atan2(player_y - self.y, player_x - self.x)
            if aimed or forward or backward:
                angle = math.atan2(player_y - self.y, player_x - self.x) if aimed else self.sweep_angle
                self._volley(angle, max(0, 640 - active_bullets))
                if forward or backward:
                    self.sweep_angle += SWEEP_STEP if forward else -SWEEP_STEP
            if phase in (24, 167):
                self._start_random_move()
        ready = [shot for shot in self.pending if shot.active_frame <= self.frame]
        self.pending = [shot for shot in self.pending if shot.active_frame > self.frame]
        return ready
