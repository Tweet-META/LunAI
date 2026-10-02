import math
import random
import unittest
from collections import Counter

from rl.scarlet_meister import ScarletMeisterPattern


class ScarletMeisterTests(unittest.TestCase):
    def test_native_first_burst_activation_timing_and_hitbox_sizes(self):
        pattern = ScarletMeisterPattern(rng=random.Random(7))
        shots = {}
        for frame in range(1, 216):
            shots[frame] = pattern.step(192, 384)
        self.assertTrue(all(not shots[f] for f in range(1, 184)))
        # Native Hard snapshot: small bullets at 184, large at 207, medium at 215.
        self.assertEqual(Counter(s.radius for s in shots[184]), {3.0: 10})
        self.assertEqual(Counter(s.radius for s in shots[207]), {16.0: 1})
        self.assertEqual(Counter(s.radius for s in shots[215]), {8.0: 8})
        big = shots[207][0]
        self.assertAlmostEqual(big.angle, math.pi / 2)
        self.assertAlmostEqual(big.speed, 6.2)
        self.assertAlmostEqual(big.x, 192)
        # Normal Bullet.move adds one full-speed step on the activation frame.
        self.assertAlmostEqual(big.y + big.speed, 192.6)
        self.assertEqual(big.spawn_frame, 184)

    def test_schedule_has_two_opposite_sweeps_and_reaims(self):
        pattern = ScarletMeisterPattern(rng=random.Random(1))
        all_shots = []
        for frame in range(1, 650):
            all_shots.extend(pattern.step(320 if frame < 220 else 64, 384))
        centers = {s.spawn_frame: s.angle for s in all_shots if s.radius == 16}
        self.assertEqual(sorted(f for f in centers if f < 552),
                         [184,190,196,202,208] + list(range(214,263,3))
                         + [345,351] + list(range(357,406,3)))
        for start, sign in [(214,1),(357,-1)]:
            for frame in range(start+3,start+49,3):
                self.assertAlmostEqual(centers[frame]-centers[frame-3], sign*math.pi/8)
        self.assertIn(552, centers)
        self.assertNotAlmostEqual(centers[208], centers[345])

    def test_seeded_randomness_movement_bounds_and_full_timeout(self):
        def sample(seed):
            pattern = ScarletMeisterPattern(rng=random.Random(seed))
            shots = []
            positions = []
            for _ in range(1800):
                shots.extend(pattern.step(192,384))
                positions.append((pattern.x,pattern.y))
            return shots, positions
        shots, positions = sample(8)
        self.assertEqual((shots,positions), sample(8))
        self.assertNotEqual(shots, sample(9)[0])
        self.assertEqual(len(shots), 3534)
        self.assertTrue(all(32<=x<=352 and 48<=y<=120 for x,y in positions))
        self.assertGreater(len(set(positions[208:])), 100)
        for shot in shots:
            self.assertTrue(math.isfinite(shot.x+shot.y+shot.angle+shot.speed))
            if shot.radius==3:
                self.assertTrue(2<=shot.speed<=3)
            elif shot.radius==8:
                self.assertTrue(3<=shot.speed<=6)
            else:
                self.assertEqual(shot.speed,6.2)

    def test_native_bullet_limit_includes_pending_spawn_effects(self):
        pattern = ScarletMeisterPattern(rng=random.Random(1))
        for _ in range(184):
            ready = pattern.step(192,384,active_bullets=635)
        self.assertEqual(len(ready) + len(pattern.pending), 5)


if __name__ == '__main__':
    unittest.main()
