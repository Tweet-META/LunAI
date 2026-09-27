import random
import unittest

import numpy as np

from observation_builder import BulletState, ObservationBuilder, PlayerState
from rl.yellow_gap_diagnostic import (
    GAP_CENTERS,
    GAP_HALF_WIDTH,
    MAX_GAP_SHIFT,
    MIN_GAP_SHIFT,
    WALL_SPEED,
    WAVE_COUNT,
    WAVE_INTERVAL_FRAMES,
    choose_gap_centers,
    wall_points,
)


class YellowGapDiagnosticTests(unittest.TestCase):
    def test_seeded_gaps_change_without_leaving_the_middle(self):
        for seed in range(30):
            random.seed(seed)
            centers = choose_gap_centers(WAVE_COUNT)
            self.assertEqual(len(centers), WAVE_COUNT)
            self.assertTrue(all(center in GAP_CENTERS for center in centers))
            for previous, current in zip(centers, centers[1:]):
                self.assertLessEqual(MIN_GAP_SHIFT, abs(current - previous))
                self.assertLessEqual(abs(current - previous), MAX_GAP_SHIFT)
            for center in centers:
                points = wall_points(center)
                self.assertEqual(len({y for _, y in points}), 1)
                self.assertEqual(len(points), 42)
                self.assertTrue(all(abs(x - center) > GAP_HALF_WIDTH for x, _ in points))

    def test_yellow_can_distinguish_midrange_gaps(self):
        builder = ObservationBuilder()
        player = PlayerState(x=192.0, y=384.0, radius=1.92)
        observations = []
        for center in (128.0, 256.0):
            bullets = [
                BulletState(x, y, 4.0, 0.0, WALL_SPEED)
                for x, y in wall_points(center, 300.0)
            ]
            observations.append(builder.build(bullets, player))
        left, right = observations
        np.testing.assert_allclose(left["red_occupancy"], right["red_occupancy"], atol=1e-7)
        np.testing.assert_allclose(left["red_pccm"], right["red_pccm"], atol=1e-7)
        self.assertGreater(
            float(np.max(np.abs(left["yellow_pccm"] - right["yellow_pccm"]))),
            0.1,
        )

        aligned = [
            BulletState(x, y, 4.0, 0.0, WALL_SPEED)
            for x, y in wall_points(192.0, 384.0)
        ]
        observation = builder.build(aligned, player)
        self.assertEqual(float(observation["red_occupancy"][32, 32]), 0.0)
        self.assertLess(float(np.min(observation["yellow_pccm"][7:9, 7:9])), 0.8)

    def test_seeded_oracle_can_cross_full_sequence(self):
        player_speed = 370.0 * 384.0 / 600.0 / 60.0
        bullet_speed = WALL_SPEED / 60.0
        collision_radius = 4.0 + 3.0 * 384.0 / 600.0
        for seed in range(12):
            random.seed(seed)
            player_x = 192.0
            centers = choose_gap_centers(WAVE_COUNT)
            walls = [wall_points(center) for center in centers]
            for frame in range(1800):
                wave = min(frame // WAVE_INTERVAL_FRAMES, WAVE_COUNT - 1)
                target = centers[wave]
                player_x += float(np.clip(target - player_x, -player_speed, player_speed))
                first_active = max(0, (frame - 100) // WAVE_INTERVAL_FRAMES)
                for active in range(first_active, wave + 1):
                    traveled = (frame - active * WAVE_INTERVAL_FRAMES) * bullet_speed
                    for bullet_x, bullet_y in walls[active]:
                        distance_sq = (player_x - bullet_x) ** 2 + (384.0 - bullet_y - traveled) ** 2
                        self.assertGreater(
                            distance_sq,
                            collision_radius**2,
                            f"seed={seed} frame={frame} wave={active}",
                        )


if __name__ == "__main__":
    unittest.main()
