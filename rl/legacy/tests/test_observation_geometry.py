import unittest

from observation_builder import (
    BulletState,
    PlayerState,
    expand_hazard_for_player,
    is_aabb_hazard,
    make_occupancy_map,
    pccm_sample_components,
)
from rl.th06_adapter import (
    Th06BulletSnapshot,
    Th06EnemySnapshot,
    Th06ObservationAdapter,
)


class ObservationGeometryTests(unittest.TestCase):
    # Check that AABB corners remain inside the collision map.
    def test_aabb_occupancy_keeps_rectangle_corners(self) -> None:
        box = BulletState(
            x=5.0,
            y=5.0,
            radius=2.0,
            vx=0.0,
            vy=0.0,
            half_width=2.0,
            half_height=1.0,
        )
        circle = BulletState(x=5.0, y=5.0, radius=2.0, vx=0.0, vy=0.0)
        box_map = make_occupancy_map(11, 11, [box])
        circle_map = make_occupancy_map(11, 11, [circle])
        self.assertEqual(float(box_map[6, 7]), 1.0)
        self.assertEqual(float(circle_map[6, 7]), 0.0)

    # Check that rectangle expansion adds each player half extent separately.
    def test_aabb_minkowski_expansion_preserves_axes(self) -> None:
        box = BulletState(
            x=5.0,
            y=5.0,
            radius=4.0,
            vx=2.0,
            vy=3.0,
            half_width=4.0,
            half_height=2.0,
        )
        player = PlayerState(
            x=0.0,
            y=0.0,
            radius=1.0,
            half_width=1.0,
            half_height=0.5,
        )
        expanded = expand_hazard_for_player(box, player)
        self.assertTrue(is_aabb_hazard(expanded))
        self.assertAlmostEqual(expanded.half_width, 5.0)
        self.assertAlmostEqual(expanded.half_height, 2.5)
        self.assertAlmostEqual(expanded.radius, 5.0)
        self.assertAlmostEqual(expanded.vx, box.vx)
        self.assertAlmostEqual(expanded.vy, box.vy)

    # Check that PCCM hard collision uses the rectangle instead of a circle.
    def test_aabb_pccm_uses_rectangle_hard_region(self) -> None:
        box = BulletState(
            x=5.0,
            y=5.0,
            radius=2.0,
            vx=0.0,
            vy=0.0,
            half_width=2.0,
            half_height=1.0,
        )
        _, _, _, hard = pccm_sample_components(
            [box],
            (0, 0, 10, 10),
            (10, 10),
            10,
            10,
            1,
            2.0,
            0.1,
            upper_field_cost=0.0,
        )
        self.assertEqual(float(hard[5, 6]), 1.0)
        self.assertEqual(float(hard[5, 7]), 0.0)

    # Check that the TH06 adapter keeps exported bullet and enemy dimensions.
    def test_th06_adapter_keeps_hitbox_width_and_height(self) -> None:
        adapter = Th06ObservationAdapter()
        bullet = Th06BulletSnapshot(
            x=10.0,
            y=20.0,
            vx_per_frame=0.0,
            vy_per_frame=1.0,
            hitbox_width=8.0,
            hitbox_height=4.0,
            speed_per_frame=1.0,
            angle=0.0,
            state=1,
            flags=0,
        )
        enemy = Th06EnemySnapshot(
            x=30.0,
            y=40.0,
            vx_per_frame=0.0,
            vy_per_frame=0.0,
            hitbox_width=24.0,
            hitbox_height=12.0,
            life=10,
            max_life=10,
            is_boss=False,
            contact_active=True,
        )
        bullet_hazard = adapter._bullet_hazards((bullet,))[0]
        enemy_hazard = adapter._enemy_hazards((enemy,))[0]
        self.assertEqual((bullet_hazard.half_width, bullet_hazard.half_height), (4.0, 2.0))
        self.assertEqual((enemy_hazard.half_width, enemy_hazard.half_height), (8.0, 4.0))


if __name__ == "__main__":
    unittest.main()
