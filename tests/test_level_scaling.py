import json
import unittest
from pathlib import Path

from assets.scripts.math_and_data.level_scaling import ATTACK_SPATIAL_FIELDS, scale_level


LEVEL_DIR = Path(__file__).resolve().parent.parent / "assets" / "levels"


class LevelScalingTests(unittest.TestCase):
    def test_every_existing_level_scales_without_changing_source(self):
        for path in LEVEL_DIR.glob("*.json"):
            with self.subTest(level=path.name):
                original = json.loads(path.read_text(encoding="utf-8"))
                scaled = scale_level(original)
                self.assertEqual(scaled["playfield_size"], [384, 448])
                self.assertNotIn("playfield_size", original)
                for before, after in zip(original["enemies"], scaled["enemies"]):
                    self.assertAlmostEqual(after["start_position"][0], before["start_position"][0] * 0.64)
                    self.assertAlmostEqual(after["start_position"][1], before["start_position"][1] * 0.64)
                    self.assertAlmostEqual(after["collider"]["radius"], before["collider"]["radius"] * 0.64)
                    for old_attack, new_attack in zip(before["attacks"], after["attacks"]):
                        for index in ATTACK_SPATIAL_FIELDS[old_attack[0]]:
                            self.assertAlmostEqual(new_attack[index], old_attack[index] * 0.64)
                        self.assertAlmostEqual(new_attack[3][3], old_attack[3][3] * 0.64)
                        self.assertAlmostEqual(new_attack[3][5], (old_attack[3][5] if len(old_attack[3]) > 5 else 1.0) * 0.64)

    def test_native_level_does_not_scale_twice(self):
        native = {"playfield_size": [384, 448], "enemies": []}
        self.assertEqual(scale_level(native), native)


if __name__ == "__main__":
    unittest.main()
