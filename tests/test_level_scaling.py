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
                source_width = original.get("playfield_size", [600, 700])[0]
                scale = 384 / source_width
                self.assertEqual(scaled["playfield_size"], [384, 448])
                for before, after in zip(original["enemies"], scaled["enemies"]):
                    self.assertAlmostEqual(after["start_position"][0], before["start_position"][0] * scale)
                    self.assertAlmostEqual(after["start_position"][1], before["start_position"][1] * scale)
                    self.assertAlmostEqual(after["collider"]["radius"], before["collider"]["radius"] * scale)
                    for old_attack, new_attack in zip(before["attacks"], after["attacks"]):
                        for index in ATTACK_SPATIAL_FIELDS[old_attack[0]]:
                            self.assertAlmostEqual(new_attack[index], old_attack[index] * scale)
                        self.assertAlmostEqual(new_attack[3][3], old_attack[3][3] * scale)
                        self.assertAlmostEqual(
                            new_attack[3][5],
                            (old_attack[3][5] if len(old_attack[3]) > 5 else 1.0) * scale,
                        )

    def test_native_level_does_not_scale_twice(self):
        native = {"playfield_size": [384, 448], "enemies": []}
        self.assertEqual(scale_level(native), native)


if __name__ == "__main__":
    unittest.main()
