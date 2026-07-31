import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "assets" / "levels" / "level_6.json"
OUTPUT_PATH = ROOT / "assets" / "levels" / "level_eval_alternating_pressure.json"
MIRROR_OUTPUT_PATH = ROOT / "assets" / "levels" / "level_eval_alternating_pressure_mirror.json"


# Keep an enemy on one side of the playfield during its attack.
def set_stationary_side_path(enemy: dict, x_position: int) -> None:
    enemy["start_position"] = [x_position, 0]
    enemy["trajectory"] = [
        [x_position, 70],
        [x_position, 110],
        [x_position, 150],
        [x_position, 120],
        [x_position, -80],
    ]


# Configure one downward rectangular bullet wall.
def set_attack_pressure(enemy: dict) -> None:
    bullet_data = enemy["attacks"][0][3]
    enemy["attacks"] = [
        [
            "rectangle_wall",
            25,
            6,
            bullet_data,
            360,
            100,
            180,
            145,
            0.35,
            0,
        ]
    ]


# Create one enemy from the Level 6 attack template.
def make_enemy(template: dict, time: float, x_position: int) -> dict:
    enemy = copy.deepcopy(template)
    enemy["time"] = time
    set_stationary_side_path(enemy, x_position)
    set_attack_pressure(enemy)
    return enemy


# Build a balanced level that repeatedly swaps the safer side.
def build_level() -> dict:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    template = source["enemies"][0]
    enemies = []
    phase_starts = [0.5, 7.0, 13.5, 20.0, 26.5, 33.0]

    for phase_index, phase_start in enumerate(phase_starts):
        left_is_heavy = phase_index % 2 == 0
        wall_center = 180 if left_is_heavy else 420
        enemies.append(make_enemy(template, phase_start, wall_center))

    return {
        "length": 46.0,
        "name": "evaluation_alternating_pressure",
        "description": (
            "Alternating one-sided pressure that repeatedly moves all projectile "
            "sources between the left and right halves of the playfield."
        ),
        "enemies": enemies,
    }


# Mirror all horizontal coordinates while preserving the attack schedule.
def mirror_level(level: dict) -> dict:
    mirrored = copy.deepcopy(level)
    mirrored["name"] = "evaluation_alternating_pressure_mirror"
    mirrored["description"] = (
        "Horizontally mirrored alternating-pressure evaluation level."
    )

    for enemy in mirrored["enemies"]:
        enemy["start_position"][0] = 600 - enemy["start_position"][0]
        for point in enemy["trajectory"]:
            point[0] = 600 - point[0]

    return mirrored


# Write the generated levels using the repository JSON style.
def main() -> None:
    level = build_level()
    mirrored_level = mirror_level(level)
    OUTPUT_PATH.write_text(json.dumps(level, indent=2) + "\n", encoding="utf-8")
    MIRROR_OUTPUT_PATH.write_text(
        json.dumps(mirrored_level, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote two levels with {len(level['enemies'])} enemies each.")


if __name__ == "__main__":
    main()
