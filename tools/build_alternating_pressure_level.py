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


# Configure downward random-cone bullets for one pressure source.
def set_attack_pressure(enemy: dict) -> None:
    random_attack = enemy["attacks"][0]

    random_attack[0] = "long_random_cone"
    random_attack[1] = 7
    random_attack[2] = 13
    random_attack[4:4] = [180, 20]
    random_attack[7] = 0.35
    random_attack[8] = 3.8 / 12
    enemy["attacks"] = [random_attack]


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
        heavy_positions = [100, 220] if left_is_heavy else [380, 500]

        for offset, x_position in enumerate(heavy_positions):
            enemies.append(make_enemy(template, phase_start + offset * 0.35, x_position))

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
