"""Load legacy 600x700 levels into the current playfield coordinates."""

from copy import deepcopy

from playfield_config import LEGACY_PLAYFIELD_SIZE, PLAYFIELD_SIZE


ATTACK_SPATIAL_FIELDS = {
    "wide_ring": (4,),
    "long_random": (4,),
    "long_random_cone": (6,),
    "wide_cone": (5,),
    "rectangle_wall": (4, 5, 7),
    "th06_aimed_circle": (4, 5),
    "th06_aimed_fan": (4, 5),
}


def scale_level(level: dict, target_size: tuple[int, int] = PLAYFIELD_SIZE) -> dict:
    """Return a scaled copy; source JSON stays intact for reproducibility."""
    result = deepcopy(level)
    source_size = tuple(result.get("playfield_size", LEGACY_PLAYFIELD_SIZE))
    sx = target_size[0] / source_size[0]
    sy = target_size[1] / source_size[1]
    if abs(sx - sy) > 1e-9:
        raise ValueError("Level scaling requires the same horizontal and vertical scale.")
    scale = sx

    for enemy in result["enemies"]:
        enemy["start_position"] = [value * scale for value in enemy["start_position"]]
        enemy["trajectory"] = [[value * scale for value in point] for point in enemy["trajectory"]]
        enemy["collider"]["radius"] *= scale
        enemy["collider"]["offset"] = [value * scale for value in enemy["collider"]["offset"]]
        for attack in enemy["attacks"]:
            kind = attack[0]
            if kind not in ATTACK_SPATIAL_FIELDS:
                raise ValueError(f"Unknown attack geometry: {kind}")
            for index in ATTACK_SPATIAL_FIELDS[kind]:
                attack[index] *= scale
            bullet = attack[3]
            bullet[3] *= scale  # Collision radius; sprite frame dimensions stay unchanged.
            bullet[4] = [value * scale for value in bullet[4]]
            if len(bullet) > 5:
                bullet[5] *= scale
            else:
                bullet.append(scale)

    result["playfield_size"] = list(target_size)
    return result
