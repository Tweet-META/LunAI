# Hard Scarlet Meister / 红符「绯红之主」

Level: `level_th06_stage6_scarlet_meister_hard.json` (384 × 448, 1800 frames).

## Source and scope

Reconstructed from the local original `th06_ST.DAT` archive, `ecldata6.ecl`:

- Subroutine 39: spell sequence, random boss movement and 1800-frame timeout.
- Subroutine 37: one volley. Apply the Hard difficulty mask `0x04`.
- Spell ID 49 (zero-based), displayed spell number 50; stage-local spell ordinal 4.
- Native movement: `EclManager.cpp` and `EnemyEclInstr.cpp`.
- Native bullet generation, activation and size: `BulletManager.cpp`.

The simulator uses its existing player, reward, observation and collision code.
This is an isolated survival task, without shooting down the boss or native spell
scoring. The boss starts at a standardized (192, 64), moves to (192, 112) over 120
frames, and then follows the spell's random movement. In the original full stage,
its initial position depends on the preceding attack. Python's seeded random
stream replaces TH06's RNG: equivalent parameter ranges do not imply identical
trajectories for an identically numbered seed. Sprites use the simulator's neutral
bullet and fairy graphics; original charge and spawn animations are omitted.

## Attack parameters

Every volley contains 19 bullets in Hard:

| Count | Speed (px/frame) | Angle around volley center | Simulator radius | Delay before activation |
|---:|---:|---:|---:|---:|
| 1 | 6.2 | 0° | 16 px | 23 frames |
| 3 | uniform 4–6 | uniform ±5.625° | 8 px | 31 frames |
| 5 | uniform 3–4 | uniform ±9° | 8 px | 31 frames |
| 10 | uniform 2–3 | uniform ±22.5° | 3 px | 0 frames |

The extra five slow bullets under mask `0x08` are Lunatic-only and excluded.
Native full hitbox widths 32/16/6 are mapped to simulator radii 16/8/3; the
simulator's existing collision calculation is retained, not replaced by native
TH06 collision geometry.

Delayed bullets move at half speed during their non-collidable spawn animation.
They are queued outside the active bullet/observation list, and enter it at the
correct advanced position. On activation the regular full-speed step also runs,
as in the native engine. The active-plus-pending bullet budget is 640.

First volley is at frame 184. Each cycle is 368 frames, with these relative times:

- 0, 6, 12, 18, 24: independently aimed volleys.
- 30, 33, …, 78: 17 volleys, starting at the current player angle and increasing
  by 22.5° each time; the remaining sweep does not re-aim.
- 161, 167: independently aimed volleys.
- 173, 176, …, 221: 17 volleys sweeping in the opposite direction.
- At 24 and 167: begin a 90-frame decelerating random move, initial speed 2.5
  px/frame. Bounds are x=32–352 and y=48–120, with the original heading reflection.

These timings include the elapsed time introduced by ECL's `JUMPDEC` loops;
reading the displayed ECL timestamps without unfolding loops gives wrong timing.

## Validation

- A native headless run (stage 6, difficulty 2, ordinal 4) returned spell ID 49.
- Its first active small bullets appeared at frame 184; the first large bullet
  at 207, and the first eight medium bullets at 215.
- The first native large bullet at frame 207 was at (192, 192.6001), moving down
  at 6.2 px/frame. The port reproduces (192, 192.6), including spawn travel.
- Unit checks cover this native reference, the unfolded sweep schedule, speed
  and size ranges, random movement bounds, seeded repeatability and bullet budget.
- Full simulator smoke runs use invincibility only to inspect all 1800 frames;
  this does not establish difficulty or a trained policy's performance.

No training or evaluation groups automatically include the new level. Existing
three-spell experiments continue to select their original level files.

## Preview

From `LunAI-experiments`, with the same Python environment used for training:

```powershell
python tools/preview_scarlet_meister.py --invincible
```

Move with arrows/WASD; Esc or closing the window exits. Invincibility is only for
previewing a whole cycle. Omit `--invincible` to stop at the first collision.
Use `--policy stay` to inspect aimed volleys without moving, `--debug` for
observation panels, or `--seed 10002` for another random realization. If Numba is
not installed on the preview machine, add `--pccm-implementation reference`.
