# LunAI

The current main project uses the Pygame environment and code copied from LunAI-experiments. Run commands from this directory so imports, level files, assets, checkpoints, and logs resolve locally. Existing checkpoints, training history, legacy code and older levels are retained. The native TH06 adapter files are retained separately; the current training and evaluation entry points use Pygame.

## Setup

```powershell
pip install -r requirements.txt
```

## Observation-scale ablations

```powershell
python rl/train_ppo_cnn.py --config config_red_only.json
python rl/train_ppo_cnn.py --config config_red_blue.json
python rl/train_ppo_cnn.py --config config.json
```

The inactive scales are zero-masked inside the unchanged three-branch network, so `red_only`, `red_blue`, and `full` retain the same trainable parameter count.

## PCCM ablations

Use distinct output paths for every run:

```powershell
python rl/train_ppo_cnn.py --config config.json --pccm-observation-mode occupancy_only --model-path checkpoints/occupancy_seed0.pt --log-path training_logs/occupancy_seed0.csv
python rl/train_ppo_cnn.py --config config.json --pccm-observation-mode static --model-path checkpoints/static_seed0.pt --log-path training_logs/static_seed0.csv
python rl/train_ppo_cnn.py --config config.json --pccm-observation-mode trajectory --model-path checkpoints/trajectory_seed0.pt --log-path training_logs/trajectory_seed0.csv
```

## Evaluation and profiling

```powershell
python rl/evaluate_ppo_cnn.py --model-path checkpoints/trajectory_seed0.pt --episodes 150 --level-file level_6.json --log-path evaluation_logs/trajectory_seed0_level6.csv
python rl/evaluate_random_agent.py --episodes 150 --level-file level_6.json --log-path evaluation_logs/random_level6.csv
python tools/benchmark_pccm_roi.py
python tools/benchmark_pccm_level.py
```

The project now includes observation-scale and PCCM ablation switches, reconstructed TH06 spell-card levels, and profiling implementations. The copied `config.json` uses a 1M-frame budget and a 2160-frame episode cap; the formal paper experiment instead uses the 2M/1800-frame settings in `tools/run_formal_9_1_seeds.ps1`.

## Optional PyTorch PCCM prototype

The default PCCM implementation remains `reference`. On a CUDA machine, first
compare the PyTorch output with the reference and measure complete observation
builds on a real level:

```powershell
python -m unittest tests.test_torch_pccm -v
python tools/benchmark_pccm_torch.py --level-file level_th06_stage3_spell2.json
```

The benchmark stops if any PCCM value differs by more than `5e-4`. To try the
optional backend in a separate training run, add `--pccm-implementation torch_cuda`
and use a new checkpoint and log path. This prototype still builds
one observation per environment process and returns NumPy maps; a reported GPU
kernel time alone does not establish a training speedup.

## Optional Numba CPU PCCM

The `numba` backend fuses distance, halo and probability-product operations into
a single-threaded compiled loop, avoiding the large bullet/time/grid temporary
arrays. It also compiles full-field and red-zone circular occupancy rasterization,
preserving the original float64 geometry, clipping, rounding and pixel-center
rules. Occupancy masks and density inputs must match exactly. PCCM uses float32
without fast-math. Collision detection is unchanged and `reference` remains the
default. Install only the optional dependency:

```powershell
python -m pip install -r requirements-numba.txt
python -m unittest tests.test_numba_pccm tests.test_numba_occupancy -v
python tools/benchmark_pccm_numba.py
```

The benchmark checks every sampled snapshot against the reference (maximum
absolute PCCM error `3e-6`) and compares complete observation builds on all three
stage-3 spells. It prints bullet counts and separates first-build compilation
or cache loading from steady-state timings. For one level, pass
`--level-file level_th06_stage3_spell2.json`; optionally save a report with
`--json-path evaluation_logs/numba_benchmark.json`.
Add `--compare-occupancy` to also measure `numba_pccm_only`, reproducing the
earlier backend with reference occupancy. This isolates the extra speedup from
compiled rasterization on the same snapshots and in alternating measurement order.

After checking speed on the training machine, add `--pccm-implementation numba`
to training or evaluation. PPO's `--device` remains independent. Each environment
process uses one compiled CPU thread; first use can pause for compilation/cache
loading. Observation timing does not establish the full PPO training speedup.

To include game updates, rewards and four-frame CNN input preparation, and
separately attribute CPU hot paths:

```powershell
python tools/profile_env_cpu.py
```

This writes `evaluation_logs/cpu_environment_profile.json`. Timing passes run
without instrumentation; a separate pass measures components (entries marked
`nested` are contained in other components and must not be added again). The
workload uses a stationary invincible player and continues after collision
signals to compare the same dense frames. This is a performance diagnostic,
not survival evaluation. It excludes model inference, PPO updates and IPC.

## Yellow-gap diagnostic preview

`level_yellow_gap_diagnostic.json` is a synthetic 30-second level with 30 seeded,
single-row descending walls. Each wall has one gap in the middle of the field.
The next gap is 32–80 pixels from the previous one, so the passage changes
without jumping beyond the Yellow observation window. The player starts centered.
It is separate from the three TH06 spell-card training levels.

```powershell
python tools/preview_yellow_gap.py --policy oracle --seed 10001
python tools/preview_yellow_gap.py --policy manual --seed 10001
python tools/preview_yellow_gap.py --policy oracle --seed 10001 --debug
```

The oracle reads the hidden answer to check physical reachability. It is not an
agent result. In manual mode, use arrow keys or WASD. The debug view shows the
three observation scales. A collision ends the run; repeat with another seed
to inspect a different left/right sequence.
