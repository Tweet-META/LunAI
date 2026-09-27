# LunAI Experiments

This directory is a standalone copy of the pygame environment and the code used for paper experiments. Run commands from this directory so imports, level files, assets, checkpoints, and logs resolve locally.

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

`level_7.json` and `level_8.json` are the increased-intensity extrapolation levels. The root project intentionally omits experimental switches and keeps only the production observation path.

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
