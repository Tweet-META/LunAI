# LunAI Experiments

This directory is a standalone copy of the pygame environment and the code used for paper experiments. Run commands from this directory so imports, level files, assets, checkpoints, and logs resolve locally.

## Setup

```powershell
cd experiments
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
