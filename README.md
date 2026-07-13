# LunAI

LunAI is a reinforcement learning project for training bullet-hell game agents in a pygame-based Touhou-style environment.

The current project focuses on a multi-scale, multi-frame CNN PPO agent. The MLP PPO agent is kept as a baseline, and the earlier DQN implementation remains under `rl/legacy_dqn/`.

## Project Scope

This repository includes:

- a modified pygame Touhou-style game environment
- blue/yellow/red multi-scale observation maps
- Potential Collision Cost Maps (PCCM) with short-horizon bullet prediction
- playable-area masks for player-centered local maps
- RL environment wrappers
- reward function design
- MLP PPO baseline training scripts
- CNN PPO training and evaluation scripts
- legacy DQN baseline code under `rl/legacy_dqn/`
- curriculum levels for staged training
- evaluation and visualization tools

## RL Entry Points

MLP PPO baseline:

```powershell
python rl/train_ppo.py --episodes 300 --level-file level_1.json
```

CNN PPO main training path:

```powershell
python rl/train_ppo_cnn.py --config config.json
```

`config.json` is the versioned baseline for the current experiment. Command-line values override the file, so a temporary budget change is concise:

```powershell
python rl/train_ppo_cnn.py --config config.json --max-total-frame-steps 1000000
```

Each new PPO log begins with a `# run_config:` JSON line containing the final effective parameters, including any command-line overrides. The trend viewer skips this metadata line automatically.

`global_step` means a policy decision. `total_frame_steps` means actual game frames and is the recommended unit for comparing training budgets. The CNN trainer currently contains an experimental invincible-contact mode: a bullet or enemy-body contact does not end the training episode, but receives the collision penalty on every contact frame. CNN evaluation remains lethal. This training-evaluation mismatch is recorded as a failed v6 experiment and is not the recommended main training route.

### PCCM (Potential Collision Cost Map) Observation

The current `pccm` observation schema keeps the three CNN map branches and gives every scale three channels per frame:

- bullet occupancy or density
- projected PCCM risk
- playable-area mask

PCCM uses each bullet's own position, hitbox, and velocity to estimate current soft danger and the next five game frames. Bullet buffers, predicted trajectories, and four wall costs use soft probabilistic composition capped below hard collision. Current collision regions are then restored to `1.0`.

The implementation does not build a full-screen PCCM. It samples the same world-space cost rule directly at each scale, uses internal supersampling for yellow and blue maps, and preserves the exact `64x64` red occupancy. The full-grid NumPy implementation remains the training default. An exact floating-point ROI implementation and an experimental `auto` hybrid are retained for profiling, but real-level benchmarks did not show an end-to-end observation-building speedup. Multi-frame stacking remains enabled so the CNN can still learn non-linear changes that the short constant-velocity prediction cannot describe.

Run `python tools/benchmark_pccm_roi.py` to compare the reference, pure ROI, and `auto` implementations at 80, 200, and 500 synthetic bullets. The command also checks maximum and mean absolute error and requires zero hard-collision mismatches. Run `python tools/benchmark_pccm_level.py` for a complete observation-building comparison on the dedicated 500-bullet pygame level.

Old CNN checkpoints automatically use the legacy `motion` schema. New PCCM checkpoints store their observation schema, prediction horizon, halo width, and wall margin to prevent silent evaluation mismatches.

### Parallel Environment Sampling

`--num-envs` defaults to `1`. It can use several headless game environments on a CUDA-capable machine while keeping one shared CNN PPO model in the main process:

```powershell
python rl/train_ppo_cnn.py --config config.json
```

The environment workers run pygame and observation building on CPU. The main process batches their observations for one GPU model, so workers do not create separate models or checkpoints. `rollout_steps` is the total PPO batch size and must be divisible by `num_envs`. Parallel training cannot use `--render`, and CNN logs include an `env_id` column.

PCCM logs also record mean local risk, total PCCM shaping reward, wall-time ratio, and all nine episode action counts.

## Acknowledgements

The game environment in this project is adapted from [`NumPix/pygame-touhou`](https://github.com/NumPix/pygame-touhou), which is licensed under the MIT License.

The reinforcement learning components, including the observation representation, reward design, environment wrapper, curriculum levels, training scripts, evaluation tools, and visualization utilities, were developed for the LunAI project.

All Touhou Project characters, music, and related intellectual property belong to ZUN and Team Shanghai Alice.

## Controls

### Game

- Move: arrow keys
- Shoot: `Z`
- Slow movement: `Shift`

### Menu

- Select: `Enter` or `Z`
- Cancel: `X`

## License

This project is released under the MIT License. The original game environment copyright notice is preserved in [LICENSE](LICENSE).
