# LunAI

LunAI is a reinforcement learning project for training bullet-hell game agents in a pygame-based Touhou-style environment.

The current project focuses on building a usable RL environment, multi-scale bullet observations, reward functions, curriculum stages, and PPO baseline agents. The MLP PPO agent is kept as the baseline, while the CNN PPO agent is the next experimental training path.

## Project Scope

This repository includes:

- a modified pygame Touhou-style game environment
- blue/yellow/red multi-scale observation maps
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

CNN PPO experiment:

```powershell
python rl/train_ppo_cnn.py --config config.json
```

`config.json` is the versioned baseline for the current experiment. Command-line values override the file, so a temporary budget change is concise:

```powershell
python rl/train_ppo_cnn.py --config config.json --max-total-frame-steps 1000000
```

Each new PPO log begins with a `# run_config:` JSON line containing the final effective parameters, including any command-line overrides. The trend viewer skips this metadata line automatically.

`global_step` means a policy decision. `total_frame_steps` means actual game frames and is the recommended unit for comparing training budgets. The CNN trainer currently uses an invincible-contact curriculum: a bullet or enemy-body contact does not end the training episode, but receives the collision penalty on every contact frame. CNN evaluation remains lethal, so reported evaluation survival is still based on real collisions.

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
