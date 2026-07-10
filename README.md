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
python rl/train_ppo_cnn.py --episodes 300 --level-file level_1.json
```

To train for a fixed environment-frame budget, set a large episode ceiling and add `--max-total-frame-steps`. For example, this stops close to one million game frames:

```powershell
python rl/train_ppo_cnn.py --episodes 1000000 --max-total-frame-steps 1000000 --level-file level_1.json
```

`global_step` means a policy decision. `total_frame_steps` means actual game frames and is the recommended unit for comparing training budgets.

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
