# LunAI

LunAI is a reinforcement learning project for training bullet-hell game agents in a pygame-based Touhou-style environment.

The current project focuses on building a usable RL environment, multi-scale bullet observations, reward functions, curriculum stages, and PPO/DQN baseline agents.

## Project Scope

This repository includes:

- a modified pygame Touhou-style game environment
- blue/yellow/red multi-scale observation maps
- RL environment wrappers
- reward function design
- DQN and PPO baseline training scripts
- curriculum levels for staged training
- evaluation and visualization tools

## Acknowledgements

The game environment in this project is adapted from [`NumPix/pygame-touhou`](https://github.com/NumPix/pygame-touhou) by Leonid Ushakov, which is licensed under the MIT License.

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
