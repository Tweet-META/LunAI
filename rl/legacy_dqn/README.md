# Legacy DQN Baseline

This folder keeps the earlier DQN baseline implementation for reference.

The current LunAI training path uses PPO from the parent `rl/` package. The DQN code is kept here as a historical baseline and ablation reference, but it is no longer the main training path.

Example commands:

```powershell
python rl/legacy_dqn/train_dqn.py --episodes 200 --level-file level_1.json
python rl/legacy_dqn/evaluate_dqn.py --model-path checkpoints/dqn_baseline.pt --level-file level_1.json
```
