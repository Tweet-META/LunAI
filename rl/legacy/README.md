# Legacy RL Baselines

This folder archives the earlier MLP PPO, DQN, and random-agent baselines.

The active LunAI training path uses CNN PPO from the parent `rl/` package. Files in this folder are not part of the current training pipeline.

Example commands:

```powershell
python rl/legacy/train_dqn.py --episodes 200 --level-file level_1.json
python rl/legacy/evaluate_dqn.py --model-path checkpoints/dqn_baseline.pt --level-file level_1.json
python rl/legacy/train_ppo.py --episodes 300 --level-file level_1.json
python rl/legacy/evaluate_ppo.py --model-path checkpoints/ppo_baseline.pt --level-file level_1.json
```
