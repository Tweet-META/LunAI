# Tian 射线 DQN 基线（统一环境下的重新实现）

来源：Dongxin Tian, *Artificial Intelligent Player for Bullet Hell Games Based on
Deep Q-Networks*, ICMLCA 2023, pp. 931–935.
DOI: https://doi.org/10.1145/3650215.3650381

这是根据论文重新实现并适配 LunAI 环境的基线，不是作者代码，也不声称完全复现
原文实验数值。本次检索未找到可以确认的作者公开实现。训练结果尚未验证；请先
检查学习曲线和行为，再决定是否把这套超参数冻结为正式比较设置。

## 隔离范围

所有新增源码、配置、测试、模型、训练记录和评估 CSV 都在本目录。独立的
`RayTouhouEnv` 子类调用现有模拟器及奖励函数，不修改 PPO、观察构造器、关卡或
碰撞代码。游戏素材、符卡和游戏逻辑继续依赖原项目，因此这个目录不能脱离
LunAI-experiments 单独运行。游戏引擎仍会打开其已有的排行榜数据库；本基线不提交
分数或更改排行榜。无需 Numba，也不会构造 PCCM 或占用图。

## 论文设定与适配

| 内容 | 实现 | 来源/说明 |
|---|---|---|
| 输入 | 36 条射线 × 5 特征 = 180 维 | 论文 4.1.3、4.2.1 |
| 每条射线 | 最近相交子弹中心的 dx、dy、vx、vy、detected | 论文 4.1.3 |
| 网络 | 180 → 128 → 64 → 9 | 论文 Model B |
| 算法 | 普通 DQN；均匀经验回放、独立目标网络 | 按论文 DQN 描述实现；不用 Double/Dueling/PER |
| 动作 | 八方向＋不动，每帧一个动作 | 与论文语义相同；数字编号沿用 LunAI |
| 空间输入 | 不加玩家坐标、墙距离、地图、PCCM、历史帧 | 网络严格接收 180 维射线数据 |
| 射线几何 | 玩家判定中心发射，向上为第 0 条，顺时针每 10°一条 | 具体方向为本实现选择 |
| 长度/选择 | 到场地边界；选择射线与圆的最近入口，不是角度分桶 | 原文未充分说明，明确选定 |
| 半径 | 子弹原有判定半径，不叠加 halo 或玩家半径 | 本实现选择；物理碰撞仍用现有模拟器 |
| 空射线 | 五项全 0；相交为 detected=1 | 本实现约定 |
| 归一化 | dx/W、dy/H、vx/W、vy/H；速度使用 px/s，无裁剪 | 原文未报告，属于适配 |
| 危险对象 | 子弹＋可碰撞敌机 | 与现有地图使用相同对象集合；原文描述的是子弹 |
| 场景 | 三张三面符卡随机混合，单局 1800 帧 | 本项目共同测试平台，不是原文关卡 |
| 奖励 | 与现有无 PCC 惩罚组相同：存活 0.1，贴墙衰减，死亡 0 | 原文有碰撞惩罚且未披露数值；此处为明确适配 |
| 终止 | 首次碰撞或到 1800 帧；都不自举 | 有限时长任务设定，保留现有任务边界 |

环境字典中的 `player_features` 仅供共享墙壁奖励和日志使用，**不会送进 DQN**。
环境不含隐藏的 PCC 奖励成本，PCC 权重固定为 0。

### 未披露参数的实现选择

`config.json` 是可审阅配置：Adam 学习率固定 1e-4；gamma=0.99；ReLU；Huber loss；
梯度范数上限 10；回放池 100000；batch=128；10000 帧后开始更新；每 4 帧一次梯度
更新；每 10000 帧硬同步目标网络；epsilon 从 1 在 500000 帧内线性降至 0.05，之后
固定。单环境采样，Torch 默认 1 CPU 线程，设备 auto。以上数值不是从论文复原的。
回放池主要数组约占 139 MiB，不包括 Torch、模拟器等开销。

DQN 训练连续进行到 2000000 帧，并在恰好 1000000、2000000 帧各存一个检查点；
没有按验证结果选择最佳检查点。与 PPO 比较时统一的是环境帧预算，不是梯度更新
次数、并行环境数或学习率计划；因此比较的是整套系统，不能把差距全归因于 PCCM。

## 运行（在 LunAI-experiments 根目录）

沿用已有训练 Python 环境。缺依赖时：

```powershell
python -m pip install -r .\baselines\tian2023_dqn\requirements.txt
```

三个 seed 顺序运行，每个到 200 万帧：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\baselines\tian2023_dqn\run_seeds.ps1
```

先单跑 seed 0：

```powershell
python -u -m baselines.tian2023_dqn.train --seed 0
```

输出在 `baselines/tian2023_dqn/runs/seed_0/`：

- `run.json`：实际配置、版本、共享代码和关卡的 SHA256、训练状态。
- `training.csv`：已完成回合的存活帧数、奖励、epsilon、最近一次 loss 等。
- `tian_dqn_1000000.pt`、`tian_dqn_2000000.pt`：精确预算处的模型。

目录已存在会停止，避免覆盖实验。检查点用于评估，不包含回放池或模拟器状态，
**当前没有断点续训功能**。中断时尽量保存 `interrupted.pt`；这不是可精确续训的状态。
达到训练预算时尚未结束的回合不写成完整回合，长度记录在 `run.json`。

### 评估

默认 greedy，环境 seeds=10001…10050，与现有实验一致；同一确定性符卡下的重复
轨迹不能视为独立场景样本。读取检查点里的观察配置和时长，不使用 PPO 的评估入口。

```powershell
python -m baselines.tian2023_dqn.evaluate --model runs/seed_0/tian_dqn_2000000.pt --level level_th06_stage3_spell1.json
```

三个 seed × 三张符卡连续评估：

```powershell
foreach ($seed in 0,1,2) {
    foreach ($spell in 1,2,3) {
        python -m baselines.tian2023_dqn.evaluate --model "runs/seed_$seed/tian_dqn_2000000.pt" --level "level_th06_stage3_spell$spell.json"
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: seed=$seed spell=$spell" }
    }
}
```

CSV 在本目录 `eval/`，列名与现有评估兼容。相对 `--model`、`--log`、`--output-dir`
路径均相对于本基线目录；输出不允许跳到原项目的 checkpoints/training_logs。
已有 CSV 不会被覆盖。

看一局：

```powershell
python -m baselines.tian2023_dqn.evaluate --model runs/seed_0/tian_dqn_2000000.pt --level level_th06_stage3_spell1.json --episodes 1 --render --log eval/preview_seed0_spell1.csv
```

按 Esc 或关闭窗口退出。重复预览时换一个日志名。

## 验证

```powershell
python -m unittest discover -s baselines/tian2023_dqn/tests -v
```

测试覆盖射线几何、缺失检测、边界、共享环境物理和奖励一致、禁止调用 PCCM 构造器、
DQN 终止目标、目标网络同步、实际梯度更新和模型保存加载。
完整训练效果不由这些测试保证；论文应把本方法标为统一环境中的重新实现，报告
原文未披露参数及上述适配。

本地验证（2026-10-02，Python 3.12 / Torch 2.5.1 CPU）：7 项测试通过；额外完成
1000 帧短训练、219 次梯度更新、500/1000 帧检查点保存，以及加载后两局评估与
CSV 动作计数核对。短训练使用专门的测试配置，不能作为正式训练效果证据。
尚未运行默认 200 万帧、三个 seed 的正式实验，也未验证 CUDA 运行。
