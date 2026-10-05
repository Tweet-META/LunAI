# LunAI 项目上下文

本文档供后续接手本仓库的 AI 编程助手阅读。开始修改代码前，应先阅读本文、`README.md`、`config.json` 和 `git diff`。不要仅凭文件名或旧对话推测当前实验状态。

## 项目目标

LunAI（读作“露奈”）是一个用于弹幕游戏避弹研究的强化学习项目。当前主线是多尺度、多帧 CNN + PPO，研究重点是用蓝、黄、红三个尺度模拟人类从全局规划到近距离反应的视觉过程。

环境基于 `NumPix/pygame-touhou` 修改。2026-10-05 已将 `LunAI-experiments` 的当前代码、配置、关卡、测试和工具复制到本仓库，后续在本仓库继续改进。旧 MLP PPO、DQN 和历史测试保留在 `rl/legacy/`；论文使用的射线 DQN 是 `baselines/tian2023_dqn/` 下的新实现。

当前训练和评估入口使用 Pygame 模拟器，没有 `--environment th06` 选项。main 原有的 `rl/th06_adapter.py`、相关文档和验证脚本仅保留为历史适配代码，尚未与迁移后的观察结构重新整合。

## 术语

### PCCM

PCCM 的准确全称是 **Potential Collision Cost Map**，中文为“潜在碰撞代价图”。

- `P` 是 `Potential`，不是 `Predictive`。
- PCCM 包含对子弹未来位置的短期预测，但“预测”只是代价图的一个组成部分。
- 文档、代码注释和提交信息不得把 PCCM 展开为 `Predictive Collision Cost Map`。

### Step

- `decision_step`：策略网络选择一次动作。
- `frame_step`：游戏实际推进一帧。
- 实验预算和不同设置之间的比较应优先使用 `total_frame_steps`。
- `action_repeat=1` 时二者基本一致；不要因此混淆日志字段的定义。

## 当前观察结构

主线使用三个尺度的 density/occupancy、PCCM 和 playable mask，不再保留直接速度通道。支持 `trajectory`、`static` 和 `occupancy_only` 观察消融；支持 `full`、`red_blue` 和 `red_only` 尺度消融。关闭的输入置零，网络参数量保持一致。三个尺度由独立 CNN 分支编码：

| 尺度 | 世界范围 | 输出大小 | 每帧通道 |
| --- | --- | --- | --- |
| 蓝区 | 完整 `384x448` 游戏区域 | `8x8` | density、PCCM、playable mask |
| 黄区 | 玩家中心 `204x204` | `16x16` | density、PCCM、playable mask |
| 红区 | 玩家中心 `64x64` | `64x64` | occupancy、PCCM、playable mask |

红区用于精细反应，黄区用于中距离规划，蓝区用于全局态势。红、黄窗口始终以玩家为中心，可以超出游戏区域；场外部分由 playable mask 表示，不应通过移动窗口或吸附网格来隐藏。

每帧每个分支有三个通道。当前 `frame_stack=4`，因此每个分支有十二个输入通道；三种尺度仍分别进入各自 CNN，不要误写为一张普通的 36 通道全屏图。

玩家的八维特征通过单独的 MLP 分支输入。敌人本体具有体术碰撞，因此与敌弹一样作为 hazard 写入观察。几何尺寸集中在 `playfield_config.py`；旧关卡由 `level_scaling.py` 按比例载入，源 JSON 不改写。

直接速度通道已从 PCCM 主线输入中删除。每颗子弹的 `vx/vy` 仍在 observation builder 内部用于未来代价预测。不要在没有新实验依据时恢复 `red_speed`、平均速度或方向通道。

## 当前 CNN 网络

新训练默认使用 `architecture_version=2`。三个尺度仍分别编码，先保留空间结构，再拼接玩家特征进入共享全连接层：

| 分支 | v2 卷积结构 | 池化后大小 |
| --- | --- | --- |
| 红区 | `12 -> 32 -> 32 -> 64 -> 64`，中间一次 `2x2` 最大池化 | `64x8x8` |
| 黄区 | `12 -> 32 -> 64` | `64x4x4` |
| 蓝区 | `12 -> 32 -> 64` | `64x2x2` |

在当前 `frame_stack=4`、每帧三通道的输入下，v2 共 `813,674` 个可训练参数，其中三个 CNN 编码器共 `112,128` 个参数。融合后的特征维度为 `5,408`，共享层仍为 `5,408 -> 128 -> 64`，没有直接把 hidden dimension 扩大到 256。

旧版 v1 共 `323,802` 个参数，其中 CNN 编码器仅 `7,280` 个参数。旧 checkpoint 若没有 `architecture_version` 元数据，会自动按 v1 加载；加载 v1 checkpoint 不会把网络升级为 v2。需要验证 v2 时必须从头训练，并使用新的 checkpoint 和日志文件名。

## PCCM 不变量

当前实现位于 `observation_builder.py`，修改时必须保持以下规则：

1. 不生成完整屏幕的 PCCM。
2. 蓝、黄、红分别在各自的世界坐标采样网格上计算同一个代价规则。
3. 子弹位置、速度和碰撞半径使用精确浮点世界坐标，不吸附到网格。
4. 每颗子弹逐颗计算，不对同一格内的速度求平均。
5. 当前配置预测未来 `5` 个游戏帧，并包含当前时刻 `t=0`。
6. 软代价使用 `1 - (1 - old) * (1 - new)` 合并，并限制在 `0.8` 以下。
7. 当前真实碰撞区域最后覆盖为 `1.0`，不能被 soft cap 降低。
8. 场外区域不写成 PCCM 的硬危险，只由 playable mask 表示不可到达。
9. 四面墙分别贡献软代价，因此角落会自然叠加。
10. 上方 70% 区域从分界线的 0 线性增加到顶部的 0.3，并与墙壁代价软叠加。
11. 保留 NumPy `reference`，以及 ROI、Numba 和 Torch 后端。Numba 同时加速 PCCM 和 occupancy；PPO 的 `--device` 与地图计算后端独立。

当前 halo 为 `20` 像素。蓝区在 `16x16` 采样后投影为 `8x8`，黄区在 `32x32` 采样后投影为 `16x16`，红区在 `32x32` 采样后插值为 `64x64`。奖励使用隐藏的完整 `_reward_red_pccm`，不随观察消融置零。

未来轨迹在普通慢速子弹上可能不明显。例如 `145 px/s` 的子弹在五帧内只移动约 `12 px`；`600 px/s` 的高速子弹会移动约 `50 px`。这不是预测失效。

## 当前奖励结构

完整奖励定义在 `rl/reward.py`，每个真实游戏帧计算一次：

```text
collision frame = 0.0
surviving frame = max(0.0, 0.1 - 0.1 * min(1.0, wall_proximity)
                      - pccm_reward_weight * local_PCCM)
pccm_reward_weight 默认 = 0.0；论文 PCC 惩罚组 = 0.1
action change penalty = 0.0
blocked movement = 只记录 blocked ratio，不扣分
```

墙壁接近度在距边界不足对应宽高的 12% 时线性增加，横纵贡献相加后在奖励中限制为 1。所有存活帧奖励非负，碰撞帧没有额外负惩罚。`rl/reward.py` 集中定义奖励；`rl/touhou_rl_env.py` 负责测量实际运动、推进环境和统计指标。

默认训练和评估中，第一次有效碰撞结束 episode。

## 当前训练原则

- 新观察形状变更后必须从头训练，旧 checkpoint 不能静默加载。
- 新实验不得覆盖旧 checkpoint 或 CSV。
- `config.json` 只保存当前实验有意覆盖的参数；省略项使用脚本默认值，命令行参数用于临时覆盖。
- 每个 CSV 首行保存最终生效的 `# run_config`。
- 正式训练关闭 `render` 和 `render_debug`；渲染只用于短暂验收。
- 直接复制的 `config.json` 是 1M 帧、单局 2160 帧的旧配置。论文正式三面实验使用 `tools/run_formal_9_1_seeds.ps1`：累计 2M 帧、单局 1800 帧；绯红之主转训使用 `tools/finetune_scarlet_meister.py`，追加 300k 帧。不要把根配置误认为论文最终设置。

## 常用入口

训练主线：

```powershell
python rl/train_ppo_cnn.py --config config.json
```

## 修改与验证规则

- 保持修改范围紧凑，不顺手重构无关代码。
- 用户可能正在修改 `config.json`、奖励和训练历史；先读 `git diff`，不要覆盖用户改动。
- 不自动执行 `git commit`。只有用户明确要求时才提交。
- 新增函数要写简短、简单的英文 `#` 注释，说明函数用途。
- 用户偏好普通 `#` 注释，不要用冗长 docstring 代替所有注释。
- 手工代码修改后至少运行相关 `py_compile`、目标测试和 `git diff --check`。
- 改 observation 时必须验证 shape、有限数值、PCCM 硬碰撞一致性和 PPO smoke training。
- 改可视化时应实际生成或打开截图，检查尺寸、对齐和文字重叠。
- 不删除失败实验和历史记录。实验代码现已迁入本仓库；`LunAI-experiments` 原目录保留。

## 关键文件

- `observation_builder.py`：主线多尺度地图与 trajectory-aware PCCM。
- `observation_sources.py`：从 pygame scene 提取玩家、敌弹和敌人状态。
- `rl/cnn_observation_utils.py`：多帧、多通道 CNN 输入堆叠。
- `rl/ppo_cnn_agent.py`：三分支 CNN PPO 网络及 checkpoint 元数据。
- `rl/touhou_rl_env.py`：环境推进、奖励统计、frame history 和渲染。
- `rl/reward.py`：完整奖励数值与计算函数。
- `config.json`：当前主线训练配置。
- `baselines/tian2023_dqn/`：论文中按同帧预算比较的射线 DQN 基线。
- `docs/scarlet_meister_finetuning.md`：绯红之主转训与评估流程。
- `training_logs/plots/reward_and_training_history.md`：按时间记录的实验历史。
- `tools/visualization_debug.py`：完整游戏区域上的合成 PCCM 调试图。

LunAI 是项目名，智能体称为“露奈”，使用女性代词。对实验结果应保持客观：区分理论动机、已实现功能、诊断验证和正式训练结论，不要把尚未训练验证的设计写成已经证明有效。
