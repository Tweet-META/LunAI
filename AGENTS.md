# LunAI 项目上下文

本文档供后续接手本仓库的 AI 编程助手阅读。开始修改代码前，应先阅读本文、`README.md`、`config.json` 和 `git diff`。不要仅凭文件名或旧对话推测当前实验状态。

## 项目目标

LunAI（读作“露奈”）是一个用于弹幕游戏避弹研究的强化学习项目。当前主线是多尺度、多帧 CNN + PPO，研究重点是用蓝、黄、红三个尺度模拟人类从全局规划到近距离反应的视觉过程。

环境基于 `NumPix/pygame-touhou` 修改。MLP PPO、DQN 和随机 baseline 已移入 `rl/legacy/`，不要将它们恢复为主线。

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

主线观察固定为 PCCM，不再保留可选 motion schema。三个尺度由独立 CNN 分支编码：

| 尺度 | 世界范围 | 输出大小 | 每帧通道 |
| --- | --- | --- | --- |
| 蓝区 | 完整 `600x700` 游戏区域 | `8x8` | density、PCCM、playable mask |
| 黄区 | 玩家中心 `320x320` | `16x16` | density、PCCM、playable mask |
| 红区 | 玩家中心 `128x128` | `64x64` | occupancy、PCCM、playable mask |

红区用于精细反应，黄区用于中距离规划，蓝区用于全局态势。红、黄窗口始终以玩家为中心，可以超出游戏区域；场外部分由 playable mask 表示，不应通过移动窗口或吸附网格来隐藏。

每帧每个分支有三个通道。若 `frame_stack=2`，每个分支有六个输入通道；三种尺度仍分别进入各自 CNN，不要误写为一张普通的 18 通道全屏图。

玩家特征通过单独的 MLP 分支输入。敌人本体具有体术碰撞，因此与敌弹一样作为 hazard 写入观察。

直接速度通道已从 PCCM 主线输入中删除。每颗子弹的 `vx/vy` 仍在 observation builder 内部用于未来代价预测。不要在没有新实验依据时恢复 `red_speed`、平均速度或方向通道。

## PCCM 不变量

当前实现位于 `observation_builder.py`，修改时必须保持以下规则：

1. 不生成完整的 `600x700` PCCM。
2. 蓝、黄、红分别在各自的世界坐标采样网格上计算同一个代价规则。
3. 子弹位置、速度和碰撞半径使用精确浮点世界坐标，不吸附到网格。
4. 每颗子弹逐颗计算，不对同一格内的速度求平均。
5. 当前配置预测未来 `5` 个游戏帧，并包含当前时刻 `t=0`。
6. 软代价使用 `1 - (1 - old) * (1 - new)` 合并，并限制在 `0.8` 以下。
7. 当前真实碰撞区域最后覆盖为 `1.0`，不能被 soft cap 降低。
8. 场外区域不写成 PCCM 的硬危险，只由 playable mask 表示不可到达。
9. 四面墙分别贡献软代价，因此角落会自然叠加。
10. 上方 70% 区域从分界线的 0 线性增加到顶部的 0.3，并与墙壁代价软叠加。
11. `reference` NumPy broadcasting 实现必须保留，并且目前仍是训练默认实现。
12. ROI optimized 和 `auto` 仅作为精确实现及性能研究保留，不得为了“优化”删除 reference。

未来轨迹在普通慢速子弹上可能不明显。例如 `145 px/s` 的子弹在五帧内只移动约 `12 px`；`600 px/s` 的诊断弹会移动约 `50 px`。这不是预测失效。

## 当前奖励结构

完整奖励定义在 `rl/reward.py`，每个真实游戏帧计算一次：

```text
survival reward       = +0.1
collision penalty     = -30.0（发生碰撞时）
action change penalty =  0.0
PCCM state penalty    = -0.05 * local PCCM cost
```

PCCM state penalty 在碰撞帧跳过，避免和碰撞惩罚重复。奖励数值只在 `rl/reward.py` 修改；`rl/touhou_rl_env.py` 只调用完整奖励函数并记录各项统计。

默认训练和评估中，第一次有效碰撞结束 episode。

## 当前训练原则

- 新观察形状变更后必须从头训练，旧 checkpoint 不能静默加载。
- 新实验不得覆盖旧 checkpoint 或 CSV。
- `config.json` 只保存当前实验有意覆盖的参数；省略项使用脚本默认值，命令行参数用于临时覆盖。
- 每个 CSV 首行保存最终生效的 `# run_config`。
- 正式训练关闭 `render` 和 `render_debug`；渲染只用于短暂验收。
- 先在诊断课程验证最短链路，再进入复杂随机弹幕。
- 高速关 `level_diagnostic_fast_aimed.json` 只用于肉眼检查 PCCM 未来轨迹，不加入正式训练池。

当前基础诊断课程包含：

- 空场；
- 左、中、右三种小型自机狙；
- 左、中、右三种大型自机狙。

基础课程的验收目标是：无威胁时停住、来弹时短移、躲开后停下、左右来弹均能处理，并且不形成固定向下、固定单侧或冲角策略。

## 常用入口

训练主线：

```powershell
python rl/train_ppo_cnn.py --config config.json
```

查看高速子弹的 PCCM 预测：

```powershell
python tools/realtime_observation_map.py --level-file level_diagnostic_fast_aimed.json
```

PCCM 一致性与微基准：

```powershell
python tools/benchmark_pccm_roi.py
```

真实关卡 observation 基准：

```powershell
python tools/benchmark_pccm_level.py
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
- 不删除失败实验和历史记录；它们是论文实验过程的一部分。

## 关键文件

- `observation_builder.py`：多尺度地图、PCCM、reference/ROI 实现。
- `observation_sources.py`：从 pygame scene 提取玩家、敌弹和敌人状态。
- `rl/cnn_observation_utils.py`：多帧、多通道 CNN 输入堆叠。
- `rl/ppo_cnn_agent.py`：三分支 CNN PPO 网络及 checkpoint 元数据。
- `rl/touhou_rl_env.py`：环境推进、奖励统计、frame history 和渲染。
- `rl/reward.py`：完整奖励数值与计算函数。
- `config.json`：当前实验配置。
- `training_logs/plots/reward_and_training_history.md`：按时间记录的实验历史。
- `tools/visualization_debug.py`：完整游戏区域上的合成 PCCM 调试图。

LunAI 是项目名，智能体称为“露奈”，使用女性代词。对实验结果应保持客观：区分理论动机、已实现功能、诊断验证和正式训练结论，不要把尚未训练验证的设计写成已经证明有效。
