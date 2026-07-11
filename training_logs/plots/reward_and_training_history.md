# LunAI 奖励函数与训练版本历史记录

这份记录按时间顺序整理 LunAI 从最早 DQN baseline 到 PPO 分阶段训练的主要变化。
文件放在 `training_logs/plots`，方便和趋势图一起查阅。

说明：

```text
1. 有些早期奖励函数没有单独保留旧版源码，因此只能记录当时的设计方向和日志统计。
2. 能从 CSV 或当前代码确认的地方，全部写具体数值。
3. 这里不把任何版本写成最终模型，只记录历史过程、参数、结果和判断。
```

## 2026-07-02：DQN 初始 baseline

相关日志：

```text
dqn_curriculum.csv
```

训练结果：

```text
episodes=300
mean_frame=462.8
min_frame=120
max_frame=1038
>=900: 6/300
>=1200: 0/300
mean_reward=0.860
collisions=300/300
```

奖励设计方向：

```text
以存活奖励为主。
撞弹给惩罚。
撞弹后结束 episode。
```

观察结果：

```text
DQN 可以跑通，但训练非常不稳定。
几乎每局都会撞弹。
平均生存帧只有 462.8。
作为正式 baseline 不够稳。
```

阶段判断：

```text
保留为最早 baseline。
后续改向 PPO。
```

## 2026-07-03 00:02：DQN 距离奖励实验

相关日志：

```text
exp_001_distance_reward.csv
```

训练结果：

```text
episodes=300
mean_frame=422.1
min_frame=126
max_frame=1064
>=900: 6/300
>=1200: 0/300
mean_reward=2.626
collisions=300/300
```

奖励设计方向：

```text
在基础生存/撞弹奖励之外，尝试加入和子弹距离相关的 shaping。
目标是让 agent 学会远离危险。
```

观察结果：

```text
mean_reward 从 0.860 上升到 2.626。
mean_frame 反而从 462.8 降到 422.1。
说明 reward 分数变高不等于真的更会躲。
```

阶段判断：

```text
距离奖励方向可用，但直接用于 DQN 不稳定。
```

## 2026-07-03 00:55：DQN 慢 epsilon 实验

相关日志：

```text
exp_002_slow_epsilon.csv
```

训练结果：

```text
episodes=300
mean_frame=581.5
min_frame=219
max_frame=1192
>=900: 8/300
>=1200: 0/300
mean_reward=7.033
collisions=300/300
```

奖励/训练变化：

```text
继续使用 DQN。
放慢 epsilon 衰减，让随机探索保留更久。
```

观察结果：

```text
mean_frame 从 422.1 提升到 581.5。
mean_reward 提升到 7.033。
max_frame 达到 1192。
但 collisions 仍然是 300/300。
```

阶段判断：

```text
慢 epsilon 有帮助，但 DQN 仍然没有稳定学会避弹。
```

## 2026-07-03 08:54：DQN 慢 epsilon 继续实验

相关日志：

```text
exp_003_slow_epsilon.csv
```

训练结果：

```text
episodes=282
mean_frame=579.2
min_frame=219
max_frame=1000
>=900: 8/282
>=1200: 0/282
mean_reward=-81.259
collisions=277/282
```

观察结果：

```text
mean_frame 与 exp_002 接近，约 579。
mean_reward 变成 -81.259，说明奖励尺度或惩罚项严重影响训练读数。
```

阶段判断：

```text
训练读数很难解释。
DQN 路线继续降优先级。
```

## 2026-07-03 20:52：DQN 固定撞弹惩罚实验

相关日志：

```text
exp_004_fixed_collision_penalty.csv
```

训练结果：

```text
episodes=300
mean_frame=580.6
min_frame=198
max_frame=1192
>=900: 11/300
>=1200: 0/300
mean_reward=-2.986
collisions=300/300
```

奖励设计方向：

```text
把 collision penalty 从 -100 改成 -10。
目的不是让撞弹变得不重要，而是避免 -100 过大，直接淹没前面每一步的存活奖励和距离奖励。
这次调整希望让 episode 的总回报不再完全由最后一次撞弹决定。
```

观察结果：

```text
mean_frame 约 580，和慢 epsilon 实验接近。
>=900 次数从 8 增加到 11。
mean_reward 从 exp_003 的 -81.259 回到 -2.986，说明 reward 尺度明显正常了。
但 300 局仍然全部撞弹。
```

阶段判断：

```text
把撞弹惩罚从 -100 降到 -10 是必要的尺度修正。
但 DQN 仍然没有稳定学会避弹。
DQN baseline 可以作为对照，但不适合作为主线。
```

## 2026-07-03 21:38：PPO 初始实验

相关日志：

```text
ppo_exp_001.csv
```

训练结果：

```text
episodes=300
mean_frame=679.7
min_frame=235
max_frame=1437
>=900: 60/300
>=1200: 5/300
mean_reward=-0.112
collisions=300/300
```

算法变化：

```text
从 DQN 切换到 PPO。
训练使用 on-policy 采样。
```

观察结果：

```text
mean_frame 从 DQN 最好约 581 提升到 679.7。
>=900 从 11/300 提升到 60/300。
max_frame 达到 1437。
说明 PPO 明显更适合当前环境。
```

阶段判断：

```text
PPO 成为后续 baseline 主线。
```

## 2026-07-03 22:35：PPO 近弹惩罚实验

相关日志：

```text
ppo_exp_002_near_penalty.csv
```

训练结果：

```text
episodes=300
mean_frame=733.4
min_frame=288
max_frame=1511
>=900: 88/300
>=1200: 9/300
mean_reward=-3.216
collisions=300/300
```

奖励设计变化：

```text
加入 near bullet penalty。
目标是让 agent 不要贴着子弹走。
```

观察结果：

```text
mean_frame 从 679.7 提升到 733.4。
>=900 从 60/300 提升到 88/300。
mean_reward 变负到 -3.216，但生存表现更好。
```

阶段判断：

```text
近弹惩罚是有效方向。
reward 数值不能单独作为好坏标准。
```

## 2026-07-03 23:14：PPO 边界惩罚实验

相关日志：

```text
ppo_exp_003_boundary_penalty_1200.csv
```

训练结果：

```text
episodes=300
mean_frame=736.4
min_frame=261
max_frame=1200
>=900: 83/300
>=1200: 6/300
mean_reward=-5.108
collisions=294/300
```

奖励设计变化：

```text
加入 boundary penalty。
训练上限使用 1200 frame。
目标是减少贴墙/贴角挂机。
```

观察结果：

```text
mean_frame 约 736.4，和近弹惩罚版本接近。
collisions 从 300/300 降到 294/300。
但 reward 进一步降低到 -5.108。
```

阶段判断：

```text
边界惩罚有必要，但数值需要很小心。
边界惩罚过强可能导致左右脑互搏。
```

## 2026-07-04 08:44：边界特征版 PPO，也就是 LunAI v0

相关日志：

```text
ppo_exp_004_boundary_features_1200.csv
lunai_v0.csv
```

训练结果：

```text
episodes=300
mean_frame=784.6
min_frame=291
max_frame=1200
>=900: 101/300
>=1200: 61/300
mean_reward=-0.477
collisions=239/300
```

观察/奖励设计变化：

```text
在 player_features 中加入边界距离特征。
继续使用近弹惩罚、边界惩罚、存活奖励、撞弹惩罚。
```

观察结果：

```text
mean_frame 从 736.4 提升到 784.6。
>=1200 从 6/300 提升到 61/300。
collisions 从 294/300 降到 239/300。
这是第一次比较明显看到 PPO baseline 学到东西。
```

阶段判断：

```text
v0 是第一个有效 baseline。
但它主要学会处理早期自机狙，对后续复杂弹幕仍然不足。
```

## 2026-07-04 13:36：LunAI v1，训练上限改为 2400

相关日志：

```text
lunai_v1_2400.csv
```

训练结果：

```text
episodes=500
mean_frame=999.8
min_frame=249
max_frame=1949
>=900: 276/500
>=1200: 203/500
mean_reward=-4.023
collisions=500/500
```

训练变化：

```text
max_steps 从 1200 提高到 2400。
目标是让模型接触更靠后的弹幕。
```

观察结果：

```text
mean_frame 提升到 999.8。
max_frame 达到 1949。
但没有 episode 达到 2400。
全部 500 局最终都撞弹。
后半段随机弹/复杂弹幕仍然处理不好。
```

阶段判断：

```text
单纯拉长 episode 不够。
需要 curriculum。
```

## 2026-07-04 15:04：LunAI v2 短训

相关日志：

```text
lunai_v2.csv
```

训练结果：

```text
episodes=83
mean_frame=1113.7
min_frame=356
max_frame=1835
>=900: 53/83
>=1200: 49/83
mean_reward=-9.260
collisions=83/83
```

奖励/课程方向：

```text
继续调整生存奖励、撞弹惩罚和边界相关惩罚。
尝试让 agent 面对更长关卡。
```

观察结果：

```text
短期 mean_frame 较高，达到 1113.7。
但 mean_reward=-9.260。
全部 episode 最终仍撞弹。
```

阶段判断：

```text
短训表现不能说明稳定学会。
需要长训验证。
```

## 2026-07-04 16:33：LunAI v2_long，长训失败样本

相关日志：

```text
lunai_v2_long.csv
```

训练结果：

```text
episodes=894
mean_frame=937.0
min_frame=282
max_frame=1970
>=900: 540/894
>=1200: 124/894
mean_reward=-18.603
collisions=894/894
```

训练命令关键参数：

```text
--load-path checkpoints/lunai_v2.pt
--model-path checkpoints/lunai_v2_long.pt
--log-path training_logs/lunai_v2_long.csv
```

观察结果：

```text
长训后 mean_frame 从 v2 短训的 1113.7 降到 937.0。
mean_reward 降到 -18.603。
greedy 容易贴角/贴边，学到局部坏策略。
```

阶段判断：

```text
v2_long 是重要失败样本。
说明 reward shaping 和长关卡混训会导致局部最优。
需要把弹幕模式拆开训练。
```

## 2026-07-05：关卡课程拆分

关卡文件：

```text
level_0.json = 旧版完整关卡备份，暂时弃用
level_1.json = 简单单发自机狙
level_2.json = 轻量 3-way 自机狙
level_3.json = 自机狙 + 随机弹混合
level_4.json = 中等随机弹
level_5.json = 高密度随机弹
```

训练规则：

```text
训练时一次只训练一个 level。
实际游玩时 level_1 -> level_2 -> level_3 -> level_4 -> level_5 连续切换。
observation 中不加入 stage_id。
```

设计理由：

```text
如果加入 stage_id，实际完整游玩时可能依赖隐藏标签。
分关训练只改变训练数据分布，不改变模型输入结构。
```

阶段判断：

```text
这是 v3 路线的起点。
目标是先分别学会不同弹幕模式，再进入混合/完整关卡。
```

## 2026-07-05 11:30：LunAI v3 stage1 初训

相关日志：

```text
lunai_v3_stage1.csv
```

训练结果：

```text
episodes=200
mean_frame=642.8
min_frame=326
max_frame=960
>=900: 20/200
mean_reward=5.732
collisions=185/200
```

关卡设置：

```text
level_file=level_1.json
max_steps=960
action_repeat=3
弹幕类型=简单单发自机狙
```

观察结果：

```text
能学到一些引狙，但满 960 次数较少。
```

阶段判断：

```text
需要更积极的 stage1 训练设置。
```

## 2026-07-05 11:49：LunAI v3 stage1_fast

相关日志：

```text
lunai_v3_stage1_fast.csv
```

训练结果：

```text
episodes=200
mean_frame=610.3
min_frame=203
max_frame=960
>=900: 36/200
mean_reward=6.606
collisions=168/200
```

观察结果：

```text
mean_frame 比 stage1 初训低。
但 >=900 从 20/200 提升到 36/200。
collisions 从 185/200 降到 168/200。
greedy render 中能看到靠边引狙和微移。
```

阶段判断：

```text
策略方向是对的，但离子弹较近。
stochastic 抖一下容易撞死。
```

## 2026-07-05 12:20：LunAI v3 stage1_refine

相关日志：

```text
lunai_v3_stage1_refine.csv
```

训练结果：

```text
episodes=200
mean_frame=580.4
min_frame=195
max_frame=960
>=900: 47/200
mean_reward=7.828
collisions=160/200
```

观察结果：

```text
>=900 从 36/200 提升到 47/200。
collisions 从 168/200 降到 160/200。
但 mean_frame 降到 580.4，说明早死尾巴仍然明显。
```

阶段判断：

```text
不能只靠继续 refine 解决。
需要调整近弹惩罚形状。
```

## 2026-07-05 12:46：LunAI v3 stage1_linear，近弹惩罚改为线性

相关日志：

```text
lunai_v3_stage1_linear.csv
```

奖励函数具体数值：

```text
survival_reward = 0.045
collision_penalty = 15.0
SAFE_STAY_REWARD = 0.003
NEAR_DANGER_PENALTY_SCALE = 0.03
EDGE_MARGIN = 0.14
SIDE_EDGE_PENALTY_SCALE = 0.025
VERTICAL_EDGE_PENALTY_SCALE = 0.015
CORNER_PENALTY_SCALE = 0.04
action_change_penalty = 0.01
reversal_penalty = 0.03
```

奖励公式：

```text
reward =
    survival_reward
    + safe_stay_reward
    - near_danger_penalty
    - boundary_penalty
    - collision_penalty
    - action_change_penalty
    - reversal_penalty
```

近弹惩罚：

```text
near_danger_penalty = 0.03 * danger
danger = 0.8 * occupancy_danger + 0.2 * speed_danger
```

线性权重：

```text
weight = 1 - distance_to_player / max_red_map_distance
```

这次变化：

```text
旧版本：高斯权重，参数 NEAR_DANGER_SIGMA=3.0，NEAR_DANGER_PENALTY_SCALE=0.02
新版本：线性权重，移除 sigma，NEAR_DANGER_PENALTY_SCALE=0.03
```

训练结果：

```text
episodes=200
mean_frame=652.5
min_frame=219
max_frame=960
>=900: 61/200
mean_reward=8.061
collisions=148/200
```

和 stage1_fast 对比：

```text
stage1_fast:
mean_frame=610.3
>=900: 36/200
collisions=168/200

stage1_linear:
mean_frame=652.5
>=900: 61/200
collisions=148/200
```

观察结果：

```text
线性近弹惩罚明显改善 stage1。
greedy 能执行靠边引狙策略。
stochastic 仍然会早死，但 checkpoint 可以转入 stage2。
```

阶段判断：

```text
接受 checkpoints/lunai_v3_stage1_linear.pt 作为 stage2 初始 checkpoint。
```

## 2026-07-05 13:18：LunAI v3 stage2 首次训练

相关日志：

```text
lunai_v3_stage2.csv
```

训练结果：

```text
episodes=250
mean_frame=611.4
min_frame=210
max_frame=900
>=900: 75/250
mean_reward=5.618
collisions=175/250
```

训练命令关键参数：

```text
level_file=level_2.json
load_path=checkpoints/lunai_v3_stage1_linear.pt
model_path=checkpoints/lunai_v3_stage2.pt
max_steps=900
action_repeat=3
entropy_coef=0.004
entropy_coef_final=0.0015
learning_rate=0.00012
learning_rate_final=0.00004
rollout_steps=1024
```

观察结果：

```text
CSV 中有 75/250 次满 900。
但是 greedy render 开局死亡。
说明 stochastic 训练表现不能代表 greedy 策略可用。
```

阶段判断：

```text
拒绝 checkpoints/lunai_v3_stage2.pt。
不能用它继续 stage3。
```

## 2026-07-05 14:11：LunAI v3 stage2_det

相关日志：

```text
lunai_v3_stage2_det.csv
```

训练命令关键参数：

```text
level_file=level_2.json
load_path=checkpoints/lunai_v3_stage1_linear.pt
model_path=checkpoints/lunai_v3_stage2_det.pt
max_steps=900
action_repeat=3
episodes=250
entropy_coef=0.0015
entropy_coef_final=0.0003
learning_rate=0.00008
learning_rate_final=0.000025
rollout_steps=2048
target_kl=0.01
```

训练结果：

```text
episodes=250
mean_frame=620.7
median_frame=578
min_frame=210
max_frame=900
>=900: 79/250
>=800: 102/250
>=700: 114/250
<480: 112/250
<360: 45/250
mean_reward=6.825
collisions=171/250
```

和 stage2 首次训练对比：

```text
stage2:
mean_frame=611.4
>=900: 75/250
mean_reward=5.618
collisions=175/250

stage2_det:
mean_frame=620.7
>=900: 79/250
mean_reward=6.825
collisions=171/250
```

观察结果：

```text
数值提升不大。
CSV 仍然有大量早死。
但 greedy render 观察为可接受。
因此它不是展示用最终模型，但可以作为 stage3 的迁移起点。
```

阶段判断：

```text
接受 checkpoints/lunai_v3_stage2_det.pt 作为 stage3 初始 checkpoint。
```

## 2026-07-05 15:01：LunAI v3 stage3

相关日志：

```text
lunai_v3_stage3.csv
```

训练命令关键参数：

```text
level_file=level_3.json
load_path=checkpoints/lunai_v3_stage2_det.pt
model_path=checkpoints/lunai_v3_stage3.pt
max_steps=1200
实际满帧=900，因为 level_3.json 的 length=15.0 秒
action_repeat=3
episodes=300
entropy_coef=0.002
entropy_coef_final=0.0005
learning_rate=0.00008
learning_rate_final=0.000025
rollout_steps=2048
target_kl=0.01
```

训练结果：

```text
episodes=300
mean_frame=628.2
median_frame=661
min_frame=219
max_frame=900
finish900=105/300
>=800: 128/300
>=700: 145/300
<480: 121/300
<360: 78/300
mean_reward=8.21
collisions=195/300
```

分段结果：

```text
001-050: mean=614.3, finish900=17/50
051-100: mean=591.6, finish900=17/50
101-150: mean=658.9, finish900=14/50
151-200: mean=640.0, finish900=21/50
201-250: mean=597.6, finish900=17/50
251-300: mean=666.6, finish900=19/50
```

和 stage2_det 对比：

```text
stage2_det:
mean_frame=620.7
>=900: 79/250
mean_reward=6.825
collisions=171/250

stage3:
mean_frame=628.2
finish900=105/300
mean_reward=8.21
collisions=195/300
```

观察结果：

```text
CSV 比 stage2_det 更好，最后 50 集 mean_frame=666.6，没有尾段崩坏。
stage3 加入了自机狙和随机弹混合，LunAI 仍然能保持一定满帧率。
greedy render 中仍然偏爱贴墙/贴角，但已经能做微移，整体策略方向可接受。
贴角倾向可能来自前几关自机狙训练，不在 stage3 立即修 reward。
```

阶段判断：

```text
接受 checkpoints/lunai_v3_stage3.pt 作为 stage4 初始 checkpoint。
stage3 通过，但带有明显 aimed-bullet bias。
后续 stage4 和 stage5 需要重点观察随机弹下是否愿意离开墙角。
```

## 2026-07-05 15:45：LunAI v3 stage4，CSV 通过但 greedy 失败

相关日志：

```text
lunai_v3_stage4.csv
```

训练命令关键参数：

```text
level_file=level_4.json
load_path=checkpoints/lunai_v3_stage3.pt
model_path=checkpoints/lunai_v3_stage4.pt
max_steps=840
实际满帧=840，因为 level_4.json 的 length=14.0 秒
action_repeat=3
episodes=300
entropy_coef=0.002
entropy_coef_final=0.0005
learning_rate=0.00008
learning_rate_final=0.000025
rollout_steps=2048
target_kl=0.01
```

训练结果：

```text
episodes=300
mean_frame=699.2
median_frame=807
min_frame=221
max_frame=840
finish840=137/300
>=800: 156/300
>=700: 174/300
<480: 57/300
<360: 7/300
mean_reward=3.19
collisions=163/300
```

分段结果：

```text
001-050: mean=681.6, finish840=21/50
051-100: mean=674.6, finish840=19/50
101-150: mean=653.8, finish840=16/50
151-200: mean=738.9, finish840=29/50
201-250: mean=709.3, finish840=25/50
251-300: mean=737.0, finish840=27/50
```

和 stage3 对比：

```text
stage3:
mean_frame=628.2
finish900=105/300
<480: 121/300
<360: 78/300
mean_reward=8.21
collisions=195/300

stage4:
mean_frame=699.2
finish840=137/300
<480: 57/300
<360: 7/300
mean_reward=3.19
collisions=163/300
```

观察结果：

```text
CSV 指标看起来明显变好，早死次数大幅减少。
但 greedy render 发现 LunAI 一直贴右下角，并且随机弹过来时也不主动离开。
用 checkpoints/lunai_v3_stage4.pt 回测 level_1 时，原本学会的纯自机狙微移也退化成缩角落。
这说明 stage4.pt 不是随机弹能力提升，而是学到了角落局部最优，并且覆盖了 stage3 的好策略。
```

阶段判断：

```text
拒绝 checkpoints/lunai_v3_stage4.pt。
保留它作为失败样本，不继续用于 stage5。
失败原因记录为：灾难性遗忘 + 角落局部最优。
下一步从 checkpoints/lunai_v3_stage3.pt 回退，调强边界惩罚后重新训练 stage4。
```

## 2026-07-05：边界惩罚 wallfix 调整

调整前边界惩罚：

```text
EDGE_MARGIN = 0.14
SIDE_EDGE_PENALTY_SCALE = 0.025
VERTICAL_EDGE_PENALTY_SCALE = 0.015
CORNER_PENALTY_SCALE = 0.04
贴满角最大边界惩罚 = 0.025 + 0.015 + 0.04 = 0.08
```

调整后边界惩罚：

```text
EDGE_MARGIN = 0.14
SIDE_EDGE_PENALTY_SCALE = 0.025
VERTICAL_EDGE_PENALTY_SCALE = 0.03
CORNER_PENALTY_SCALE = 0.08
贴满角最大边界惩罚 = 0.025 + 0.03 + 0.08 = 0.135
```

调整理由：

```text
只惩罚 stay 会产生漏洞：模型可以在右下角一直按 down_right，位置不变但不算 stay。
因此这次不新增 stay 惩罚，而是直接调强现有线性边界惩罚。
侧边惩罚 SIDE_EDGE_PENALTY_SCALE 保持 0.025，尽量保留纯自机狙中的靠边引狙能力。
主要增加 VERTICAL_EDGE_PENALTY_SCALE 和 CORNER_PENALTY_SCALE，用来打掉右下角冻结策略。
```

后续观察：

```text
0.135 版本过重。
训练早期出现大量 840 满帧但 reward=-15 到 -23 的情况，说明模型即使活到关底也收到强负反馈。
这会让 reward 信号变得不清晰：活着很亏，撞死也亏，模型难以判断什么策略是好的。
因此停止使用 0.135 重墙版本，不继续训练 checkpoints/lunai_v3_stage4_wallfix.pt。
```

## 2026-07-05：边界惩罚 softwall 回退

调整后边界惩罚：

```text
EDGE_MARGIN = 0.14
SIDE_EDGE_PENALTY_SCALE = 0.025
VERTICAL_EDGE_PENALTY_SCALE = 0.02
CORNER_PENALTY_SCALE = 0.055
贴满角最大边界惩罚 = 0.025 + 0.02 + 0.055 = 0.10
```

调整理由：

```text
原始版本 0.08 太容易让模型学到右下角局部最优。
重墙版本 0.135 又会让满帧 episode 大量变成强负分。
softwall 版本取中间值 0.10，希望让角落明显亏，但不至于让活到关底也完全没有正反馈。
仍然从 checkpoints/lunai_v3_stage3.pt 回退重训 stage4，不使用已经退化的 stage4 checkpoint。
```

后续观察：

```text
softwall 版本仍然会直接去右下角。
回测多个 v3 checkpoint 时，右下角倾向都存在，只是严重程度不同。
因此问题不只是单个 stage4 checkpoint，而是固定出生点 + 固定关卡训练下形成的稳定局部最优。
继续调大边界惩罚会导致满帧也大负分，继续调小又不能打掉角落策略。
```

阶段判断：

```text
停止 v3 stage4 修补路线。
保留 v3_stage1_linear / v3_stage2_det / v3_stage3 作为有效中间结果和论文分析材料。
v3_stage4 / v3_stage4_wallfix / v3_stage4_softwall 作为失败样本。
进入 v4，从头训练，但训练条件必须改变，否则新模型仍可能学到右下角局部最优。
```

## 2026-07-05：LunAI v4 重启计划

代码变化：

```text
在 TouhouRLEnv 中加入 random_player_start 训练选项。
默认关闭，不影响正常游戏和普通 evaluate。
开启后，reset 时把玩家随机放在下半场区域，打破固定出生点到右下角的固定路线。
```

随机出生范围：

```text
x 范围：GAME_ZONE 左右各留 player_start_margin，默认 80 像素
y 范围：GAME_ZONE 高度的 55% 到底部 player_start_margin 之前
```

设计理由：

```text
右下角局部最优的核心原因之一是固定出生点和固定弹幕。
如果每局出生位置略有变化，模型不能只背一条逃向右下角的路线。
它必须根据当前 observation 选择局部避弹和找空隙策略。
```

v4 初始训练建议：

```powershell
python rl/train_ppo.py --episodes 300 --max-steps 960 --action-repeat 3 --level-file level_1.json --random-player-start --player-start-margin 80 --model-path checkpoints/lunai_v4_stage1.pt --log-path training_logs/lunai_v4_stage1.csv --entropy-coef 0.006 --entropy-coef-final 0.002 --learning-rate 0.00012 --learning-rate-final 0.00004 --rollout-steps 2048 --target-kl 0.01
```

v4 验收重点：

```text
1. level_1 是否重新学会微移引狙。
2. greedy 是否不再开局固定跑向右下角。
3. render 时如果靠边，必须能解释为引狙，而不是冻结挂机。
4. 进入 stage4 前，必须回测 level_1 和 level_2，确认没有角落退化。
```

## 2026-07-05 17:56：LunAI v4 random-first 初训

相关日志：

```text
lunai_v4_stage_random.csv
```

训练命令关键参数：

```text
level_file=level_4.json
random_player_start=True
player_start_margin=80
model_path=checkpoints/lunai_v4_stage_random.pt
max_steps=840
episodes=300
entropy_coef=0.006
entropy_coef_final=0.002
learning_rate=0.00012
learning_rate_final=0.00004
rollout_steps=2048
target_kl=0.01
```

训练结果：

```text
episodes=300
mean_frame=539.5
median_frame=492
finish840=57/300
finish840_reward_mean=20.33
>=700: 84/300
<480: 142/300
<360: 69/300
mean_reward=3.46
collisions=243/300
entropy≈2.18
```

观察结果：

```text
随机出生打破了固定右下角路线，模型不再稳定跑向右下角。
但是 greedy 仍然喜欢先靠墙，敌人还没出来时也会往边上走。
这说明模型没有学到“无威胁时站定”的期望行为。
日志没有显示 reward 崩坏，满帧 episode 的平均 reward=20.33，说明不是 wallfix 那类强负反馈问题。
问题更像是 safe stay 奖励太弱，没能压住先靠边的习惯。
```

阶段判断：

```text
v4 random-first 初训没有失败，但行为不够理想。
保留 checkpoints/lunai_v4_stage_random.pt 作为 random-start 对照。
下一步只调整 SAFE_STAY_REWARD，不改其他 reward 结构。
```

## 2026-07-05：safe stay 奖励调整

调整前：

```text
SAFE_STAY_REWARD = 0.003
```

调整后：

```text
SAFE_STAY_REWARD = 0.006
```

调整理由：

```text
当前 reward 理论上希望“红区无弹时 stay 更优”。
但 0.003 太弱，模型仍会在无威胁时提前靠边。
提高到 0.006 是小幅强化，不希望训练出危险时也不动的挂机策略。
```

## 通用验收标准

```text
1. 不只看 CSV。
2. 必须 render greedy。
3. greedy 开局死则拒绝 checkpoint。
4. greedy 行为必须能解释为合理避弹策略。
5. stochastic 早死可以接受，但不能掩盖 greedy 坏策略。
```

## 到目前为止的经验总结

```text
1. PPO 明显比 DQN 更适合当前环境。
2. reward 分数不能单独代表策略质量。
3. 高斯近弹惩罚不如线性近弹惩罚直观稳定。
4. 边界惩罚必须有，但不能太大，否则会造成策略冲突。
5. 长关卡直接混训容易学到贴角/贴边局部最优。
6. 分阶段 curriculum 是必要的。
7. CSV 高分不能保证 greedy 可用，必须做确定性 render 验收。
8. baseline 目标不是立刻通关，而是证明观察表示 + PPO 能学到可解释避弹行为。
```

## 可用于论文的对比点

```text
DQN baseline: dqn_curriculum.csv
DQN reward shaping: exp_001/002/003/004
PPO baseline: ppo_exp_001.csv
PPO near penalty: ppo_exp_002_near_penalty.csv
PPO boundary penalty: ppo_exp_003_boundary_penalty_1200.csv
PPO boundary features / v0: lunai_v0.csv
Long episode failure: lunai_v2_long.csv
Staged curriculum stage1: lunai_v3_stage1_linear.csv
Staged curriculum stage2: lunai_v3_stage2_det.csv
Staged curriculum stage3: lunai_v3_stage3.csv
Staged curriculum stage4 failure: lunai_v3_stage4.csv
CNN self-aim pilot: lunai_cnn_selfaim_pilot.csv
CNN self-aim with aim bug (invalid result): lunai_v5_test2.csv
CNN self-aim reward comparison: lunai_v5_test3.csv, lunai_v5_test4.csv
CNN random bullets with safe stay: lunai_v5_test5.csv
CNN random bullets without safe stay: lunai_v5_test6.csv
CNN two-frame random bullets: lunai_v6_test7-stack2.csv
```

## 2026-07-10：CNN PPO 单帧自机狙试运行

### 实验配置

```text
模型：CNN PPO，多尺度网格的单帧观察
训练关卡：level_1.json（自机狙弹幕）
随机自机初始位置：开启
单局最大帧数：960
action_repeat：3
总训练预算：200,000 环境帧
随机种子：0
rollout_steps：1024
minibatch_size：256
update_epochs：4
学习率：0.00012 -> 0.00004
熵系数：0.004 -> 0.0015
模型：checkpoints/lunai_cnn_selfaim_pilot.pt
日志：training_logs/lunai_cnn_selfaim_pilot.csv
```

### 本轮使用的奖励函数参数

```text
survival_reward = 0.045
collision_penalty = 15.0
SAFE_STAY_REWARD = 0.006
NEAR_DANGER_PENALTY_SCALE = 0.03
SIDE_EDGE_PENALTY_SCALE = 0.025
VERTICAL_EDGE_PENALTY_SCALE = 0.02
CORNER_PENALTY_SCALE = 0.055
action_change_penalty = 0.01
reversal_penalty = 0.03
```

### 训练结果

```text
最后一个完整写入日志的 episode：310
最后一个完整写入日志的累计环境帧：199,296

前 50 局：
平均存活帧数 = 505.8
存活帧数中位数 = 493
平均 reward = 1.411
存活至 960 帧 = 1 / 50

后 50 局：
平均存活帧数 = 765.3
存活帧数中位数 = 739
平均 reward = 8.863
存活至 960 帧 = 17 / 50

最后 20 局：
平均存活帧数 = 819.2
存活帧数中位数 = 960
平均 reward = 18.278
存活至 960 帧 = 11 / 20

最后一次更新指标：
entropy = 1.449614
approx_kl = 0.000086
value_loss = 7.347075
```

### 行为观察与阶段结论

```text
在采样执行时，LunAI 已能通过小幅方向调整引导自机狙弹幕，操作方式在视觉上比旧 MLP baseline 更接近人类的微移避弹。因此保留 CNN 空间观察架构，后续不将 CNN 视为本轮问题来源。

在一个未参与训练的随机种子上进行 greedy 验收时，模型表现出较强的停留偏好：
初始动作概率：stay=0.628，down_left=0.068，right=0.065，left=0.056，down_right=0.052，up_left=0.041，up_right=0.040，down=0.040，up=0.011
greedy 动作统计：stay=116
结果：存活 348 帧，collisions=1

解释：模型推理和动作传递均正常。随机策略已经可以产生有意义的局部移动，但确定性策略仍过度偏好 stay。下一轮只调整奖励函数对“停留”的偏好，CNN 架构保持不变；在此之后，再重新进行 1M 环境帧的正式自机狙实验。
```

## 2026-07-10：LunAI v5 自机狙连续试验记录

为避免临时文件名造成混淆，v5 的 test2、test3、test4 按实际代码状态记录如下。`lunai_cnn_selfaim_pilot.csv` 可视作本系列的 test1。

### test2：修复前的高分样本，不能作为正式结果

```text
日志：training_logs/lunai_v5_test2.csv
模型：checkpoints/lunai_v5_test2.pt
最后完整日志：episode 302，total_frame_steps 199,155

整体：
平均存活 = 659.5 帧
中位存活 = 614 帧
平均 reward = 12.342
存活至 960 帧 = 76 / 302

最后 50 局：
平均存活 = 912.7 帧
中位存活 = 960 帧
平均 reward = 34.657
存活至 960 帧 = 41 / 50
entropy = 1.529138
value_loss = 2.467606
```

```text
训练时的自机狙角度计算仍使用旧的 arccos 逻辑。该逻辑在玩家位于敌人上方时会错误地让子弹继续向下飞，左上角形成假安全区。因此 test2 的高存活数据受到环境漏洞污染，只保留作“发现自机狙方向 bug 前的样本”，不进入正式横向比较。
```

### test3：修复自机狙方向后的奖励函数试验

```text
日志：training_logs/lunai_v5_test3.csv
模型：checkpoints/lunai_v5_test3.pt
最后完整日志：episode 332，total_frame_steps 199,776

环境：自机狙角度已改为 atan2，玩家在敌人的左上、右上时也会被正确瞄准。
奖励：survival_reward=0.1，collision_penalty=50.0，action_change_penalty=0.005，reversal_penalty=0.02；safe stay 当时仍只检查红区。

整体：
平均存活 = 601.7 帧
中位存活 = 573 帧
平均 reward = 4.696
存活至 960 帧 = 29 / 332

最后 50 局：
平均存活 = 689.9 帧
中位存活 = 636 帧
平均 reward = 14.593
存活至 960 帧 = 10 / 50
entropy = 1.965948
value_loss = 50.337753
```

```text
该版本没有利用左上角漏洞，但 greedy 仍容易选择 stay。一次回测的初始概率为 stay=0.218、up_left=0.166、down_right=0.162、left=0.160；因为 greedy 总取最大值，实际动作统计仍以 stay 为主并可能原地撞死。

结论：提高生存奖励和碰撞惩罚不足以消除 stay 的局部偏好。test3 保留为“瞄准修复后的 reward 数值调整样本”，不作为最终模型。
```

### test4：红区与黄区同时为空才奖励停留，已完成

```text
日志：training_logs/lunai_v5_test4.csv
模型：checkpoints/lunai_v5_test4.pt
最后完整日志：episode 322，total_frame_steps 199,707

本轮只在 safe_stay_reward 的安全条件上修改：
1. action 必须为 stay。
2. red_occupancy 必须为空。
3. yellow_density 也必须为空。

只要黄区或红区任一出现子弹，就立即取消 SAFE_STAY_REWARD。
```

```text
训练曲线：
001-050：平均存活 517.9 帧，平均 reward -4.06，满帧 0 / 50。
151-200：平均存活 642.2 帧，平均 reward 12.35，满帧 7 / 50。
201-250：平均存活 721.9 帧，平均 reward 28.15，满帧 19 / 50。
251-300：平均存活 631.3 帧，平均 reward -14.58，满帧 0 / 50。
301-322：平均存活 622.0 帧，平均 reward -19.67，满帧 0 / 22。

训练后段的 stochastic rollout 发生退化，但 value_loss 从早期约 139 降至 30.61，KL 保持很小；这不是数值爆炸，而是策略分布仍然较宽且后段采样质量下降。

Greedy 30 局验收：mean_frame_steps=898.17，mean_reward=62.480，mean_collisions=0.30。行为以 stay 为主，只在需要时做少量 down_left 微移；不同随机出生点的最终横坐标并不完全固定。

Stochastic 30 局验收：mean_frame_steps=621.93，mean_reward=-19.860，mean_collisions=1.00。采样动作包含大量不同方向，并更容易漂到左下边界后碰撞。

阶段结论：test4 证明“红黄都空才奖励 stay”可以产生当前最好的 deterministic 自机狙行为。实际部署和主要论文指标应以 greedy 为准；stochastic 表现较差，记录为策略方差仍高，而非否定该 checkpoint。
```

### test5：保留安全停留奖励的随机弹训练

```text
日志：training_logs/lunai_v5_test5.csv
模型：checkpoints/lunai_v5_test5.pt
关卡：level_5.json，高密度随机弹
训练预算：200,000 环境帧
最后完整日志：episode 402，total_frame_steps 199,933

整体：平均存活 = 497.3 帧，中位数 = 443 帧，平均 reward = -4.202，
      存活至关卡上限 840 帧 = 57 / 402。
最后 50 局：平均存活 = 568.4 帧，平均 reward = 4.681，满帧 = 12 / 50。
最后 20 局：平均存活 = 580.8 帧，平均 reward = 8.250，满帧 = 6 / 20。
最后一次更新：entropy = 1.775931，value_loss = 50.701445。
```

```text
本轮仍保留 SAFE_STAY_REWARD：只有黄区和红区均没有子弹时，stay 才获得很小的额外奖励。
训练曲线虽然起伏明显，但后 50 局和后 20 局的存活帧数、满帧次数、平均 reward 都高于整体水平。
实际 render 观察中，LunAI 已经可以在随机弹下根据局部空隙做一些移动；仍会偶发撞弹，策略尚未完全收敛。
该实验作为“带安全停留奖励”的随机弹 CNN PPO 对照保留。
```

### test6：移除安全停留奖励的随机弹消融

```text
日志：training_logs/lunai_v5_test6.csv
模型：checkpoints/lunai_v5_test6.pt
关卡：level_5.json，高密度随机弹
训练预算：200,000 环境帧
最后完整日志：episode 398，total_frame_steps 199,878

整体：平均存活 = 502.2 帧，中位数 = 456 帧，平均 reward = -3.779，
      存活至关卡上限 840 帧 = 45 / 398。
最后 50 局：平均存活 = 509.7 帧，平均 reward = -1.148，满帧 = 7 / 50。
最后 20 局：平均存活 = 514.0 帧，平均 reward = -3.658，满帧 = 2 / 20。
最后一次更新：entropy = 1.942183，value_loss = 57.762933。
```

```text
本轮仅移除了 SAFE_STAY_REWARD，其余训练预算和 level_5 任务保持可比。
虽然整体平均存活与 test5 接近，但训练后段明显较弱：后 50 局比 test5 少 58.7 帧，
满帧次数为 7 / 50（test5 为 12 / 50），后 20 局也只有 2 / 20 满帧。
render 中模型出现过度 stay、面对弹幕不移动的问题。说明单纯删除停留奖励并不能自然得到更积极的避弹策略；
在当前随机弹环境与参数下，test5 是更值得保留的 checkpoint 和训练设置。
```

## 2026-07-11：LunAI v6 两帧输入与 level_6 课程变更

### 两帧输入实现

```text
CNN PPO 新增 --frame-stack 超参数，范围为 1 到 5，默认值为 1。
新增 --frame-stack-interval 超参数，范围为 1 到 5，默认值为 1。
环境在每个真实 frame_step 更新完成后保存当前子弹 map；不会只在 action_repeat 结束后采样。
frame_stack=2 时，输入为“前一游戏帧 + 当前游戏帧”。
frame_stack=2、frame_stack_interval=5 时，输入为“5 帧前 + 当前帧”。
frame_stack=5、frame_stack_interval=5 时，环境保留 21 张连续 map，并抽取第 0/5/10/15/20 张作为输入。

red：4 个通道变为 8 个通道。
yellow：2 个通道变为 4 个通道。
blue：2 个通道变为 4 个通道。
player_features 保持当前帧的 8 维特征，不做时间堆叠。

reset 时用初始 map 重复填满历史，保证每一步输入形状固定。
多帧 checkpoint 会保存 frame_stack 和 frame_stack_interval；不同帧数或不同间隔的模型不能互相加载。
```

### test7：两帧输入的 level_5 随机弹试验

```text
日志：training_logs/lunai_v6_test7-stack2.csv
模型：checkpoints/lunai_v6_test7-stack2.pt
关卡：level_5.json，高密度纯随机弹
观察：CNN PPO，frame_stack=2
训练预算：200,000 环境帧
最后完整日志：episode 334，total_frame_steps 199,574

整体：平均存活 = 597.5 帧，中位数 = 578 帧，平均 reward = -4.094，
      存活至关卡上限 840 帧 = 79 / 334。
最后 50 局：平均存活 = 671.9 帧，中位数 = 678 帧，平均 reward = -1.994，满帧 = 17 / 50。
最后 20 局：平均存活 = 708.9 帧，中位数 = 725 帧，平均 reward = -0.944，满帧 = 8 / 20。
最后一次更新：entropy = 1.357140，value_loss = 44.568279，approx_kl = 0.000104。
```

```text
训练后段的存活帧数上升，说明两帧输入没有导致数值训练失败。
但 greedy render 仍学到了向角落移动的局部策略；在纯随机弹环境中，角落仍可能成为可利用的局部安全区。
因此不能把该结果解释为“两帧输入无效”，而应记录为“仅增加时间信息不能消除任务本身的角落局部最优”。
该 checkpoint 保留为纯随机弹、两帧输入的对照样本，不继续在其上训练。
```

### level_6：随机弹加低频自机狙

```text
文件：assets/levels/level_6.json
来源：完整复制 level_5.json，保留随机弹密度、敌人轨迹和关卡时长。
单轮新增：6 个敌人各发射 2 枚单发自机狙，共 12 枚。
自机狙：kunai_0.png，16x16，碰撞半径 4，速度 145；速度与 level_5 随机弹一致。
发射时机：敌人出现后 0.55 秒开始，间隔 1.0 秒。
```

```text
设计目的不是单纯提高弹幕数量，而是让长期静止或缩在角落的策略逐渐失效；
随机弹仍是主任务，自机狙只提供温和且持续的移动压力。
level_5 保留为纯随机弹对照组。正常游玩关卡序列扩展为 level_1 到 level_6。
level_6 尚未训练；它属于新的环境分布，后续应从头初始化模型训练。
```

### 弹幕内容翻倍并连续衔接

```text
level_1 到 level_6 的原始敌人和弹幕完整复制第二遍。
第二轮中，每个敌人的出生 time 都等于第一轮 time 加第二轮偏移；攻击配置、轨迹、贴图、弹速均不变。
第二轮偏移经过计算，使第二轮第一发弹的发射时刻恰好等于第一轮最后一次发射时刻。

level_1：16 秒 -> 30.70 秒，敌人 8 -> 16。
level_2：15 秒 -> 27.90 秒，敌人 8 -> 16。
level_3：15 秒 -> 27.50 秒，敌人 6 -> 12。
level_4：14 秒 -> 24.00 秒，敌人 6 -> 12。
level_5：14 秒 -> 23.20 秒，敌人 6 -> 12。
level_6：14 秒 -> 23.20 秒，敌人 6 -> 12；全关自机狙为 24 枚，每轮 12 枚。
```

```text
此调整不会在两轮之间制造无弹空白时间，也不改变单轮弹幕密度。
实际时间不再严格等于两倍原 length，因为原关卡末尾本就存在发射结束后的等待时间；去除这段等待才可以保证连续。
训练使用 max_steps 时应按新长度换算：level_1=1842 frame，level_2=1674 frame，level_3=1650 frame，level_4=1440 frame，level_5/6=1392 frame。
```

## 2026-07-11：LunAI v6 实验协议与非终止接触训练失败记录

### 可复现实验配置与日志

```text
配置文件位置：项目根目录 config.json
每轮实际参数来源：对应 CSV 首行的 # run_config: {...}

本文整理时 config.json 指向 lunai_v6_test2.2：
load_path=checkpoints/lunai_v6_test2.1.pt
model_path=checkpoints/lunai_v6_test2.2.pt
log_path=training_logs/lunai_v6_test2.2.csv

启动命令：python rl/train_ppo_cnn.py --config config.json
临时覆盖示例：python rl/train_ppo_cnn.py --config config.json --max-total-frame-steps 1000000
```

```text
config.json 作为一轮实验的版本化基线，应在每次创建新实验时随代码一起提交。
命令行参数优先于 config.json；训练日志首行会自动写入 # run_config: {...}，记录实际生效的完整参数，
因此临时覆盖不会丢失。training_trend_viewer.py 会跳过该元数据行，旧 CSV 也仍可正常读取。
checkpoint、CSV 和趋势图继续保持在 .gitignore 中，避免仓库随训练产物膨胀。
```

### v6_test1 参数

```text
任务：level_6.json（随机弹 + 低频自机狙，双波连续）
训练预算：max_total_frame_steps=100,000；episodes=1,000,000 仅作为安全上限
每次决策：action_repeat=3
观察：CNN，frame_stack=2，frame_stack_interval=2
初始状态：random_player_start=true，player_start_margin=80.0，seed=0
PPO：rollout_steps=1024，minibatch_size=256，update_epochs=4
学习率：0.00012 -> 0.00004
熵系数：0.004 -> 0.0015
其余关键值：gamma=0.99，gae_lambda=0.95，clip_range=0.2，target_kl=0.03，hidden_dim=128
从头训练：load_path 为空
```

```text
frame_stack=2、frame_stack_interval=2 的输入是“2 个真实游戏帧前的 map + 当前 map”。
地图历史在每个真实 frame_step 更新，不受 action_repeat 的决策频率限制。
本节记录 v6_test1 的启动方案；其后的实际结果见本节末尾，不能仅由训练中间指标判断效果。
```

### 非终止接触模式与 test1 奖励定义

```text
CNN 训练环境启用 training_invincible=true：玩家与敌弹或敌人本体接触时不会掉血、复活或重开，
episode 仅在关卡 max_steps 或全局训练预算到达时结束。每个接触的真实游戏帧都会单独记一次 contact frame。
evaluate_ppo_cnn.py 不启用该模式，验收仍是一次真实碰撞即结束。

因此无敌训练阶段不能用 frame_steps 判断是否学会避弹，因为大多数局都会走到关卡上限；
应重点观察 episode_reward 与每局 contact_frames 是否下降，随后再用致死评估检验真实存活表现。
本节 v6 CSV 的 collisions 列实际记录的是 contact_frames；旧式致死训练日志中的 collisions 则表示终局前的碰撞统计，两者不能混为同一指标。
```

```text
test1 每个真实游戏帧的奖励：
survival_reward = +0.100
near_danger_penalty = 0.03 * danger，danger 范围为 [0, 1]
boundary_penalty 最大为 0.100（侧边 0.025 + 上下边 0.020 + 角落 0.055）
collision_penalty = -5.000 / 接触帧
action_change_penalty = -0.005
reversal_penalty = -0.020（反向时还会同时产生动作切换惩罚）

安全且静止的单帧最高回报为 +0.100；接触一帧的基础回报约为 -4.900，
连续接触会线性累计惩罚。该设置的目的，是让模型在不被终局截断的情况下获得“离开碰撞区域”的连续反馈。
```

### test1：每接触帧固定惩罚导致尺度崩坏

```text
日志：training_logs/lunai_v6_test1.csv
模型：checkpoints/lunai_v6_test1.pt
最后完整日志：episode 57，total_frame_steps=79,344
训练模式：training_invincible=true；每个 contact frame 扣 5.0。

整体：mean_reward=-315.808，mean_contact_frames=75.40。
最后 20 局：mean_reward=-689.727，mean_contact_frames=142.25。
最后一次更新：entropy=0.343851，value_loss=1,018.500。
```

```text
早期存在随机的较好样本，例如 episode 21 的 reward=73.614、contact_frames=5；
但当时 entropy 仍接近 2.20，策略基本均匀随机，不能当作已学会躲弹的证据。
之后 contact_frames 持续升高，reward 降至约 -800，value_loss 升至约 1,000，
greedy render 收缩到右下角。原因是 -5.0 会按真实接触帧重复累计：
接触 160 帧单项即约 -800，而完整关卡的生存奖励总量仅为 +139.2。

结论：在原地无敌、但每接触帧重罚的定义下，回报尺度和 critic 拟合均失稳；
test1 作为“连续接触惩罚失败”的负对照保留，不继续训练。
```

### test2、test2.1、test2.2：数值未崩坏，但任务定义仍可被利用

```text
共同设置：level_6.json，frame_stack=2，frame_stack_interval=2，action_repeat=3，
random_player_start=true，max_total_frame_steps=100,000，training_invincible=true。
每个 CSV 的最后完整 episode 均为 71，total_frame_steps=98,832；
全局帧预算在下一局的未写入部分达到上限，因此最后完整日志小于 100,000。

test2：从头训练。
mean_reward=46.122，last20_reward=40.087，mean_contact_frames=24.77，last20=22.75，
最后 entropy=1.893730，value_loss=8.858。

test2.1：从 checkpoints/lunai_v6_test2.pt 继续训练。
mean_reward=55.277，last20_reward=63.056，mean_contact_frames=19.93，last20=16.95，
最后 entropy=1.622650，value_loss=9.134。

test2.2：从 checkpoints/lunai_v6_test2.1.pt 继续训练。
mean_reward=50.763，last20_reward=53.578，mean_contact_frames=23.49，last20=22.65，
最后 entropy=1.447210，value_loss=12.498。
```

```text
这三轮的 reward、contact_frames 和 value_loss 看起来比 test1 健康，
但渲染验收显示策略仍不能可靠地在真实致死规则下避弹。非终止碰撞改变了环境动力学：
模型可以把“承受一次或多次碰撞后继续推进关卡”作为可接受甚至可利用的路径，
而最终游戏并不存在这个选项。因此正 reward 或较少 contact_frames 不能等价于真实通关能力。

早期 CSV 只保存了命令行参数，没有保存当时 reward.py 的源码快照；
故本记录不补写 test2 系列各轮的精确奖励数值，以免把无法验证的信息伪装成实验事实。
```

### 阶段结论

```text
“训练时碰撞不终止、最终验收时碰撞终止”形成了明显的训练-评估动力学不一致。
连续接触重罚会造成回报尺度爆炸；降低惩罚虽可保持数值稳定，却会留下利用受击继续推进的空间。
因此非终止接触训练不再作为 LunAI 的主训练路线，相关代码仅保留为失败对照。

后续主线恢复真实致死碰撞，并通过拆分后期弹幕为独立短关、或改变训练起点分布，
让模型直接获得后期弹幕样本，而不是依赖“无敌后继续活着”才能看到后续内容。
```

## 2026-07-11：终止碰撞主线恢复与并行采样准备

### 训练规则恢复

```text
主训练路径恢复为真实致死碰撞：training_invincible=false。
玩家碰撞敌弹或敌人本体后，该 episode 立即结束；非终止接触模式的代码仅作为 v6 失败消融保留，
不再用于新的主线 checkpoint。

当前 reward.py 的关键数值：
survival_reward = +0.100 / 真实游戏帧
collision_penalty = -50.000 / 终局碰撞
near_danger_penalty 最大为 -0.030 / 帧
boundary_penalty 最大为 -0.100 / 帧
action_change_penalty = -0.005
reversal_penalty = -0.020
```

### 八环境并行采样

```text
config.json 新增 num_envs=8。
训练时创建 8 个无渲染 pygame 环境进程，但只在主进程保留 1 个 CNN PPO 模型与 1 份优化器；
这不是 8 个彼此独立的 LunAI。

当前配置 rollout_steps=1024，因此每次 PPO 更新由 8 个环境各采集 128 个决策组成，
总计 1024 条 transition。global_step 记录所有环境 transition 的总数，
total_frame_steps 记录所有环境实际游戏帧的总数。

并行训练不支持 render；新 CSV 额外记录 env_id，以标识每个完成 episode 的来源环境。
```

### 可复现实验修复

```text
环境 reset(seed) 现在同时调用 numpy.random.seed(seed) 与 random.seed(seed)。
此前随机出生点受 NumPy seed 控制，但 level_5/level_6 的随机弹幕角度使用 Python 标准库 random，
同一 seed 不能严格重现同一局。修复后两类随机源均随 reset seed 固定。

本训练历史继续放在 training_logs/plots/，但 .gitignore 已为
training_logs/plots/reward_and_training_history.md 留出例外，使其会随代码推送；
checkpoint、CSV 与生成趋势图仍被忽略，避免仓库包含大型训练产物。
```

## 2026-07-11：精细红区观察版本

```text
红区空间范围保持 128×128 游戏像素，不扩大观察半径；red_map 从 32×32 改为 64×64。
因此每个红区单元从约 4×4 像素变为约 2×2 像素，红区继续保留 occupancy、vx、vy、speed 四类信息。

红区 CNN encoder 的 AdaptiveAvgPool 从 4×4 改为 8×8，避免新增的近身空间细节被直接平均回旧分辨率。
yellow 和 blue 分支在本版本不变，分别保持 8×8 / pool 2×2 与 6×6 / pool 2×2。

在 frame_stack=5、hidden_dim=128 下，CNN 可训练参数约为 302,394。
该版本不能加载旧 32×32 红区 checkpoint，后续训练必须从头初始化。
```

## 2026-07-11：玩家中心红黄区与可活动区域掩码

```text
红区和黄区改为始终以玩家为中心。当局部窗口超出游戏边界时，图外位置的弹幕数据保持为 0，
但新增 valid mask 明确标识该格子不可活动，避免将图外零值误读为安全空区域。

新增 red_valid 与 yellow_valid：它们的值为每个格子落在游戏场地内的面积比例，范围为 [0, 1]。
红区每帧输入通道变为 occupancy、vx、vy、speed、valid；黄区每帧输入通道变为 density、speed、valid。
多帧输入会为每个历史时刻保留对应的 valid mask。

因此，玩家在红黄图中始终位于中心；局部 CNN 可以直接将子弹格子解释为相对玩家的位置，
并从 valid mask 获知哪一侧是不可通行的场地外区域。旧 checkpoint 与本表示不兼容，必须从头训练。
```
