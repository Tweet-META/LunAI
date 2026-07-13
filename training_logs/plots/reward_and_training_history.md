# Lunai的学习日志

这份学习日志按时间顺序整理Lunai学习中的奖励函数和各代的设计

说明：

```text
1. 部分早期奖励函数没有单独保留旧版源码，因此只能记录当时的设计方向和日志统计
2. 能从 CSV 或当前代码确认的地方，全部写具体数值
3. 这里不把任何版本写成最终模型，只记录历史过程、参数、结果和判断
```

## 2026-07-02：dqn_curriculum

奖励设计方向：

```text
以存活奖励为主
撞弹给惩罚
撞弹后结束 episode
```

训练结果：

```text
episodes=300
mean_frame=462.8
mean_reward=0.860
collisions=300/300
```

观察结果：

```text
DQN 可以跑通，但训练非常不稳定
几乎每局都会撞弹
平均生存帧只有 462.8
作为正式 baseline 不够稳
```

阶段判断：

```text
保留为最早 baseline
```

## 2026-07-03 00:02：exp_001_distance_reward

改动：

```text
距离子弹较近时给予惩罚
子弹越近惩罚越大，线性增长
```

结果：

```text
方向可用，但直接用于 DQN 不稳定
```

## 2026-07-03 00:55：exp_002_slow_epsilon

改动：

```text
放慢 epsilon 衰减，让随机探索保留更久
```

结果：

```text
慢 epsilon 有帮助，但没起到决定性作用
```

## 2026-07-03 08:54：exp_003_slow_epsilon

改动：

```text
调大collision_penalty
```

结果：

```text
reward崩坏，对训练也没有帮助
```

## 2026-07-03 20:52：exp_004_fixed_collision_penalty


改动：

```text
把collision penalty调回了-10
避免-100过大直接淹没前面每一步的存活奖励和距离奖励
```

结果：

```text
DQN仍然没有学会避弹
猜测DQN不适合Lunai
```

## 2026-07-03 21:38：ppo_exp_001

改动：

```text
从 DQN 切换到 PPO
训练使用 on-policy 采样
```

结果：

```text
平均存活时间和最大存活时间均增加
PPO 明显更适合当前环境
PPO 定为后续 baseline 主线
```

## 2026-07-03 22:35：ppo_exp_002_near_penalty

改动：

```text
加入near bullet penalty
目标是让Lunai不要贴着子弹走
```

结果：

```text
reward变低
但实际表现更好
```

## 2026-07-03 23:14：ppo_exp_003_boundary_penalty_1200

改动：

```text
加入 boundary penalty
训练上限使用 1200 frame
目标是减少贴墙/贴角挂机
```

结果：

```text
边界惩罚有效
但过强可能导致左右脑互搏
```

## 2026-07-04 08:44：ppo_exp_004_boundary_features_1200/lunai_v0

改动：

```text
在 player_features 中加入边界距离特征
```

结果：

```text
collisions 从 294/300 降到 239/300
v0 是第一个有效 baseline
但她主要学会处理早期自机狙，对后续复杂弹幕仍然不足
```

## 2026-07-04 13:36：lunai_v1_2400

改动：

```text
max_steps 从 1200 提高到 2400
目标是让Lunai接触更靠后的弹幕
```

结果：

```text
单纯拉长 episode 不够
需要 curriculum
```

## 2026-07-04 15:04：lunai_v2

改动：

```text
继续调整生存奖励、撞弹惩罚和边界相关惩罚
```

结果：

```text
短训表现不能说明稳定学会
需要长训验证
```

## 2026-07-04 16:33：lunai_v2_long

改动：

```text
-
```

阶段判断：

```text
v2_long 是重要失败样本
说明 reward shaping 和长关卡混训会导致局部最优
需要把弹幕模式拆开训练
```

## 2026-07-05 11:30：lunai_v3_stage1

改动：
```text
关卡拆分如下：
level_0.json = 旧版完整关卡备份，暂时弃用
level_1.json = 简单单发自机狙
level_2.json = 轻量 3-way 自机狙
level_3.json = 自机狙 + 随机弹混合
level_4.json = 中等随机弹
level_5.json = 高密度随机弹
```

结果：

```text
能学到一些引狙，但满 960 次数较少
需要更积极的 stage1 训练设置
```

## 2026-07-05 11:49：lunai_v3_stage1_fast

改动：
```text
增加学习速度
```

结果：

```text
greedy render 中能看到靠边引狙和微移
策略方向是对的，但离子弹较近
stochastic 抖一下容易撞死
```

## 2026-07-05 12:20：lunai_v3_stage1_refine

改动：

```text
调整奖励函数（？）
```

阶段判断：

```text
不能只靠继续 refine 解决
需要调整近弹惩罚形状
```

## 2026-07-05 12:46：lunai_v3_stage1_linear

改动：

```text
高斯权重改为线性权重

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

near_danger_penalty = 0.03 * danger
danger = 0.8 * occupancy_danger + 0.2 * speed_danger

reward =
    survival_reward
    + safe_stay_reward
    - near_danger_penalty
    - boundary_penalty
    - collision_penalty
    - action_change_penalty
    - reversal_penalty
```

结果：

```text
线性近弹惩罚明显改善 stage1
greedy 能执行靠边引狙策略
stochastic 仍然会早死，但 checkpoint 可以转入 stage2
```

## 2026-07-05 13:18：lunai_v3_stage2

改动：

```text
-
```

训练参数：
```text
entropy_coef=0.004
entropy_coef_final=0.0015
learning_rate=0.00012
learning_rate_final=0.00004
rollout_steps=1024
```

结果：

```text
CSV 中有 75/250 次满 900
但是 greedy render 开局死亡
```

## 2026-07-05 14:11：lunai_v3_stage2_det

训练参数：

```text
entropy_coef=0.0015
entropy_coef_final=0.0003
learning_rate=0.00008
learning_rate_final=0.000025
rollout_steps=2048
target_kl=0.01
```

结果：

```text
greedy render 观察为可接受
可以作为 stage3 的迁移起点
```

## 2026-07-05 15:01：lunai_v3_stage3

训练参数：

```text
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
满帧=105/300
最后 50 局 mean_frame=666.6
最后 50 局满帧=19/50
mean_reward=8.209
最后 entropy=2.042841
```

结果：

```text
加入自机狙和随机弹混合后仍能保持一定满帧率
greedy 仍然偏爱贴墙/贴角，但能做微移
接受为 stage4 的迁移起点
```

## 2026-07-05 15:45：lunai_v3_stage4

训练参数：

```text
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
满帧=137/300
最后 50 局 mean_frame=737.0
最后 50 局满帧=27/50
mean_reward=3.188
最后 entropy=1.437866
```

结果：

```text
CSV 看起来比 stage3 更好
greedy 一直贴右下角，随机弹过来时也不主动离开
回测 level_1 时，原本学会的纯自机狙微移也退化成缩角落
拒绝该 checkpoint
```

## 2026-07-05 16:56：lunai_v3_stage4_wallfix

改动：

```text
VERTICAL_EDGE_PENALTY_SCALE = 0.015 -> 0.03
CORNER_PENALTY_SCALE = 0.04 -> 0.08
```

训练结果：

```text
episodes=103
mean_frame=721.9
满帧=44/103
mean_reward=-11.983
最后 50 局 mean_reward=-19.953
最后 entropy=1.126772
```

结果：

```text
大量满帧 episode 仍然得到强负分
活着和撞死都会得到强负反馈
边界惩罚过强，reward 信号不清晰
```

## 2026-07-05 17:22：lunai_v3_stage4_softwall

改动：

```text
EDGE_MARGIN = 0.14
SIDE_EDGE_PENALTY_SCALE = 0.025
VERTICAL_EDGE_PENALTY_SCALE = 0.02
CORNER_PENALTY_SCALE = 0.055
贴满角最大边界惩罚 = 0.10
```

训练结果：

```text
episodes=112
mean_frame=675.8
满帧=46/112
mean_reward=-1.488
最后 50 局 mean_reward=-4.785
最后 entropy=1.340531
```

结果：

```text
从 lunai_v3_stage3 回退重训
softwall 仍然会直接去右下角
继续调大边界惩罚会让满帧也得到负分，调小则压不住角落策略
停止 v3 stage4 修补路线
```

## 2026-07-05 17:58：lunai_v4_stage_random

改动：

```text
加入 random_player_start
玩家随机出生在游戏区下半场
player_start_margin=80
从头训练 level_4
```

训练结果：

```text
episodes=300
mean_frame=539.5
满帧=57/300
mean_reward=3.460
collisions=243/300
最后 entropy=2.183966
```

结果：

```text
随机出生打破了固定出生点到右下角的固定路线
greedy 仍喜欢在敌人出现前先靠墙
没有学到无威胁时站定
```

## 2026-07-05 18:29：lunai_v4_random_stay

改动：

```text
SAFE_STAY_REWARD = 0.003 -> 0.006
只有红区无弹且 action=stay 时获得
```

训练结果：

```text
episodes=300
mean_frame=536.2
满帧=55/300
mean_reward=3.187
最后 50 局 mean_frame=494.5
最后 50 局满帧=6/50
最后 entropy=2.183255
```

结果：

```text
提高 safe stay 没有改善整体存活
策略熵仍接近最大值
无威胁时站定的问题没有被稳定解决
```

## 2026-07-05 19:42：lunai_v4_stage4_fromscratch

改动：

```text
不加载旧 checkpoint
重新从头训练 level_4
保留 random_player_start
```

训练结果：

```text
episodes=300
mean_frame=537.5
满帧=54/300
mean_reward=3.128
最后 50 局 mean_frame=540.1
最后 entropy=2.181865
```

结果：

```text
从头训练仍然没有明显改善
entropy 几乎没有下降
说明问题不是旧 checkpoint 污染
```

## 2026-07-05 21:09：lunai_v4_stage4_more

改动：

```text
把 stage4 训练延长到 500 episodes
```

训练结果：

```text
episodes=500
mean_frame=530.3
满帧=99/500
mean_reward=3.340
最后 50 局 mean_frame=529.4
最后 50 局满帧=11/50
最后 entropy=2.153860
```

结果：

```text
增加训练量后 entropy 只小幅下降
平均存活没有增长
MLP PPO 在随机弹上的学习效果有限
v4 路线停止
```

## 2026-07-10 13:10：lunai_cnn_selfaim_pilot

改动：

```text
从 MLP PPO 改为多尺度 CNN PPO
单帧 observation
level_1 自机狙
random_player_start=true
训练预算 200,000 frame
```

奖励参数：

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

训练结果：

```text
episodes=310
mean_frame=642.9
满帧=48/310
后 50 局 mean_frame=765.3
后 50 局满帧=17/50
最后 entropy=1.449614
最后 value_loss=7.347075
```

结果：

```text
采样执行时能通过小幅方向调整引导自机狙
视觉上比 MLP baseline 更接近人类微移
greedy 在未参与训练的 seed 上过度偏好 stay
CNN 输入方向有效，但策略仍有局部最优
```

## 2026-07-10 14:32：lunai_v5_test2

训练结果：

```text
episodes=302
mean_frame=659.5
满帧=76/302
最后 50 局 mean_frame=912.7
最后 50 局满帧=41/50
最后 entropy=1.529138
```

结果：

```text
发现自机狙角度计算仍使用旧 arccos 逻辑
玩家位于敌人上方时子弹方向错误，左上角形成假安全区
高存活数据被环境 bug 污染
不进入正式横向比较
```

## 2026-07-10 15:30：lunai_v5_test3

改动：

```text
自机狙角度改为 atan2
survival_reward = 0.1
collision_penalty = 50.0
action_change_penalty = 0.005
reversal_penalty = 0.02
safe stay 仍然只检查红区
```

训练结果：

```text
episodes=332
mean_frame=601.7
满帧=29/332
最后 50 局 mean_frame=689.9
最后 50 局满帧=10/50
最后 entropy=1.965948
最后 value_loss=50.337753
```

结果：

```text
不再利用左上角瞄准漏洞
greedy 仍容易选择 stay 并原地撞死
提高生存奖励和撞弹惩罚不足以消除 stay 偏好
```

## 2026-07-10 16:23：lunai_v5_test4

改动：

```text
只有 action=stay 且红区、黄区同时无弹时才给予 SAFE_STAY_REWARD
红区或黄区任一区域出现子弹就取消停留奖励
```

训练结果：

```text
episodes=322
mean_frame=620.2
满帧=37/322
最后 50 局 mean_frame=620.0
最后 50 局满帧=0/50
最后 entropy=1.610652
```

验收结果：

```text
greedy 30 局 mean_frame=898.17
greedy collisions=0.30
stochastic 30 局 mean_frame=621.93
stochastic collisions=1.00
```

结果：

```text
训练后段 stochastic rollout 退化
greedy 以 stay 为主，只在需要时做少量微移
test4 是当时 deterministic 自机狙表现最好的版本
```

## 2026-07-10 17:42：lunai_v5_test5

改动：

```text
切换到 level_5 高密度随机弹
保留红区和黄区同时为空时的 SAFE_STAY_REWARD
训练预算 200,000 frame
```

训练结果：

```text
episodes=402
mean_frame=497.3
满帧=57/402
最后 50 局 mean_frame=568.4
最后 50 局满帧=12/50
最后 entropy=1.775931
最后 value_loss=50.701445
```

结果：

```text
训练后段好于整体
render 中能根据局部空隙做一些移动
仍会偶发撞弹
保留为带 safe stay 的随机弹对照
```

## 2026-07-10 23:46：lunai_v5_test6

改动：

```text
移除 SAFE_STAY_REWARD
其余训练预算和 level_5 与 test5 保持可比
```

训练结果：

```text
episodes=398
mean_frame=502.2
满帧=45/398
最后 50 局 mean_frame=509.7
最后 50 局满帧=7/50
最后 entropy=1.942183
最后 value_loss=57.762933
```

结果：

```text
整体 mean_frame 与 test5 接近，但训练后段更弱
render 中出现过度 stay 和面对弹幕不移动
单纯删除停留奖励没有得到更积极的策略
```

## 2026-07-11 01:24：lunai_v5_test7

改动：

```text
新增 frame_stack
frame_stack=2
每个真实 frame_step 保存一次 map
red 通道 4 -> 8
yellow 通道 2 -> 4
blue 通道 2 -> 4
player_features 保持当前帧 8 维
level_5 高密度随机弹
```

训练结果：

```text
episodes=334
mean_frame=597.5
满帧=79/334
最后 50 局 mean_frame=671.9
最后 50 局满帧=17/50
最后 entropy=1.357140
最后 value_loss=44.568279
```

结果：

```text
两帧输入训练后段明显上升
greedy 仍然学到向角落移动的局部策略
增加时间信息不能单独消除关卡中的角落局部最优
保留为两帧随机弹对照
```

## 2026-07-11 12:56：lunai_v5_test8

改动：

```text
frame_stack=2
frame_stack_interval=2
使用加长后的弹幕关卡
checkpoint 确认输入仍为 32x32 红区
```

训练结果：

```text
episodes=127
total_frame_steps=56,515
mean_frame=445.0
mean_reward=-19.648
collisions=127/127
最后 50 局 mean_frame=463.2
最后 50 局 mean_reward=-28.319
最后 entropy=1.159604
最后 value_loss=67.299019
```

结果：

```text
训练提前停止
所有完整 episode 最终都撞弹
策略熵快速下降，但存活没有同步改善
该版本没有形成可用策略
```

## 2026-07-11 22:58：lunai_v6_test1

改动：

```text
red_map 从 32x32 改为 64x64
红区范围仍为 128x128 游戏像素
red encoder pool 从 4x4 改为 8x8
新增 num_envs=8
rollout_steps=1024，每个环境每轮采集 128 个决策步
reset 同时设置 NumPy random 和 Python random
从头训练 level_1
frame_stack=2
frame_stack_interval=2
```

训练结果：

```text
episodes=145
total_frame_steps=99,495
mean_frame=668.7
满 1392 帧=7/145
最后 50 局 mean_frame=803.0
最后 50 局满帧=6/50
mean_reward=7.170
最后 entropy=1.899747
最后 value_loss=43.792219
```

结果：

```text
数值上后段比前段好
高分和满帧 episode 开始出现
greedy 仍然会利用墙边策略
64x64 红区本身没有消除贴边局部最优
```

## 2026-07-11 23:27：lunai_v6_test2

改动：

```text
从 lunai_v6_test1 继续训练
learning_rate 固定为 0.00004
entropy_coef 固定为 0.0015
其余环境和 observation 不变
```

训练结果：

```text
episodes=107
total_frame_steps=99,436
mean_frame=897.6
满 1392 帧=17/107
最后 50 局 mean_frame=960.0
最后 50 局满帧=9/50
mean_reward=20.825
最后 entropy=1.610083
最后 value_loss=17.073160
```

结果：

```text
CSV 数值明显高于 test1
greedy render 仍会缩到角落
数值没有崩坏，但行为上仍是不可接受的局部最优
```

## 2026-07-12 11:51：lunai_v7_test1

改动：

```text
红区和黄区始终以玩家为中心
窗口超出边界时图外弹幕数据填 0
新增 red_valid 和 yellow_valid
valid 表示格子落在游戏区内的面积比例
red 每帧通道变为 occupancy、vx、vy、speed、valid
yellow 每帧通道变为 density、speed、valid
num_envs=1
旧 checkpoint 不兼容，从头训练
```

当前记录：

```text
episodes=122
total_frame_steps=61,406
mean_frame=503.3
满 1392 帧=0/122
mean_reward=-19.965
collisions=122/122
最后 50 局 mean_frame=383.5
最后 50 局 mean_reward=-37.356
最后 entropy=1.231844
最后 value_loss=56.678904
```

结果：

```text
valid mask 修复了贴边时局部地图坐标含义变化的问题
当前训练仍然出现贴墙和不主动避弹
说明角落策略不只来自输入歧义
本轮日志仍未达到原定训练预算，暂时不能作为最终实验结果
```

## 2026-07-12：lunai_v7_diagnose5

改动：
```text
action_repeat=1
ACTION_CHANGE_PENALTY=0
固定玩家出生位置
训练关卡随机选择空场和左、中、右三种大狙
敌人出现时间在 0～2.5 秒内随机
加入较弱的持续贴墙惩罚和上半场惩罚
大狙贴图放大到约 48x48，碰撞半径从 10 增加到 20
frame_stack=2
frame_stack_interval=2
```

训练结果：
```text
total_frame_steps=99,962
episodes=222
前 50 局 mean_frame=426.6，满 500 帧 27/50
后 50 局 mean_frame=464.9，满 500 帧 43/50
后 20 局 mean_frame=456.6，满 500 帧 17/20
最终 entropy=1.665668
```

greedy 验收：
```text
左侧大狙：10/10 存活 500 帧
中间大狙：10/10 存活 500 帧
右侧大狙：10/10 存活 500 帧
三组贴墙比例均为 0
三组向下动作比例均为 0
策略通常等待子弹接近，再向左移动约 6～8 帧并重新停下
```

结论：
```text
首次明确验证“多帧 occupancy -> CNN 识别来弹 -> PPO 短移避弹 -> 停下”的完整链路
策略仍有固定向左的动作偏好，对小狙的泛化不足
该 checkpoint 保留为 PCCM 前的 motion-schema 基线
```

## 2026-07-13：多尺度 PCCM 主线实现

改动：
```text
新增 observation_schema=pccm，同时保留旧 motion schema 的 checkpoint 兼容
红、黄、蓝三个 CNN 分支统一使用 occupancy/density、PCCM、playable_mask
两帧时三个分支均为 6 通道，总地图通道数仍为 18
速度不再直接输入 CNN，但每颗子弹的 vx/vy 继续用于 PCCM 预测
PCCM 预测未来 5 帧，halo_width=24，wall_margin=0.12
软风险使用 1-(1-old)*(1-new) 合并并限制到 0.8
真实碰撞区域考虑 bullet.radius+player.radius，最终覆盖为 1.0
新增 PCCM transition shaping，weight=0.05，碰撞帧不计算差值
训练日志新增平均局部 PCCM、PCCM shaping、贴墙比例和 9 个动作计数
调试图分开展示 occupancy、当前缓冲、未来预测、墙壁、最终 PCCM 和 playable mask
```

性能与验证：
```text
红区硬 occupancy 保持 64x64
平滑 PCCM 内部采样：红 32x32 后双线性放大，黄 32x32 -> 16x16，蓝 12x12 -> 6x6
level_6、最多 86 个危险体时，本机 PCCM 环境约 65.3 FPS
保留全采样 NumPy broadcasting 作为 reference，新增精确浮点 ROI 实现
纯 PCCM 微基准中蓝区 12x12 使用 reference 更快，黄区和红区 32x32 使用 ROI 更快
500 弹、7 次中位数基准中 reference 三尺度总耗时约 23.82 ms，auto 约 18.38 ms
随机玩家一致性测试中 max_abs_error=2.38e-7，mean_abs_error=5.30e-9，hard_collision_mismatch_count=0
新增 level_benchmark_pccm.json，固定生成 500 颗慢速环形弹进行真实场景压力测试
501 个危险体时完整 observation：reference 22.98 FPS，auto 22.63 FPS，auto 慢约 1.5%
level_6 平均 68.92 个危险体时：reference 153.57 FPS，auto 152.24 FPS，auto 慢约 0.9%
结论是 ROI 子步骤优化未转化为端到端收益，训练默认继续使用 reference，ROI 与 auto 保留作后续分析
旧 motion 环境和 lunai_v7_diagnose5 checkpoint 仍可正常加载
单环境和双环境 PPO smoke training 均通过
该条目仅记录实现完成，尚不是正式训练结果
```
