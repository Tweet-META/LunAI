# 原生 TH06 环境适配

LunAI 可以把 [`AgentMystia/th6_web`](https://github.com/AgentMystia/th6_web) 中复刻的红魔乡 C++ 游戏逻辑编译为独立的 `th06_rl_server`。训练不启动网页、浏览器或游戏窗口；Python 通过 stdin/stdout 二进制协议逐帧发送动作并读取状态。

当前适配是实验性第二环境，不替代论文实验使用的 Pygame 环境。

## 数据通路

1. `rl_server_main.cpp` 启动原始 TH06 游戏逻辑，并等待固定长度的二进制命令。
2. `rl_api.*` 从游戏对象读取玩家、子弹、激光、敌人和关卡状态。
3. `rl/th06_adapter.py` 启动服务进程，将快照转换为 LunAI 的红黄蓝三区和 PCCM。
4. `Th06RLEnv` 复用当前 reward、帧堆叠和 CNN PPO 接口。

每次 `step()` 只运行一个原始游戏帧，不调用显示刷新，也不按 60 FPS 等待。训练默认完全无窗口。`--render` 只让 Python 根据同一份状态快照绘制观察窗口，不会改变游戏推进逻辑。

## 构建

TH06 反编译代码依赖 32 位布局。在 Visual Studio 开发者终端中执行：

```powershell
cd ..\external\th6_web
cmake -S . -B build-native -A Win32
cmake --build build-native --config Release --target th06_rl_server
```

服务程序默认位于：

```text
external/th6_web/build-native/Release/th06_rl_server.exe
```

运行还需要合法取得的原版 `th06_*.DAT` 数据文件。它们不属于 LunAI 仓库，也不应提交到 GitHub。可以把数据文件放在服务程序旁边，或用 `--th06-assets-dir` 指定目录。

## 无窗口训练

```powershell
python rl/train_ppo_cnn.py --environment th06 --num-envs 1 --action-repeat 1 --th06-stage 1 --th06-difficulty 1 --th06-assets-dir D:\path\to\th06 --frame-stack 4 --max-steps 3600 --max-total-frame-steps 1000000 --model-path checkpoints/lunai_th06.pt --log-path training_logs/lunai_th06.csv
```

原生 TH06 环境目前只支持单进程和 `action_repeat=1`。未指定 `--render` 时不会初始化 Pygame 显示窗口。

## 可视化训练或评估

训练时临时查看：

```powershell
python rl/train_ppo_cnn.py --environment th06 --action-repeat 1 --render --th06-assets-dir D:\path\to\th06
```

评估 checkpoint：

```powershell
python rl/evaluate_ppo_cnn.py --environment th06 --model-path checkpoints/lunai_th06.pt --action-repeat 1 --th06-stage 1 --th06-assets-dir D:\path\to\th06 --render
```

`--render-debug` 会在右侧额外显示当前红区 PCCM 和状态计数。按 `Esc` 或关闭窗口会结束当前运行。

## 快照内容

固定 39000 字节的 ABI 快照包含：

- 玩家坐标、碰撞箱、残机、死亡计数和边界阻挡位移
- 最多 640 颗活动子弹的精确位置、碰撞箱和每帧速度
- 最多 64 条激光的方向、长度、宽度和碰撞状态
- 最多 256 个敌人的位置、速度、碰撞箱和体术判定
- 原始关卡、难度、游戏帧和结束状态

TH06 的速度单位是像素/游戏帧。适配器在生成 PCCM 前换算为像素/秒。九个动作编号与 Pygame 环境一致。

普通子弹使用原作导出的宽高作为轴对齐矩形判定，敌人体术使用原作实际的 `hitboxDimensions / 1.5`。观察构建时会分别把矩形横纵半径与自机判定相加，再生成 density、红区 occupancy 和 PCCM，因此硬碰撞区域与原作的 AABB 判定一致。旋转激光目前仍使用沿激光轴排列的圆形样本近似，原生 C++ 游戏本体继续负责最终死亡判定。

TH06 场地是 Pygame 环境尺寸的约 `0.64` 倍。适配器默认使用 `204×204` 黄区、`64×64` 红区和 `20px` PCCM halo；输出张量仍保持蓝区 `8×8`、黄区 `16×16`、红区 `64×64`，因此不会改变 CNN 输入 shape。`--render` 会把三个尺度按实际世界坐标叠加到游戏区域，方便检查投影和弹幕是否对齐。

## 验证

无需 TH06 数据文件即可验证 ABI 解析、观察构建和环境接口：

```powershell
python tools/validate_th06_adapter.py
```

完整运行仍需在安装了 32 位 C++ 工具链的机器上构建服务程序，并提供原版数据文件。
