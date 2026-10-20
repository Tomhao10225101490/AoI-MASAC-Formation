# AoI-MASAC：链路会丢包时的无人机编队

这是一个给课程论文用的研究代码，题目是：

**AoI-Aware Multi-Agent Soft Actor-Critic for Robust UAV Formation and Obstacle Avoidance under Lossy Links**

中文可以叫：面向不可靠通信链路的、能感知信息时效的多智能体软演员-评论家无人机编队避障。

目标会议是 CCGrid 2027 或 WoWMoM 2027（二选一，不能同时投）。截稿按计划是 2026-12-01（AoE）。CCGrid 还要在 2026-11-24 交摘要。

这个仓库现在是**能在 CPU 上跑通的第一版**。它把环境、算法、消融开关、训练、评估、丢包率扫描和画图都接好了。它**还不是**论文里的最终实验。下面凡是写了数字的地方，都会标明那是本机上的 smoke（冒烟，意思是“先确认没有炸”）或 short run（短跑，步数很少），**不能写进论文当结果**。

---

## 0. 先看懂这件事在干什么（给高中生的版本）

想象四架小飞机跟着一架领队飞向终点，路上有一个圆形障碍。领队自己看得到终点。僚机（跟随者）要站在领队的斜后方，组成一个 V 字形。

以前的做法默认：僚机每一步都能从无线电里听到领队**现在**在哪、飞多快、朝哪转。真实无线电不是这样。飞得越远，包越容易丢。丢了的话，僚机脑子里还是几步之前的位置，但它自己可能不知道这条消息已经旧了，于是照着旧位置猛追，队形就散了，还可能撞上障碍。

这篇工作只改一件事：让僚机不但拿着“最后一次听到的领队状态”，还拿着一个数字——**这条消息已经旧了几步**。这个数字叫信息年龄，英文缩写 AoI（Age of Information）。训练的时候，每一回合随机换一个“信号好不好”的参数，让同一个策略在信号好和信号差时都能飞。

打个比方：以前僚机戴着实时耳机听领队。现在耳机会断。我们要它知道“这句话是三秒前说的”。消息旧了就别冲太猛，可以先往领队靠近，因为靠近了信号会变好。

---

## 1. 和本科那个私有仓库的关系

本科仓库 `Tomhao10225101490/MASAC` 是私有的。搭这个新仓库时**没有读到它的源码**（当前这台机器的 GitHub 身份打不开那个库），所以这里是按你写好的方案**重新实现**的，不是把旧文件拷过来。旧仓库不要改。

方案里写明、并且这版照着做的设定：

- 平面地图默认 700×600（单位当成米）
- 1 架领队 + 默认 4 架僚机
- 每架飞机每步 2 个连续动作：加减速、转弯
- 稠密奖励：靠近目标/靠近槽位、航向、编队、动作别太猛
- PyTorch，每架飞机自己的 actor，双 Q critic，自动调熵温度 α
- 训练前 50 个回合可以在动作上加 OU 噪声（Ornstein–Uhlenbeck，一种平滑的随机扰动）

方案里点名要改、这版已经改的：

- 僚机的观测里加上最近障碍物的相对位置和半径（旧观测“看不见”障碍）
- 飞机和飞机之间也判碰撞
- Critic 在训练时看全局状态，并且看**所有飞机的动作**（旧 critic 只用全局状态加自己的动作，不算标准的多智能体 critic）
- 领队状态走一条会丢包的链路，观测里带 AoI

旧论文里的 87.4% 成功、11% 碰撞，是**旧仓库**在完美通信下测出来的，不是这个仓库跑出来的。不要把那个数字写成本仓库的结果。

奖励的具体系数旧代码里看不到，所以 `configs/default.yaml` 里的权重是新选的起点，**还没调好**。论文实验之前要专门调。

---

## 2. AoI 是什么

AoI 是“我手里这份领队状态，已经多少步没有被新包刷新了”。

- 回合开始时，每架僚机免费拿到一次领队的真实状态，AoI = 0。这一次**不计入**后面的链路成功率，免得把开局的福利算进信道。
- 某一步如果这架僚机收到了包：把领队的位置、速度、航向换成**这一步移动之后**的新值，AoI 归零。
- 如果没收到：位置、速度、航向保持上次收到的值，AoI 加 1。
- AoI 最大到 `aoi_cap`（默认 30）。再丢包也不再增大。观测里送给网络的是 `AoI / aoi_cap`，所以大约落在 0 到 1。

例子：连续丢了 3 个包，AoI = 3。`aoi_cap = 30` 时，网络看到的是 `3/30 = 0.1`。

注意两件事：

1. 奖励里的“你离槽位有多远”用的是领队的**真实**位置。观测里的“你该飞去的槽位”用的是这架僚机**被允许知道**的领队位置。这两个可以不一样。这个差别就是任务本身。
2. 障碍物、以及最近的另一架飞机，当成本机传感器（雷达/相机）能看到的，**不走**这条会丢包的链路。会过时的只有领队状态。审稿人如果问“是不是所有通信都丢包”，答案是：这版只建模领队到僚机的下行状态包。

---

## 3. 信道模型

每一架僚机、每一步，独立抛一次硬币，决定这一个包到不到。成功概率：

```
distance 模式（默认，论文主模型）:
    p_succ = p0 * exp(-d / d0)

independent 模式（消融：丢包和距离无关）:
    p_succ = p0

perfect 模式（训练 vanilla MASAC 用）:
    p_succ = 1
```

- `d`：这一步大家移动完之后，该僚机到领队的距离（米）
- `p0`：距离为 0 时的成功概率（independent 模式下则是任何距离的成功概率）
- `d0`：距离尺度，默认 250 米。只在 distance 模式里使用

数字例子（`p0 = 0.9`，`d0 = 250`）：

| 距离 d | p_succ |
|---|---|
| 0 米 | 0.90 |
| 250 米 | 0.90 / e ≈ 0.331 |
| 500 米 | 0.90 × exp(-2) ≈ 0.122 |

`randomize_channel: true` 时，每个回合开始从 `[p0_low, p0_high]`（默认 0.3 到 1.0）里均匀抽一个新的 `p0`。perfect 模式不抽。评估脚本如果在 `options` 里指定了 `p0`，就用指定值，不再随机。这就是丢包率扫描能把横轴固定住的原因。

---

## 4. 飞机怎么动

每架飞机的状态是 `(x, y, v, heading)`。航向 0 表示朝 +x（向右）。

动作 `a = (a_v, a_h)`，两个数都被截断到 [-1, 1]：

```
v       <- clip(v + a_v * max_speed_delta, v_min, v_max)
heading <- wrap(heading + a_h * max_heading_delta)     # 折回 [-pi, pi]
x       <- x + v * cos(heading) * dt
y       <- y + v * sin(heading) * dt
```

先改速度和航向，再用**新的**速度和航向位移。默认 `dt = 1` 秒，`v` 在 0 到 6 米/秒，每步速度最多变 1.5，航向最多变 0.4 弧度（大约 23 度）。

飞出地图会被夹回边界，并给这架飞机一个小的出界惩罚，回合不因此结束。

障碍物是圆。`vx, vy` 都是 0 就是静止；否则每步移动，碰到边界就反向（弹回）。

碰撞（回合立刻失败）：

- 飞机中心到障碍中心的距离 < 障碍半径 + 飞机半径
- 或者两架飞机中心距离 < 2 × 飞机半径

成功（回合立刻结束，算赢）：

- 领队到目标的距离 ≤ `goal_radius`（默认 40 米）
- 并且每一架僚机到自己槽位的距离 ≤ `formation_tol`（默认 45 米）
- 并且本回合没有碰撞

槽位在领队机体坐标系里：x 向前，y 向左。默认 4 架僚机是两排 V 字：

```
僚机0: (-55, -40)    僚机1: (-55, +40)
僚机2: (-110, -80)   僚机3: (-110, +80)
```

世界坐标：

```
dx = ox * cos(h) - oy * sin(h)
dy = ox * sin(h) + oy * cos(h)
槽位 = 领队位置 + (dx, dy)
```

这里的领队位置，奖励用真实的，观测用僚机知道的那一份。

超时：步数到 `max_steps` 还没成功也没撞，算超时。

默认出生点大约在 `(200, 300)`，目标在 `(640, 300)`，静止圆障碍在 `(420, 300)`、半径 40 米，正好挡在直线上，领队得绕。

---

## 5. 观测向量（actor 能看见的）

每架飞机一条向量，长度一样。领队在下标 0，僚机接着排。默认只看最近 **1** 个障碍时，长度是 **23**。公式：`19 + 4 * n_nearest_obstacles`。

下面的“参考点”：僚机是它知道的领队，领队是目标点。

| 下标 | 含义 | 怎么归一化 |
|---|---|---|
| 0, 1 | 自己的 x, y | 除以地图宽、高 |
| 2 | 自己的速度 | 除以 v_max |
| 3, 4 | sin(航向), cos(航向) | 已经在 [-1, 1] |
| 5, 6 | 参考点相对自己的 x, y | 除以地图宽、高 |
| 7 | 参考速度。领队看目标时这一维是 0 | 除以 v_max |
| 8, 9 | 参考航向的 sin, cos。领队这里是“指向目标的方位角” | |
| 10 | AoI / aoi_cap。不该看 AoI 时写成 0 | |
| 11, 12 | 本机槽位偏移（机体坐标）。领队是 0 | 除以 formation_scale |
| 13, 14 | 指向“我现在认为该去的点”的相对向量 | 除以地图宽、高 |
| 15, 16, 17 | 最近另一架飞机的相对 x, y, 距离 | 距离除以地图对角线 |
| 18 | 是不是领队，1 或 0 | |
| 19 起，每 4 个数一个障碍 | 相对 x, 相对 y, 半径/对角线, 有没有这个障碍（1 或 0） | |

四种观测模式只改变“参考点”从哪来，**不改变向量长度**，所以同一个网络结构能加载不同模式的权重（前提是僚机数量、最近障碍个数没变）。

| 模式 | 参考点里的领队位置 | 第 10 维 AoI |
|---|---|---|
| `aoi` | 最后一次收到的位置、速度、航向 | 若 `use_aoi=true` 就写入，否则写 0 |
| `hold` | 最后一次收到的，不做预测 | 永远 0 |
| `extrapolate` | 用最后一次的速度和航向，沿直线外推 `AoI` 步：`pos + v * [cos h, sin h] * dt * AoI` | 永远 0 |
| `perfect` | 领队这一步的真实状态（上帝视角） | 永远 0 |

外推用的步数是**已经封顶**的 AoI。默认最多外推 30 步。

`use_aoi=false` 且模式仍是 `aoi` 时，僚机看到的东西和 `hold` 一样。`configs/ablation_no_aoi.yaml` 就是这个开关；`configs/masac_hold.yaml` 是同一种信息、单独命名的基线。两个都跑一遍，可以检查开关没有接错。

---

## 6. 奖励（每一架飞机各自一条）

这些权重都在 yaml 的 `env:` 下面，默认值是起点，不是论文定稿。

领队，记上一步到目标的距离为 `d_prev`，这一步为 `d`：

```
r = w_progress * (d_prev - d)
  + w_heading * cos(指向目标的方位角 - 自己的航向)
  - w_smooth * (a_v^2 + a_h^2)
  - w_step
```

僚机 i，记上一步槽位误差为 `e_prev`，这一步为 `e`（米，用真实领队算）：

```
r = w_form_progress * (e_prev - e)
  - w_formation * e
  + w_heading * cos(真实领队航向 - 自己的航向)
  + formation_bonus          （仅当 e ≤ formation_tol）
  - w_smooth * (a_v^2 + a_h^2)
  - w_step
```

然后所有飞机一起：

- 只要有任何碰撞，每架飞机再减 `collision_penalty`（默认 40）
- 如果这一步满足成功条件，每架飞机再加 `success_bonus`（默认 50）
- 被边界夹住的那一架再减 `oob_penalty`（默认 2）

默认权重：`w_progress = w_form_progress = 0.05`（每靠近 1 米大约 +0.05），`w_formation = 0.01`（误差 50 米大约 -0.5/步），`w_heading = 0.1`，`w_smooth = 0.02`，`w_step = 0.02`，在槽位里额外 +0.2。

碰撞惩罚是全队一起扣的，这样领队不会为了自己冲线把僚机丢掉。

---

## 7. 算法：谁在训练时能看见什么

这是 CTDE：训练时评论家（critic）是“中央教练”，执行时每架飞机的演员（actor）只看自己的观测。

**Critic 的全局状态**（`env.global_state()`，actor 拿不到）包括：

- 每架飞机的真实位置、速度、航向
- 目标点
- 每个障碍槽的位置、速度、半径（槽的个数由 `max_obstacles` 固定，不够就补 0）
- 每个僚机的 AoI，以及它最后收到的领队状态

也就是说，就算某个消融把 actor 的 AoI 维清零，**critic 仍然看得到 AoI**。这是中央教练的特权信息，要在论文里写清楚。如果你希望“去掉 AoI”连教练也看不见，现在的代码还没做这个开关，见文末路线图。

**Actor** 只吃第 5 节那条向量。每架飞机一个网络，不共享参数。

### 7.1 AoI-MASAC（主方法）

软演员-评论家（SAC）的多智能体版本，类在 `algos/masac.py`。

- 每架飞机一个高斯 actor，输出经过 tanh，动作落在 [-1, 1]。评估时用 tanh(均值)，不再采样。
- 每架飞机一对 Q 网络。输入是全局状态拼上**所有飞机的动作**。目标值用两个 Q 的较小者，再减去 `α * log π`。
- 每架飞机一个温度 α，用梯度自动调，目标熵是 `-动作维数 = -2`。
- 更新 actor 时，只把“我自己”的动作换成当前策略新采样的动作，队友的动作仍用回放池里的旧动作。这样梯度不会穿过队友的网络。
- 回合因成功或碰撞结束时，不把下一步的价值加回来（`terminated=1`）。只是超时（`truncated`）时，下一步价值仍然加上，因为超时不是环境说“这件事结束了”，只是我们停止了模拟。
- `ou_episodes`（默认 50）之内，采样动作上再叠加 OU 噪声，然后截断到 [-1, 1]。设成 0 就关掉。

`vanilla_masac`、`masac_hold`、`masac_extrapolate` 用的是**同一个类**。差别只在环境往观测里填什么，见各自的 yaml。

### 7.2 Vanilla MASAC（完美通信训练）

`configs/vanilla_masac.yaml`：信道 `perfect`，观测 `perfect`。它训练时从没见过丢包。

测试时若你把它放到有丢包的信道上，`evaluate.py` 和扫描脚本在 `--obs-mode auto`（默认）下会把观测改成 `hold`：网络没有上帝视角，只能吃过期的领队状态，也没有 AoI。这就是“按完美通信训练，放到差信道里考”的基线。

如果你故意要看“作弊的上帝视角”，评估时加 `--obs-mode perfect`。

丢包率为 0（`p0 = 1`）时不会切换，因为包其实都能到，perfect 和 hold 填出来的数一样。

### 7.3 MASAC-hold

在会丢包的信道上训练，观测用最后一次收到的领队状态，AoI 维是 0。它知道消息会旧，但不知道旧了多久。

### 7.4 MASAC-extrapolate

用来回答“为什么不直接预测”。丢包时按最后一次的速度和航向做匀速直线外推，AoI 维仍然是 0。这不是卡尔曼滤波，只是审稿人最容易提出的那个简单预测器。

### 7.5 MADDPG 和 MATD3

在 `algos/deterministic_ac.py`。Actor 是确定性的，输出 tanh。探索靠高斯噪声 `expl_noise`。

- MADDPG：一个 Q。每步都更新 actor。
- MATD3：两个 Q，目标取较小值；每 `policy_delay` 步才更新 actor（默认 2）；目标动作上加截断过的高斯噪声（target policy smoothing）。目标网络只在更新 actor 的那一步做软更新。

这两个基线的 yaml 用的是和 MASAC-hold 一样的观测：会丢包，但没有 AoI。这样比的是“同样的烂信道，有没有把年龄告诉策略”，而不是“一个看见真相、一个看不见”。

### 7.6 MAPPO

`algos/mappo.py` 会直接抛出 `NotImplementedError`。它是 on-policy 的，不能塞进现在的回放池。文件顶部写了以后该怎么做（去中心化 actor、中心化 V(全局状态)、GAE、PPO 裁剪）。**在实现并跑通之前，不要在论文表格里写 MAPPO 的数字。**

---

## 8. 消融开关（三件套）

| 你想关掉什么 | 配置文件 | 关键字段 |
|---|---|---|
| 不给 actor 看 AoI | `configs/ablation_no_aoi.yaml` | `obs_mode: aoi` 且 `use_aoi: false` |
| 训练时不随机信道 | `configs/ablation_no_randomize.yaml` | `randomize_channel: false`，`p0: 0.6` |
| 丢包不随距离变化 | `configs/ablation_independent_loss.yaml` | `channel_mode: independent`，`p0: 0.6` |

命令行也能临时改，不必另存 yaml（下面用 `python3`，和你本机的 `python` 是同一个的话可以换掉）：

```bash
python3 train.py --config configs/default.yaml --use-aoi 0
python3 train.py --config configs/default.yaml --randomize-channel 0 --p0 0.6
python3 train.py --config configs/default.yaml --channel-mode independent --p0 0.6
```

另外几个只影响训练循环、不改任务的开关：`--episodes`、`--max-steps`、`--warmup-steps`、`--start-steps`、`--batch-size`、`--hidden`、`--seeds`、`--device`、`--log-dir`。

动态障碍不是第四个消融，而是可选场景：`configs/moving_obstacles.yaml`。它把“最近障碍个数”改成 2，**观测长度会变**，不能拿默认配置训出来的权重直接加载。

---

## 9. 指标

定义以 `docs/METRICS.md` 为准，代码在 `metrics.py` 的 `summarize_episodes`。这里用白话再说一遍。

一个回合恰好是下面三种结局之一：成功、碰撞、超时。

- **成功率**：成功回合占多少。成功要同时到达目标、全队在槽位里、并且全程没撞。
- **碰撞率**：出现过碰撞的回合占多少。另外拆成障碍碰撞率和机间碰撞率，一个回合可以两个都算。
- **超时率**：走到步数上限、没成功也没撞的回合占多少。
- **编队保持率**：先在一个回合里算“有多少比例的步，**所有**僚机都在槽位容差内”，再对回合取平均。撞了的回合、超时的回合也算进去，不会只挑成功回合。
- **平均编队误差**：槽位误差，单位米。一步内先对僚机平均，再对步平均，再对回合平均。
- **平均 AoI / 最大 AoI**：抽完这一步的包之后的计数。平均 AoI 先在回合内对“步 × 僚机”平均，再对回合平均。最大 AoI 是该回合里最差的那一下，再对回合平均。
- **链路成功率**：收到的包 / 尝试的包。开局那一次免费同步不算。
- **平均中断长度**：连续丢包的段有多长（步）。从没丢过包就不贡献段；一整回合都没丢，这个数是 0。

汇总是**按回合宏平均**：一个 50 步的回合和一个 200 步的回合权重相同。标准差是样本标准差（除以 n-1）。只有 1 个回合时标准差写 0。

丢包率扫描图的横轴**不是** distance 模式里的 `p0`，而是把信道临时改成 independent 之后的名义丢包率 `loss_rate = 1 - p0`，从 0% 到 70%。图上另外有一条“仿真里真正收到包的比例”，它应该贴近 `1 - loss_rate`。训练用的是距离相关信道，测试用独立丢包，这是为了让横轴是一个数。论文里要写明这个区别。

---

## 10. 每个文件是干什么的

```
train.py                      训练入口。读 yaml，支持多种子，写 CSV / TensorBoard / 检查点
evaluate.py                   加载检查点，按指定信道评估
metrics.py                    指标怎么从一串回合汇总成一个数
requirements.txt              Python 依赖的最低版本
pytest.ini                    让测试能 import 到本仓库
LICENSE                       MIT

envs/channel.py               p_succ 的三个公式
envs/geometry.py              角度折返、机体坐标转世界坐标、默认 V 字槽位
envs/formation_env.py         环境本身。reset / step 的接口像 Gym，但不依赖 Gym

algos/networks.py             高斯 actor、确定性 actor、单 Q、双 Q
algos/buffer.py               回放池
algos/masac.py                AoI-MASAC 以及三个 MASAC 基线共用的更新
algos/deterministic_ac.py     MADDPG 和 MATD3
algos/maddpg.py               上面的别名，方便 from algos.maddpg import MADDPG
algos/matd3.py                同上
algos/mappo.py                占位，一构造就报错，并写明以后该怎么做
algos/factory.py              按名字构造算法
algos/common.py               软更新、OU 噪声

utils/config.py               yaml 合并。具体配置盖在 configs/default.yaml 上
utils/seeding.py              同时设定 random / numpy / torch 的种子
utils/logger.py               metrics.csv、updates.csv、TensorBoard
utils/rollout.py              评估时跑一个回合，可选记录轨迹

configs/default.yaml          CPU 上的小规模 AoI-MASAC，不是论文预算
configs/smoke.yaml            几秒钟的冒烟配置
configs/vanilla_masac.yaml    完美通信训练
configs/masac_hold.yaml       只保留旧值
configs/masac_extrapolate.yaml 直线外推
configs/maddpg.yaml
configs/matd3.yaml
configs/ablation_*.yaml       三个消融
configs/moving_obstacles.yaml 一个静止障碍 + 一个上下移动的障碍
configs/paper_*.yaml          论文规模（800 回合 × 最多 400 步，隐藏层 256）。还没在本机跑完

scripts/sweep_packet_loss.py  丢包率 0%→70%
scripts/plot_results.py       训练曲线、丢包率曲线、轨迹图（英文标注，方便贴进英文论文）
scripts/run_key_experiment.sh 扫描然后画图
scripts/train_paper.sh        论文规模的一串训练。非常久，别拿来冒烟

tests/                        环境、指标、每种算法的短训练
docs/METRICS.md               指标的正式定义
```

训练一个回合会在运行目录里留下：

- `config.yaml`：这次真正用到的全部配置
- `metrics.csv`：每个回合一行
- `updates.csv`：隔若干步记一次损失
- `eval_metrics.csv`：训练中途的固定信道小评估（`eval_every > 0` 才有）
- `tb/`：TensorBoard
- `ckpt_epXXXX.pt` 和 `ckpt_final.pt`
- `speed.json`：这一次在本机测到的每回合、每步、每次更新耗时

`runs/` 和 `results/` 在 `.gitignore` 里，不会被提交。论文数字必须能从检查点和种子重跑出来，不要只保存一张图。

---

## 11. 安装

需要 Python 3.10 或更高。下面的命令在 Ubuntu 上用 `python3`。如果你的环境里 `python` 就是 3.10+，把 `python3` 换成 `python` 即可。

```bash
cd AoI-MASAC-Formation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

有 NVIDIA GPU 时，不要装 CPU 版 torch。先按 PyTorch 官网的 CUDA 版本装 torch，再 `pip install -r requirements.txt`。代码里 `device: auto` 会在有 CUDA 时用 GPU。

检查安装：

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -m pytest -q
```

---

## 12. 怎么跑

所有命令都在仓库根目录执行。

### 12.1 先确认没写坏（大约一分钟，见第 13 节的实测）

```bash
python3 -m pytest -q
python3 train.py --config configs/smoke.yaml --device cpu
```

`smoke.yaml` 只有 2 架僚机、每回合最多 40 步、一共 3 个回合、网络很窄。它只证明“能走完训练并写出检查点”。

### 12.2 CPU 上的小规模调试（默认配置）

```bash
python3 train.py --config configs/default.yaml --seed 0 --device cpu
```

这是 4 架僚机、每回合最多 200 步、30 个回合、隐藏层 128、会随机 `p0`。仍然不是论文实验。多种子：

```bash
python3 train.py --config configs/default.yaml --seeds 0,1,2 --device cpu
```

### 12.3 每个对比算法

```bash
python3 train.py --config configs/vanilla_masac.yaml --seeds 0,1,2
python3 train.py --config configs/masac_hold.yaml --seeds 0,1,2
python3 train.py --config configs/masac_extrapolate.yaml --seeds 0,1,2
python3 train.py --config configs/maddpg.yaml --seeds 0,1,2
python3 train.py --config configs/matd3.yaml --seeds 0,1,2
```

消融把第 8 节的 yaml 换上去即可。动态障碍：

```bash
python3 train.py --config configs/moving_obstacles.yaml --seed 0
```

### 12.4 评估某一个检查点

```bash
python3 evaluate.py \
  --checkpoint runs/<某次>/ckpt_final.pt \
  --episodes 20 \
  --channel-mode independent \
  --p0 0.5 \
  --out results/eval_one \
  --save-traj results/eval_one/traj.npz
```

`--p0 0.5` 在 independent 模式下就是名义丢包率 50%。`--obs-mode auto` 是默认：完美通信训练出来的策略会被改成 hold。距离相关信道也可以测，例如 `--channel-mode distance --p0 0.6`，但这时横轴不该再叫“丢包率 40%”，因为真实丢包还和距离有关，要看输出里的“链路成功率”。

### 12.5 论文里那张核心图：丢包率从 0% 到 70%

```bash
bash scripts/run_key_experiment.sh runs/<某次>/ckpt_final.pt 30 results/sweep_aoi
```

第二个数是每个丢包率上的评估回合数。冒烟时可以写 2；论文图建议至少 50，并且检查点来自 `configs/paper_*.yaml` 那种长训练。脚本会调用：

```bash
python3 scripts/sweep_packet_loss.py --checkpoint ... --episodes 30 --out results/sweep_aoi
python3 scripts/plot_results.py sweep --csv results/sweep_aoi/sweep.csv --out results/sweep_aoi/figs
```

每个算法各训一个检查点，就各扫一次，得到多条曲线。这版绘图脚本一次画一个 CSV。要把多条曲线叠在一张图上，需要你后面加一个对比脚本（路线图里有）。

### 12.6 论文规模训练（很久，先看第 13 节再决定跑不跑）

```bash
DEVICE=cpu SEEDS=0 bash scripts/train_paper.sh
```

默认种子是 `0,1,2`，六个算法，每个 800 回合、每回合最多 400 步。**先用短跑测出每回合秒数，再乘这个预算**，不要直接开跑。MAPPO 不在这个脚本里。

### 12.7 看 TensorBoard

```bash
python3 -m tensorboard.main --logdir runs --port 6006
```

CSV 永远是画图用的正式记录。TensorBoard 打不开时，训练不会停。

---

## 13. 在这台机器上实测到的速度

本节只记录**已经跑完**的命令。没有的配置就不要脑补。

机器：本仓库的云端开发环境，Linux，**4 个 CPU**，没有 GPU（`torch.cuda.is_available()` 为 false）。PyTorch 是 CPU 轮子 `2.14.1+cpu`。耗时含环境模拟和梯度更新，写在每次运行目录的 `speed.json` 里。

### 13.1 测试套件

命令：`python3 -m pytest -q`

2026-10-20 在这台机器上：**24 项测试全部通过，墙钟时间 10.9 秒**。其中每种可训练算法（AoI-MASAC、vanilla、hold、extrapolate、MADDPG、MATD3）各跑 4 个回合、每回合最多 50 步（2 架僚机，隐藏层 32，批大小 32）。碰撞会让回合提前结束，所以实际步数可以少于 200；测试要求至少 120 步，并且最后一回合的 critic loss 是有限数。MAPPO 被断言为“一构造就报未实现”。

### 13.2 smoke 配置

命令：

```bash
python3 train.py --config configs/smoke.yaml --device cpu --log-dir runs/measured_smoke
```

AoI-MASAC，2 架僚机，隐藏层 32，批大小 32，3 个回合、每回合 40 步，固定 `p0=0.8`。`speed.json`：

| 项 | 测到的值 |
|---|---|
| 总步数 / 参数更新次数 | 120 / 89 |
| 平均每回合 | 0.288 秒 |
| 平均每步（含还没开始更新的步） | 0.0072 秒 |
| 平均每次梯度更新 | 0.0088 秒 |
| 总墙钟 | 0.86 秒 |

控制台里三个回合分别是 0.10 秒、0.33 秒、0.44 秒。第一回合几乎还在热身，所以比后面慢的更新回合短。成功率是 0。这只说明训练循环能走完。

### 13.3 默认宽度的短跑（隐藏层 128，批大小 128，5 架飞机）

命令把热身改短了，否则 5 个短回合填不满默认的 1000 步热身，测到的会只是“环境空转”：

```bash
python3 train.py --config configs/default.yaml --algo aoi_masac \
  --episodes 5 --max-steps 80 --warmup-steps 40 --start-steps 20 \
  --device cpu --log-dir runs/measured_short_aoi

python3 train.py --config configs/maddpg.yaml \
  --episodes 5 --max-steps 80 --warmup-steps 40 --start-steps 20 \
  --device cpu --log-dir runs/measured_short_maddpg
```

| | AoI-MASAC | MADDPG |
|---|---|---|
| 总步数 / 更新次数 | 294 / 167 | 334 / 207 |
| 平均每回合（含热身回合） | 0.696 秒 | 0.488 秒 |
| 平均每步（含热身） | 0.0118 秒 | 0.0073 秒 |
| 平均每次梯度更新 | 0.0192 秒 | 0.0110 秒 |
| 总墙钟 | 3.48 秒 | 2.44 秒 |

“平均每回合”被前两个几乎没有更新的回合拉低了，不要拿 0.7 秒去乘 800。看已经在更新的回合更靠谱：AoI-MASAC 最后一回合 80 步用了 1.58 秒（约 0.020 秒/步）；MADDPG 最后一回合 80 步用了 0.89 秒（约 0.011 秒/步）。这和上表的“平均每次更新”对得上，因为热身之后每步更新一次。

这两次的成功率都是 0，编队保持率掉到接近 0。随机策略会撞，回合经常不到 80 步就结束。**不要把这里的成功率和碰撞率写进论文。**

### 13.4 论文网络宽度的短跑（隐藏层 256，批大小 256）

只计了 AoI-MASAC。MADDPG / MATD3 没有按这个宽度单独计时。

```bash
python3 train.py --config configs/paper_aoi_masac.yaml \
  --episodes 12 --max-steps 80 --warmup-steps 256 --start-steps 10 \
  --batch-size 256 --device cpu --log-dir runs/measured_h256_b256
```

`speed.json`：387 步，132 次更新，平均每次更新 **0.0299 秒**，总墙钟 4.41 秒。已经在更新的最后一回合是 80 步、2.49 秒。没有更新时，环境本身大约 0.001 秒/步甚至更少（同一配置下，更新开始前的回合只有 0.01–0.07 秒）。

同一个宽度、批大小改成 128 时，34 次更新的平均是 0.0285 秒/次。批大小从 128 加到 256，在这台 CPU 上几乎没有变慢，时间主要花在宽度 256 的网络上。样本只有几十到一百次更新，当个数量级看，不要写成精确基准。

若一回合真的走满论文配置的 400 步，并且每步更新一次，用 0.030 秒/次来乘：一回合约 12 秒，800 回合约 2.7 小时/种子。这是**用短跑的更新耗时做的乘法，不是把 800 回合跑完了**。未训练时碰撞多，回合更短，墙钟会小于这个数；等策略学会少撞、回合变长，就会靠近这个上限。6 个算法 × 3 个种子若都按满步数估，是十几个小时量级的 CPU 时间。GPU 没有测过，不要自己乘一个加速比写进计划里当事实。

`configs/paper_*.yaml` 里还有 `eval_every: 25` 的评估。评估只做前向、不反传，按上面“环境约 0.001 秒/步”来看，它比更新便宜，但论文跑的时候它会算在总时间里。

---

## 14. 怎么把图画出来

图的文字是英文，方便贴进 CCGrid / WoWMoM。同时写 png（300 dpi）和 pdf。

训练曲线（回报、成功、碰撞、编队保持率、编队误差、平均 AoI、链路成功率）：

```bash
python3 scripts/plot_results.py training \
  --csv runs/<某次>/metrics.csv \
  --out figs/train \
  --window 5
```

丢包率曲线：见 12.5。轨迹：

```bash
python3 scripts/plot_results.py trajectory \
  --npz results/eval_one/traj.npz \
  --out figs/traj
```

轨迹图里，圆点是起点，叉是终点，星是目标，灰圆是障碍（障碍在动的话会隔几帧再画一个虚线圆）。领队是黑色，僚机是其他颜色。

短跑画出来的曲线**不会**好看，也不该出现在论文里。它们只说明画图脚本读得懂 CSV。

---

## 15. 论文还要做什么（路线图）

按你自己的时间表（截稿 2026-12-01，CCGrid 摘要 2026-11-24）。下面是代码和实验上还没做完的事，不是日历承诺。

1. 把方案第五节里标了星号的论文打开读完，再引用。没读过原文的不要只凭调研笔记引用。
2. 调奖励权重和 `formation_tol`。现在的权重能让代码学习，不保证编队保持率够写论文。建议先固定种子，只改一项奖励，看 `metrics.csv`。
3. 用 `configs/paper_*.yaml` 把 AoI-MASAC、vanilla、hold、extrapolate、MADDPG、MATD3 各跑至少 3 个种子。确认 GPU 时间后再加到 5 个种子。
4. 每个最终检查点跑 `scripts/run_key_experiment.sh`，回合数建议 ≥ 50。这是全文核心图。
5. 三个消融各跑同样的种子和同样的扫描。
6. 加一组动态障碍（`moving_obstacles.yaml`）作为附加实验，不要和主图混在同一条曲线里，因为观测长度不同。
7. 训练曲线要画多种子的均值和标准差。现在的绘图脚本一次只吃一个 CSV，多种子叠加还没写。
8. 可选：实现 MAPPO（见 `algos/mappo.py` 顶部的步骤），或者在论文里明确写“MAPPO 留给扩展”。不要伪造。
9. 可选：再加一个“critic 也看不到 AoI”的开关，堵住“你只是让教练看见年龄”这种质疑。当前消融只清零 actor 的那一维。
10. 可选：把外推基线换成常速度卡尔曼滤波。现在的直线外推是最低限度的回答。
11. 投稿前用知网和 Google Scholar 再查一次 “AoI + formation + MARL”。调研笔记说这个组合当时没查到，但不能保证之后没有新论文。
12. 本科工作如果已经公开（含学位论文上网），新论文里引用它，并写清改了信道、AoI、障碍观测、机间碰撞和集中式动作 critic。图和句子不要从旧文照搬。
13. WoWMoM 和 CCGrid 只能投一个。和导师确认署名、单位和是否要组内审批之后再传。

---

## 16. 已知限制

- 运动学是平面上的“速度 + 航向”，不是真实四旋翼的动力学，没有风、没有姿态、没有能耗。
- 只有领队→僚机这一条链路会丢包。机间相对位置和障碍是本机传感器，默认可靠。
- 默认场景是一个静止圆障碍。动态障碍要换配置，而且最近障碍个数一变，旧检查点就不能加载。
- 网络是普通的多层感知机，没有注意力，僚机之间不共享参数。飞机一多，参数量线性涨。
- 地图大小、飞机数量写进了网络的输入维度。改 `n_followers`、`n_nearest_obstacles` 或 `max_obstacles` 之后必须重新训练。
- 奖励还没系统调过。短跑里的成功率和碰撞率没有科学意义。
- 丢包率扫描为了有单一横轴，用的是与距离无关的丢包，和训练分布不完全一样。
- MAPPO 没有实现。
- 没有在 GPU 上跑过。没有多种子的论文图。
- 评估时的 `--obs-mode auto` 只对“训练时是 perfect”的检查点做 hold 切换。别的模式保持训练时的观测。

---

## 17. 学术上不要做的事

- 不要把 smoke、short run、或者只跑了一个种子的曲线放进论文，也不要只挑好看的种子。
- 不要编造 MAPPO 或任何还没跑的算法的数字。
- 不要同时把同一篇投给 WoWMoM 和 CCGrid。
- 每个表格里的数都要能用检查点、配置文件和种子重做出来。

---

## 18. English summary

This repository is a first runnable version of an AoI-aware multi-agent Soft Actor-Critic for 2D leader–follower UAV formation under a lossy downlink. One leader and several followers fly a unicycle kinematic model on a 700×600 map. Each follower receives the leader state with success probability `p_succ = p0 * exp(-d / d0)`. On a loss it reuses the last packet and increments an AoI counter (capped, then normalized into the actor observation). Training randomizes `p0` per episode. Actors are decentralized; critics are centralized and condition on the privileged global state plus every agent’s action. Obstacle geometry and the nearest teammate are treated as onboard sensing, not as traffic on the lossy link.

The same trainer covers four observation builders, selected from YAML: AoI-MASAC, vanilla MASAC trained with a perfect link (deployed at test time with a hold-last-packet observation), MASAC-hold, and MASAC with linear extrapolation. MADDPG and MATD3 are implemented as hold-observation baselines. MAPPO is an explicit `NotImplementedError` stub. Ablations turn off the AoI feature, turn off channel randomization, or make packet loss independent of distance.

Metrics (success, collision split into obstacle vs inter-UAV, timeout, formation-keeping rate, formation error, AoI, packet success, outage length) are defined in `docs/METRICS.md` and `metrics.py`. The key figure sweeps a nominal independent packet-loss rate from 0% to 70%.

Reward weights are an untuned starting point. The private undergraduate MASAC repository was not readable from this environment, so this code is a reimplementation of the written spec, not a copy. Numbers in Section 13 are smoke tests and short CPU runs on a 4-core machine with no GPU. They are not paper results. Do not report MAPPO numbers from this repository.

---

## 19. 这次实际跑了什么

全部在第 13 节那台 4 核 CPU、无 GPU 的机器上，2026-10-20。

1. `python3 -m pytest -q`：24 项通过，10.9 秒。
2. `configs/smoke.yaml` 的 AoI-MASAC：3 回合 × 40 步，写出检查点。成功率 0。
3. 默认宽度（隐藏层 128，批大小 128）的 AoI-MASAC 和 MADDPG：各 5 回合、每回合最多 80 步。成功率 0。
4. 论文宽度（隐藏层 256，批大小 256）的 AoI-MASAC：12 回合、每回合最多 80 步，132 次更新。成功率 0。用来看一次更新要多久，不是用来看会不会飞。
5. 用第 3 步那个**没训练好**的 AoI-MASAC 检查点做了一次评估和一次很小的扫描，只为确认脚本能出图：
   - `evaluate.py --episodes 3 --channel-mode independent --p0 0.6`：成功率 0，碰撞率 1，编队保持率 0.399，平均 AoI 0.587，实测链路成功率 0.629。3 个回合全部撞在障碍上。
   - 扫描名义丢包率 0%、30%、70%，每个点只评 2 个回合：实测链路成功率分别是 1.000、0.739、0.310（2 个回合的抽样，所以不会刚好等于 1、0.7、0.3）。三个点的成功率都是 0。
   - `scripts/plot_results.py` 写出了训练曲线、丢包率曲线和一张轨迹图。图上能看到地图、挡在路中间的圆障碍、右侧目标和五条轨迹。策略会往障碍里飞，这是没训练够的表现。

这些数都**不是**论文结果。许可证：MIT，见 `LICENSE`。
