# 杂乱空域里的信息年龄编队：AoI-MASAC

论文题目：

**AoI-Aware Multi-Agent Soft Actor-Critic for Robust UAV Formation and Obstacle Avoidance under Lossy Links**

中文：在会丢包、还会被建筑物挡住的链路下，用能感知信息年龄的多智能体软演员-评论家做无人机编队和避障。

目标会议是 CCGrid 2027 或 WoWMoM 2027，两个只能投一个。按目前的计划，全文截稿是 2026-12-01（AoE），CCGrid 还要在 2026-11-24 交摘要。

这个仓库是这条研究线的可运行代码。奖励还没有调完，下面凡是出现数字的地方，都是这台机器上的冒烟或短跑，**不能写进论文**。

---

## 1. 这篇论文到底在问什么

近年的多无人机编队强化学习，大多默认一件事：僚机每一步都能拿到领队**现在**的位置、速度和航向。代表工作包括带动态障碍的编队（IROS 2025 的编队避障、2025 的 MASAC-CA 这一类）。它们把力气花在注意力、课程学习、奖励搜索上，通信被当成是好的。

另有一小支工作承认通信不完美，但场景很窄：Sensors 2023 的 delay-informed formation 用的是**固定时延**，而且没有障碍；2024–2026 的时延车队控制是在**一条直线**上跟车。OJVT 2025 的综述把这个空白说得很直：多无人机强化学习里，真实信道很少被放进环境。

本文只改这一处，但把环境做成这些工作正在用的那种空域，而不是一块空地：

领队要带楔形编队穿过一片**既有楼、又有横穿交通**的区域，沿 4 个航路点飞。僚机靠一条下行链路听领队。这条链路有两件事会让包到不了：

1. 距离越远，成功概率越低（指数衰减）。
2. 领队和僚机之间的直线如果穿过建筑物圆盘，再乘一个明显更小的非视距系数。也就是说，**挡路的楼同时挡无线电**。包丢了，僚机手里还是旧的领队状态，一个叫 AoI 的计数加 1。策略看得到这个计数。

要回答的问题：消息旧了的时候，编队该飞得保守一点，还是照着过期槽位冲。冲的话，队形散、也更容易在楼群里撞上。靠近领队，链路又会变好。这个行为应该是学出来的，不是我们在奖励里直接写“请减小 AoI”。

打个比方：以前僚机戴着实时耳机。现在耳机在楼后会断，而且飞得越远断得越勤。我们让它知道“这句话是几步之前的”，自己决定还敢不敢跟。

正式写相关工作之前，请把上面提到的论文打开核对。这里的定位是按题目和公开摘要来的，不是替代你自己的阅读。

---

## 2. 场景里有什么

地图 **1200 m × 800 m**。默认 1 架领队 + 4 架僚机。僚机数可以改。

航路点（米），按顺序飞：

| 序号 | 坐标 |
|---|---|
| 1 | (280, 420) |
| 2 | (520, 560) |
| 3 | (800, 320) |
| 4 | (1080, 500) |

领队从大约 (150, 420) 出发，朝 +x。进入航路点周围 `wp_radius`（默认 30 m）就算捕获，然后转向下一个点。

障碍（定义在 `envs/scenario.py`，默认配置和测试用的是同一份）：

- 8 个静止圆盘，当成建筑物。其中三个正好压在航路点之间的连线上，所以直线既会撞、也没有视距。
- 3 个会动的圆盘，速度大约 4–8 m/s，碰到边界就反向。这是近年编队避障论文里的“静态 + 动态障碍”，不是可有可无的开关。
- `configs/moving_obstacles.yaml` 再加一架更快的横穿目标，用来做更挤的附加实验。

风：每个回合抽一次，风速 0 到 `wind_max`（默认 4 m/s），方向均匀。风会推动地速。Actor 只能看到带噪声的风速估计（模拟空速管/风估计），Critic 在训练时看真实风。

探测距离 `sensor_range` 默认 250 m。超出这个距离的障碍和队友，Actor 看不到。领队状态不走这个传感器，走下面的链路。

---

## 3. 飞机怎么动

这是外环控制：假定里面已经有一个飞控，强化学习只发纵向加速度和偏航角速度。近年的无人机强化学习大多是这么做的，而不是让策略直接瞬移速度和航向。

状态是地速坐标系里的 `(x, y, v, heading)`。航向 0 是 +x。动作 `(a_v, a_ω)` 都在 [-1, 1]：

```
加速度   = a_v * max_accel          默认 max_accel = 6 m/s²
偏航速率 = a_ω * max_yaw_rate       默认 max_yaw_rate = 1.2 rad/s
v        ← clip(v + 加速度 * dt, 0, 16 m/s)
heading  ← wrap(heading + 偏航速率 * dt)
x        ← x + (v * cos(heading) + wind_x) * dt
y        ← y + (v * sin(heading) + wind_y) * dt
```

默认 `dt = 0.2` s（外环 5 Hz）。以 16 m/s 飞、偏航顶满时，转弯半径大约 `v / yaw_rate ≈ 13` m，和建筑物的尺度是可比的，不能原地甩头绕过去。

机身半径 4 m。两机中心距离小于 8 m 算互撞。飞出地图会被夹回来，并给该机一个小惩罚，回合不因此结束。

编队是领队机体坐标下的楔形，间距 `formation_spacing = 32` m（+x 朝前，+y 朝左）：

```
第 1 排: (-32, -24) 和 (-32, +24)
第 2 排: (-64, -48) 和 (-64, +48)
```

世界坐标：

```
dx = ox * cos(h) - oy * sin(h)
dy = ox * sin(h) + oy * cos(h)
槽位 = 领队位置 + (dx, dy)
```

奖励用**真实**领队位姿算槽位。观测里的槽位用这架僚机**被允许知道**的领队位姿。这两个可以不一样，这就是任务。

---

## 4. 信道和 AoI

每一架僚机、每一步，独立决定这个包到不到。先看几何，再看概率。

建筑物遮挡：领队位置和僚机位置的线段如果碰到某个障碍圆盘（半径再加上 `blockage_margin`，默认 0），这一跳就是非视距（NLOS）。

```
distance 模式（主模型）:
    p = p0 * exp(-d / d0)
    若这一跳是 NLOS 且 use_blockage 为真:  p ← p * nlos_factor

independent 模式（消融，也是扫描图的横轴）:
    p = p0          距离和楼都不看

perfect 模式:
    p = 1
```

默认 `p0 = 0.95`，`d0 = 200` m，`nlos_factor = 0.2`。数字例子：

| 情况 | 成功概率 |
|---|---|
| 贴着领队、视距 | 0.95 |
| 200 m、视距 | 0.95 / e ≈ 0.35 |
| 200 m、被楼挡住 | 0.35 × 0.2 ≈ 0.07 |
| 400 m、视距 | 0.95 × exp(-2) ≈ 0.13 |

`randomize_channel: true` 时，每个回合从 `[0.45, 1.0]` 里重新抽 `p0`。perfect 模式不抽。评估时在选项里写死 `p0` 就不再随机，扫描脚本靠这个把横轴固定住。

AoI（信息年龄）：这份领队状态已经多少步没被新包刷新。

- 回合开始免费同步一次，AoI = 0。这一次**不计入**链路成功率。
- 收到包：写成移动之后的领队位置、速度、航向，AoI 归零。
- 没收到：三个量保持上次的值，AoI + 1，最大 `aoi_cap`（默认 40 步，也就是 8 秒）。
- 送给网络的是 `AoI / aoi_cap`。

外推基线用的“过了几步”也是封顶之后的 AoI。

---

## 5. 观测

每架飞机一条向量，长度相同。领队在下标 0。默认看最近 3 个**探测距离内**的障碍时，长度是 `23 + 6 × 3 = 41`。

| 下标 | 含义 |
|---|---|
| 0, 1 | 自己的 x, y，除以地图宽、高 |
| 2 | 自己的速度 / v_max |
| 3, 4 | sin(航向), cos(航向) |
| 5, 6 | 参考点相对自己的位置。僚机的参考点是它知道的领队；领队的参考点是当前航路点 |
| 7 | 参考速度 / v_max。领队看航路点时这一维是 0 |
| 8, 9 | 参考航向的 sin, cos。领队这里是指向当前航路点的方位角 |
| 10 | AoI / aoi_cap。不该看 AoI 时写成 0 |
| 11, 12 | 本机槽位偏移 / formation_scale。领队是 0 |
| 13, 14 | 指向“我现在认为该去的点”。僚机是用已知领队位姿算出来的槽位；领队是当前航路点 |
| 15 | 已完成航路点的比例。只有领队有；僚机写 0，因为这是领队的任务，不能白给 |
| 16, 17 | 带噪声的风 / wind_max |
| 18–21 | 探测距离内最近一架其他飞机的相对位置、距离、有没有看到 |
| 22 | 是不是领队 |
| 23 起，每 6 个数一个障碍 | 相对 x、y，障碍速度 vx、vy，半径，以及“在不在探测距离内” |

四种观测模式只改“领队那一段从哪来”，不改向量长度：

| 模式 | 领队位置从哪来 | 第 10 维 |
|---|---|---|
| `aoi` | 最后一次收到的位置、速度、航向 | `use_aoi` 为真才写入 |
| `hold` | 最后一次收到的，不预测 | 永远 0 |
| `extrapolate` | `pos + v · [cos h, sin h] · dt · AoI` | 永远 0 |
| `perfect` | 这一步的真实领队 | 永远 0 |

`use_aoi: false` 且模式仍是 `aoi` 时，僚机看到的和 `hold` 一样。`configs/ablation_no_aoi.yaml` 就是这个开关。

---

## 6. 奖励

权重在 yaml 里，是按 0.2 秒步长估的起点，不是定稿。

领队，上一步到**当前航路点**的距离是 `d_prev`，这一步是 `d`：

```
r = w_progress * (d_prev - d)
  + w_heading * cos(指向航路点的方位角 - 自己的航向)
  - w_smooth * (a_v² + a_ω²)
  - w_step
```

僚机 i，槽位误差（米，真实领队）从上一步的 `e_prev` 变成 `e`：

```
r = w_form_progress * (e_prev - e)
  - w_formation * e
  + w_heading * cos(真实领队航向 - 自己的航向)
  + formation_bonus          仅当 e ≤ formation_tol（默认 18 m）
  - w_smooth * (a_v² + a_ω²)
  - w_step
```

然后：

- 每架飞机如果离最近障碍的表面间隙小于 `clearance_margin`（20 m），按缺口比例扣 `w_clearance`。这是密集的避障信号，不是控制屏障函数。
- 每捕获一个航路点，全队加 `wp_bonus`。
- 只要有碰撞，全队减 `collision_penalty`。
- 航路走完且全队在容差内且没撞，全队加 `success_bonus`。
- 被边界夹住的那一架再减 `oob_penalty`。

默认大致是：每靠近航路点或槽位 1 米，`+0.25`；槽位误差每米 `-0.02`；碰撞 `-30`；完成全部航路且编队合格 `+40`；每个航路点 `+8`。

奖励里**没有**“AoI 越小越好”这一项。链路变好应该是飞得更合理之后的结果，不是优化目标本身。

---

## 7. 算法

训练时 Critic 是中央教练：看每架飞机的真实位姿、真实风、全部航路点、全部障碍、以及每个僚机的 AoI、最后收到的领队状态、这一跳是不是被楼挡住。执行时 Actor 只看第 5 节那条向量。

每架飞机一个网络，不共享参数。

**AoI-MASAC**（`algos/masac.py`）：高斯 Actor，tanh 到 [-1, 1]；每架飞机一对 Q，输入是全局状态拼上所有飞机的动作；目标用两个 Q 的较小值再减去 `α log π`；α 自动调，目标熵是 `-2`。更新某个 Actor 时，只替换它自己的动作，队友动作用回放池里的。成功或碰撞结束时不自举；仅仅因为步数上限停下时仍然自举。

`vanilla_masac`、`masac_hold`、`masac_extrapolate` 是同一个训练器，差别只在观测。

- **Vanilla**：用 `perfect` 链路训练。放到有丢包的链路上评估时，`--obs-mode auto` 会改成 `hold`，不把真实领队偷偷塞给它。想看上帝视角就显式写 `--obs-mode perfect`。丢包率为 0 时不切换，因为包其实都能到。
- **Hold**：在真实链路上训练，只保留旧值，没有 AoI。
- **Extrapolate**：丢包时按上次速度和航向做匀速直线外推，仍然不给 AoI。用来回答“为什么不直接预测”。它不是卡尔曼滤波。
- **MADDPG / MATD3**：确定性 Actor，观测和 Hold 一样，这样比的是同一条烂链路下有没有把年龄告诉策略。MATD3 用双 Q、延迟 Actor 更新、目标动作平滑。
- **MAPPO**：`algos/mappo.py` 会直接报未实现。不要在表格里写它的数字。文件顶部写了以后该怎么接（去中心化 Actor、中心化 V、GAE、PPO 裁剪）。

探索靠 SAC 的熵，以及 MADDPG/MATD3 的高斯噪声。`ou_episodes` 默认是 0。

---

## 8. 消融

| 关掉什么 | 配置 | 关键字段 |
|---|---|---|
| Actor 不看 AoI | `configs/ablation_no_aoi.yaml` | `use_aoi: false` |
| 训练时不随机 p0 | `configs/ablation_no_randomize.yaml` | `randomize_channel: false`，`p0: 0.6` |
| 丢包不看距离、也不看楼 | `configs/ablation_independent_loss.yaml` | `channel_mode: independent` |
| 楼只挡飞机、不挡无线电 | `configs/ablation_no_blockage.yaml` | `use_blockage: false` |

```bash
python3 train.py --config configs/default.yaml --use-aoi 0
python3 train.py --config configs/default.yaml --randomize-channel 0 --p0 0.6
python3 train.py --config configs/default.yaml --channel-mode independent --p0 0.6
```

还有只影响训练循环的开关：`--episodes`、`--max-steps`、`--warmup-steps`、`--start-steps`、`--batch-size`、`--hidden`、`--seeds`、`--device`、`--log-dir`。

改 `n_followers`、`n_nearest_obstacles`、`max_obstacles`、`max_waypoints` 会改变网络输入长度，旧检查点不能加载。

---

## 9. 指标

正式定义在 `docs/METRICS.md`。一个回合恰好是成功、碰撞、超时三者之一。

- **成功**：四个航路点都捕获了，结束时每个僚机离真实槽位不超过 18 m，并且全程没撞。
- **碰撞率**：分障碍碰撞和机间碰撞，一个回合可以两个都算。
- **航路点完成比例**：捕获数 / 4。撞了或超时也保留它飞到哪。
- **编队保持率**：有多少比例的步，**所有**僚机同时在容差内。失败回合也算。
- **平均编队误差**：米。
- **平均 AoI、最大 AoI、链路成功率、平均中断长度**：口径见 METRICS。开局那次免费同步不算进链路成功率。
- **NLOS 比例**：有多少比例的发包尝试，线段在几何上穿过了楼。这是路径的几何，不等于“遮挡开关有没有把概率乘小”。
- **最小间隙**：飞机表面到障碍表面的最近距离，撞进去是负数。

汇总是按回合宏平均。

丢包率扫描把信道临时改成 independent，横轴是 `loss_rate = 1 - p0`，从 0% 到 70%。训练用的是“距离 × 遮挡”。论文里要写明测试分布和训练分布不是同一个，横轴这样选是为了能画成一条曲线。

---

## 10. 文件

```
train.py  evaluate.py  metrics.py  requirements.txt  pytest.ini  LICENSE
envs/scenario.py        默认航路点和楼、横穿目标的坐标
envs/channel.py         距离衰减 × NLOS 系数
envs/geometry.py        楔形槽位、线段和圆盘是否相交
envs/formation_env.py   环境
algos/masac.py          AoI-MASAC 和三个 MASAC 基线
algos/deterministic_ac.py   MADDPG、MATD3
algos/mappo.py          未实现，调用即报错
configs/default.yaml    杂乱空域，CPU 上的中等预算
configs/smoke.yaml      空地、两架僚机，只用来确认能跑
configs/paper_*.yaml    800 回合 × 最多 500 步，隐藏层 256
configs/ablation_*.yaml 四个消融，含“楼不挡无线电”
scripts/sweep_packet_loss.py
scripts/plot_results.py 训练曲线、丢包率曲线、带着航路点的轨迹
docs/METRICS.md
```

每次训练在运行目录写 `config.yaml`、`metrics.csv`、`updates.csv`、`eval_metrics.csv`、TensorBoard、检查点和 `speed.json`。`runs/` 不进 git。

---

## 11. 安装和运行

Python 3.10 或更高。下面用 `python3`。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest -q
python3 train.py --config configs/smoke.yaml --device cpu
```

有 GPU 时先装对应 CUDA 版的 torch，配置里 `device: auto` 会用它。本仓库没有在 GPU 上跑过。

默认场景的调试训练（仍然不是论文预算）：

```bash
python3 train.py --config configs/default.yaml --seed 0 --device cpu
python3 train.py --config configs/vanilla_masac.yaml --seeds 0,1,2
python3 train.py --config configs/masac_hold.yaml --seeds 0,1,2
python3 train.py --config configs/masac_extrapolate.yaml --seeds 0,1,2
python3 train.py --config configs/maddpg.yaml --seeds 0,1,2
python3 train.py --config configs/matd3.yaml --seeds 0,1,2
```

评估和核心图：

```bash
python3 evaluate.py \
  --checkpoint runs/<某次>/ckpt_final.pt \
  --episodes 20 \
  --channel-mode independent --p0 0.5 \
  --out results/eval_one \
  --save-traj results/eval_one/traj.npz

bash scripts/run_key_experiment.sh runs/<某次>/ckpt_final.pt 30 results/sweep_aoi
```

论文规模（先用第 12 节的秒数乘一遍再决定）：

```bash
DEVICE=cpu SEEDS=0 bash scripts/train_paper.sh
```

TensorBoard：`python3 -m tensorboard.main --logdir runs --port 6006`。

画图（英文标注，png + pdf）：

```bash
python3 scripts/plot_results.py training --csv runs/<某次>/metrics.csv --out figs/train
python3 scripts/plot_results.py trajectory --npz results/eval_one/traj.npz --out figs/traj
```

---

## 12. 这台机器上重新测过的速度

2026-10-20，4 核 CPU，无 GPU，PyTorch `2.14.1+cpu`。场景已经是上面的 1200×800 杂乱空域（smoke 那一行除外，smoke 是空地）。成功率都是 0，这些数只说明循环和耗时。

| 跑法 | 测到的更新耗时 |
|---|---|
| `pytest`：27 项 | 全部通过，约 12.3 秒 |
| smoke：2 僚机，隐藏层 32，空地，3×40 步 | 每次更新 0.010 秒；平均每回合 0.33 秒；总墙钟 1.0 秒 |
| AoI-MASAC，5 架，隐藏层 128，批大小 128，杂乱地图，5 回合、最多 80 步 | 195 次更新，平均 **0.021 秒/次**。正在更新的最后一回合 80 步用了 1.77 秒 |
| 同样预算的 MADDPG | 170 次更新，平均 **0.013 秒/次**。最后一回合 80 步用了 1.00 秒 |
| AoI-MASAC，隐藏层 256，批大小 256 | 180 次更新，平均 **0.036 秒/次**。满步的回合大约 2.2 秒走 60 步 |

前几个回合还在热身，几乎没有更新，“平均每回合”会被拉低。估算用“每次更新”。

论文配置是每回合最多 500 步、800 回合、每步更新一次。用 0.036 秒/次来乘：一回合约 18 秒，800 回合约 **4 小时/种子**。这是乘法，不是把 800 回合跑完。未训练时经常撞、回合更短；学会少撞之后会靠近这个上限。六个算法、三个种子，按满步数是几十个 CPU 小时。GPU 没有测。

---

## 13. 还要做的事

1. 把相关工作里的原文读完再引用，尤其是 IROS 2025 编队避障、Sensors 2023 固定时延编队、OJVT 2025 通信综述、以及车队时延 MARL。正文里写清楚：他们要么通信理想，要么时延固定且没有障碍和遮挡。
2. 只改一项奖励，把航路完成比例和编队误差拉到能看的水平。现在的权重只保证尺度和动力学匹配，不保证学得会。
3. 用 `configs/paper_*.yaml` 把六个已实现的算法各跑至少 3 个种子。
4. 每个检查点做 0%→70% 扫描，每个点至少 50 个评估回合。这是核心图。再单独报一条“测试时仍用距离×遮挡”的曲线，避免审稿人说横轴和训练分布不一致就完了。
5. 四个消融都按同样的种子跑。`ablation_no_blockage` 用来说明“楼挡无线电”不是摆设。
6. `moving_obstacles.yaml` 作为更挤的附加实验，不要和主图混在一条曲线里。
7. 多种子曲线的均值和方差还要在画图脚本里接上。现在一次只吃一个 CSV。
8. MAPPO 要么按 `algos/mappo.py` 顶部的步骤做完，要么在论文里写明不做。不要编数字。
9. 可选：Critic 也看不到 AoI 的更严消融；把直线外推换成常速度卡尔曼滤波。
10. 投稿前再查一次 “AoI + formation + MARL” 和 “blockage + multi-UAV RL”。和导师确认署名之后再传。WoWMoM 和 CCGrid 只能投一个。

---

## 14. 限制

- 运动是平面外环（加速度和偏航速率加风），不是六自由度四旋翼，没有桨、没有姿态、没有能耗。
- 遮挡是“线段碰圆盘再乘一个系数”，不是真实的菲涅尔区和信噪比。它够用来把楼和链路绑在一起，不够当通信论文里的信道测量。
- 只有领队到僚机这一条链路会丢。机间感知和障碍探测是有距离限制的本机传感器。
- 奖励没调完。短跑成功率是 0。
- 扫描图的横轴是与距离无关的丢包，和训练用的距离×遮挡不是同一个分布。
- MAPPO 没有实现。没有 GPU 计时。没有多种子的论文图。

---

## 15. English summary

The task is a cluttered 1200 m × 800 m transit: one leader tracks four waypoints through eight static buildings and three crossing intruders while four followers hold a 32 m wedge, under episode-wise wind. Followers receive the leader state on a downlink with `p = p0 exp(-d/d0)`, multiplied by an NLOS factor when a building intersects the segment. Packet loss keeps the last state and increments AoI; the actor observes that age. Onboard obstacle and teammate sensing is range-limited. Actions are longitudinal acceleration and yaw rate at 5 Hz.

This is aimed at formation MARL that assumes perfect communication (recent MASAC / IROS-style obstacle fields) and at delay-aware formation or platoon RL that uses a constant delay without obstacles. The single modeling increment is geometry-dependent loss — distance and building blockage — together with AoI as a policy input. Training randomizes `p0`. Baselines are perfect-link MASAC deployed with hold-last, MASAC-hold, linear extrapolation, MADDPG, and MATD3. Ablations remove AoI, channel randomization, distance dependence, or blockage. MAPPO is an explicit stub.

Section 12 records only smoke and short CPU runs on a 4-core machine with no GPU. Success rate on those runs is 0. They are not paper results.

许可证：MIT，见 `LICENSE`。
