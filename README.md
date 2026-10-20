# 带屋顶遮挡的三维编队：Belief-MASAC

论文题目：

**Belief-Aware Multi-Agent Soft Actor-Critic for UAV Formation under Geometry-Dependent Packet Loss**

中文：在距离和建筑物屋顶一起决定丢包的三维空域里，用带预测不确定性的多智能体软演员-评论家做编队和避障。

目标会议是 CCGrid 2027 或 WoWMoM 2027，两个只能投一个。按目前的计划，全文截稿是 2026-12-01（AoE），CCGrid 还要在 2026-11-24 交摘要。

这个仓库是这条研究线的可运行代码。更完整的公式、文献判断和已修错误在 `docs/MODEL.md`。指标定义在 `docs/METRICS.md`。奖励还没有调完。下面凡是耗时，都是这台机器上的冒烟或短跑，**不能写进论文**。

---

## 1. 二维地图还能发吗，算法还要不要往前走

二维还能发。2026 年 10 月 Drones 上的 DI-MATD3 就是平面窄门里的编队，信道是有界时延。2025 年也有论文把二维点模型当成正式的一档实验。所以“二维一律发不出来”不成立。

编队加障碍、以及“楼会不会挡住链路”这两支，近两年已经把高度写进模型：

- Xie 等人 2025 年的编队避障（arXiv:2410.18495）明确说，从二维走到三维会放大探索空间，他们做了仿真和真机。
- 2025 年底的三维城市编队把建筑物当成有高度的盒子。
- 2025 年初的遮挡感知数据采集用建筑物高度判断射线能不能从屋顶上面过去。

没有高度，就没有“爬高恢复视线”这个动作。地图因此改成三维：楼有屋顶，有的矮于升限，可以飞越；航路中间有一栋高于升限，必须绕开，而且只要脚印还夹在两机之间，无线电就继续被挡。

算法也不能停在“观测里加一个 AoI”。DI-MATD3 已经做了每条链路的 AoI，以及由 AoI 驱动的卡尔曼均值，并和“不预测”“常速度外推”比过。本文在这之上多三件事，而且三件事绑在一起：

1. 丢包是随机的，概率由三维距离和屋顶遮挡一起决定，不是一个事先给定的时延上界。
2. 跟随者本地的常速度卡尔曼把**位置标准差**交给演员。预测槽位落在楼后时，过程噪声放大，因为这时领导者多半在转弯，常速度最不可信。
3. 骨干是 MASAC。信道是随机的，策略的熵针对这个随机性。更新本身是标准的集中训练、分布执行，没有另造一个策略梯度。

奖励不写“请减小 AoI”。爬高会拉大高度上的编队误差，所以恢复视线是有代价的。这个代价要由策略自己权衡。

正式写相关工作之前，把上面提到的论文打开核对。这里的定位按题目和公开摘要来写，不代替你自己的阅读，也不引用他们的成功率当本文的结果。

---

## 2. 场景

地图水平 **1200 m × 800 m**，高度 **15 m 到 120 m**。默认 1 架领队 + 4 架僚机。僚机数可以改。

航路点在 `envs/scenario.py`，和测试用的是同一份：

| 序号 | 坐标 (m) | 这一段 |
|---|---|---|
| 1 | (280, 420, 40) | |
| 2 | (520, 560, 80) | 中途有一座高 50 m 的楼，巡航高度会撞，爬过屋顶就恢复视线 |
| 3 | (800, 320, 40) | 中途有一座高 160 m 的楼，高于升限，只能绕，绕开之前链路保持非视距 |
| 4 | (1080, 500, 40) | 中途有一座高 70 m 的楼，同样是飞越才能同时避开碰撞和遮挡 |

另有 5 栋不压在航段上的楼，高度 40–110 m。3 个交通球体以大约 4–8 m/s 横穿，高度约 35–60 m，只参与碰撞，不挡无线电。`configs/moving_obstacles.yaml` 再加一个更快的球体。

领队从大约 (150, 420, 40) 出发，速度 10 m/s，朝 +x。进入航路点周围 30 m 的球就算捕获。最后一个点还要等所有僚机进入真实槽位 18 m 以内，避免“点已经记完、队形在别处收拢”被算成成功。

风每个回合抽一次，0–4 m/s，只推动水平地速。演员看到带噪声的估计，评论家看真实风。

障碍和最近一架队友的探测距离默认 250 m。领队状态不走这个传感器，走第 4 节的链路。滤波器用的建筑底图视为已知，和这个有限距离的感知不是同一路。

---

## 3. 飞机怎么动

外环控制。动作 `(a_v, a_ω, a_z)` 都在 [-1, 1]，每步 0.2 s：

```
加速度     = a_v * 6 m/s²
偏航速率   = a_ω * 1.2 rad/s
垂直加速度 = a_z * 3 m/s²
水平速度限制 0–16 m/s，垂直速度限制 ±6 m/s
x, y 再加上这一回合的风
```

16 m/s、偏航顶满时转弯半径大约 13 m，和楼的半径可比，不能原地掉头。从 40 m 爬到能越过 50 m 屋顶，垂直速度顶满也要好几步。

机身半径 4 m。两机三维中心距离小于 8 m 算互撞，只在水平面上重叠、高度错开则不算。飞出地图或高度带会被夹回来，并给该机一个小惩罚。

编队是领队机体坐标下的平底楔形，间距 32 m（+x 朝前，+y 朝左，高度偏移 0）：

```
第 1 排: (-32, -24) 和 (-32, +24)
第 2 排: (-64, -48) 和 (-64, +48)
```

槽位高度等于用来计算槽位的那个领队高度。奖励用**真实**领队。观测里的槽位用这架僚机**被允许知道**的领队。爬高离开编队高度，会直接变成编队误差。

---

## 4. 信道、遮挡和信念

每一架僚机、每一步，独立决定这个包到不到。

建筑物遮挡：领队和僚机的三维线段若切到某栋楼的圆柱（地面到屋顶），这一跳是非视距。线段从屋顶上面过去则不是。交通球体永远不是。

```
distance 模式（主模型）:
    p = p0 * exp(-d / d0)          d 为三维距离
    若非视距且 use_blockage 为真:  p ← p * nlos_factor

independent 模式（消融，也是扫描图横轴）:
    p = p0

perfect 模式:
    p = 1
```

默认 `p0 = 0.95`，`d0 = 200` m，`nlos_factor = 0.2`。距离 200 m、不被挡时 p ≈ 0.37；再被挡则约 0.074。两机都高于屋顶后，遮挡项消失。训练时每个回合把 `p0` 抽成 0.45 到 1 之间的均匀分布，除非这一回合把 `p0` 写死。

丢了包，AoI +1，上限 40，并做一步常速度预测。收到包，AoI 归零，信念重置成这次测量。回合开始的免费同步不计入链路成功率。

预测用的风是这一步的噪声估计，不是真实风。预测落点如果仍被楼挡住，过程噪声乘 4。观测里除了预测位置，还有三个轴的位置标准差。公式和默认方差在 `docs/MODEL.md` 第 7 节。

六种观测，网络结构相同：

| 模式 | 槽位依据 | 年龄 | 标准差 | 配置 |
|---|---|---|---|---|
| `belief` | 卡尔曼均值 | 给 | 给 | 本文方法 |
| `belief_point` | 卡尔曼均值 | 给 | 不给 | 消融：只给均值 |
| `aoi` | 上一包测量，不外推 | 给 | 不给 | 只告诉它消息有多旧 |
| `extrapolate` | 卡尔曼均值 | 不给 | 不给 | 只外推，不给年龄和不确定性 |
| `hold` | 上一包测量 | 不给 | 不给 | 丢了就停住 |
| `perfect` | 真实位姿 | 不给 | 不给 | 理想通信训练 |

完美链路训练的策略，在丢包测试里会自动改成 `hold`。否则它看到的仍是真实领队，丢包不起作用。

---

## 5. 奖励

领队的进度是“离**当前**航路点又近了多少”，三维距离。捕获发生时，先记这个差值，再把目标换成下一个点。若先换点再算，一两百米的间距会变成一大笔负奖励，把过点奖金抵消掉。

僚机的进度是真实槽位误差的下降，另有一个与误差成正比的小惩罚。全队同时在 18 m 内有一小笔保持奖金。间隙小于 20 m 按比例罚该机。过一个点、整条航路完成、碰撞，是全队共享的奖金或惩罚。权重是起点，不是调完的结果。完整式子在 `docs/MODEL.md` 第 9 节。

成功 = 全部航路点已捕获，且这一步全队在真实槽位内，且全程没有碰撞。碰撞和超时把剩下的回合分开，三者加起来是 1。

---

## 6. 算法

集中训练、分布执行。每个演员只看自己的观测。每个评论家看全局状态和所有飞机的动作。更新某个演员时，只替换它自己的那一段动作。

| 名字 | 演员 | 观测 |
|---|---|---|
| `belief_masac` | 压缩高斯 SAC，自动温度，目标熵 −3 | `belief` |
| `aoi_masac` | 同上 | `aoi` |
| `vanilla_masac` | 同上，理想链路训练 | `perfect` |
| `masac_hold` | 同上 | `hold` |
| `masac_extrapolate` | 同上 | `extrapolate` |
| `belief_point_masac` | 同上 | `belief_point` |
| `maddpg` | 确定性策略，单个 Q | `hold` |
| `matd3` | 确定性策略，双 Q，延迟更新，目标噪声 | `hold` |
| `mappo` | 未实现，构造时直接报错 | |

SAC 靠熵探索，`ou_episodes` 保持 0。成功和碰撞不自举，超时要自举。

消融配置：

| 文件 | 关掉的东西 |
|---|---|
| `configs/ablation_no_uncertainty.yaml` | 标准差通道 |
| `configs/ablation_no_aoi.yaml` | 年龄通道（槽位仍是上一包测量） |
| `configs/ablation_no_blockage.yaml` | 楼对无线电的惩罚，楼本身还在，还是要躲开 |
| `configs/ablation_no_randomize.yaml` | 回合间的 `p0` 随机，固定 0.6 |
| `configs/ablation_independent_loss.yaml` | 距离和遮挡，丢包率就是 `1 - p0` |

---

## 7. 文件

```
envs/scenario.py          默认航路点、楼高、交通球体
envs/geometry.py          航向旋转、线段与圆柱相交
envs/belief.py            单链路常速度卡尔曼
envs/channel.py           p = p0 * exp(-d/d0)，可选遮挡系数
envs/formation_env.py     环境、观测、奖励
algos/                    SAC、MADDPG、MATD3；MAPPO 是占位
train.py  evaluate.py     训练和评估
metrics.py                按回合宏平均
scripts/sweep_packet_loss.py
scripts/plot_results.py
scripts/train_paper.sh
configs/                  含 paper_belief_masac.yaml
docs/MODEL.md             公式、文献、已修错误
docs/METRICS.md           每一列指标
tests/                    环境机制和每种算法的短训练
```

---

## 8. 安装和运行

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest -q
python3 train.py --config configs/smoke.yaml --device cpu
```

机器上如果没有名为 `python` 的命令，用 `python3`。有 GPU 时先装对应 CUDA 版的 torch，配置里 `device: auto` 会用它。本仓库没有在 GPU 上跑过。

调试训练（不是论文预算）：

```bash
python3 train.py --config configs/default.yaml --seed 0 --device cpu
python3 train.py --config configs/vanilla_masac.yaml --seeds 0,1,2
python3 train.py --config configs/masac_hold.yaml --seeds 0,1,2
python3 train.py --config configs/masac_extrapolate.yaml --seeds 0,1,2
python3 train.py --config configs/maddpg.yaml --seeds 0,1,2
python3 train.py --config configs/matd3.yaml --seeds 0,1,2
```

评估和 0%→70% 的独立丢包扫描：

```bash
python3 evaluate.py \
  --checkpoint runs/<某次>/ckpt_final.pt \
  --episodes 20 \
  --channel-mode independent --p0 0.5 \
  --out results/eval_one \
  --save-traj results/eval_one/traj.npz

bash scripts/run_key_experiment.sh runs/<某次>/ckpt_final.pt 30 results/sweep_belief
```

扫描的横轴是与距离无关的丢包率，这样横轴是一个数。训练分布是距离乘遮挡。论文里还要另报一条测试时仍用距离×遮挡的曲线。

论文规模（先用第 9 节的秒数乘一遍再决定）：

```bash
DEVICE=cpu SEEDS=0 bash scripts/train_paper.sh
```

TensorBoard：`python3 -m tensorboard.main --logdir runs --port 6006`。

画图（英文标注，png + pdf）：

```bash
python3 scripts/plot_results.py training --csv runs/<某次>/metrics.csv --out figs/train
python3 scripts/plot_results.py trajectory --npz results/eval_one/traj.npz --out figs/traj
```

轨迹图是俯视图。npz 里另有每架飞机的高度和每栋楼的高度，图上还没把高度画出来。

---

## 9. 这台机器上重新测过的速度

2026-10-20，4 核 CPU，无 GPU，PyTorch `2.14.1+cpu`。场景是上面的三维杂乱空域（smoke 那一行是空地）。成功率都是 0。这些数只说明循环和耗时。

| 跑法 | 测到的更新耗时 |
|---|---|
| `pytest`：34 项 | 全部通过，约 12.6 秒 |
| smoke：2 僚机，隐藏层 32，空地，3×40 步 | 89 次更新，每次 **0.0086 秒**；平均每回合 0.29 秒；总墙钟 0.86 秒 |
| Belief-MASAC，5 架，隐藏层 128，批大小 128，三维城市场景，5 回合、最多 80 步 | 243 次更新，平均 **0.021 秒/次**。正在更新的最后一回合 80 步用了 1.97 秒 |
| 同样预算的 MADDPG | 145 次更新，平均 **0.014 秒/次**。一次走满的 80 步回合用了 1.16 秒 |
| Belief-MASAC，隐藏层 256，批大小 256 | 166 次更新，平均 **0.040 秒/次**。走满的 60 步回合大约 2.3–2.5 秒 |

前一两回合还在热身，几乎没有更新，“平均每回合”会被拉低。估算用“每次更新”。

论文配置是每回合最多 500 步、800 回合、每步更新一次。用 0.040 秒/次来乘：一回合约 20 秒，800 回合约 **4.4 小时/种子**。这是乘法，不是把 800 回合跑完。未训练时经常撞、回合更短；学会少撞之后会靠近这个上限。`scripts/train_paper.sh` 里有 7 个已实现算法，三个种子，按满步数是几十个 CPU 小时。GPU 没有测。

---

## 10. 已经修掉的错误

细节和测试在 `docs/MODEL.md` 第 11 节。

- 过航路点的同一步，进度奖励曾把“离下一个点还有多远”记成负分。
- 最后一个航路点曾在领队刚进球时就捕获，队形可以事后在别处收拢并被记成成功。
- 外推以前不加风，侧风会把槽位带偏。现在加上这一步的风估计。
- 风噪声以前每读一次观测就重新抽，同一步的信念不稳定。
- 碰撞和遮挡以前是没有高度的圆盘。现在矮楼可以飞越并恢复视线，高楼不行；上下错开的两架飞机不再被判成互撞；交通球体不挡无线电。

---

## 11. 还要做的事

1. 把相关工作的原文读完再引用。正文写清楚：他们要么通信理想，要么时延有界且没有屋顶，要么只把卡尔曼均值交给策略、不给标准差。
2. 只改一项奖励，把航路完成比例和编队误差拉到能看的水平。
3. 用 `configs/paper_*.yaml` 把已实现的算法各跑至少 3 个种子。
4. 每个检查点做 0%→70% 的独立丢包扫描，每个点至少 50 个评估回合。再单独报一条距离×遮挡的测试曲线。
5. 五个消融按同样的种子跑。`ablation_no_uncertainty` 用来说明“只外推均值”不够；`ablation_no_blockage` 用来说明屋顶遮挡不是摆设。
6. `moving_obstacles.yaml` 作为更挤的附加实验，不要和主图混在一条曲线里。
7. 多种子的均值和方差还要在画图脚本里接上。俯视图还没有把高度画出来。
8. MAPPO 要么按 `algos/mappo.py` 顶部的步骤做完，要么在论文里写明不做。
9. 投稿前再查一次 “AoI + formation + MARL” 和 “blockage + multi-UAV RL”。和导师确认署名之后再传。WoWMoM 和 CCGrid 只能投一个。

---

## 12. 限制

- 运动是外环加水平风，不是六自由度四旋翼，没有桨、姿态和能耗。
- 遮挡是线段切圆柱，不是菲涅尔区或信噪比测量。
- 只有领队到僚机这一条链路会丢。丢包是信道抽出来的，策略不能选择这一步不发。
- 建筑底图对滤波器是已知的。障碍感知本身仍有距离限制。
- 奖励没调完。短跑成功率是 0。
- 扫描图横轴是与距离无关的丢包，和训练用的距离×遮挡不是同一个分布。
- MAPPO 没有实现。没有 GPU 计时。没有多种子的论文图。

---

## 13. English summary

Published 2025–2026 formation papers still include planar maps, so a 2D setting is not by itself a reason for rejection. The formation-and-obstacle papers a reviewer will cite have moved to 3D, and blockage-aware UAV papers decide line of sight by whether the ray clears the roof. This repo therefore uses a 1200 m × 800 m × [15, 120] m transit: eight buildings with heights, three of them sitting on the route, one taller than the altitude ceiling, and three spherical intruders that collide but do not block radio.

The closest communication-aware formation result (DI-MATD3, Drones, 11 Oct 2026) already feeds per-link AoI and a Kalman mean into MATD3 under a bounded delay, on a planar gate, with no buildings. The increment here is a Bernoulli downlink `p = p0 exp(-d/d0)`, multiplied by an NLOS factor when the 3D segment intersects a building, plus the Kalman position standard deviation as an actor input. Process noise is inflated when the predicted leader lies behind a roof. The backbone is MASAC. The reward uses the true leader pose and does not pay the agent to reduce AoI, so climbing to restore the link costs formation altitude error.

Baselines are perfect-link MASAC deployed with hold-last, hold, constant-velocity coasting, AoI without a coast, coasting without the variance channels, MADDPG, and MATD3. Ablations also remove blockage, distance dependence, or channel randomization. MAPPO is an explicit stub.

Section 9 records only smoke and short CPU runs on a 4-core machine with no GPU (20 Oct 2026). Success rate on those runs is 0. A 256-wide update costs about 0.040 s; at one update per step that is about 4.4 hours per 800-episode seed if episodes run to 500 steps. That product is not a finished training run.

许可证：MIT，见 `LICENSE`。
