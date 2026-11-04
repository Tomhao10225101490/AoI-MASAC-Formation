# 捕获中途轨迹诊断：seed 1511，20k / 22k

**性质：post-hoc、中间快照的描述性诊断，不是 40k 最终结果，也不作疗效判断。**
本文件只解读已经生成的两份轨迹摘要：
[20k summary.json](../results/masac_v4_sync_40k_progress_20261105/trajectory_20000/summary.json)、
[22k summary.json](../results/masac_v4_sync_40k_progress_20261105/trajectory_22000/summary.json)。
两份摘要记录：所读轨迹与 JSON 的 ever-ball、closest distance、无碰撞 success
及最大捕获占用交叉检查通过，且 control/B 的地图 seed、family、scene hash 配对一致。
四组快照均为 0/12 joint success、0 个碰撞地图。

本诊断不改奖励、参数、源码或配置，不选择替代终点，不解封测试地图，不延长预算。
既定三 seed、两臂、final 40k 主终点和原 advancement rule 保持不变。
每臂每快照的 36 个 agent-map 观测来自同一训练 seed 和 12 张地图，彼此相关；
20k 与 22k 也是同一训练历史的重复测量，不能当作独立训练样本。

## 1. 先分清三个不同的团队事件

令离散 post-step 时刻为 k，采样间隔 Δt=0.2 s。对 UAV i 定义

```
b_i(k) = 1[distance_i(k) < 4 m]
c_i(k) = 1[distance_i(k) < 4 m AND speed_i(k) < 2 m/s]
A = AND_i EXISTS_k b_i(k)       # 全体分别曾进入目标球
C = AND_i EXISTS_k c_i(k)       # 全体分别曾满足位置+速度捕获条件
J = EXISTS_k AND_i c_i(k)       # 某一个相同时刻全体满足捕获条件
```

有 J ⇒ C ⇒ A，但反向不成立。这里均无碰撞，因此 J 与本次保存的 success 一致。
应将失败按以下互斥的团队条件分层，而不是统一称为“同步失败”：

1. **未完成全体位置到达：¬A。** 至少一机整个回合没有进入目标球。
2. **全体进球，但捕获条件尚未全体分别成立：A ∧ ¬C。** 至少一机虽曾进球，
   却没有在球内同时满足速度阈值。
3. **全体分别曾捕获，但捕获时间不重叠：C ∧ ¬J。** 这是已观测到的同步/保持子型，
   并不等于已经识别其唯一控制原因。

| 快照 | 任意一机曾进球 /12 | ¬A：未完成全体位置到达 | A∧¬C | C∧¬J |
|---|---:|---:|---:|---:|
| control 20k | 8 | 12 | 0 | 0 |
| B 20k | 8 | 10 | 0 | 2 |
| control 22k | 11 | 12 | 0 | 0 |
| B 22k | 10 | 12 | 0 | 0 |

这是一种按团队最早未完成条件划分的层级诊断。¬A 地图中，其他 UAV 仍可能出现
“进球但速度不达标”，因此 A∧¬C=0 **不能**推出速度问题不存在。
同样，任意一机进入的比例较高，不能推出全体到达，更不能推出只是时间协调不足。

按个体事件分类的原始计数如下；它们是相关描述，不是独立样本量：

| 快照 | never_ball | ball_without_capture | ever_capture | 合计 |
|---|---:|---:|---:|---:|
| control 20k | 27 | 4 | 5 | 36 |
| B 20k | 22 | 2 | 12 | 36 |
| control 22k | 20 | 9 | 7 | 36 |
| B 22k | 24 | 0 | 12 | 36 |

## 2. 两个“全体分别曾捕获，但从未重叠”的明确实例

两例均来自 **B / seed 1511 / 20k**，保存回合长度均为 70 s。
它们分别是 830001（urban_canyon）与 830012（dynamic_crossing）。
两例的 maximum simultaneous capture occupancy 均为 **1/3**；因此不仅没有三机
联合捕获，连双机捕获占用的重叠也未出现。

下表 UAV 编号与摘要一致，采用 0 起始编号。
“捕获帧 / 最长连续帧”均只计 post-step 的严格 4m/2mps 捕获状态。

| 地图 | UAV | 首次捕获 s | 捕获帧 / 最长连续帧 | 最小距离 m | 末距 m | 末速 m/s |
|---|---:|---:|---:|---:|---:|---:|
| 830001 | 0 | 12.0 | 11 / 11 | 3.765 | 8.712 | 0.1526 |
| 830001 | 1 | 22.6 | 1 / 1 | 2.131 | 15.105 | 0.0348 |
| 830001 | 2 | 18.0 | 17 / 17 | 3.230 | 7.112 | 0.2251 |
| 830012 | 0 | 9.6 | 21 / 21 | 2.877 | 11.194 | 0.0751 |
| 830012 | 1 | 23.6 | 4 / 4 | 1.826 | 13.198 | 0.0303 |
| 830012 | 2 | 19.2 | 16 / 16 | 2.538 | 9.112 | 0.1762 |

每机的捕获总帧数都等于最长连续帧数，因此这两例中每机只出现一段非空连续捕获帧。
可由首帧时刻和连续帧数推得以下**离散采样时刻范围**：

- 830001：UAV 0 为 [12.0, 14.0] s，UAV 2 为 [18.0, 21.2] s，
  UAV 1 仅 22.6 s 一帧。首次捕获时间跨度为 **10.6 s**。
- 830012：UAV 0 为 [9.6, 13.6] s，UAV 2 为 [19.2, 22.2] s，
  UAV 1 为 [23.6, 24.2] s。首次捕获时间跨度为 **14.0 s**。

这里的区间端点只表示捕获条件在采样时刻成立；例如一帧不构成可证明的连续时间
停留区间。报告帧数比把一帧直接称为“保持 0.2 s”更准确。
首次捕获跨度本身也不是失败证据：如果先到者能保持捕获，跨度较大仍可能成功。
本例“不重叠”的证据是完整轨迹重算得到的 maximum occupancy=1。

两例回合末，**每例的三机均已在 4 m 球外**，但末速均低于 2 m/s
（共六个 agent-episode 末帧观测）。
830001 最坏末距为 **15.105 m**、全队最大末速 **0.2251 m/s**；
830012 对应 **13.198 m**、**0.1762 m/s**。
因此不能把这两例描述成“最后只差减速”：个体曾满足捕获，但后续位置保持/重新到达
没有在联合时刻实现。仅凭这些摘要不能认定失败唯一由等待策略、奖励项或熵变化导致。

## 3. 总体仍有大量未到达；末帧低速与持续停滞应区分

B 20k 的 12 张地图中，10 张仍属于 ¬A；两个 C∧¬J 实例不能代表全部失败。
到 B 22k，12 张地图全部属于 ¬A，虽然其中 10 张有至少一机曾进入目标球，
并且一张地图曾达到双机同时捕获。该快照有 24/36 个 agent-map 观测从未进球。

摘要还给出以下末帧状态：

- B 20k 所有地图的最大末速不超过 **0.2552 m/s**；22 个 never_ball 观测
  均末速低于 2 m/s，末距范围约 **5.275–36.598 m**。
- B 22k 对应最大末速不超过 **0.7661 m/s**；24 个 never_ball 观测
  均末速低于 2 m/s，末距范围约 **4.098–16.645 m**。

准确表述是“观测到未到达 UAV 在回合末处于球外低速状态”。
这些末速与末距快照不单独证明其在一段时间内持续停滞，也不证明是 B 项诱导的停滞。
20k 的两个全体曾捕获实例在 22k 未再出现，只能报告为中途失败构成变化；
不能据此选用 20k 代替既定 40k，或作收敛、退化因果及方法疗效结论。

## 4. 可直接用于论文讨论的措辞

> At the intermediate 20k and 22k checkpoints of training seed 1511, any-agent
> goal-ball entry did not imply that every UAV had reached its goal ball.
> Most failed episodes still contained at least one UAV that never entered
> its ball. Separately, two treatment episodes at 20k contained an individual
> position-and-speed capture event for every UAV, but their maximum simultaneous
> capture occupancy was only one. Their first-capture time spans were 10.6 s
> and 14.0 s, and all UAVs ended outside their capture balls. These observations
> distinguish incomplete team arrival from non-overlapping individual capture
> events. They are post-hoc descriptions of one training seed, not the registered
> final-40k endpoint, and do not establish a treatment effect or its cause.

论文应分别报告 A、C、J 或等价的互斥失败类别，并同时保留逐 UAV 的距离、速度和
捕获连续帧；不要以 any-ever-ball 代替全体到达，也不要把所有 timeout 一概解释为
同步问题。这里的诊断用于澄清失败现象，最终方法比较仍由完整预注册实验裁决。
