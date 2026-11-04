# Method, mathematical claims and evidence audit

**2026-11-05, Asia/Shanghai.** This audit corrects the paper-facing interpretation
of saved suite 20261103_162345; it does not rewrite the historical experiments.
Canonical equations E1–E7 are in
[MATH_ARGUMENT_IMPROVEMENTS_20261103.md](MATH_ARGUMENT_IMPROVEMENTS_20261103.md).

## 1. Formula-to-source contract

| Equation / claim | Executable source and symbols | Correction / boundary |
|---|---|---|
| E1 observation and latent state | [v4_environment.py](../uav_planning/v4_environment.py): V4Mission.observe; [v4_reward_reshape.py](../uav_planning/v4_reward_reshape.py): first-entry tracker | 162-vector observations omit episode time and first-entry memory. Joint observations are not assumed Markov. |
| E2 team target / mask | [v4_maneuver_masac.py](../uav_planning/v4_maneuver_masac.py): ManeuverMASAC.update; [trainer](../scripts/train_masac_v4_maneuver.py): model.push | Reward and log-probability are each averaged over agents. True terminals mask bootstrap; timeouts do not. |
| E3 temperature | ManeuverMASAC.update: temperature loss; [shared_masac.py](../uav_planning/shared_masac.py): SharedMASAC.update | Implemented loss is \(-\log\alpha\,(\operatorname{sg}(\bar\ell)-3)\); entropy target is **−3**. Do not replace log-alpha by alpha in the source-matched loss. |
| E3 action / entropy | [v4_networks.py](../uav_planning/v4_networks.py): Actor.forward; [structured.py](../uav_planning/structured.py): limited | Tanh log-Jacobian applies before radial command projection and speed saturation. |
| E4 reward | V4MissionRewardReshape.step and ReshapeConfig | First ball entry once per UAV; unlatched capture delta can be −1. Discounted distance progress is not a pure boundary term. |
| E5 control bank / preview | structured.DIRECTIONS; [v4_prediction.py](../uav_planning/v4_prediction.py): predict_probes | Zero plus 18 unit vectors; ten integration updates and 11 sample positions. |
| E5 feature scale | ProbeConfig and predict_probes | Distances and braking reserve use 40 m units with clipping; first sampled-hit time uses 2 s units. A value of one conflates no hit with a hit at 2 s. |
| E6 full/static/zero | predict_probes mode branches; v4_networks.RiskEncoder/Features | Static retains ego forward preview and excludes moving/peer clearances from derived features. Zero removes directions and scalars; trainable constant embeddings remain possible. |
| E7 clearance relation | predict_probes sampled trajectory | Sample minimum bounds only the continuous minimum **within the same surrogate model**, not realized closed-loop clearance. |
| E7 travel bound | Speed limit, 350-step horizon and 0.2 s timestep | 700 m is an upper bound on path length, not a capture-feasibility proof. |

No soft Bellman convergence result is claimed for the neural observation critic.
The display \(\mathcal T^\pi\) is a sufficient-state **policy evaluation**
definition; twin minima and target networks do not turn it into a new theorem.

## 2. Three distinct entry/capture events

For map rollout \(m\), \(d_{i,t}\) is distance to that UAV's independent goal.
Define
\[
B_m^{\rm any}=\mathbf1[\exists i,\exists t:d_{i,t}<4],\qquad
B_m^{\rm all}=\mathbf1[\forall i,\exists t:d_{i,t}<4],
\]
\[
C_m=\mathbf1[\exists t:\forall i,\ d_{i,t}<4,\
               \|v_{i,t}\|_2<2,\ \text{and no collision}].
\]
Then \(C_m\le B_m^{\rm all}\le B_m^{\rm any}\).
Historical trainer output **ever_ball_maps** computes \(B^{\rm any}\), not
\(B^{\rm all}\). The historical timeout-after-ball category also uses any-agent
entry; its name cannot establish a timing-only failure.

| Arm | Final successes, seeds 1211/1222/1233 | All-agent eventual ball entry | All-three-seed gate |
|---|---|---|---|
| zero | 1/12, 0/12, 1/12 | 1/12, 1/12, 2/12 | FAIL 0/3 |
| full | 3/12, 4/12, 8/12 | 5/12, 4/12, 8/12 | FAIL 1/3 |
| static | 11/12, 10/12, 5/12 | 11/12, 10/12, 5/12 | FAIL 2/3 |

Of 65 final failures, **61** have only some UAVs ever entering; **four** have all
UAVs entering at some times without success; **none** collide. The four all-entry
failures comprise two full/1211, one zero/1222 and one zero/1233. These counts
motivate worst-agent approach and joint capture analysis; they do not identify
whether speed, timing, optimization or representation is the causal bottleneck.
See the independent
[saved-record analysis](../results/masac_v4_windows_audit_20261105/analysis.json).

The final static mean exceeds full and zero, but static is weaker than full on
seed 1233 and drops from 9/12 at 70k to 5/12 at 80k. Preserve the fixed final
checkpoint result and show its learning curve. Totals 26/36, 15/36 and 2/36 are
descriptive; 36 reused-map outcomes are not 36 independent training seeds.

## 3. Contribution-to-evidence map

| Candidate contribution | What the paper can currently support | Missing evidence |
|---|---|---|
| E5–E6 observation-only preview bank and pooling | Fully specified executable representation and information boundary | Distinctiveness against close prior art; mechanism-use interventions |
| Matched-capacity content comparison | Under this exact protocol, static has the strongest final mean; added CV channels have no established advantage | More independent training seeds, uncertainty, sealed unseen-map evaluation and additional controlled mechanisms |
| Frozen-capture artifact and failure taxonomy | Explicit any/all-eventual/simultaneous metrics and per-seed gates | Generalization beyond the fixed L0 development maps |
| B worst-agent approach/capture response | A separately named, source-matched design and algebraic shaping identity | Matched control/B learning evidence; no gain follows from the identity |

Standard-C L0 passed 11/12, 11/12, 10/12 with seeds 911/922/933 after 50k L0 steps.
Every seed loaded the same empty-floor seed-811 40k checkpoint with model and
optimizer weights; replay/RNG were reset. Its legacy architecture, initialization
history, seed ids and L0 budget differ from the 80k cold-start V4 arms. It proves
only an empirical learnability floor for the task/reward contract, not a fair
performance advantage or disadvantage of the candidate.

## 4. What B's potential identity can and cannot show

The [new B design](design/MASAC_V4_SYNC_CAPTURE_20261105.md) specifies the potential,
fixed coefficient, evaluation freeze and a new experiment id.
For \(F_t=\lambda(\gamma\Phi_{t+1}-\Phi_t)\),
\[
\sum_{t=0}^{T-1}\gamma^tF_t
 =-\lambda\Phi_0+\gamma^T\lambda\Phi_T.
\]
At a true absorbing terminal with \(\Phi_T=0\), the change is an initial-state
constant. At a truncated cutoff with bootstrap, the boundary is canceled only
by the corresponding shifted value \(V'=V-\lambda\Phi\). A finite truncated
reward sum alone retains the boundary term. State fixed initial distribution,
terminal convention and value-representability assumptions before making a
policy-ordering claim. This algebra also holds for history-dependent policies
under partial observability; it does not guarantee exact TD learning for the
implemented memoryless observation critic, finite-sample improvement or safety.

## 5. Current work boundary

The 2026-11-04 Mac-only pause is historical. The 2026-11-05 user request authorizes
the Windows hardware/CPU audit and B implementation/small validation. A large
launch needs a visible plan summary first. Attention, L1 and a full paper-suite
rerun remain outside authorization. Keep CPU as the default until a CUDA
benchmark establishes a stable speed benefit. Document all new results under
new ids; never overwrite or reinterpret the saved suite as B evidence.
