# Mathematical argument and code contract — Predictive Maneuverability MASAC

**Original audit:** 2026-11-03. **Corrected:** 2026-11-05, Asia/Shanghai.
These equations describe the saved suite implementation, not an untested replacement.
Read the companion [code/evidence audit](METHOD_MATH_AUDIT_20261105.md).
The suite stamp 20261103_162345 is COMPLETE: static **11/12, 10/12, 5/12**;
full **3/12, 4/12, 8/12**; zero **1/12, 0/12, 1/12**.
No arm passes the all-three-seed gate. These results do not establish predictive
superiority, safety, convergence or novelty.

## 1. Canonical equations and assumptions

### (E1) Partially observed team control and the implemented critic

Let \(S_t\) denote the simulator state, augmented with time and reward memory
(e.g. first-ball-entry flags). Agent \(i\in\{1,2,3\}\) receives
\(o_{i,t}=h_i(S_t)\in\mathbb R^{162}\). Write \(O_t=(o_{1,t},\ldots,o_{N,t})\)
and \(A_t=(a_{1,t},\ldots,a_{N,t})\). The shared actor is a decentralized,
memoryless policy; its deterministic probe features are functions of its current
observation and public configuration only. The joint policy factorizes given \(O_t\).

The two centralized networks approximate \(Q_{\psi_k}(O_t,A_t)\), \(k=1,2\),
with per-agent trunks and joint mean/max pooling (v4_networks.TeamQ).
The environment returns the mean of the per-agent rewards, \(r_t=N^{-1}\sum_i r_{i,t}\).
For fixed temperature, the entropy term uses the **mean per-agent entropy**,
not the sum. Discount \(\gamma=0.99\).

This is an observation-based critic approximation. The current observation omits
at least the first-entry reward memory and episode time. Even broad L0 sensor
coverage does not establish that \(O_t\) is a sufficient Markov state. Do not
invoke an unconditional observation-space Bellman contraction or convergence
theorem. A formal policy-value operator belongs on sufficient latent states,
beliefs or histories, with its assumptions stated.

### (E2) Implemented bootstrapped target and critic update

Define \(\bar\ell_t=N^{-1}\sum_i\log\pi_\theta(a_{i,t}\mid o_{i,t})\),
where the action-coordinate log density is summed over each UAV's three coordinates.
The code uses the target
\[
y_t=r_t+\gamma(1-\delta_t)
 \left[\min_{k=1,2}Q_{\bar\psi_k}(O_{t+1},A'_{t+1})
             -\alpha\bar\ell'_{t+1}\right],
\quad A'_{t+1}\sim\pi_\theta(\cdot\mid O_{t+1}),
\]
\[
L_Q=\mathbb E_{\mathcal D}\sum_{k=1}^2
          (Q_{\psi_k}(O_t,A_t)-\operatorname{sg}(y_t))^2.
\]
Here \(\operatorname{sg}\) means stop-gradient. Critic/actor gradient norms are
clipped at 5; target parameters use
\(\bar\psi\leftarrow(1-\tau)\bar\psi+\tau\psi\), \(\tau=0.005\).

**Saved-suite terminal convention:** train_masac_v4_maneuver.py passes only
the true-terminal flag to replay. Thus \(\delta_t=1\) on collision or success;
a time-limit truncation resets the episode but keeps \(\delta_t=0\) and bootstraps.
This is the historical continuing-value approximation at a timeout. It is not
the exact terminal-value equation for a finite 350-step task. Any change to
this convention requires a new matched control and experiment id.

On a sufficient state \(x\), a fixed-policy soft evaluation operator can be defined as
\[
(\mathcal T^\pi Q)(x,A)=
 \mathbb E[r+\gamma(1-\delta)(Q(x',A')-\alpha\bar\ell')\mid x,A].
\]
This is **policy evaluation**, not a new soft optimality theorem; the implemented
twin minimum and neural observation critic are algorithmic approximations.

### (E3) Actor, temperature sign and executed controls

With \(\beta=\log\alpha\) and the critics held fixed during the actor update,
\[
J_\pi=\mathbb E_{\mathcal D,A\sim\pi_\theta}
 \left[\operatorname{sg}(e^\beta)\bar\ell-\min_k Q_{\psi_k}(O,A)\right],
\qquad
J_\beta=-\mathbb E_{\mathcal D}
 \left[\beta\,(\operatorname{sg}(\bar\ell)-3)\right].
\]
This is exactly the implemented actor/temperature loss in both
ManeuverMASAC.update and SharedMASAC.update.
The optimized variable in the temperature loss is **log temperature**, not
\(e^\beta\). The fixed-point condition is \(\mathbb E[\bar\ell]=3\);
equivalently the differential-entropy target is
\(\mathcal H_\star=-3\), **not \(+3\)**. Indeed
\(\partial J_\beta/\partial\beta=\mathcal H-\mathcal H_\star=\mathcal H+3\).
This corrects the old document's entropy sign and \(\alpha\)-vs-\(\log\alpha\) mismatch.

The raw command and environment dynamics are
\[
z_i\sim\mathcal N(\mu_\theta(o_i),\operatorname{diag}\sigma_\theta^2(o_i)),
\quad a_i=\tanh z_i,\quad
u_i=\frac{a_i}{\max(1,\|a_i\|_2)},
\]
\[
v_{i,t+1}=\Pi_{\|v\|\le v_{\max}}
 (v_{i,t}+\Delta t\,a_{\max}u_i),\qquad
p_{i,t+1}=p_{i,t}+\Delta t\,v_{i,t+1}.
\]
The structured.limited function implements the radial command projection and
speed bound (up to its numerical epsilon). The acceleration penalty uses the
realized \((v_{i,t+1}-v_{i,t})/\Delta t\). The exact tanh Jacobian defines
\(\log\pi\) on \(a_i\in(-1,1)^3\); it is not a density for the radially projected,
speed-limited physical control.

### (E4) Path-C reward and the frozen capture event

Index a transition from \(t\) to \(t+1\), let
\(d_{i,t}=\|g_i-p_{i,t}\|_2\), and define
\[
b_{i,t}=\mathbf1[d_{i,t}<4],\quad
c_{i,t}=\mathbf1[d_{i,t}<4\ \land\ \|v_{i,t}\|_2<2].
\]
The capture flag is non-sticky; \(\Delta c_{i,t}=c_{i,t+1}-c_{i,t}\) can be negative.
Let \(e_{i,t}\) track whether agent \(i\) has previously entered its ball,
\(K_t\) denote any collision on this transition, and
\(G_t=\mathbf1[\forall i:c_{i,t+1}=1\ \land\ \neg K_t]\).
Then the frozen L0 Path-C reward is
\[
\begin{aligned}
r_{i,t}={}&(d_{i,t}-d_{i,t+1})-0.005
 -0.002\left\|\frac{v_{i,t+1}-v_{i,t}}{\Delta t}\right\|_2^2\\
&+0.4\,\operatorname{clip}(1-d_{i,t+1}/15,0,1)
 +3\,b_{i,t+1}(1-e_{i,t})\\
&+8(c_{i,t+1}-c_{i,t})+50G_t-20\mathbf1[K_t].
\end{aligned}
\]
The first-entry reward is once per UAV per episode; capture loss emits \(-8\)
when the unlatched predicate becomes false. Team bonuses/penalties are added
to every UAV before taking the mean, so they remain \(+50/-20\) in team reward.

The empty-floor diagnostic used a different capture contract: 6 m / 3.5 m/s,
latched. Under a latch, different agents may capture at different times:
\(\bigwedge_i\exists t:c^{\rm soft}_{i,t}=1\) is not simultaneous capture.
The L0 event is \(\exists t:\bigwedge_i c_{i,t}=1\), with no collision;
all maneuver-suite finals use the latter.

**No policy-invariance claim for Path-C.** For its progress reward,
\[
\sum_{t=0}^{T-1}\gamma^t(d_t-d_{t+1})
=d_0-(1-\gamma)\sum_{t=1}^{T-1}\gamma^{t-1}d_t-\gamma^{T-1}d_T.
\]
It telescopes to a pure boundary difference only when \(\gamma=1\).
The undifferenced near-goal bonus also changes the objective. Path-C is an
engineering reward design that empirically established the Standard-C L0 floor,
not a theorem preserving a sparse success objective.

### (E5) Observation-only sampled maneuverability features

The 19-element bank is
\[
U=\{0\}\ \cup\ \{\pm e_k:k=1,2,3\}\
 \cup\ \{(\sigma e_k+\rho e_l)/\sqrt2:
                  k<l,\ \sigma,\rho\in\{-1,1\}\}.
\]
There are **18 unit directions plus zero**; \(U\not\subset S^2\).
The zero command holds velocity in the probe model; it is not a braking action.
Starting at the reconstructed ego state, each command integrates
\[
\hat v_{j,l+1}=\Pi_{\|v\|\le v_{\max}}
 (\hat v_{j,l}+a_{\max}u_j\Delta),\quad
\hat p_{j,l+1}=\hat p_{j,l}+\Delta\hat v_{j,l+1},
\]
with \(\Delta=0.2\) s, \(H=2\) s, \(l=0,\ldots,9\).
Clearances include **11 samples**, at \(l=0,\ldots,10\).

Assumption (A1): each visible moving object or peer follows
\(\hat p_q(l\Delta)=p_{q,0}+v_{q,0}l\Delta\), reconstructed from current masked
observations. There is no simulator future, peer policy or joint closed-loop rollout.
Let \(c^q_{j,l}\) be the implemented signed clearance for
\(q\in\{\mathrm{static},\mathrm{moving},\mathrm{peer},\mathrm{boundary}\}\):
inflated AABB signed distance, moving-sphere distance, peer-sphere distance,
or inward distance from world bounds. Masked slots receive clearance \(L=40\) m.
Write
\[
m_j^q=\min_{0\le l\le10}c^q_{j,l},\quad
c_{j,0}=\min_q c^q_{j,0},\quad
D_{\rm brake}=\|v\|_2^2/(2a_{\max}),
\]
\[
T_j=\begin{cases}
\min\{l\Delta:\min_q c^q_{j,l}\le0\},&\text{if a sampled hit exists},\\
H,&\text{otherwise}.
\end{cases}
\]
The value \(T_j/H=1\) conflates a hit exactly at \(H\) with no sampled hit.
It is a grid-based first-hit feature, not the true continuous TTC.
With \(C(x)=\operatorname{clip}(x,-1,1)\), the 11-vector is
\[
f_j=\left[u_j;\
 C(m_j^{\mathrm{static}}/L),C(m_j^{\mathrm{moving}}/L),
 C(m_j^{\mathrm{peer}}/L),C(m_j^{\mathrm{boundary}}/L);\
 T_j/H;\
 \operatorname{clip}(D_{\rm brake}/L,0,1);\
 C((c_{j,0}-D_{\rm brake})/L);\
 \mathbf1[\min_q m_j^q>0]\right].
\]
The stopping-distance term is a scalar surrogate under ideal opposing
acceleration; it does not solve coupled 3D avoidance or guarantee a feasible stop.

### (E6) Risk pooling, fusion and precisely scoped ablations

\[
h_j=\phi(f_j)\in\mathbb R^{64},\qquad
r_{\rm risk}=[\operatorname{mean}_j h_j;\operatorname{max}_j h_j]
 \in\mathbb R^{128}.
\]
The pool SetEncoder supplies \(s_{\rm set}\in\mathbb R^{384}\);
\([s_{\rm set};r_{\rm risk}]\in\mathbb R^{512}\) feeds actor/critic trunks.
The actor and each critic have their own encoders. All three suite arms have
the same architecture and trainable dimensions.

| Mode | Exact change in predict_probes |
|---|---|
| full | All visible static/moving/peer/boundary clearance families under (A1). |
| static | Replace moving/peer clearances by \(L\) **before** first-hit/reserve/flag calculation; set the two corresponding output channels to zero. Retain ego forward integration and static/boundary preview. |
| zero | Set **all 19×11 values**, including direction coordinates, to zero. Encoder biases can still learn a constant embedding. |
| shuffle (unrun) | Keep the canonical direction coordinates fixed; permute the eight scalar features across directions. |

Static is a **static-geometry preview**, not a prediction-free input.
Full-vs-static isolates inclusion of CV moving/peer information and its derived
aggregates in this implementation; it does not isolate all prediction.
Zero is a matched-parameter constant-risk-input control, not a proof that
network capacity is useless or that the policy causally uses the risk encoder.
Mean/max pooling is invariant to permutation of complete tokens; shuffling
complete tokens would therefore be a vacuous direction-alignment control.

### (E7) Model-relative clearance and feasibility limits

For the **same** surrogate trajectory, obstacle model, inflation and visible
objects, the sample times are a subset of the continuous interval, hence
\[
c_j^{\rm sample}=\min_l c_j(l\Delta)
 \ \ge\ \inf_{\tau\in[0,H]}c_j(\tau)=c_j^{\rm continuous,\ surrogate}.
\]
This only compares two minima within the same model. CV error, occlusion and
other agents' control invalidate a direct inequality to the realized future
clearance. Positive sampled clearance is not a continuous-time or joint safety
certificate; no hard filter is applied.

For the saved horizon, a speed-bound **upper bound** on path length is
\[
\sum_{t=0}^{349}\Delta t\,\|v_{i,t+1}\|_2
 \le350(0.2)(10)=700\ {\rm m}.
\]
This does not prove capture feasibility or exclude geometric, braking or timing
infeasibility. Feasibility evidence must come from the actual per-map reference
audits or constructive trajectories, separately from learned-policy results.

## 2. Evidence and contributions that remain defensible

| Statement tied to equations | Evidence / limit |
|---|---|
| E1–E3 define the shared soft actor–critic and exact temperature convention | Source correspondence; standard learner, no new convergence theorem. |
| E4 established a Standard-C L0 learnability floor | 11/12, 11/12, 10/12, seeds 911/922/933. Warm-started 50k L0 runs, not a fair cold-start V4 performance comparison. |
| E5–E6 define an executable observation-only feature bank | Implementation/geometry checks; effectiveness and novelty are separate questions. |
| Matched-capacity content ablation | static 26/36, full 15/36, zero 2/36 final successes across three training seeds and the same fixed development maps. Descriptive totals, not 36 independent training trials. |
| Full predictive channels outperform static | **Unsupported by this suite**; mean final ordering is static > full > zero, and static loses to full on seed 1233. |
| Prediction superiority, safety, generalization, novelty | **Not established**. Independent sealed-map tests and mechanism/related-work comparisons remain necessary. |

The final-record audit separates any-agent ball entry from all-agent eventual
entry. All-agent eventual-entry counts are static 11/12, 10/12, 5/12; full
5/12, 4/12, 8/12; zero 1/12, 1/12, 2/12. Of 65 final failures, 61 involve only
some agents ever entering and four involve all agents entering without joint
capture; none collide. The historical ever_ball_maps count uses any-agent
entry. Thus a purely synchronization-caused failure is not established; the
next hypothesis targets worst-agent approach and joint capture together.

The Standard-C seeds all load the same empty-floor seed-811 checkpoint at
40k with weights and optimizer states; replay and RNG are reset. Its legacy
architecture, initialization history, seeds and 50k L0 budget differ from the
80k cold-start V4 suite. Treat it as a learnability floor only.

## 3. Next hypothesis and theorem boundaries

The current user authorization (2026-11-05) permits Windows hardware/CPU
replication, B sync-capture design/implementation and a small validation after
audit. Large launches require a plan summary first; Attention, L1 and a complete
paper-suite rerun remain outside this authorization.

See [B design](design/MASAC_V4_SYNC_CAPTURE_20261105.md) for the new experiment's
potential, terminal handling, unchanged evaluation predicate and controls.
Do not conflate a new \(\gamma\Phi_{t+1}-\Phi_t\) term with the existing Path-C reward.

For fixed shaping coefficient \(\lambda\), the pathwise identity is
\[
\sum_{t=0}^{T-1}\gamma^t\lambda(\gamma\Phi_{t+1}-\Phi_t)
 =-\lambda\Phi_0+\gamma^T\lambda\Phi_T.
\]
A policy-ordering statement requires the same initial-state distribution and
a policy-independent terminal boundary (e.g. absorbing terminal potential zero),
or a properly transformed continuing bootstrap. Partial observability alone
does not invalidate the algebra for history-dependent policies. It does prevent
unqualified claims of exact learning equivalence for a memoryless observation
critic. Timeout bootstrapping, function approximation and finite replay must
be stated; the algebra does not imply faster learning, convergence or capture gains.
