# Mathematical / theoretical argument improvements — Predictive Maneuverability MASAC

Asia/Shanghai, 2026-11-03. Paper-facing audit of formulas already implemented in
`uav_planning/shared_masac.py`, `v4_maneuver_masac.py`, `v4_networks.py`,
`v4_prediction.py`, `v4_reward_reshape.py`. Goal: tighten WowMoM-style claims,
separate **definition / lemma / experiment**, and list concrete equation upgrades.

**Status of evidence:** Path-C Standard L0 passed (seeds 911/922/933 → 11/12,
11/12, 10/12). Maneuver paper suite (`full`/`zero`/`static`) is mid-training and
**unverified**. Do not claim novelty until suite gates and ablations finish.

---

## 1. Recommended paper equation set (canonical numbering)

Use these as the spine of the Method section. Symbols match the code unless noted.

### (E1) Team MDP / centralized soft Q

Agents \(i=1\ldots N\) (\(N=3\)), observation \(o_i\in\mathbb{R}^{162}\),
raw action \(a_i\in(-1,1)^3\) (tanh space). Joint reward is the **mean** of
per-UAV shaped rewards (Path-C). Discount \(\gamma=0.99\).

Centralized twin critics estimate
\[
Q_{\psi_k}(o_{1:N}, a_{1:N}), \quad k\in\{1,2\},
\]
with team pooling mean/max over per-agent trunks (code: `TeamQ`).

### (E2) Soft Bellman target (shared entropy)

With next actions \(a'\sim\pi_\theta(\cdot\mid o')\) and mean team log-prob
\(\bar{\ell}' = \frac{1}{N}\sum_i \log\pi_\theta(a'_i\mid o'_i)\):
\[
y = r + \gamma (1-d)\Big(\min_{k} Q_{\bar\psi_k}(o',a') - \alpha\,\bar{\ell}'\Big).
\]
Matches `ManeuverMASAC.update` / Legacy `SharedMASAC.update`.

**Claimable as definition** of the learner. **Not** a new contribution.

### (E3) Actor / entropy temperature

\[
J_\pi = \mathbb{E}\big[\alpha\,\bar{\ell} - \min_k Q_{\psi_k}(o,a_\theta)\big],
\qquad
J_\alpha = -\mathbb{E}\big[\alpha(\bar{\ell} - \mathcal{H}_0)\big],
\]
with target entropy \(\mathcal{H}_0 = \dim(a_i) = 3\) (code: `lp.mean(-1) - 3`).
Log-prob includes the exact tanh Jacobian (SAC). Entropy is in **raw tanh
action space**, not post-environment radial acceleration clipping — state this
explicitly (already in `docs/design/MASAC_V4_ARCHITECTURE.md`).

### (E4) Path-C reward reshape (learnability floor)

Per UAV, with distances \(d_t=\|g_i-p_{i,t}\|\):
\[
\begin{aligned}
r^{\mathrm{prog}}_t &= \lambda_p (d_{t-1}-d_t), \quad \lambda_p=1,\\
r^{\mathrm{live}} &= -c_\ell, \quad c_\ell=0.005,\\
r^{\mathrm{acc}}_t &= -\beta\|\mathrm{acc}_t\|_2^2, \quad \beta=0.002,\\
r^{\mathrm{near}}_t &= \eta\,\mathrm{clip}(1-d_t/R,0,1), \quad \eta=0.4,\ R=15,\\
r^{\mathrm{ball}}_t &= B_b\,\mathbf{1}[\text{first }d_t<b], \quad B_b=3,\\
r^{\mathrm{cap}}_t &= B_c\,\Delta c_{i,t}, \quad B_c=8,\\
r^{\mathrm{team}} &= B_T\,\mathbf{1}[\text{joint success}], \quad B_T=50,\\
r^{\mathrm{col}} &= -P\,\mathbf{1}[\text{collision}], \quad P=20.
\end{aligned}
\]
Capture indicator \(c_{i,t}\): for L0-return / maneuver suite,
\(d_t < 4\) and \(\|v_{i,t}\| < 2\), **non-sticky** (`capture_latch=False`).
Empty-floor Path-C used soft \(b=6\), speed \(3.5\), latch on — report as a
**curriculum diagnostic**, not the evaluation predicate.

**Paper claim:** Path-C is an **engineering reward curriculum** that established
a reproducible Standard L0 floor. Do **not** claim it as the scientific novelty.

### (E5) Predictive probes (candidate contribution)

Fixed direction set \(U=\{u_j\}_{j=1}^{19}\subset S^2\) (axes + face diagonals).
From ego state \((p,v)\) and observation-reconstructed objects, integrate
\[
v_{\tau+\Delta} = \Pi_{v_{\max}}\big(v_\tau + a_{\max} u_j \Delta\big),\quad
p_{\tau+\Delta}=p_\tau+\Delta v_{\tau+\Delta},
\]
for \(\tau=0,\Delta,\ldots,H\) with \(\Delta=0.2\,\mathrm{s}\), \(H=2.0\,\mathrm{s}\)
(\(T=H/\Delta=10\) interior steps + \(t=0\)).

Moving objects / peers: **constant-velocity** extrapolation from the current
observation only (no simulator futures).

Per probe feature \(f_j\in\mathbb{R}^{11}\):
\[
f_j=\big[u_j;\; \hat{c}^{\mathrm{st}},\hat{c}^{\mathrm{mv}},\hat{c}^{\mathrm{peer}},\hat{c}^{\mathrm{bd}};\;
\widehat{\mathrm{TTC}}_j;\; \hat{s}_{\mathrm{brake}};\; \hat{\rho};\; \mathbf{1}[\underline{c}_j>0]\big],
\]
where clearances are minima over the discrete samples, clipped/normalized by
\(L=40\,\mathrm{m}\); TTC normalized by \(H\); braking surrogate
\(\hat{s}_{\mathrm{brake}}=\|v\|^2/(2a_{\max})\); reserve
\(\hat{\rho}=c_0-\hat{s}_{\mathrm{brake}}\).

### (E6) RiskEncoder + fusion

\[
h_j = \phi(f_j)\in\mathbb{R}^{64},\quad
r = [\mathrm{mean}_j h_j;\, \mathrm{max}_j h_j]\in\mathbb{R}^{128}.
\]
SetEncoder pool yields \(s\in\mathbb{R}^{384}\); actor/critic trunk input
\([s;r]\in\mathbb{R}^{512}\). Ablations keep \(\phi\) and dims fixed:
`full` / `static` (mv+peer channels zeroed) / `zero` (\(f_j=0\)) / optional `shuffle`.

### (E7) What probes are **not**

State in the paper (and code docstring): discrete CV margins are **not**
continuous-time clearance certificates, not joint multi-agent certificates, and
not a hard safety filter. Contribution, if any, is **representation for learning**.

---

## 2. Top improvements (numbered) — tighten math without changing the live suite

### Improvement 1 — Explicit soft optimality statement (lemma-ready)

**Gap:** Code implements SAC twin-Q + entropy, but docs never write the soft
Bellman optimality operator \(\mathcal{T}^\pi\).

**Upgrade:** Add one lemma-style display:
\[
(\mathcal{T}^\pi Q)(o,a)= \mathbb{E}\big[r+\gamma\big(Q(o',a')-\alpha\log\pi(a'|o')\big)\big],
\]
and state that twin min and target Polyak \(\tau=0.005\) are **bias-reduction
engineering**, not part of the novelty claim.

**Claim vs proof:** Claim “we train soft Q with twin targets” (definition).
Do not claim new convergence rates.

### Improvement 2 — Entropy domain footnote → main text equation

**Gap:** Entropy is optimized in tanh coordinates; environment then applies
radial `amax` clipping. Reviewers will ask whether \(\mathcal{H}_0=3\) is
calibrated to executed controls.

**Upgrade:** Write
\[
a^{\mathrm{cmd}}=\tanh(z),\qquad
a^{\mathrm{env}}=\mathrm{RadialClip}_{a_{\max}}(a^{\mathrm{cmd}}),
\]
and define \(\log\pi\) on \(a^{\mathrm{cmd}}\) only. Optional experiment (future
id): retarget \(\mathcal{H}_0\) after measuring empirical entropy of \(a^{\mathrm{env}}\).

**Claim vs proof/experiment:** Clarification now; retarget needs a **new**
experiment id (do not mid-flight change suite).

### Improvement 3 — Path-C as a potential-based / staged return decomposition

**Gap:** Near-goal term \(r^{\mathrm{near}}=\eta\mathrm{clip}(1-d/R,0,1)\) is a
**state potential**, not shown as shaping relative to sparse success.

**Upgrade:** Write sparse team return \(R^{\mathrm{sparse}}\) vs Path-C return
\(R^{\mathrm{C}}\) and note:
- progress term is telescoping in distance (bounded path credit);
- near-goal is a bounded potential \(\Phi(d)=\eta\mathrm{clip}(1-d/R,0,1)\) added
  **undifferenced** (not classical Ng shaping \(\gamma\Phi(s')-\Phi(s)\)) — so it
  **can** change optimal policy. Be honest: Path-C is **not** policy-invariant
  potential shaping; it is a deliberate dense curriculum.

**Claim vs proof:** Claim “dense curriculum that empirically unlocked L0” with
table of empty→L0 gates. Do **not** claim Ng-invariant shaping.

### Improvement 4 — Capture predicate as a formal success measure

**Gap:** Soft empty capture (6/3.5+latch) vs frozen gate (4/2, no latch) are easy
to conflate in prose.

**Upgrade:** Define two predicates:
\[
\mathrm{Succ}_{\mathrm{soft}} = \bigwedge_i (d_i<6\land\|v_i\|<3.5)\ \text{(latched)},
\]
\[
\mathrm{Succ}_{\mathrm{L0}} = \bigwedge_i (d_i<4\land\|v_i\|<2)\ \text{(simultaneous, unlatched)}.
\]
All maneuver-suite and L0-return numbers use \(\mathrm{Succ}_{\mathrm{L0}}\).

**Claim vs proof:** Definitional hygiene; already enforced in protocol validators.

### Improvement 5 — Probe geometry: sampled clearance vs continuous swept clearance

**Gap:** Training collisions use continuous segment/swept tests; probes use
**point samples** every \(\Delta\). Between-sample penetration can be missed
(docstring admits this).

**Upgrade:** Add inequality / remark:
\[
c^{\mathrm{sample}}_{j} \ge c^{\mathrm{true}}_{j}
\quad\text{(can be strictly greater when contact is inter-sample)},
\]
so \(\mathbf{1}[c^{\mathrm{sample}}>0]\) is **optimistic**. Optionally define a
conservative surrogate using capsule swept volume along each probe segment
(future work; new experiment if trained on).

**Claim vs proof:** Remark now. Conservative probes = future method paper item.

### Improvement 6 — Constant-velocity peer/obstacle model as an explicit assumption

**Gap:** “Predictive” can be read as learned dynamics or planner foresight.

**Upgrade:** Label assumption (A1): exogenous objects evolve as
\(p(t)=p_0+vt\) with \((p_0,v)\) from the current observation mask. Peers use the
same CV model (no joint closed-loop prediction).

**Claim vs proof:** Assumption. Ablation `static` tests value of dynamic/peer
channels under (A1). `mismatch` (wrong \(a_{\max}\)) is a sensitivity probe, not a
learned dynamics model.

### Improvement 7 — Risk pooling permutation symmetry

**Gap:** Mean/max pool over 19 probes is permutation-invariant; `shuffle`
ablation remaps scalars across **fixed** direction columns — the correct
control for “directions must align with risk”.

**Upgrade:** State that if one shuffled **entire** tokens before invariant
pooling, the ablation would be vacuous. Keep the current shuffle definition in
the paper (already in architecture doc).

### Improvement 8 — Capacity-matched identification

**Gap:** Standard-C L0 (Legacy SetEncoder ~65k-scale actor path) is **not**
parameter-matched to V4 pool+RiskEncoder (~506k actor params in suite notes).

**Upgrade:** In equations/tables:
- **Task-matched baseline:** Standard-C L0 Path-C (same maps/reward/capture).
- **Parameter-matched contrast:** `full` vs `zero` vs `static` inside the suite.

Never write “ours beats Standard-C with the same capacity” unless a true
capacity-matched Standard rebuild exists (future id).

### Improvement 9 — Soft Q scale under Path-C bonuses

**Gap:** Team success +50 and capture +8 inflate Q magnitude; live suite shows
`critic_grad` often clipped at 5 while `q_mean` can be \(O(1)`–`O(5)`.

**Upgrade:** Report reward scale and clipping rate. Optional future: reward
normalization or smaller \(B_T\) with a **new** id. Do not change live suite
mid-seed.

### Improvement 10 — Horizon / travel-budget lemma (short)

With \(H_{\mathrm{ep}}=350\) steps, \(\Delta t=0.2\,\mathrm{s}\), wall time
\(70\,\mathrm{s}\). At \(v_{\max}=10\,\mathrm{m/s}\), straight-line budget
\(700\,\mathrm{m}\) ≫ typical L0 goal range (~40 m). So timeout failures are
**behavioral / capture**, not pure travel insufficiency — consistent with
ever_ball ≫ joint success early in training.

---

## 3. What to claim vs what needs proof / experiment

| Statement | Status |
|-----------|--------|
| Soft team MASAC with twin Q + entropy (E1–E3) | Definition / standard method |
| Path-C unlocked Standard L0 multi-seed gate | **Empirical** (done) |
| Path-C is Ng-invariant shaping | **False** — do not claim |
| Probes are safety certificates | **False** — forbid |
| Full probes improve joint success vs zero/static | **Needs suite result** |
| Attention helps | Out of scope this suite |
| Conservative continuous probes | Future work |

---

## 4. Suggested Method subsection outline (WowMoM-style)

1. Problem & \(\mathrm{Succ}_{\mathrm{L0}}\) (E4 capture predicate).
2. Shared soft actor–critic (E1–E3) — short, cite SAC/MASAC.
3. Path-C dense curriculum (E4) — empirical unlock, honesty about non-invariance.
4. Predictive maneuverability features (E5–E7) + assumptions (A1).
5. Ablations: full / static / zero (/ shuffle).
6. Limitations: sampled clearance optimism, CV peers, cold start after encoder change.

---

## 5. Concrete next writing/code hygiene (safe; no live hyperparam edits)

1. Keep this file linked from README Method pointers.
2. When suite finishes, add a one-page `docs/MATH_SUITE_RESULTS_*.md` mapping
   (E5)–(E6) claims to full/zero/static tables.
3. Training-speed edits for **next** launches live in
   `docs/TRAINING_OPTIMIZATION_MANEUVER_20261103.md` (risk cache, thread policy);
   they must not interrupt PIDs on stamp `162345`.

---

## 6. Top 5 formula improvements (executive)

1. **(E2)+(soft operator)** — write \(\mathcal{T}^\pi\) explicitly; confine novelty away from SAC.
2. **Entropy domain (E3 footnote→equation)** — tanh vs env radial clip.
3. **Path-C honesty** — dense curriculum, **not** Ng-invariant potential shaping.
4. **Succ predicates** — soft vs frozen L0 success measures, never conflated.
5. **Probe optimism remark** — sampled clearance ≥ true continuous clearance; not a certificate.

Repo rename note: GitHub repository is `Tomhao10225101490/masac-maneuverability`
(formerly `AoI-MASAC-Formation`). Local folder name may still be `MASAC` / legacy AoI.
