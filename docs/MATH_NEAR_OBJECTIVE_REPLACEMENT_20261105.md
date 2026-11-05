# Near-objective replacement: exact identities and claim boundaries

2026-11-05. This is a mathematical audit of the new reward intervention, not a new PBRS theorem, a launch instruction, or a result from the proposed 40k study. It changes no source, configuration, gate, evaluation, or completed history. The implementation contract is recorded in the [2k protocol](design/MASAC_V4_NEAR_OBJECTIVE_2K_PROTOCOL_20261105.md); the [independent study proposal](design/MASAC_V4_NEAR_OBJECTIVE_CONFIRMATION_PROPOSAL_20261105.md) specifies the next comparison. This note independently checks the supplied equations and those documents, not executable source.

## 1. Variables and the two distinct reward changes

Let \(S_t\) include the physical state and whatever additional state/history is needed to specify transitions and event rewards. There are \(N=3\) agents. Define their physical goal distances \(d_i(S)\geq0\), and the fixed potential

\[
g(d)=\operatorname{clip}(1-d/R,0,1),\qquad
\Phi(S)=\frac{c}{N}\sum_{i=1}^{N}g(d_i(S)),\qquad
c=0.4,\ R=15\ {\rm m},\ \gamma=0.99.
\tag{1}
\]

Thus \(0\leq\Phi\leq0.4\). This is a **mean distance potential**. It contains no speed, worst-agent maximum, joint-capture predicate, arrival-time gap, obstacle prediction, or feasibility certificate. For example, distance vectors \((0,30,30)\) m and \((10,10,10)\) m have the same \(\Phi=0.4/3\), although their worst distances differ. It is not the earlier B distance/speed potential.

Let \(S_{t+1}^{\rm phys}\) be the actual post-step state before a reset or absorbing substitution, and let \(\delta_t=1\) mean true termination, including collision or success. Define

\[
\phi_t=\Phi(S_t),\quad
\phi_{t+1}^{\rm phys}=\Phi(S_{t+1}^{\rm phys}),\quad
\phi_{t+1}^{\rm eff}=(1-\delta_t)\phi_{t+1}^{\rm phys}.
\tag{2}
\]

A timeout alone has \(\delta_t=0\), retains the physical successor potential, and retains the existing bootstrap. True termination takes precedence if both flags are set. The original Path-C team-mean near term is \(\phi_{t+1}^{\rm phys}\). The replacement is

\[
\begin{aligned}
r_{0,t}&=r_{{\rm PathC},t}-\phi_{t+1}^{\rm phys},\\
F_t&=\gamma\phi_{t+1}^{\rm eff}-\phi_t,\\
r_{{\rm new},t}&=r_{0,t}+F_t.
\end{aligned}
\tag{3}
\]

The first line changes the base objective. Only the second line is standard PBRS relative to that new base \(r_0\). At a true terminal, removal still uses the **physical** near term; only the added shaping successor is replaced by zero. Replacing both successor terms by zero would implement a different reward.

## 2. Exact per-transition correction bounds

Relative to original Path-C, write \(C_t=r_{{\rm new},t}-r_{{\rm PathC},t}\). Equation (3) gives

\[
C_t=
\begin{cases}
-\phi_t-(1-\gamma)\phi_{t+1}^{\rm phys},&\delta_t=0,\\
-\phi_t-\phi_{t+1}^{\rm phys},&\delta_t=1.
\end{cases}
\tag{4}
\]

Consequently,

\[
-0.404\leq C_t\leq0\quad\text{for nonterminals, including timeouts};
\qquad
-0.8\leq C_t\leq0\quad\text{for true terminals}.
\tag{5}
\]

These are algebraic outer bounds; this does not assert that every extremum is reachable under the physics. The added PBRS term alone instead satisfies \(-0.4\leq F_t\leq0.396\) on nonterminals and \(F_t=-\phi_t\) on true terminals. A negative whole correction or terminal correction is not, by itself, evidence that capture has become less preferred relative to \(r_0\); policy ordering is a trajectory-level question.

## 3. Discounted trajectory identity, with the endpoint retained

Consider \(T\) transitions without a preceding true termination. If the last transition truly terminates, append a zero-potential absorbing state. Otherwise retain the final physical state. Let \(b_T=1-\delta_{T-1}\). Direct cancellation gives

\[
\begin{aligned}
\sum_{t=0}^{T-1}\gamma^t F_t
&=\sum_{t=0}^{T-1}
  \bigl(\gamma^{t+1}\phi_{t+1}-\gamma^t\phi_t\bigr)\\
&=-\phi_0+\gamma^T b_T\phi_T^{\rm phys}.
\end{aligned}
\tag{6}
\]

Here intermediate \(\phi_{t+1}\) is the next decision state's potential, and the final value is the effective endpoint from (2). This is a pathwise identity; observation Markovness is not needed for the cancellation. The two useful return identities are therefore

\[
G_{{\rm new},T}=G_{0,T}-\phi_0+
  \gamma^T b_T\phi_T^{\rm phys},
\tag{7}
\]

\[
G_{{\rm new},T}=G_{{\rm PathC},T}
-\sum_{t=0}^{T-1}\gamma^t\phi_{t+1}^{\rm phys}
-\phi_0+\gamma^T b_T\phi_T^{\rm phys},
\qquad
G_{r,T}=\sum_{t=0}^{T-1}\gamma^t r_t.
\tag{8}
\]

At true termination the boundary vanishes. At an artificial cutoff it generally does not:

\[
0\leq\gamma^T\phi_T^{\rm phys}\leq0.4\gamma^T.
\tag{9}
\]

Its boundedness is not exact finite-horizon invariance. Dropping it can change the ranking of raw cutoff returns. An undiscounted logged episode sum also does not inherit the discounted identity: for \(\gamma<1\), summing \(F_t\) without discounting leaves intermediate-potential terms.

For a nonterminal cutoff, exact cancellation with a bootstrap requires a consistently shifted continuation value:

\[
\begin{aligned}
G_{{\rm new},T}+\gamma^T V_{\rm new}^{\pi}(X_T)
&=G_{0,T}+\gamma^T V_0^{\pi}(X_T)-\phi_0,\\
V_{\rm new}^{\pi}(X_T)&=V_0^{\pi}(X_T)-\Phi(X_T).
\end{aligned}
\tag{10}
\]

The notation \(X_T\) requires sufficient state/history and a well-defined potential on it. Equation (10) is conditional on the **exact** corresponding continuation values. The collector resets after a cutoff but the learner bootstraps from the saved pre-reset next observation. That is the existing continuing-value surrogate, not a proof of equivalence to the finite-time benchmark. A done-false flag and an approximate neural bootstrap do not establish (10). The formal values must refer to the same modeled continuation, not silently switch to the reset state.

With bounded potential and a genuine infinite discounted continuation, \(\gamma^T\Phi_T\to0\). With true-terminal absorbing zero potential, the same conclusion holds after termination. Under either compatible construction,

\[
J_{\rm new}(\pi)=J_0(\pi)-\mathbb E_{\rho_0}[\Phi(S_0)].
\tag{11}
\]

This comparison requires the same initial distribution \(\rho_0\), dynamics, discount, stopping/continuation model, and admissible policy information class; the initial-potential law must be policy independent. If a controller can use reward history as information, changing that history must not silently enlarge its information class. The present actors retain their existing observation interface.

## 4. Entropy and partial-observation scope

For a fixed policy and fixed temperature \(\alpha\), let \(\mathcal H_t^\pi\) denote the same mean-agent entropy convention in both tasks. Assume the discounted reward/entropy expectations are well defined. Adding the identical term \(\alpha\mathcal H_t^\pi\) to both sides of (7) leaves its difference unchanged. Under the boundary conditions for (11),

\[
J_{{\rm new},\alpha}(\pi)
=J_{0,\alpha}(\pi)-\mathbb E_{\rho_0}[\Phi(S_0)].
\tag{12}
\]

Thus exact soft-objective policy ordering is preserved **relative to \(r_0\)** inside the common policy class, including a fixed decentralized or memoryless class. This is not a claim of observability, existence of an optimal policy in the neural class, or preservation of the original Path-C objective. Equation (8) includes the policy-dependent removed-near sum. Neither \(r_0\) nor \(r_{\rm new}\) has been proved equivalent to maximizing sparse joint-success probability.

On a suitable sufficient state/history where \(\Phi(X)\) is defined, the ideal fixed-policy soft values have the familiar shifts

\[
Q_{{\rm new},\alpha}^{\pi}(X,A)
=Q_{0,\alpha}^{\pi}(X,A)-\Phi(X),\qquad
V_{{\rm new},\alpha}^{\pi}(X)
=V_{0,\alpha}^{\pi}(X)-\Phi(X).
\tag{13}
\]

The scalar trajectory identity remains valid even if the physical potential is hidden from an actor. That does **not** justify writing \(Q_{\rm new}(o,a)=Q_0(o,a)-\Phi(o)\) for an aliased observation projection. Potential reconstructibility from critic inputs, sufficient information for transitions/reward memory, and exact value representation are separate requirements. Our joint-capture history and time-limit modeling do not become Markov merely by adding this reward.

## 5. What exact critic shifts would and would not imply for SAC

Fix a replay batch, actor, action samples and \(\alpha\). If both supplied critics share the exact action-independent shift \(Q_{{\rm new},k}(X,A)=Q_{0,k}(X,A)-\Phi(X)\), then their minimum shares it and the actor losses satisfy

\[
L_{\pi,{\rm new}}=L_{\pi,0}+\mathbb E_{\mathcal D}[\Phi(X)].
\tag{14}
\]

With fixed batch inputs and a potential independent of actor parameters/current sampled action, the added term has zero actor gradient. Likewise, if **online and target** critics all have the matching shift, true-terminal masking and retained timeout potential yield

\[
\begin{aligned}
y_{\rm new}
&=r_0+\gamma(1-\delta)\Phi(X')-\Phi(X)\\
&\quad+\gamma(1-\delta)
  [\min_k(Q_{0,\bar k}(X',A')-\Phi(X'))-\alpha\bar\ell']\\
&=y_0-\Phi(X),\qquad
Q_{{\rm new},k}-y_{\rm new}=Q_{0,k}-y_0 .
\end{aligned}
\tag{15}
\]

Here \(\bar\ell'\) is the unchanged mean-agent log probability. These are conditional function-level identities, not the learning trajectory of the two fresh arms. Identical random initialization gives equal critic functions, not the required shifted functions. Neural approximation, parameter Jacobians, replay, Adam state, clipping and parameter-space Polyak updates need not maintain the shift. The temperature loss contains no potential directly; it is unchanged only given the same actor/log-probability samples and optimizer inputs. Different learned actors can change those samples and hence the learned temperature. Equations (12)–(15) provide no finite-budget, convergence or automatic-temperature trajectory guarantee.

## 6. Local occupancy flow is not a policy-priority proof

On a nonterminal constant-potential transition, \(\phi_{t+1}=\phi_t=\phi\),

\[
F_t=-(1-\gamma)\phi=-0.01\phi,\qquad
C_t=-(2-\gamma)\phi=-1.01\phi.
\tag{16}
\]

The old near component was \(+\phi\) per step; after its removal the replacement contributes \(-0.01\phi\). This precisely removes that local positive occupancy stream. Other progress, entry/exit, acceleration, living and terminal terms remain. Their net flow is not determined by (16). Nor does (16) establish capture retention, a preferred stopping strategy, or a higher probability of joint success.

Over an infinite constant-potential segment the discounted shaping contribution is \(-\phi\), consistent with telescoping. A transient progress-shaped positive \(F_t\), or a terminal negative \(F_t\), must be interpreted through the same identity. Under the compatible exact-objective conditions above, only removal of the undifferenced near term can change policy ordering relative to Path-C; PBRS can still change finite neural learning relative to \(r_0\). Which effect dominates an empirical result is not identified by the present comparison.

## 7. What the two-arm experiment identifies

The two arms compare **original Path-C** with **Path-C minus physical near plus PBRS**. Removal and shaping are changed together. Even a positive endpoint identifies only the effect of that complete replacement at this budget/population; it cannot separately attribute improvement to removal, credit assignment or stopping behavior. A factorial comparison with \(r_{\rm PathC}\), \(r_{\rm PathC}+F\), \(r_0\), and \(r_0+F\) could separate the factors, but is only a possible future design, not an experiment authorized here.

Both new arms keep static probes and B off. There is no moving-information or predictive-content intervention. This comparison cannot establish dynamic prediction utility or identify the cause of the old full-versus-static ranking. Its mean distance potential does not encode the earlier worst-agent distance/speed potential or isolate a synchronization mechanism.

The [closed 2k pilot](NEAR_OBJECTIVE_2K_RESULTS_20261105.md) reports **0/12 success in each arm**. Treatment has **103 transitions with nonzero potential/correction**, while each history has **26 true training terminations and zero timeouts**. These are different denominators: 26 is not a count of nonzero terminal corrections. The observed treatment correction range is within (5). The pilot exercises nonzero replacement and true-terminal execution, but supplies no actual timeout coverage or performance decision; timeout algebra was tested separately. The earlier B1522 result, with reduced sampled near-stall occupancy but still 0/12 success, likewise rules out treating lower stall alone as task completion. Neither observation establishes a cause.

The independent study design uses new cold seeds **1611/1622/1633**, static/B-off arms, and a fixed **40k final endpoint**. Let \(s_{a,j}\) be final successes out of 12 development maps and \(\Delta_j=(s_{{\rm new},j}-s_{{\rm original},j})/12\). The proposed engineering planning rule is

\[
\frac13\sum_j\Delta_j\geq\frac1{12},\qquad
\#\{j:\Delta_j>0\}\geq2,\qquad
\min_j\Delta_j\geq-\frac1{12},
\tag{17}
\]

with six complete, finite, unmodified histories. This supports **planning confirmation only**. The separate absolute gate is at least **8/12 on every seed**; intermediate peaks do not replace final results. Three training seeds remain the replicates, and the 12 reused maps are development cases, not 36 independent Bernoulli trials. This note states the design and no 40k outcome or launch status. Final executable protocol/configuration and authoritative completion records determine those facts.

## 8. Checked primary-source boundaries

- **Adamczyk et al., AAAI 2023 / December 2022 preprint**, [full author paper](https://arxiv.org/html/2212.01174v2): **Preliminaries, Theorem 5/Equations 20–22, Remark 7/Equations 23–24**. The sufficient PBRS/value-shift result keeps dynamics, discount, inverse temperature and reference policy fixed and requires bounded potential and well-defined regularized values. It supports the standard shift used here, not a new theorem or neural-SAC trajectory equivalence. The paper's inverse temperature is not the repository's log-temperature variable. We do not import its necessity discussion as a general claim for this one task.
- **Vasan et al., 2024**, [full author paper](https://arxiv.org/html/2407.00324v2): **Section 3/Table 1; Section 4/Figure 5; Section 5/Figures 7–8**. Low-speed goal arrival, time to goal and time on goal are established evaluation distinctions. Their SAC task-formulation comparison changes reward, termination and reset semantics together. Their complete-interaction initial-policy goal-hit evidence does not prove that our near coefficient caused failure or that this replacement improves sample efficiency; its hit-frequency heuristic is not a threshold for our conditional actor audit.

These primary sources motivate a precise objective and explicit boundary accounting. They do not establish novelty, safety, convergence, causal reward pathology, or reliable capture for this implementation. The separate [PBRS scope note](MATH_PBRS_SCOPE_NOTE_20261105.md) documents the earlier B argument; its base objective and potential must not be substituted for the different \(r_0,\Phi\) in this note.
