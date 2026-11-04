# Discrete peer-CV and ego-probe error: model-relative bounds

**2026-11-05, Asia/Shanghai.** This is a mathematical audit only. It changes no
running source, configuration, reward, protocol or 40k decision rule. The tools
below are standard nonexpansiveness of Euclidean projection, triangle/reverse
triangle inequalities and Lipschitz minima/clipping, not a novel theorem.

The bounds compare discrete surrogate trajectories. They do **not** certify
continuous-time clearance, joint safety, capture feasibility or learned-policy
effectiveness, and do not establish that L0 identifies moving-obstacle prediction.

## 1. Source correspondence and assumptions

The relevant sources are [v4_prediction.py](../uav_planning/v4_prediction.py),
[structured.py](../uav_planning/structured.py) and
[v4_environment.py](../uav_planning/v4_environment.py).
The 40k runs' source/configuration files remain frozen.

- A probe uses the semi-implicit update: first update/cap velocity, then advance
  position with the **new** velocity. There are ten updates and eleven sample
  positions at h=0.2 s over H=2 s.
- The control bank has 18 unit vectors and zero. For a fixed command u_j, the
  probe's requested physical acceleration is A u_j, where A is decoded from the
  current observation. Zero command is constant velocity, not active braking.
- The environment first radially normalizes the raw action to norm at most one,
  then multiplies by its acceleration bound. Thus the pre-speed-cap physical
  acceleration satisfies norm at most A. It is distinct from the realized
  acceleration logged after speed clipping.
- Visible peer positions and world velocities are decoded from current relative
  position/velocity channels. The full peer model holds that world velocity
  constant; it does not predict peer policy actions.
- The argument assumes the same timestep and velocity-bound ball in the two
  compared accelerated trajectories. The actual peer starts with norm(v_0)<=V.
  Peer identities, masks and sphere inflation must be held fixed for the
  clearance comparison.
- All K updates must exist under the stated dynamics. A simulator termination
  before H requires an explicitly defined continuation model; the stopped
  rollout is not automatically a K-step trajectory.

Write P_V for exact Euclidean projection onto the closed speed ball of radius V.
The proof is in exact arithmetic. The environment's denominator adds 1e-12,
whereas the probe uses a clamped denominator; float32 observation reconstruction
and floating-point integration also introduce residuals. Those implementation
residuals are **not bounded here**. Initial reconstruction errors may be included
as epsilon_p/epsilon_v only after bounds for them have actually been established.
The displays must not be advertised as machine-verified inequalities for every
floating-point execution.

## 2. Bounded-acceleration peer versus a constant-velocity peer model

Consider the idealized actual peer dynamics
\[
v_{k+1}=P_V(v_k+h a_k),\qquad
p_{k+1}=p_k+h v_{k+1},\qquad \|a_k\|_2\le A,\quad \|v_0\|_2\le V.
\]
Since projection is nonexpansive and P_V(v_k)=v_k,
\[
\|v_{k+1}-v_k\|_2
=\|P_V(v_k+h a_k)-P_V(v_k)\|_2\le h A.
\]
Consequently \(\|v_l-v_0\|_2\le l h A\). Against the exact-initial-state CV
model \(p_k^{\rm CV}=p_0+khv_0\),
\[
\begin{aligned}
\|p_k-p_k^{\rm CV}\|_2
&=\left\|h\sum_{l=1}^k(v_l-v_0)\right\|_2\\
&\le A h^2\sum_{l=1}^k l
=\frac{A h^2}{2}k(k+1)
=\frac A2(H_k^2+H_k h),
\end{aligned}
\]
where \(H_k=kh\). Speed saturation can tighten the actual error; it does not
invalidate this upper bound under the ideal projection dynamics.

For an observed initial state \((\hat p_0,\hat v_0)\), assume
\(\|p_0-\hat p_0\|_2\le\epsilon_p\) and
\(\|v_0-\hat v_0\|_2\le\epsilon_v\). A separate triangle inequality gives
\[
E_{\rm peer}(k)=
\epsilon_p+H_k\epsilon_v+\frac{A h^2}{2}k(k+1)
\]
as an upper bound on
\(\|p_k-(\hat p_0+kh\hat v_0)\|_2\).
The actual v_0 must be feasible; the noisy CV estimate need not itself lie in
the speed ball for this last triangle-inequality argument.
No independence or probability distribution for the errors is assumed.

## 3. Ego acceleration-model perturbation

Compare two semi-implicit trajectories with common h and P_V:
\[
v_{k+1}=P_V(v_k+h a_k),\quad
\tilde v_{k+1}=P_V(\tilde v_k+h\tilde a_k).
\]
Let \(\|a_k-\tilde a_k\|_2\le d_k\), with initial position/velocity errors
\(\epsilon_p^{\rm ego},\epsilon_v^{\rm ego}\).
Nonexpansiveness and induction yield
\[
\|v_l-\tilde v_l\|_2
\le\epsilon_v^{\rm ego}+h\sum_{j=0}^{l-1}d_j.
\]
Summing the new-velocity position updates gives the more general bound
\[
E_{\rm ego}(k)=\epsilon_p^{\rm ego}+H_k\epsilon_v^{\rm ego}
       +h^2\sum_{j=0}^{k-1}(k-j)d_j.
\]
If d_j<=delta_A uniformly,
\[
E_{\rm ego}(k)\le
\epsilon_p^{\rm ego}+H_k\epsilon_v^{\rm ego}
             +\frac{\delta_A h^2}{2}k(k+1).
\]
For two probes using the **same fixed** u_j with norm at most one, but magnitudes
A and A_tilde, take delta_A=|A-A_tilde| norm(u_j)<=|A-A_tilde|.
The zero probe has zero acceleration-magnitude perturbation.

In the current prediction helper, mismatch mode multiplies the rollout
acceleration by its configured factor; the default factor 0.5 with A=4 gives
delta_A=2 for a unit direction. This is a model diagnostic, not evidence that
the executing SAC policy holds the probe direction. Actual actions may change
direction at every step. Matching only acceleration **magnitudes** is insufficient
then: one must bound the vector difference d_j, which can be as large as 2A.
Ego error is zero only for matched initial conditions and the same ideal
command/dynamics, not merely because the probe and environment share A.

## 4. Peer-sphere clearance on the same sample grid

For one peer with the same fixed inflation R in both models, define
\[
c_k=\|p_k^{\rm ego}-p_k^{\rm peer}\|_2-R,\qquad
\hat c_k=\|\hat p_k^{\rm ego}-\hat p_k^{\rm peer}\|_2-R.
\]
The reverse triangle inequality implies
\[
|c_k-\hat c_k|\le
\|p_k^{\rm ego}-\hat p_k^{\rm ego}\|_2+
\|p_k^{\rm peer}-\hat p_k^{\rm peer}\|_2
\le E_{\rm ego}(k)+E_{\rm peer}(k).
\]
The code uses R=2 body_radius+safety_margin for peer spheres.
If radii/inflation differ, their absolute error must be added separately.

For the **same finite peer set and sample indices**, the minimum is
1-Lipschitz in the sup norm:
\[
\left|\min_{k,q}c_{k,q}-\min_{k,q}\hat c_{k,q}\right|
\le\max_{k,q}|c_{k,q}-\hat c_{k,q}|.
\]
Identical padded-slot sentinels can be included in both lists. With the code's
L=40 m normalization and C(x)=clip(x,-1,1),
\[
|C(m/L)-C(\hat m/L)|\le |m-\hat m|/L.
\]
This is a bound for the corresponding clearance channel, **not** all eleven
probe features. Initial-speed/braking-distance and reserve errors need their
own argument if their inputs/configurations differ.

First-sampled-hit TTC and the positive-preview flag are discontinuous at zero
clearance. Arbitrarily small clearance perturbations can flip the flag and move
the first-hit feature from an early sample to the no-hit sentinel H. Thus the
clearance Lipschitz bound must not be propagated to those features without
additional separation-from-threshold assumptions.

## 5. The 8.8 m discrete bound and its meaning

At A=4 m/s^2, h=0.2 s, K=10 and H=2 s,
\[
\frac{A h^2}{2}K(K+1)
=4(0.2)^2(55)=8.8\ {\rm m}.
\]
The continuous-time expression A H^2/2=8 m is **not** a valid replacement for
this semi-implicit bound. The difference is A H h/2=0.8 m.
The bound is attainable in the ideal discrete system: start at rest, accelerate
along a fixed coordinate axis at A=4 for ten steps, and use V=10 m/s. The final
speed is 8 m/s, strictly below the cap, so no clipping occurs.
The CV model stays at the initial position and the discrete position advances
8.8 m. This witness also shows why omitting the h term can underestimate error.

An ego magnitude perturbation delta_A=2 under the same conditions adds at most
4.4 m, so the zero-initial-error peer-clearance bound becomes 8.8+4.4=13.2 m.
With an exact matched ego probe it is 8.8 m instead. These correspond to bounds
of 0.33 and 0.22 on the respective clipped 40 m clearance feature; they are
worst-case model bounds, not measured errors or observed policy effects.

## 6. Exclusions and research use

The result excludes unseen peers, changes in masks, wrong sphere geometry,
unbounded control/model errors, different timestep/speed bounds, termination
before the horizon, unbounded reconstruction errors and numerical residuals.
It also excludes **inter-sample** collisions. A continuous swept-clearance bound
requires a separate time-interpolation/relative-speed argument.

Do not apply the bounded peer-acceleration assumption automatically to the
simulator's moving-obstacle bounce model: an instantaneous velocity reversal
needs its own per-step error model. Actual observation partiality and peer
closed-loop reaction remain limitations; no safety certificate follows.

A valid empirical next use is an offline, same-observation comparison of
sample-grid peer prediction errors and clearance errors, with the errors,
threshold crossings and masked/saturated channels all reported. The 8.8 m
worst case alone does not explain static-versus-full performance, show that
the learned actor uses a probe, identify moving-obstacle prediction on L0, or
establish novelty.

The independent checks in tests/test_discrete_probe_error.py exercise the
original structured.limited function without modifying it: the 8.8 m tightness
witness, the 4.4 m acceleration-mismatch witness and saturated-velocity cases.
Numerical tolerances in those checks are validation of those cases, not a
replacement for the exact-arithmetic assumptions or an exhaustive machine proof.
