# Joint recourse as a falsifiable predictive-maneuverability hypothesis

Date: 2026-11-05. **UNREGISTERED / UNIMPLEMENTED / NOT AN EXPERIMENT LAUNCH.**

This document proposes a possible later research question. It registers no
training, evaluation, simulator calls, diagnostic execution, seed, test access,
budget extension, Attention/L1 module, or change to an active run. The registered
near-objective study, its twenty frozen sources, paired configs, fixed endpoints
and decisions remain separate. Nothing below interprets its intermediate results.
The proposed objects are instances of established robust reachability/viability
reasoning, not new theorems or cleared novelty.

## 1. Why this question remains open

The completed late-replay audit found full and static-fill-one tokens identical
in 349849/350208 positions (99.89748949%) on the fixed sampled population. Valid
peers were usually far enough for the distance sentinel to dominate. This is
content-degeneracy evidence on those static-policy replay samples, not prediction
accuracy, absence of peers, or a causal explanation of the old full-policy result.
The completed B cohort was mixed and failed the absolute task gate in both arms.
The current near-objective study tests a base reward objective with static probes;
even a favorable result would not establish predictive-channel utility.

A useful next claim would need to distinguish **a first action that leaves one
shared feasible recovery** from actions that appear acceptable only when each
hazard is allowed a different, mutually incompatible recovery. A second question
is how that distinction changes under honest uncertainty about future motion.
Neither question is answered by naming a scene “dynamic” or appending another
clearance scalar.

## 2. State, information, dynamics and normalized constraints

Let x_k include the positions and velocities of all controlled UAVs and whatever
external-obstacle state the proposed model actually assumes. Let

    x_(k+1) = f(x_k, u_k, w_k),  k = 0,...,H-1.

The time step, speed projection, acceleration limits and action application order
must be specified. Using the present discrete simulator would yield a statement
about that simulator, not a flight-physics guarantee. Initial-state uncertainty
belongs in the uncertainty set as well. No unknown coordinate is silently treated
as fully observed.

At real decision time t, h_t denotes the allowed observation history. A causal
recourse rule chooses u_k from information available by k, never from an entire
realized future disturbance sequence. Fix one information class Pi before making
comparisons. For decentralized execution, Pi must contain only collections of
rules pi_i(h_i,k) implementable with the declared local information and any
explicitly allowed messages. A centralized full-state solver can be an oracle
reference; it is not a deployable decentralized certificate. Its privileged
information must not enter only the candidate actor.

Hard admissibility restricts Pi using the actual input/speed limits and dynamics.
Assume a finite constraint index set, nonempty admissible Pi for the committed
first action, nonempty W, and finite margins on the fixed horizon. Otherwise
undefined/empty optimization domains must be reported explicitly, not as safe
scores. Geometric safety is evaluated by the margins below, not imposed as an
extra all-hazard admissibility restriction that would erase the relaxation gap.
For the safety margin itself, use dimensionless geometric constraints, for example

    g_ij(x) = ( ||p_i-p_j|| - r_i-r_j ) / L_ij,
    g_iO(x) = ( signed_distance(p_i, O) - r_i ) / L_iO,
    g_iB(x) = signed_boundary_clearance(p_i) / L_iB.

Every L is a positive, physically documented reference length, fixed across arms,
states and outcomes. It is not learned or tuned after observing results. If speed
or actuation slack is also included in a minimum, it needs its own dimensionless
normalization, e.g. (V_i-||v_i||)/V_i; mixing metres with m/s is invalid. Adding
such a slack changes the numerical object and must be separately specified.
The counterexample below uses geometric margins only and enforces action bounds
as hard admissibility.

Let W(h_t) be a nonempty set of external disturbance sequences, fixed before
optimizing recourse. For exogenous scripted obstacles, their paths do not depend
on the ego action. For interactive agents, a fixed recorded future is generally
invalid under counterfactual actions. One must instead specify causal response
maps and primitive noise, e.g. w_k=F_theta(history_k,u_k,noise_k), and quantify over
theta/noise consistently. Controlled peers inside x use the same joint causal Pi;
they are not simultaneously granted cooperative controls and treated as unrelated
adversarial trajectories. Changing the information class, response assumptions or
disturbance set between the candidate and a baseline invalidates the comparison.

## 3. Object A: the shared-recourse margin and its relaxation

Fix the **same first joint action** u_0=a for every constraint. For each constraint
ell define

    F_ell(pi,w;h_t,a) = min_(k=0,...,H) g_ell(x_k),

with x generated by the specified dynamics, initial information, first action and
causal recourse pi from step 1 onward. Define

    R_joint(h_t,a)
      = sup_(pi in Pi) inf_(w in W(h_t)) min_ell F_ell(pi,w;h_t,a),

    R_pair(h_t,a)
      = min_ell sup_(pi_ell in Pi) inf_(w in W(h_t)) F_ell(pi_ell,w;h_t,a).

“Pair” abbreviates a per-constraint relaxation; static and boundary constraints
may be included too. R_joint requires one common recovery to handle every hazard.
R_pair lets each hazard choose its own recovery. These are not interchangeable.

For any fixed ell and any pi,

    inf_w min_m F_m(pi,w) <= inf_w F_ell(pi,w).

Taking sup_pi preserves this inequality. Taking the minimum over ell on its
right-hand side gives

    R_joint(h_t,a) <= R_pair(h_t,a).

No minimax interchange, differentiability or new convergence theorem is used.
The domains Pi and W must be the same on both sides. A positive R_pair says only
that the relaxed individual problems admit margin; it does not establish common
joint feasibility. R_joint>epsilon>0 is a finite-horizon discrete-grid recovery
statement only when the model, information class and full disturbance treatment
are valid. R_joint=0 may additionally require attainment to imply an admissible
nonnegative-margin policy. Neither value alone proves arrival, simultaneous goal
capture, absence of deadlock, continuous-time safety or recursive feasibility.
An invariant terminal set and intersample/model-error arguments would be extra
work, not consequences of this definition.

### Strict gap at one fixed first action

Consider a deterministic one-dimensional lateral slice with H=2, dt=1,

    y_(k+1)=y_k+v_(k+1),  v_(k+1)=v_k+u_k,
    y_0=v_0=0,  u_0=a=0,  u_1 in [-1,1].

Thus y_1=0 and y_2=u_1. The same first action has already been committed. There
is one subsequent scalar recourse u_1, not a separate first action per hazard.
Two exogenous obstacles have longitudinal coordinate X_k=6-3k and lateral
coordinates -1/2 and +1/2; their inflated collision radius is 1. The UAV's
longitudinal coordinate is fixed at 0 in this illustrative slice. With reference
length L=1, the two binding final geometric margins are

    g_L(y_2)=|y_2+1/2|-1,
    g_R(y_2)=|y_2-1/2|-1.

At k=0,1 the longitudinal separation makes both margins greater than 1/2, so they
do not change the following minima. Ignoring the other obstacle, the left problem
chooses u_1=+1 and reaches margin +1/2; the right chooses u_1=-1 and also reaches
+1/2. Therefore R_pair=+1/2. For any y_2 in [-1,1], at least one final distance
to the obstacle centers is at most 1/2, so min(g_L,g_R)<=-1/2. This bound is
attained at y_2=0 and at the two endpoints, hence R_joint=-1/2.

The strict gap comes from incompatible **shared subsequent recourse under the
same a=0**. It is not a comparison of different fixed first actions. This is a
mathematical low-dimensional witness, not an implemented three-dimensional UAV
scene. A later 3D realization must use genuine geometry/dynamics to rule out an
unmodeled vertical or longitudinal escape, verify that individual obstacle paths
are physically valid, and retain all those constraints in both problems.

### Finite candidate libraries are a different approximation

For a fixed admissible backup library B subset Pi, exact worst-case evaluation
would give

    R_B = max_(pi in B) inf_w min_ell F_ell(pi,w) <= R_joint.

Thus restricting the recovery family gives a conservative lower approximation;
it is not the per-constraint upper relaxation above. If per-constraint solutions
also use only B, min_ell max_(pi in B) inf_w F_ell upper-bounds R_B, but need not
upper-bound unrestricted R_joint. A finite first-action grid covers only its
queried first actions, not every continuous actor action.

Enumerating a finite set of disturbance trajectories replaces inf_W by a larger,
optimistic sampled minimum. Combining that approximation with a restricted backup
library yields no general lower/upper ordering relative to exact R_joint. This
proposal therefore treats enumerated scores as diagnostic values, **not safety
certificates**. A certificate would require separate conservative verification of
the entire disturbance set, model/observation errors, every relevant constraint,
the actually selected control and any continuous-time/invariance claim. None of
that work is claimed here.

## 4. Object B: uncertainty-conditioned loss of recoverable options

One possible extension uses a whole-trajectory uncertainty set W_alpha(h_t),
giving R_joint,alpha by replacing W above. If W_1 subset W_2, then

    R_joint(h_t,a;W_2) <= R_joint(h_t,a;W_1).

For a fixed, named finite first-action proposal set A, describe the fraction of
queried actions retaining a conservative target margin:

    q_alpha(h_t) = |{a in A : lower_R_alpha(h_t,a)>=epsilon}| / |A|.

This is a finite proposal-set fraction, not continuous action-space volume or
entropy. Comparing a point forecast with a valid uncertainty set can expose
overconfident apparent escape options. The uncertainty model and the recourse
model are separate components; a larger q is not automatically better task
performance, because waiting safely can still fail the mission.

Calibrating per-frame average prediction error does not establish joint coverage
over agents, horizon steps and a planner-selected action. Exogenous obstacle
paths permit an action-independent trajectory tube; interactive predictions may
need action/policy-conditioned response sets. Ordinary split-conformal marginal
coverage needs the stated exchangeability/sampling conditions. Adaptive MARL,
correlated replay windows and action selection can violate them. Any proposed
calibration must use episode-disjoint fitting/calibration/evaluation data and
declare whether coverage concerns a frozen deployed policy, all proposed actions,
one chosen action, a complete trajectory or a long-run average. A nominal 1-alpha
per-step or marginal guarantee must not be rewritten as an episode safety bound.
Without those conditions, report empirical coverage and calibration error only.

## 5. Prior art and the actual burden of contribution

The following primary sources were opened during this review. Summary claims
marked “abstract” below are verified only at that level; they do not establish
that every implementation detail or theorem assumption has been reproduced.

- [RCRL, ICML 2022](https://proceedings.mlr.press/v162/yu22d.html): feasible-set
  safety values and reachability constraints are already an established RL
  approach (official abstract read). The object above is not a new definition of
  feasible-set RL.
- [DGPPO, ICLR 2025 official paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/b0916c29bc32d57e2c86a63e8f2df9f2-Paper-Conference.pdf):
  learned discrete graph barriers address neighborhood changes and input
  constraints. Merely adding actuation limits to MARL risk is insufficient.
- [SCOPE, IJCAI 2026](https://ijcai.org/proceedings/2026/7), method sections
  4.1–4.3 read: online learned previews, preview-fused actor/critic, joint CBF-QP
  correction and a next-state feasibility look-ahead are already explicit.
  “Prediction plus future feasibility” is not an empty research gap.
- [CC-HGCBF, arXiv:2610.41214v1, 2026-10-30](https://arxiv.org/html/2610.41214v1):
  state-dependent finite-horizon reachable velocity representations and capability
  conditioning already appear in the abstract and method outline read. The claim
  of priority for directional maneuverability/capability tokens is unavailable.
- [Conformal Reachability, arXiv:2602.03799v1](https://arxiv.org/html/2602.03799v1):
  the abstract and introduction read already combine uncertainty intervals,
  reachability analysis and policy learning with a planning-horizon tradeoff.
  Calling Object B “conformal reachability” would not differentiate it.
- [Conformalized Distance Fields, arXiv:2607.00776v1](https://arxiv.org/html/2607.00776v1):
  the abstract/introduction read describe field-level functional conformal
  uncertainty, adaptive updates and MPC, including a 3D quadrotor evaluation.
  An uncertainty-aware clearance field by itself is not a fresh contribution.
- [Who Moved My Distribution?, arXiv:2511.11567v1](https://arxiv.org/html/2511.11567v1):
  the abstract/introduction read explicitly discuss controller-induced changes in
  reactive agents and iterative conformal adaptation. Ignoring this interaction
  issue would invalidate a broad calibration/safety claim.
- [AVO author page, ICRA 2011](https://gamma-web.iacs.umd.edu/AVO/), foundational
  context outside the 2022–2026 frontier: acceleration-constrained reciprocal
  avoidance predates this project. A scalar stopping-distance term is not a new
  collision-avoidance principle.

A defensible contribution would need a useful, computationally affordable
action-conditioned representation/approximation of shared recovery, clearly
separated from those existing methods, with both geometry and closed-loop
evidence at matched information and compute. The elementary inequality above
does not supply that contribution. No novelty clearance is asserted.

## 6. Smallest future falsification sequence, not a launch plan

Any future execution first needs its own reviewed, registered protocol, owned
outputs and visible concrete commands. It must not modify current sources,
configs, active runs or sealed maps. The following sequence identifies evidence
that would be needed, without authorizing execution.

1. **New development geometry witnesses.** Specify exogenous near-range crossing
   objects with verified nonzero motion and clearance/recourse conflicts inside
   the actual horizon and sensing range. Include static, single-moving-hazard
   and shared-recourse-conflict controls. Validate actual kinematic limits,
   valid masks, nonsaturated channels, scene hashes and action-set coverage.
   A scene name is not evidence of dynamics. The strict-gap example above must
   be realized with all 3D escape directions accounted for before it counts.
2. **Information-identifiable counterfactuals.** Keep current geometry comparable
   while varying a genuinely observed motion history, known actuation limit or
   declared motion law. Give every learned baseline the same history and motion
   information. If two futures differ but all available histories are identical,
   no predictor can be required to distinguish them; their branches belong in
   uncertainty/coverage tests instead. Oracle future information is a separately
   labeled reference and never enters only the candidate's deployable input.
3. **Mechanism/content controls before training.** Compare current static/full,
   constant-filled channels of equal width, an action-conditioned point/CV risk
   model, per-constraint relaxation and the shared-recourse approximation.
   Keep feature dimension, capacity, preprocessing, information and per-decision
   compute measured and matched as far as the question permits. Break the
   action-to-margin association while retaining channel distributions as a
   separate diagnostic; do not silently relabel shuffled values as geometry.
   Report disagreements, unknowns and false-safe/false-unsafe classifications,
   not only an average feature norm.
4. **Prediction versus decision value.** On disjoint new development episodes,
   compare the same downstream recourse method using a point/CV predictor,
   the proposed predictor/uncertainty model and a labeled oracle-motion
   reference. Conversely hold one predictor fixed and change only the recourse
   representation. Report trajectory error, joint coverage where meaningful,
   action-ranking/regret diagnostics and task outcomes separately. An oracle
   gain with no real-predictor gain suggests a prediction/calibration bottleneck;
   no oracle gain challenges relevance of the representation or scene. Those
   are diagnostic interpretations, not automatic causal proofs.
5. **Only then, separately proposed closed-loop comparison.** All arms require
   identical task reward/capture rules, fresh matched training seeds, budgets,
   deterministic or stochastic evaluation contract, maps and failure accounting.
   Include a viable common MASAC baseline and computable acceleration-aware
   avoidance/MPC or CBF references under documented information/compute. Compare
   mechanistically with SCOPE/DGPPO/CC-HGCBF without assuming a port is already
   fair or importing unapproved modules. Report joint success and collision/
   timeout tradeoffs; safe immobility or reduced stall alone is insufficient.

Stop the proposed line before adding training if the designed witness has no
genuine dynamic conflict, input identifiability is absent, valid signals remain
saturated/constant, or shared-recourse scores fail to distinguish the predeclared
conflict from their relaxation. Stop or revise the hypothesis rather than add
budget if oracle content gives no decision value on an appropriate witness, the
approximation systematically calls relaxed conflicts safe, uncertainty coverage
is unsupported, or matched closed-loop results remain negative/mixed. Thresholds,
seeds, scene population and decisions would have to be fixed before those tests;
no threshold is selected from the active near-objective results.
