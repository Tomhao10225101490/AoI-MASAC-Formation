# Direct prior art narrows the joint-recourse hypothesis

Date:2026-11-05. Primary-source review; no experiment launched and no novelty
clearance. Complements the existing2022–2026 contribution criteria and the
unimplemented joint-recourse hypothesis. Negative near-objective results do not
validate a different representation by elimination.

## Three direct precedents newly checked

**Yang and He,2026-10-12 arXivv1**, [Joint-Feasibility Margin Feedback for
Bounded-Input Second-Order Control Barrier Functions](https://arxiv.org/html/2610.16978v1).
Read SectionsIII–V, Equations9–13 and Proposition1. They maximize minimum affine
HOCBF slack over bounded inputs, derive a support-function dual and predict
joint margin for nine candidate commands before a hardQP. This is direct prior
art against claiming the first signed shared-control margin or predicting
incompatibility despite individually feasible constraints. Their object is a
local pointwise barrier margin with specified neighbor accelerations and a
one-step forecast, rather than our proposed multi-step causal recovery object.
That distinction is a question, not evidence of advantage. Preprint; no accepted
venue is asserted here.

**Sah and Keshavan,2026-09-14 arXivv1**, [Exact Feasibility Certification and
Optimal Responsibility Allocation for Multi-Robot CBF Safety Filters](https://arxiv.org/html/2609.14935v1).
Read formulation and SectionIII through the feasibility-reserve definition.
They treat heterogeneous control-affine agents, compact convex input sets,
stacked barrier constraints and support-function feasibility reserve; their
method also allocates shared responsibility. Thus actuator demand/supply,
constraint-conflict localization and a common reserve are not empty gaps.
We have not reproduced their experiments or independently audited every theorem
and its conditions. Their pointwise encoded-barrier certificate must not be
equated to our uncomputed finite-horizon geometric recoverability. Preprint;
no accepted venue is asserted here.

**Choi et al.,RSS2025**, [Resolving Conflicting Constraints in Multi-Agent
Reinforcement Learning with Layered Safety](https://www.roboticsproceedings.org/rss21/p094.html),
DOI10.15607/RSS.2025.XXI.094. Official proceedings abstract verified: MARL reduces
multi-agent conflict exposure; pair urgency and reachability/CBVF filtering
provide additional layers, with drone hardware and aerial-mobility evaluations.
This round does not claim a complete proof or implementation review. It rules
out “MARL anticipates higher-order conflicts” as our general new contribution.
Its safety-filter interface differs from our current observation-only MASAC.

## Mathematical distinction that still needs to earn its cost

For fixed information, disturbance assumptions and committed first action a,
write F_l(pi,w)=min_k g_l(x_k) over a fixed discrete horizon. Our proposed object
is R_joint=sup_pi inf_w min_l F_l(pi,w), versus
R_pair=min_l sup_pi_l inf_w F_l(pi_l,w). Same domains imply R_joint<=R_pair;
restricting recourse to a library B gives R_B<=R_joint only when the same complete
disturbance treatment is retained. These are standard optimization order and
set-inclusion facts, not a new theorem or the implementation of the cited CBFs.
Sampled disturbances do not confer robust safety; central state or oracle peer
futures cannot enter only our deployable candidate.

A possible useful difference would be **a cheap action-conditioned, multi-step
common-recovery feature whose measured decision benefit survives matched
information, feature capacity and compute controls**. No such benefit or new
feature has been established. More horizon can also increase model mismatch and
cost, so it cannot be declared superior to pointwise/one-step methods by design.

## Next minimal decision, before training

First review whether a faithful, tractable comparison can be built inside the
current bounded-acceleration3D task. A protocol must specify actual limiter/
integration order, safety normalization, sensing, action/backup sets, exogenous
motion, no privileged futures, genuine near-range conflicts and all vertical/
longitudinal escape directions. Include no-conflict and one-hazard controls.
Measure candidate support, relaxation/shared-recovery disagreements, discretized
false-safe/false-unsafe errors and real latency against a common reference.
No scene name, analytic1Dwitness or low-dimensional controller is a3DUAV result.

Advance only if the intended disagreement exists, input information identifies
it, approximations retain it and cost is reasonable. Otherwise revise the
hypothesis without launching learned arms. Closed-loop efficacy would still
require separately registered static/constant-fill/full and shared-recovery
controls, matched MASAC capacity/budgets, all seed endpoints, collision/timeout
accounting and credible external references. No direct performance comparison
or “first/SOTA” claim is currently defensible.
