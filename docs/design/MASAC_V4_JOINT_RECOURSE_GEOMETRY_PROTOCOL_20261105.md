# Registered synthetic 3D shared-recovery geometry diagnostic

Date2026-11-05. New experiment id`masac_v4_joint_recourse_geometry_20261105`.
Run only after root source/math review, selected-file commit and push. This is
not training, a learned-policy evaluation, an old study repair or a novelty claim.
The closed near-objective study and its20sources/configs remain untouched.
All scenarios are newly specified synthetic development inputs; no old weights,
replay, trajectory, development-map file or sealed test is accessed.

## Scientific question and prior art

Can a source-aligned3Dgeometry diagnostic distinguish individually recoverable
hazards from a conflict with no common recovery after the same committed first
action? Can acting earlier retain a common recovery? The
[direct prior-art audit](../FRONTIER_JOINT_FEASIBILITY_GAP_20261105.md) already
finds joint-feasibility margins and MARL conflict handling. This diagnostic is
instrument validation and a relevance check, not a new theorem/method or evidence
of learned benefit. A later finite-budget MASAC feature comparison would require
a separate protocol and credible matched baselines. No automatic launch follows.

## Source-aligned model and information

Use dt=.2s,amax=4m/s2,vmax=10m/s,body radius1m,margin0,world240x240x90m.
Normalized action components lie in[-1,1]; the effective acceleration direction
is Euclidean unit-ball projection P(a). Update velocity first with source
`structured.limited`, then p_next=p+dt*v_next. Motion within each tick is linear.
Source`environment.swept_distance` measures closest relative separation per
segment; its denominator regularization1e-12 is preserved and independently
checked. Boundaries use their endpoint minima, exact for each affine segment.
The float64 diagnostic kernel must match these NumPy source functions to1e-9
on independent generic states, including active speed limiting. No rigid-body
flight or continuous-acceleration integration is claimed.

There are three UAVs, independent fixed goal coordinates, all initial velocities
zero. Only ego0's second action varies. Peer1and2 hold zero for both ticks;
their actual positions are known and stationary. This is an **ego recovery
slice with fixed peer backups**, not optimization over three unconstrained
cooperative policies. No environmental constructor/reset/step or learned actor
is used. A source pure`V4Mission.observe` call on a synthetic record produces
the same162Dlayout and visibility masks; it is not an episode. Obstacles move
exogenously at a specified constant3m/s, do not react to controls and stay far
from source bounce thresholds over the two ticks. Their current velocity is
observed by all feature comparisons. No method receives oracle hidden futures.
No buildings, communications, attention or L1training are introduced.

For each of19sourceDIRECTIONS as the committed first action a, choose one
subsequent ego action b. At H=2ticks, define each dimensionless constraint
F_l(a,b)=minimum swept clearance or boundary-face clearance divided by1m.
Include every moving sphere, both peers and all six boundary faces, with common
dynamics/information. Denote

    J(a)=max_(b in effective unit ball) min_l F_l(a,b),
    P(a)=min_l max_(b in same ball) F_l(a,b).

Then J<=P by optimization order. On the19backup-action library B, compute
J_B=max_b min_l F_l and P_B=min_l max_b F_l. J_B<=J and J_B<=P_B, but P_B is
not generally an upper bound on unrestricted J. Keep these quantities distinct.
There is no disturbance uncertainty in this diagnostic and hence no robust
uncertainty or causal-feedback guarantee. Recovery here is one subsequent action.

## Fixed cases, complete3Descape accounting

Config`configs/masac_v4/joint_recourse_geometry_20261105.json` freezes all geometry.
Relative to ego's initial location, the conflict case has two raw radius1m
spheres initially at(1.2,+/-1.95,0)m, velocity(-3,0,0)m/s. Their raw separation
3.9m exceeds their2m summed radii throughout: external obstacles do not overlap.
The ego-inflated radius is2m. At the first zero-action tick, their relative
centers are(.6,+/-1.95,0); at the second, (0,+/-1.95,0). Initial and first-tick
ego clearances are strictly positive. Boundaries and peers are remote and still
included in all optimization problems.

With a=0, reachable final ego displacement has norm at most
r=amax*dt^2=.16m. For any3Ddisplacement q in that ball, the distance to at least
one final sphere center is at most sqrt(s^2+||q||^2)<=sqrt(s^2+r^2),s=1.95.
Therefore J(0)<=sqrt(s^2+r^2)-2<0. This bound covers vertical and longitudinal
escape as well as lateral escape; no component is frozen. Single-obstacle
recovery can instead use opposite lateral full acceleration. Its exact swept
segments supply a positive per-constraint lower witness, checked numerically
against source functions. For a=-approach and b=-approach, final displacement
is-.48m along approach and minimum clearance is sqrt(s^2+.48^2)-2>0; segment
and all other constraints must also validate. These expectations are analytic
design predictions, not measured outcomes or hyperparameters selected from data.

Nine fixed cases: empty, each single obstacle, two static at the same initial
geometry, two receding at the same initial geometry, wider closing(s=2.3),
and the conflict repeated with approach axes x,y,z. Every19first action is
reported in every case, not just the favorable actions. Changing motion is
observationally identifiable through velocities. All control outcomes are kept.

## Full-domain numerical reference and its limitations

Enumerate the65^3Cartesian grid q in[-1,1]^3 and apply the same effective ball
projection via the limiter. Chunk65536, float64; keep maximizing actions and
all per-constraint maxima. The grid includes all19libraryeffective actions.
Any effective action has a grid representative within
rho=sqrt(3)/(65-1) in Euclidean norm, since projection is nonexpansive. With the
first action fixed, second-segment position is r-Lipschitz in effective b;
speed projection is nonexpansive. Each normalized minimum clearance is therefore
at most L=r/1m times that distance. In exact arithmetic,

    J_G <= J <= J_G + L*rho,
    P_G <= P <= P_G + L*rho.

Report these brackets with explicit1e-9numerical allowance, not just grid signs.
This allowance and source comparisons are engineering numerical checks, **not
interval arithmetic or formally machine-verified rounding certification**.
The separate analytic endpoint bound is the mathematical no-escape argument.
Do not generalize a one-backup-action bound to longer recourse, arbitrary
obstacle laws, a learned policy, all continuous first actions or real flight.

## Observation-content and source checks

On the same synthetic source observations, save raw full/static/static-fill-one
19x11tokens using a declared.4sprobe horizon matching the diagnostic duration.
Static-fill-one changes only moving/peer output columns4and5to1. Check valid
moving slots, observed velocities, nonsaturation and full-vs-fill differences.
Repeat canonical CPU probes on selected device and require maxdifference<=1e-4.
These source probes hold one action for both ticks, unlike committed-a then
variable-b recovery. They are **content diagnostics only**; do not compare their
action ranks to J as if their control families were matched or claim actor use.
No training-mode source default/horizon is changed.

## Compute, provenance and visible progress

Source/config/protocol/tests must be tracked and clean at launch. Record exact
gitHEAD, actual source hashes and source byte snapshots before measurement.
Fresh named output is mandatory; refuse overwrite. Never edit old20sources.
Performance options CPUthreads3,CPUthreads6 and CUDAthreads1 run the same full
65^3conflict/zero-first kernel,1warmup+3timed repeats, including grid construction
and reductions with synchronization. Log all repeats, agreement and peak memory;
select lowest measured median for this diagnostic only. This choice affects
execution, never cases/endpoints/thresholds. CUDAabsence is recorded, not a
training failure; no old trainer device default is changed.

All171case/first-action combinations then receive both library and full-grid
analysis. Print case/action, counts, bracket, throughput, elapsed time and any
errors with flush; tee disk/screen. There is no episode seed, step budget,
success/ANY/ALL-ball or checkpoint: log these as NOT_APPLICABLE, not zero.
Append progress JSONL/logs; write each manifest/result/error/completion path
exclusively, without replacing an open status target. Initial status is a start
record; final completion or error is authoritative. This avoids introducing
the previous writer pattern but does not repair the old coordinator. Copy
source bytes; write artifact-hashed completion last. Unexpected runtime/numeric
errors produce a preserved stack/error and no success completion. A negative
scientific result still produces a truthful diagnostic completion and FAIL flag.

## Fixed decision and stopping rules

Call diagnostic relevance PASS only if all source/finite/grid containment and
metadata checks pass; all three conflict rotations at first0 have P lower>0
and J upper<0; all three admit a positive common-recovery lower witness for
first retreat; no-hazard, single-hazard, static, receding and wide controls admit
positive J lower for first0; moving content is valid/nonsaturated/different from
static-fill-one; and every grid-selected witness agrees with independent NumPy
source replay of the two mathematical transitions within1e-9. Report every
failed condition and all171rows. No held-out success or novelty inference.

If any condition fails, preserve it and diagnose source/geometry/information/
approximation before proposing more work. Do not alter this config, add cases,
choose a different horizon/grid or train based on favorable subsets. Changing
the design requires a new experiment identity and recorded rationale. Even a
PASS only supports designing the next minimal mechanism comparison; it does
not authorize large training or sealed access. Published reference controllers
must not be relabeled as reproduced or defeated by this synthetic check.
