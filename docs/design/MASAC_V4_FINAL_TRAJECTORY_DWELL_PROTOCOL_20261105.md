# Final-trajectory dwell and Path-C reward decomposition protocol

**Root-reviewed final protocol; first execution requires this version to be
committed and pushed.** The earlier interrupted draft is archived in
`../FINAL_TRAJECTORY_DRAFT_RECOVERY_20261105.md`; its hash snapshot describes the
old draft, not this reviewed version. Root and an independent read-only reviewer
checked reset memory, realized acceleration, shared terminal reward, time
conventions and missing-data handling before execution.

Date: 2026-11-05. Status: specified before executing this diagnostic, after the
40k study endpoint and the separate actor diagnostic were known. This is a
post-hoc descriptive analysis, not a new efficacy endpoint or a modification of
the registered advancement rule. No further training or evaluation is authorized
by this document.

## Fixed population and inputs

Analyze all 72 existing final development trajectories: static control and
static+B, training seeds 1511, 1522 and 1533, evaluation step 40000, map seeds
830001--830012. Read only each history's `evaluation_40000.json`, saved
`eval_40000_<map>.npz` and map metadata JSON, the existing per-seed trajectory
summary, the two frozen configurations and the named source/protocol files.
There is no checkpoint loading, policy execution, new environment step, new map,
sealed-test access, reward search or change of the C/B implementation.

The prior actor diagnostic used late-replay states. Its mean inward acceleration
and mean one-step distance change do not describe the final policy's actual
on-policy dwell. This diagnostic instead describes the saved final episodes;
it cannot identify a cause of their failures.

## Frames, predicates and denominators

The saved positions and velocities contain reset frame 0 and T post-step frames.
Use h=0.2 seconds, d[t,i]=norm(goal[i]-position[t,i]), speed[t,i]=norm(v[t,i]),
and realized acceleration a[t,i]=(v[t,i]-v[t-1,i])/h. Only t=1,...,T enters
dwell, occupancy, acceleration and reward summaries. Frame 0 initializes the
progress and acceleration differences. The shaper's reset memory is explicitly
false for both ever-in-ball and captured, regardless of the reset geometry.

For every agent-map, report all eight disjoint distance/speed bands:
distance [0,4), [4,5), [5,10), [10,infinity), each crossed with speed [0,2)
or [2,infinity). Report frame count, sampled occupancy time (count*h), fraction
of that agent's post-step frames, and longest uninterrupted sampled run in
frames and occupancy seconds. Retain zero-count bands and every map.

Fix the descriptive near-stall predicate to 4<=d<10 metres and speed<0.5 m/s.
Report total occupancy, maximum run, and every run containing at least 25
consecutive post-step samples (5 seconds of sampled occupancy). A run of N
samples has an endpoint-to-endpoint span (N-1)*h; N*h is the occupancy convention,
not proof of continuous residence between samples. This label does not establish
hovering, obstacle clearance, an attractor, reward preference, or causation.

Agent-frames, agent-maps, and repeated maps from a training seed are correlated.
Report their actual denominators, without independent-trial confidence intervals
or significance claims. Pooling is descriptive; retain history and map results
so different episode lengths and terminal censoring remain visible.

## Capture and consistency checks

Use the strict post-step predicates ball=(d<4) and capture=(d<4 and speed<2).
For every map recompute A=all agents ever enter the ball, C=all agents ever
individually capture, and J=any frame with all three simultaneously captured.
Keep these three events distinct. Retain first ball/capture times, capture-entry
and capture-exit times, capture-frame counts and longest capture runs. Entry/exit
transitions use the shaper's false reset capture memory; the unobserved interval
after termination contributes no exit event.

Cross-check available evaluation fields and the existing trajectory summary:
per-agent ever-ball (post-step only) and closest distance (including reset, as in the writer),
final distances, A/C/J, maximum simultaneous capture occupancy, first capture
times, capture-frame counts and longest runs. Capture events newly measured here
have no prior exit-time field to validate against. Verify equal map family and
scene hash across all histories. Preserve all four nonexclusive collision flags;
the fixed 72-episode population is expected to have no collisions. Inconsistent
numeric data, metadata or existing summaries fails the audit with a traceback;
it is not silently omitted.

When actions and scene metadata exist, independently transcribe the frozen
`limited` update using clipped float64 actions, unit-norm action limiting,
speed projection and semi-implicit position update. Require maximum absolute
velocity and position discrepancy <=1e-9. Read dt, amax and vmax from each saved
scene; require dt=0.2 and the frozen horizon. This verifies saved transition
consistency, not collision feasibility. Missing action/metadata makes only this
cross-check unavailable. JSON identity, success and collision flags are checked
even when a trajectory is missing. An unsuccessful collision-free trajectory
must reach the horizon; success may coincide with horizon truncation.

## Formula-derived reward components and unavailable return reference

Reconstruct the source Path-C **base evaluation reward**, with B shaping disabled
in evaluation. At each post-step, retain each agent's progress d[t-1]-d[t], living
cost, squared realized-acceleration cost, near-goal term, first-ever ball-entry
bonus, signed capture-state change, joint terminal bonus and collision term.
Use the recorded `reshape` coefficients after verifying them against the frozen
configuration. Ball memory is cumulative; ball entry pays once per agent per
episode. Unlatched capture pays signed entry/exit changes. Team reward is the
mean of the three per-agent values; a successful terminal transition adds the
same terminal bonus to every agent, thus only one terminal bonus to the mean.
There are no collision events in this population; if a collision flag is found,
stop rather than guess its timing from episode-level flags.

Report component totals both undiscounted and with raw weights 0.99^(t-1),
including per-agent totals and their team mean. A truncated raw discounted sum
is not a bootstrapped soft-Q target and does not include terminal value or entropy.
The saved evaluation writer has **no return/reward/episode-return field**.
Therefore `reward_sum_reference` and reward-reference validation must be
`UNAVAILABLE`; formula-derived totals are not represented as verified against a
logged evaluation return. The preregistered comparison tolerance, if an audited
evaluation base-return reference were available, is
1e-4 + 1e-5*abs(reference). Training `base_return` is a different episode
population and must never be substituted. This script does not add such a
reference after inspecting the totals.

Missing trajectory arrays make their dependent measurements `UNAVAILABLE` and
retain the map in the output. A missing reward-only field must not suppress an
otherwise valid dwell analysis. Shape/type/nonfinite violations are errors,
not missing-data imputations.

## Reproducibility and completion

`scripts/audit_sync_final_trajectory_dwell.py` accepts only an output directory;
its input population is fixed relative to the repository. It creates a fresh
directory, streams one line per map to stdout and `audit.log` with flushing, and
writes per-map records, summary JSON/Markdown, manifest and status. The manifest
records SHA-256 hashes of each history's evaluation/config/prior summary, each
map's NPZ/metadata, this protocol, this script and the named reward/writer/helper
sources. Before measurements it also checks both original run manifests, all
six seed completion records, all 14 original frozen training-source hashes and
both original config hashes. Prior summaries must match seed, step, paired-map
verification and map identity. The input manifest and numerical environment are
written before measurements; input hashes are rechecked before completion. Missing inputs remain
explicitly unavailable. The completion marker is written last, after all other
artifacts, and hashes all six final output artifacts; exceptions write FAILED status and a full traceback without a complete
marker. Completion of a descriptive audit does not turn unavailable reward
validation into a pass.

The original 40k efficacy decision, final endpoints, source/configuration freeze,
and prohibition on automatic training extension remain unchanged. Neither dwell
nor formula-derived reward components establish that staying near a goal is
preferred to completing a feasible path, or that a policy exploited a reward.
