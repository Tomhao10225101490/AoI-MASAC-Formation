# Completed fixed-state offline actor-response diagnosis

2026-11-05, Asia/Shanghai. Protocol and code were pushed as `3615a66` before
the first checkpoint load. All six histories completed; the actor values and
global CPU/CUDA Torch RNG states were unchanged. No training, Q/TD evaluation,
environment construction/step, new rollout or sealed evaluation occurred.

## Population and response definition

Exactly the PhaseA6144 joint observations/18432 agent observations and their
hashes/indices were reused. Each final40k actor responds only to its own late
replay. These are correlated states from static control/B training, not final
on-policy trajectories or a paired cross-arm state intervention.

The deterministic response is `tanh(mu)`, not expected squashed action. Exactly
64 common3D Gaussian vectors from isolated CPU seed20261105 supply **per-agent
marginal** responses; reusing them across agents does not sample an independent
team product policy. No joint probability or confidence interval is computed.

Float32 actor outputs precede float64 decoding/kinematics. The one-step strict
set is distance<4m and speed<2m/s after the frozen acceleration/speed limiter;
this is not collision feasibility or episode success. Original strict predicates
are unchanged. Near-boundary quantization counts remain available, and a0/64
sample count is not a true-zero-probability claim.

## All six histories: primary near-goal subset

The prespecified subset is4<=distance<10m and speed<2m/s.
MC-minus-deterministic differences are formed per observation-agent before
pooling; percentage points below are descriptive, not inferential.

| Actor / own replay | Agent observations | Deterministic memberships | Marginal MC memberships /64n | MC minus deterministic, pp | Mean inward acceleration, m/s² | Mean distance change, m/step |
|---|---:|---:|---:|---:|---:|---:|
| Control1511 | 1115 | 9 | 635/71360 | +0.08268 | 1.78907 | -0.06622 |
| Control1522 | 1364 | 18 | 1164/87296 | +0.01375 | 1.38974 | -0.04932 |
| Control1533 | 1418 | 43 | 2553/90752 | -0.21928 | 1.91328 | -0.06732 |
| B1511 | 1563 | 34 | 2054/100032 | -0.12196 | 1.97158 | -0.07412 |
| B1522 | 1101 | 18 | 1227/70464 | +0.10644 | 2.11768 | -0.07398 |
| B1533 | 1439 | 46 | 2807/92096 | -0.14876 | 2.24075 | -0.09473 |

Pooled descriptive total:168/8000 deterministic memberships versus10440/512000
marginal MC memberships, difference **-0.0609375 percentage points**. Three
histories have positive and three negative MC-minus-deterministic differences.
Mean deterministic inward acceleration is1.90513m/s² and distance change
-0.0712755m/step; the64-draw mean distance change exceeds the deterministic
change by+0.00255531m/step. These conditional averages do not say every state
has an inward response or show that one arm controls better than the other.

Thus this sample does not support a uniform one-step rescue by stochastic
actions. It does not refute a multi-step stochastic/deterministic occupancy
difference or identify the source of the72 final evaluation outcomes.

## Necessary distance bounds and the boundary layer

By triangle inequality a step ending below2m/s moves less than0.4m at h=.2,
so starting distance>=4.4m cannot strictly capture in that step. The stronger
state-dependent necessary bound is

    distance < 4 + h*min(2, vmax, speed + amax*h).

It is not sufficient for a feasible capture: velocity direction and collision
constraints are not resolved by this bound.7359/8000 primary states fail this
necessary distance bound. Their zero memberships do not diagnose actor failure.

The predeclared4–4.4m/speed<2 boundary layer has802 states,161 excluded by the
tighter bound. Deterministic membership is168/802; marginal MC membership is
10440/51328, paired mean difference **-0.6078554 percentage points**. All
distance/speed bins, including empty groups, remain in the JSON/CSV. Do not
reinterpret the remaining641 as certified feasible opportunities.

## Integrity and the next evidence gap

Six checkpoint/replay identities and selected-input hashes were checked before
actor extraction; only own actor parameters were consumed from trusted local
containers. Four fixed-vector tests match the original limiter. All48 chunks
passed geometry/numerical checks. Afterward, independent output verification
checked completion-listed file sizes/SHA256s, current script/protocol hashes,
actor hashes/RNG checks, complete bins and paired differences against raw counts.
`completion.json` is authoritative and was written last by the audit process.

Next inspect the **already saved72 final trajectories** for consecutive near-goal
low-speed dwell and source-derived reward components. This addresses the mismatch
between replay-state response and actual final trajectories without another
training budget or learned-policy rollout. Evaluation JSONs lack accumulated
return, so any reward decomposition must label reference-return verification
UNAVAILABLE instead of inventing it. No reward adjustment is yet justified by
these descriptive results; the registered40k decision and sealed test stay fixed.

- Protocol: `design/MASAC_V4_NEAR_GOAL_ACTOR_OFFLINE_PROTOCOL_20261105.md`.
- Code: `scripts/audit_sync_near_goal_actor.py`.
- Output: `results/masac_v4_sync_40k_near_goal_actor_audit_20261105/`:
  audit.json/audit.md, groups.csv, manifest/status/completion, indices and noise.
- Tee/verification: Windows audit root's near_goal_actor_console.log,
  near_goal_actor_summary_console.log, near_goal_actor_output_verification.json
  and near_goal_actor_verification_console.log. No binaries uploaded.
