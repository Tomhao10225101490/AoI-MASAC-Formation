# Static control / B: complete registered 40k development result

Date: 2026-11-05, Asia/Shanghai. Protocol registered and pushed in `eaf83e2`
before the04:49 launch. Both workers finished normally at07:22/07:23.
All six cold histories reached40,000 transitions and39,000 updates.
No learning source, configuration, capture rule or budget changed in the study.

## Primary endpoint and fixed decision

| Training seed | Static control /12 | Static+B /12 | B minus control /12 |
|---|---:|---:|---:|
| 1511 | 2 | 6 | +4 |
| 1522 | 1 | 0 | -1 |
| 1533 | 4 | 5 | +1 |

Mean paired success-fraction difference is `(4-1+1)/(3*12) = 1/9`, or
11.11 percentage points. The original engineering rule is **MET**:
mean difference >=1/12, two strictly positive pairs, no pair below -1/12,
and all six histories completed with finite recorded diagnostics and unchanged
contracts. The independent analyzer reports `COMPLETE`, `decision_ready=true`,
`ADVANCEMENT_RULE_MET`. The negative seed1522 is retained; its loss is exactly
the preregistered tolerance, not a reason to alter that rule.

**Both arms FAIL the separate absolute gate: zero of three seeds reach8/12.**
Meeting the relative engineering rule supports designing confirmation only.
It does not launch confirmation, extend these runs, open the sealed maps, or
establish significance, reliability, novelty, safety or predictive-feature value.

There are three independent training replicates. The same twelve development
maps and repeated checkpoints are repeated measurements. Totals7/36 versus11/36
are descriptive and must not be treated as36 independent training trials.
Training scenes are paired by seed/episode-index generator, but policy-dependent
episode lengths change transition-wise exposure. The development maps are exactly
paired. These are new CUDA40k cold runs; old CPU80k V4 and warm-start Standard-C
histories have different budgets/initialization and are not matched controls.

## Complete final-trajectory diagnosis

For strict post-step predicates, let `A = forall i exists t: distance_i(t)<4m`,
`C = forall i exists t: distance_i(t)<4m and speed_i(t)<2m/s`, and
`J = exists t forall i: distance_i(t)<4m and speed_i(t)<2m/s`.
Capture is unlatched, so `J => C => A`. Collision-free J is success. All72 final
episodes have no collision flags; that observation is not a safety guarantee.

| Arm | Success J | Failure: not A | Failure: A and not C | Failure: C and not J | Any ball | A | C |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static control | 7 | 25 | 0 | 4 | 33 | 11 | 11 |
| Static+B | 11 | 18 | 1 | 6 | 35 | 18 | 17 |

Thus43 of54 failures lack even all-agent eventual ball entry; one has all-agent
ball entry without all-agent individual capture, and ten have individual
captures without temporal overlap. This is primarily incomplete approach with
additional speed/retention/timing failures, not a purely synchronization problem.
More B episodes reach the later event classes, so raw conditional failure counts
do not show that B causally improves or worsens retention. No timing gap replaces J.

All72 stored final trajectories were independently checked against evaluation
JSONs: paired scene hashes, any/all ball entry, reset-inclusive closest distance,
post-step maximum capture occupancy, and collision-free success. Per-agent
distance, speed, capture frames/streaks and first-capture times remain available.
The helper's historical `POSTHOC_INTERMEDIATE_TRAJECTORY_DESCRIPTION` label in
each seed JSON describes its original diagnostic interface; these selected
step40000 records are the final endpoint trajectories. The efficacy decision
comes only from the independent registered-study analyzer.

## Integrity, numerical evidence and limitations

All12 final checkpoint/replay binary SHA256 checks match their manifests, all14
declared source hashes match, and both frozen protocol hashes match. The six
histories each have195 finite sampled post-warmup progress rows. The trainer also
applies a per-update finite guard during all39,000 updates/history; the195 rows
are not retained diagnostics for every update. No crash/error artifact occurred.
These checks do not validate exact CUDA resumption. No binary is uploaded to Git.

Sampled critic clipping fraction is1.0 in all six histories. This is a numerical
diagnostic, not proof of instability or a causal explanation of a performance
drop. Entropy/alpha, common-grid AUC, peak-to-last values and base/shaping reward
scales are retained in the canonical JSON; none overrides the primary endpoint.
Shaped training return is not a common cross-arm performance metric.

B remains a fixed bounded team-potential increment, not a scheduled curriculum
or an explicit arrival-gap penalty. Its usual telescoping identity requires the
information/boundary conditions in `MATH_PBRS_SCOPE_NOTE_20261105.md` and provides
no finite-data neural-SAC learning guarantee. Current L0 development maps contain
no external moving obstacles. Static+B cannot identify moving-obstacle prediction
utility, and peer motion must be distinguished from absent external movers.

## Durable artifacts and next analysis

- Protocol: `docs/design/MASAC_V4_SYNC_CAPTURE_40K_PROTOCOL_20261105.md`.
- Canonical decision: `results/masac_v4_sync_40k_analysis_20261105/analysis.{json,md}`;
  curves in the same directory. Full failure aggregation: `failure_taxonomy.json`.
- Final raw endpoints: each named arm's `maneuver_static_{1511,1522,1533}/evaluation_40000.json`,
  seed completion, checkpoint manifest and metrics; root completion covers all3.
- Integrity: `results/masac_v4_sync_40k_progress_20261105/first_pair_integrity.json`,
  `seed1522_integrity/verification.json`, `seed1533_integrity/verification.json`.
- Final trajectory checks: the same progress root's
  `trajectory_40000_seed{1511,1522,1533}/summary.{json,md}`.

The next minimal diagnostic is a separately frozen offline full/static/constant-fill
feature audit on the late replay states of all six completed histories, extending
the unrepresentative early2k sample. It loads no policy, performs no training or
new map evaluation, and cannot causally explain the historical full/static gap.
See `design/MASAC_V4_POST40K_NEXT_MINIMAL_PLAN_20261105.md`. Sealed test seeds remain
unopened. No confirmation training or budget extension is queued.
