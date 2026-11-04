# Registered40k study: two completed pairs

**Snapshot:4/6 histories complete; study IN_PROGRESS.** Both original workers
have cold-started the registered third seed1533. Its final40k endpoints are
pending. No source, configuration, budget, endpoint or decision rule is changed.
The sealed map set has not been evaluated.

## 1. Fixed endpoints observed so far

| Training seed | Control success/12 | B success/12 | B minus control, success count |
|---|---:|---:|---:|
| 1511 | 2 | 6 | +4 |
| 1522 | 1 | 0 | -1 |
| 1533 | pending final40k | pending final40k | pending |

The first improvement does not reproduce on the second seed. This is a mixed
partial result, not the full three-seed treatment-effect or advancement decision.
The independent analyzer returns `IN_PROGRESS`, `decision_ready=False`, and
`decision=None`. Do not apply the full rule to a partial sum, replace40k by a
checkpoint peak, select the favorable seed or extend a history's budget.
Both completed seeds in both arms miss the per-seed8/12 gate; nevertheless all
registered histories must finish. The relative engineering advancement rule and
the absolute all-seed task gate are distinct and neither implies publication
readiness, predictive-content utility or statistical significance.

## 2. Second pair: exact failure quantifiers

The seed1522 `completion.json` files report step40000 and success fractions1/12
and0/12. Their final evaluation rows agree. The saved trajectories were recomputed
with the same offline analyzer as seed1511, with metric and paired-map checks.

Let A mean every UAV enters its ball at some time; C mean every UAV individually
meets distance<4m and speed<2m/s at some time; J mean every UAV meets that predicate
at one common post-step time. Then J implies C implies A. All24 final1522 episodes
are collision-free, so J equals the terminal success event for this description.

| Final40k, seed1522 | Static control | Static+B |
|---|---:|---:|
| Success, J | 1/12 | 0/12 |
| Any-agent eventual ball entry | 9/12 | 11/12 |
| All-agent eventual ball entry, A | 1/12 | 2/12 |
| All-agent individual capture, C | 1/12 | 2/12 |
| Failure: not A | 11 | 10 |
| Failure: A but not C | 0 | 0 |
| Failure: C but not J | 0 | 2 |
| Individual never-ball records | 20/36 | 17/36 |
| Individual ball-without-capture records | 1/36 | 0/36 |
| Individual ever-capture records | 15/36 | 19/36 |

B's two C-but-not-J maps are830001 and830003, each with maximum simultaneous
capture occupancy2/3. Control succeeds only on830009, on which B fails.
Although B has more individual capture records and more all-agent eventual-entry
maps, its joint-success count is lower in this pair. These observed proxy rankings
cannot substitute for the registered target. They do not identify a synchronization
mechanism or establish that B causes a particular failure mode.

The36 individual/map records per arm are correlated measurements from one
training seed. The same12 development maps recur across seeds/checkpoints.
Neither pooling those records nor comparing selected checkpoints supplies extra
independent training replicates or unseen-map evidence.

## 3. Numerical and artifact scope

The initial monitor found both frozen protocol hashes and all14 declared source
hashes unchanged, no error.json, and finite current learning diagnostics. Each
completed1522 history reports39000 updates. The canonical analysis verifies its
available final logged finite diagnostics. The dedicated read-only artifact audit
also returned **PASS** on both1522 histories: all195 recorded post-warmup progress
diagnostics per arm are finite; all14 declared source hashes, both registered
protocol hashes and all four checkpoint/replay byte hashes match their manifests.
The195 samples are not39000 individually retained update records. Byte matching
does not establish exact CUDA resume or learner-path equivalence.

The canonical partial report also records critic clipping on every sampled
post-warmup update in the four completed histories. This is a numerical observation,
not proof of instability or an identified reason for the outcome. No clipping,
reward, entropy target or optimizer change is made during the study.

## 4. Evidence and next fixed action

- [Second-pair trajectory report](../results/masac_v4_sync_40k_progress_20261105/trajectory_40000_seed1522/summary.md)
- [Second-pair trajectory details](../results/masac_v4_sync_40k_progress_20261105/trajectory_40000_seed1522/summary.json)
- [Second-pair artifact verification](../results/masac_v4_sync_40k_progress_20261105/seed1522_integrity/verification.json)
- [Reusable read-only verifier](../scripts/verify_sync_seed_artifacts.py)
- [Canonical partial analysis](../results/masac_v4_sync_40k_analysis_20261105/analysis.md)
- [First-pair evidence](SYNC_CAPTURE_FIRST_PAIR_20261105.md)
- [Registered protocol](design/MASAC_V4_SYNC_CAPTURE_40K_PROTOCOL_20261105.md)

Continue only the already-running1533 histories. When both arm root completions
and all six seed-local completions exist, run the independent analyzer and report
all three fixed pairs, all failures, the original relative advancement rule and
the absolute gate. No automatic continuation, new coefficient sweep or sealed-map
evaluation is triggered by this partial result. Finite neural-learning differences
also do not refute or verify the exact PBRS result; its implementation boundaries
remain those in [the scope note](MATH_PBRS_SCOPE_NOTE_20261105.md).
