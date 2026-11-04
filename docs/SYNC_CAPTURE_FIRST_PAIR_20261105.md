# Registered40k study: first completed pair, seed1511

**Historical snapshot after the first pair: IN_PROGRESS,2/6 histories complete.** This is a record
of the first seed's registered final40k endpoints, not the three-seed efficacy
decision. Seeds1522 and1533 retain their original budgets and rules. No parameter,
source, gate, continuation budget or sealed-map evaluation has been changed.

## 1. Authoritative endpoint records

Both seed-local `completion.json` files report seed1511 and step40000. Their
success fractions match the12 rows of `evaluation_40000.json`. Both final
progress records report39000 learner updates, after1000 warmup transitions.

| Final40k, seed1511 | Static control | Static+B |
|---|---:|---:|
| Joint success | 2/12 | 6/12 |
| Any UAV ever in its goal ball | 12/12 | 12/12 |
| Every UAV ever in its goal ball, A | 5/12 | 8/12 |
| Every UAV individually captured at some time, C | 5/12 | 8/12 |
| Simultaneous capture, J | 2/12 | 6/12 |
| Collision maps | 0/12 | 0/12 |
| Failure: incomplete all-agent ball entry, not A | 7 | 4 |
| Failure: A but not C | 0 | 0 |
| Failure: C but not J | 3 | 2 |
| Per-seed gate, at least8/12 | false | false |

The seed-level success-count difference is **+4/12**. It is one of the three
registered paired differences. It does not satisfy the full advancement rule
by itself, establish statistical significance or identify a causal mechanism.
Both first-seed gates fail; this does not permit early stopping or changing the
remaining runs. Report the complete fixed-budget experiment as registered.

The two policies also succeed on different maps. Control succeeds on830007 and
830008. B succeeds on830001,830003,830004,830005,830010 and830011. Thus6 maps succeed
only with B,2 only with control, and4 with neither; there is no map on which both
succeed in this pair. This is a paired descriptive breakdown on reused development
maps, not12 independent training replicates or a claim of uniform improvement.

## 2. Completion, snapshot and numerical checks

- Both `checkpoint_040000/manifest.json` records have `complete=true`, seed1511,
  step40000, replay_size40000 and replay_index40000.
- Independently recomputed SHA256s match both checkpoint.pt files and both
  replay.npz files. Each checkpoint is26,682,500 bytes; each replay357,921,742
  bytes. The binaries remain local. The hashes and sizes are retained in
  [first_pair_integrity.json](../results/masac_v4_sync_40k_progress_20261105/first_pair_integrity.json).
- All14 declared source hashes remain unchanged. The snapshot source dictionaries
  match their run manifests; both protocol hashes match the frozen files.
- All195 recorded post-warmup progress observations per arm have finite actor,
  critic, alpha, gradient, Q/target, TD-error and entropy diagnostics. Each history
  completed39000 updates under the frozen trainer's per-update finite guard.
  The195 logged observations are not39000 individually retained update records.
- Raw-byte integrity does **not** establish exact CUDA resume. This stage neither
  loaded these weights nor resumed a policy, and never evaluated sealed maps.

The last progress JSON is emitted **before** the final evaluation. Its
`last_eval_step=38000` is therefore expected: it is not the final endpoint.
Do not edit that historical record. Use seed completion plus evaluation40000
for the endpoint, and distinguish seed-local completion from whole-arm completion.

The independent study analyzer currently reports
`IN_PROGRESS / decision_ready=False / decision=None`. A40k checkpoint alone is
not seed completion. A single completed pair is not study completion.

## 3. Limited trajectory diagnosis and continuation

The same offline analyzer used for20k/22k recomputed all24 final40k trajectories,
verified saved metrics and paired map metadata, and retained per-UAV distance,
speed, capture frames and timing. It shows A=C for these maps: the remaining
failures include both incomplete team arrival and nonoverlapping capture events.
Individual never-ball counts are7/36 control and6/36 B, whereas all-agent-entry
maps differ5 versus8. These correlated individual/map counts cannot be treated
as independent trials or establish that B specifically improves synchronization.

Artifacts:

- [Trajectory summary](../results/masac_v4_sync_40k_progress_20261105/trajectory_40000_seed1511/summary.md)
- [Trajectory details](../results/masac_v4_sync_40k_progress_20261105/trajectory_40000_seed1511/summary.json)
- [Canonical partial study analysis](../results/masac_v4_sync_40k_analysis_20261105/analysis.md)
- [Registered protocol](design/MASAC_V4_SYNC_CAPTURE_40K_PROTOCOL_20261105.md)

Both original workers automatically cold-started seed1522 after saving1511's
completion. This is the registered seed loop, not an additional training launch
or continuation from1511's weights. Seed1533 remains queued in the same loop.
Keep sources/configurations and decision rules frozen until all six histories
complete. Evaluate the full registered rule only then; do not extrapolate from
this first positive pair or replace an endpoint with an intermediate peak.
