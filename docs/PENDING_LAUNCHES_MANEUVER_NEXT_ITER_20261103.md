# PENDING — next iteration after / beside Maneuver paper suite (2026-11-03)

Asia/Shanghai. Authorized to **prepare** under **t8u** (keep training; optimize; iterate formulas from results).
**Not authorized to launch** a disruptive fourth heavy arm while stamp `162345` is live.

## Why wait

Live suite healthy (full/zero/static PIDs 45262/45327/45356). Free RAM ≈0.08 GB on 16 GB Mac.
Launching another 80k maneuver process would risk swap thrash / killing the paper suite.

## Mid-suite evidence → method target

From status snapshot `20261103_2105`:

1. Ever-ball is strong (often ≥9/12); joint success is not (0–1/12).
2. Dominant fail = timeout_after_ball + high near-goal low-speed uncaptured steps; not collisions.
3. Math hygiene already separates soft Bellman vs success predicates (`docs/MATH_ARGUMENT_IMPROVEMENTS_20261103.md`).
4. Training opts landed for next process: vectorized probes, `cache_risk=True`, `--torch-threads 1` (`docs/TRAINING_OPTIMIZATION_MANEUVER_20261103.md`).

**Hypothesis for next formula/method pass:** Path-C reshape teaches approach + ball entry, but **simultaneous capture credit** is too sparse under frozen 4/2 without latch when three agents desynchronize near the goal. Probe features may help avoidance yet are not yet translating into coordinated final approach (full/static only flickered 1/12; zero also reaches balls → capacity alone ≠ coordination).

## Candidate next experiments (pick after suite gate or user auth)

Use a **new experiment id** for any protocol/number change. Prefer serial or ≤2 arms until free RAM ≥~2 GB.

### A — Throughput-only re-run (no formula change) — FIRST if suite underperforms on wall-time

- Id sketch: `masac_v4_maneuver_optpack_2026110X`
- Arms: preferably **full only** (or full+zero) with `--torch-threads 1`, `cache_risk=True` (default), same C reshape + 4/2
- Goal: more updates/hour; cleaner paper compute appendix; not a claim of new theory
- Launch only when suite exits or RAM clearly allows

### B — Simultaneous-capture curriculum (formula/method) — PRIMARY scientific next step if gate fails or barely passes late

- Id sketch: `masac_v4_maneuver_sync_capture_2026110X`
- Keep frozen **eval** predicate 4/2 latch=False for apples-to-apples L0 gate
- Train-time dense shaping candidates (one change-set per id):
  - bonus for ↑ count of agents currently in-ball (team occupancy)
  - bonus for ↓ pairwise arrival-time gap / sync potential near goal
  - slightly longer horizon or mild living-cost anneal only inside near-goal radius
- **Do not** silently reintroduce soft 6/3.5+latch on the L0 gate; if a soft train predicate is tested, document dual metrics (train vs frozen eval) per math doc
- Map to math doc: success predicates, Ng vs dense curriculum disclaimer, probe optimism not a certificate

### C — Probe-use / RiskEncoder pressure (if full ≰ zero at equal budget)

- Id sketch: `masac_v4_maneuver_risk_aux_2026110X`
- Auxiliary loss or info-bottleneck so RiskEncoder cannot be ignored; or shuffle ablation arm when RAM free
- Ties to math doc: probe gap is optimistic CV estimate — report clearance/TTC vs realized collisions

### D — Post-suite continuation (if seed 1211 shows late non-zero but <0.6)

- Continue **same** protocol +50k only with **explicit user auth** and new continue id; no silent restack
- Prefer warm-start from best `checkpoint_*` of the same arm

### Explicitly deferred

- Attention encoder, L1 maps, CloudAgent — require separate user auth
- Fourth parallel 80k arm on this Mac while three live — **no**

## Launch checklist (when authorized)

1. Confirm suite PIDs exited or RAM free ≥2 GB (`vm_stat` / Activity Monitor)
2. New protocol JSON + auth JSON + PENDING doc flip to ACTIVE
3. `--torch-threads 1` if ≥2 arms; enable `cache_risk`
4. Gate doc template ready; `do_not_auto_launch_further=true`
5. Push protocol before/with launch commit

## Current action

**PENDING only.** Suite keeps running. Re-evaluate which of A/B/C/D after seed-1211 end-evals and again at suite completion.
