# PENDING — next iteration after Maneuver paper suite (updated 2026-11-04 18:40 Asia/Shanghai)

Suite stamp `20261103_162345` is **COMPLETE** (zero/full/static all finished). Machine idle.
Authorized to **prepare** under standing intensity / t8u-style iterate-from-results.
**Not auto-launching** further stages: suite contract + `do_not_auto_launch_further` — pick A/B/C/D with user (or explicit auth) before start.

## Suite gate summary

| Arm | Gate | Finals |
|-----|------|--------|
| zero | FAIL 0/3 | 1/12, 0/12, 1/12 |
| full | FAIL 1/3 | 3/12, 4/12, 8/12 |
| static | FAIL 2/3 | **11/12**, **10/12**, 5/12 |

Ablation: **static ≫ full ≫ zero**. Ever-ball often high; joint capture remains the bottleneck (esp. full on 1211/1222; static 1233 late collapse).

## Mid-suite → post-suite method target (unchanged hypothesis)

1. Ever-ball strong; joint success sparse under frozen 4/2 latch=False when agents desync near goal.
2. Dominant fail modes historically timeout_after_ball / near-goal uncaptured — not collisions.
3. Math hygiene: soft Bellman vs success predicates (`docs/MATH_ARGUMENT_IMPROVEMENTS_20261103.md`).
4. Training opts ready for next process: vectorized probes, `cache_risk=True`, `--torch-threads 1` (`docs/TRAINING_OPTIMIZATION_MANEUVER_20261103.md`).

## Candidate next experiments (NEW experiment id required)

Prefer serial or ≤2 arms; free RAM should be comfortable now that suite exited.

### A — Throughput-only re-run (no formula change) — FIRST if wall-time / compute appendix is the goal

- Id sketch: `masac_v4_maneuver_optpack_20261104`
- Arms: preferably **static+full** (or full-only) with `--torch-threads 1`, `cache_risk=True`, same C reshape + 4/2
- Goal: more updates/hour; cleaner paper compute appendix; **not** a new theory claim
- Lowest risk to start while deciding B

### B — Simultaneous-capture curriculum (formula/method) — PRIMARY scientific next

- Id sketch: `masac_v4_maneuver_sync_capture_20261104`
- Keep frozen **eval** predicate 4/2 latch=False for apples-to-apples L0 gate
- Train-time dense shaping (one change-set per id): team occupancy-in-ball bonus; sync/arrival-gap bonus; mild near-goal living-cost anneal
- **Do not** silently reintroduce soft 6/3.5+latch on the L0 gate; dual metrics if soft train predicate tested
- Best scientific response to static≫full≫zero + joint-capture sparsity

### C — Probe-use / RiskEncoder pressure (full ≰ static at equal budget — already true)

- Id sketch: `masac_v4_maneuver_risk_aux_20261104`
- Aux loss / info-bottleneck / shuffle ablation so RiskEncoder cannot be ignored

### D — Post-suite continuation (same protocol +50k)

- Only with **explicit user auth** and new continue id; warm-start from best same-arm checkpoint
- Candidates: static 1233 (late collapse from 9/12) or full 1233 (only full PASS)

### Explicitly deferred

- Attention encoder, L1 maps, CloudAgent — require separate user auth

## Launch checklist (when authorized)

1. Confirm no live suite PIDs; RAM free ≥2 GB
2. New protocol JSON + auth JSON + PENDING flip to ACTIVE
3. `--torch-threads 1` if ≥2 arms; enable `cache_risk`
4. Gate doc template ready; `do_not_auto_launch_further=true`
5. Push protocol before/with launch commit

## Current action

**PENDING user choice (A / B / C / D / other).** Suite complete @18:34 Asia/Shanghai; docs recorded; no auto further launched this monitor tick.
