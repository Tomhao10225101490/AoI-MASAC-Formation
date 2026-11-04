# Maneuver paper suite — status (2026-11-04 18:24 Asia/Shanghai)

Standing order: keep TomMac-mini experiments running; iterate after results.
**Do not kill** PID **45356** (static) (+ caffeinate 45359). Full + zero arm PIDs are gone (clean complete).

Gate notes:
- zero: `docs/MASAC_V4_MANEUVER_ZERO_GATE_20261103.md` (+ `.json`)
- full: `docs/MASAC_V4_MANEUVER_FULL_GATE_20261103.md` (+ `.json`)

## Full arm — FINAL (this tick)

| Field | Value |
|-------|-------|
| Arm | maneuver-full |
| Out | `results/masac_v4_maneuver_paper_suite_20261103_full_20261103_162345` |
| Finished | **2026-11-04 18:21 Asia/Shanghai** (~26.0 h) |
| `completion.json` / `status.json` | `complete=true`, `phase=complete`, `all_seeds_gate_met=false` |
| Gate | ≥0.60 each seed — **FAIL 1/3** (only 1233 met) |
| Finals | **1211=3/12**, **1222=4/12**, **1233=8/12** |
| Bests | **4/12@76k**, **4/12@80k**, **9/12@72k** |

## Zero arm — FINAL (earlier)

| Field | Value |
|-------|-------|
| Finished | **2026-11-04 14:52 Asia/Shanghai** |
| Gate | **FAIL 0/3** — finals 1/12, 0/12, 1/12 |

## Health (static still running)

| Arm | PID | Seed | Step | step/s | Last eval / gate | Alive |
|-----|-----|------|------|--------|------------------|-------|
| full | — | done | 80000×3 | — | gate FAIL 1/3 | complete |
| zero | — | done | 80000×3 | — | gate FAIL 0/3 | complete |
| static | 45356 | 1233/3 | ~77000/80000 | ~2.67 | 7/12@76k; 1211/1222 PASS 11/12 & 10/12 | yes |

- Suite stamp: `20261103_162345`
- Contract: C reshape + frozen capture 4/2 latch=False; cold start; gate ≥0.60
- `do_not_auto_launch_further=true` (no Attention / L1 / silent restack)

## Stance

- Full arm closed as **gate FAIL (1/3)**; evidence committed under docs gate note.
- Suite continues undisturbed through remaining **static** seed 1233 (~ETA 19m at doc write).
- Ablation narrative so far: zero collapses; static strong; full mixed/weak on 2/3 seeds.
- Next-iter plan stays **PENDING**: `docs/PENDING_LAUNCHES_MANEUVER_NEXT_ITER_20261103.md`.
- After suite gate: prefer documented A/B candidates — ask user if claim-changing; no auto further.


## Paper narrative implications (2026-11-04 Asia/Shanghai)

Provisional only — **not** a final contribution claim (static still running):

1. **zero FAIL** shows RiskEncoder *capacity* without probe content cannot pass the joint-capture gate.
2. **static** strong on 1211/1222 (11/12, 10/12) remains the best within-suite signal; seed 1233 still open (~7/12@76k mid).
3. **full FAIL 1/3** (only 1233 PASS 8/12; 1211/1222 weak 3–4/12) warns that predictive/full CV channels are not yet a reliable win vs static margins under this protocol.
4. Paper docs: `docs/PAPER_PLAN_WOWMOM2027.md`, `docs/RELATED_WORK_FRONTIER_20261104.md`, `docs/PAPER_OUTLINE_DRAFT.md`.
