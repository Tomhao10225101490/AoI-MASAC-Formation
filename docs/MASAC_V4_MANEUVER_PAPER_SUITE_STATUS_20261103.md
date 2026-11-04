# Maneuver paper suite — status (2026-11-04 15:08 Asia/Shanghai)

Standing order: keep TomMac-mini experiments running; iterate after results.
**Do not kill** PIDs **45262** (full) / **45356** (static) (+ caffeinates 45265/45359). Zero arm PIDs are gone (clean complete).

Zero-arm gate note: `docs/MASAC_V4_MANEUVER_ZERO_GATE_20261103.md` (+ `.json`)

## Zero arm — FINAL (this tick)

| Field | Value |
|-------|-------|
| Arm | maneuver-zero |
| Out | `results/masac_v4_maneuver_paper_suite_20261103_zero_20261103_162345` |
| Finished | **2026-11-04 14:52 Asia/Shanghai** (~22.5 h) |
| `completion.json` / `status.json` | `complete=true`, `phase=complete`, `all_seeds_gate_met=false` |
| Gate | ≥0.60 each seed — **FAIL 0/3** |
| Finals | **1211=1/12**, **1222=0/12**, **1233=1/12** |
| Bests | **1/12@56k**, **2/12@60k**, **5/12@48k** |

## Health (full/static still running)

| Arm | PID | Seed | Step | step/s | Last eval (status.json) | Alive |
|-----|-----|------|------|--------|-------------------------|-------|
| full | 45262 | 1233/3 | ~47600/80000 | ~2.61 | 1/12 @46k | yes |
| zero | — | done | 80000×3 | — | gate FAIL 0/3 | complete |
| static | 45356 | 1233/3 | ~44200/80000 | ~2.59 | 8/12 @44k | yes |

- Suite stamp: `20261103_162345`
- Contract: C reshape + frozen capture 4/2 latch=False; cold start; gate ≥0.60
- `do_not_auto_launch_further=true` (no Attention / L1 / silent restack)

## Stance

- Zero arm closed as **gate FAIL**; evidence committed under docs gate note.
- Suite continues undisturbed through remaining **full** + **static** seed 1233.
- Next-iter plan stays **PENDING**: `docs/PENDING_LAUNCHES_MANEUVER_NEXT_ITER_20261103.md`.
- After suite gate: prefer documented A/B candidates — ask user if claim-changing; no auto further.
