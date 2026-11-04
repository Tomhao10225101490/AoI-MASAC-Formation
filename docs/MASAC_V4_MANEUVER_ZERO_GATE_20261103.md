# Maneuver paper suite — zero arm gate — 2026-11-04 Asia/Shanghai

**Result: FAIL (0/3 seeds)** — `maneuver-zero` completed cleanly; all seeds below gate ≥0.60 on 12 L0 maps.

| Field | Value |
|-------|-------|
| Arm | maneuver-zero (`probe_mode=zero`) |
| Out | `results/masac_v4_maneuver_paper_suite_20261103_zero_20261103_162345` |
| Experiment | `masac_v4_maneuver_paper_suite_20261103_zero` |
| Suite | `masac_v4_maneuver_paper_suite_20261103` stamp `20261103_162345` |
| Auth | widget **t4s17** |
| Seeds / steps | 1211, 1222, 1233 × 80k sequential cold-start |
| Gate | ≥0.60 on 12 L0 development maps **each** seed |
| Finals | **1211=1/12**, **1222=0/12**, **1233=1/12** |
| Bests | **1211=1/12@56k**, **1222=2/12@60k**, **1233=5/12@48k** |
| Capture (train+gate) | ball 4.0 m, speed 2.0 m/s, latch=False |
| Rewards | Path-C reshape magnitudes |
| Warm-start | cold |
| Finished | **2026-11-04 14:52 Asia/Shanghai** (~22.5 h wall; `elapsed_s≈80940`) |
| Files | `completion.json` / `status.json`: `complete=true`, `all_seeds_gate_met=false` |
| Stance | `do_not_auto_launch_further=true` — **do not kill** full/static; no Attention/L1/silent restack |

## Per-seed (from `train.log` + seed `completion.json`)

| Seed | Final @80k | Gate | Best mid-eval | Notes |
|------|------------|------|---------------|-------|
| 1211 | 1/12 | FAIL | 1/12 @56k (also 1/12 @78k) | ever_ball often 12/12 late; joint success rare |
| 1222 | 0/12 | FAIL | 2/12 @60k (also 2/12 @64/66/68k) | same pattern: ball entry ≠ simultaneous capture |
| 1233 | 1/12 | FAIL | **5/12 @48k** | strongest zero mid-peak; still collapsed by final |

## Sibling arms (live at doc write ~2026-11-04 15:08 Asia/Shanghai; do not disturb)

| Arm | PID | Live | Last eval (from status.json) |
|-----|-----|------|------------------------------|
| full | 45262 (+caffeinate 45265) | seed 1233 ~47600/80000 (~2.61 step/s) | 1/12 @46k |
| static | 45356 (+caffeinate 45359) | seed 1233 ~44200/80000 (~2.59 step/s) | 8/12 @44k |
| zero | — | **COMPLETE** (PIDs gone) | gate FAIL 0/3 |

## Stance

- Zero arm evidence recorded; suite continues with **full** + **static** only.
- Next-iter plan remains PENDING: `docs/PENDING_LAUNCHES_MANEUVER_NEXT_ITER_20261103.md`.
- Forbidden without new auth: Attention arms / L1 / silent budget restack / auto further / Cursor cloud agents / kill of active arms.
