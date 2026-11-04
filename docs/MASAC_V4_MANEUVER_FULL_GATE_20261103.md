# Maneuver paper suite — full arm gate — 2026-11-04 Asia/Shanghai

**Result: FAIL (1/3 seeds)** — `maneuver-full` completed cleanly; only seed **1233** met gate ≥0.60 on 12 L0 maps (`all_seeds_gate_met=false`).

| Field | Value |
|-------|-------|
| Arm | maneuver-full (`probe_mode=full`) |
| Out | `results/masac_v4_maneuver_paper_suite_20261103_full_20261103_162345` |
| Experiment | `masac_v4_maneuver_paper_suite_20261103_full` |
| Suite | `masac_v4_maneuver_paper_suite_20261103` stamp `20261103_162345` |
| Auth | widget **t4s17** |
| Seeds / steps | 1211, 1222, 1233 × 80k sequential cold-start |
| Gate | ≥0.60 on 12 L0 development maps **each** seed |
| Finals | **1211=3/12**, **1222=4/12**, **1233=8/12** |
| Bests | **1211=4/12@76k**, **1222=4/12@80k**, **1233=9/12@72k** |
| Capture (train+gate) | ball 4.0 m, speed 2.0 m/s, latch=False |
| Rewards | Path-C reshape magnitudes |
| Warm-start | cold |
| Finished | **2026-11-04 18:21 Asia/Shanghai** (~26.0 h wall; `elapsed_s≈93449`) |
| Files | `completion.json` / `status.json`: `complete=true`, `all_seeds_gate_met=false` |
| Stance | `do_not_auto_launch_further=true` — **do not kill** static; no Attention/L1/silent restack |

## Per-seed (from `train.log` + arm `completion.json`)

| Seed | Final @80k | Gate | Best mid-eval | Notes |
|------|------------|------|---------------|-------|
| 1211 | 3/12 | FAIL | 4/12 @76k | weak throughout; ever_ball often high, joint capture rare |
| 1222 | 4/12 | FAIL | 4/12 @80k | same weak band as 1211 |
| 1233 | **8/12** | **PASS** | **9/12 @72k** (also 7/12@78k → 8/12@80k) | only full seed above gate; late rebound |

## Sibling arms (live at doc write ~2026-11-04 18:24 Asia/Shanghai; do not disturb)

| Arm | PID | Live | Last eval / gate |
|-----|-----|------|------------------|
| full | — | **COMPLETE** (PIDs gone) | gate FAIL 1/3 (only 1233 PASS) |
| zero | — | COMPLETE @14:52 | gate FAIL 0/3 |
| static | 45356 (+caffeinate 45359) | seed 1233 ~77000/80000 (~2.67 step/s, ETA ~19m) | last 7/12@76k; priors **1211 PASS 11/12**, **1222 PASS 10/12** |

## Stance

- Full arm evidence recorded; suite continues with **static** seed 1233 only.
- Ablation so far: **zero collapses**, **static strong (2/2 seeds PASS so far)**, **full mixed (1/3 PASS)** — wait static close before final paper claim.
- Next-iter plan remains PENDING: `docs/PENDING_LAUNCHES_MANEUVER_NEXT_ITER_20261103.md`.
- Forbidden without new auth: Attention arms / L1 / silent budget restack / auto further / Cursor cloud agents / kill of active arms.
