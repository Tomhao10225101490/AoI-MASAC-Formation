# Maneuver paper suite — static arm gate — 2026-11-04 Asia/Shanghai

**Result: FAIL (2/3 seeds)** — `maneuver-static` completed cleanly; seeds **1211** and **1222** met gate ≥0.60 on 12 L0 maps; seed **1233** final 5/12 (`all_seeds_gate_met=false`).

| Field | Value |
|-------|-------|
| Arm | maneuver-static (`probe_mode=static`) |
| Out | `results/masac_v4_maneuver_paper_suite_20261103_static_20261103_162345` |
| Experiment | `masac_v4_maneuver_paper_suite_20261103_static` |
| Suite | `masac_v4_maneuver_paper_suite_20261103` stamp `20261103_162345` |
| Auth | widget **t4s17** |
| Seeds / steps | 1211, 1222, 1233 × 80k sequential cold-start |
| Gate | ≥0.60 on 12 L0 development maps **each** seed |
| Finals | **1211=11/12**, **1222=10/12**, **1233=5/12** |
| Bests | **1211=11/12@80k**, **1222=11/12@58k**, **1233=9/12@70k** |
| Capture (train+gate) | ball 4.0 m, speed 2.0 m/s, latch=False |
| Rewards | Path-C reshape magnitudes |
| Warm-start | cold |
| Finished | **2026-11-04 18:34 Asia/Shanghai** (~26.2 h wall; `elapsed_s≈94238`) |
| Files | `completion.json` / `status.json`: `complete=true`, `all_seeds_gate_met=false` |
| Stance | `do_not_auto_launch_further=true` — no Attention/L1/silent restack; suite now fully closed |

## Per-seed (from `train.log` + arm `completion.json`)

| Seed | Final @80k | Gate | Best mid-eval | Notes |
|------|------------|------|---------------|-------|
| 1211 | **11/12** | **PASS** | **11/12 @80k** (10/12@76k → 11/12@78–80k) | strongest static seed; late climb |
| 1222 | **10/12** | **PASS** | **11/12 @58k** (final 10/12) | solid; slight late dip from peak |
| 1233 | 5/12 | FAIL | **9/12 @70k** (8/12@78k → **5/12@80k**) | late collapse after strong mid; volatile |

## Sibling arms (suite COMPLETE at doc write)

| Arm | Live | Gate |
|-----|------|------|
| full | COMPLETE @18:21 | FAIL 1/3 (3/12, 4/12, **8/12**) |
| zero | COMPLETE @14:52 | FAIL 0/3 (1/12, 0/12, 1/12) |
| static | COMPLETE @18:34 | FAIL 2/3 (**11/12**, **10/12**, 5/12) |

## Stance

- Static arm evidence recorded; **entire paper suite stamp `162345` is COMPLETE**.
- Ablation ranking under matched V4 pool+RiskEncoder capacity: **static ≫ full ≫ zero**.
- Machine idle (no train/caffeinate PIDs). Next-iter candidates: `docs/PENDING_LAUNCHES_MANEUVER_NEXT_ITER_20261103.md` (A throughput / B sync-capture / C risk-aux / D continue).
- Forbidden without new auth: Attention arms / L1 / silent budget restack / Cursor cloud agents.
