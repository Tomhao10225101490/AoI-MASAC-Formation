# Maneuver paper suite — launch record (2026-11-03)

Authorized: t4s17 + phone steering (comprehensive paper scope). Local Mac only.

## Suite id
`masac_v4_maneuver_paper_suite_20261103` @ stamp `20261103_162345`

## Arms (parallel)
| Arm | Results dir | PID | Role |
|-----|-------------|-----|------|
| full | `results/masac_v4_maneuver_paper_suite_20261103_full_20261103_162345` | 45262 | innovation |
| zero | `results/masac_v4_maneuver_paper_suite_20261103_zero_20261103_162345` | 45327 | capacity-matched control |
| static | `results/masac_v4_maneuver_paper_suite_20261103_static_20261103_162345` | 45356 | dynamic/peer ablation |

Seeds 1211/1222/1233 × 80k; C reshape; frozen capture 4/2 latch=False; set_mode=pool; **cold start**.

## Baseline
Task-matched Standard-C: `results/masac_v4_c_reshape_l0_frozen42_20261103_145043` (11/12, 11/12, 10/12).

## Do not auto-launch
Attention / L1 / silent restack / further stages — await user.

## Mid-run status (2026-11-03 21:05 Asia/Shanghai)

Suite **still RUNNING** (t8u). Snapshot: `docs/MASAC_V4_MANEUVER_PAPER_SUITE_STATUS_20261103.md` + `results/masac_v4_maneuver_paper_suite_20261103_status_20261103_2105.json`.

Seed 1211 progress ≈41–49k/80k; joint success still 0–1/12 while ever_ball often ≥11/12; dominant fail timeout_after_ball. Next-iter plan (not launched): `docs/PENDING_LAUNCHES_MANEUVER_NEXT_ITER_20261103.md`.

