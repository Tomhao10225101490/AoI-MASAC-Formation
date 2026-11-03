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
