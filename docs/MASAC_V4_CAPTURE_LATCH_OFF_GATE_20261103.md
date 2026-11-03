# Capture latch-OFF twin gate — FAILED (2026-11-03 Asia/Shanghai)

**Run:** `results/masac_v4_capture_latch_off_20261103_132438`  
**Seed:** 712 · **Steps:** 20k · **capture_latch:** false  
**Finished:** ~13:30 Asia/Shanghai (~337 s wall, ~60 step/s)  
**Final:** **0/12** (need ≥0.6) · **gate_met=false**

## Mid-evals
| Step | Success |
|------|---------|
| 2000 | 0/12 |
| 4000 | 0/12 |
| 6000 | 0/12 |
| 8000 | 0/12 |
| 10000 | 0/12 |
| 12000 | 0/12 |
| 14000 | 0/12 |
| 16000 | 0/12 |
| 18000 | 0/12 |
| 20000 | 0/12 |

## Readout
- Twin of latch-on also fails at 20k: **0/12** every eval.
- Learnability floor not shown on L0 urban family at this budget whether latch is on or off.
- Final episodes: remaining_distance mean ≈ 16.9 m; closest-goal min mean ≈ 6.1 m; ever-arrived any UAV in 1/12 eps.
- Buckets@20k: `{'timeout_far': 10, 'near_goal_or_entered_not_simultaneous': 2}`.

## Stance
- No silent +50k continue restack.
- Forbidden from this monitor: Attention / Predictive Maneuverability / L1.
- **User decision still required:** C capture/reward reshape vs new audited longer-sanity protocol.
