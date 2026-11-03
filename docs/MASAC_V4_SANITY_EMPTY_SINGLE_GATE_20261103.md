# Sanity empty/single 20k gate diagnosis (2026-11-03 Asia/Shanghai)

- Run: `results/masac_v4_sanity_empty_single_20261103_131315`
- Protocol: `configs/masac_v4/l0_sanity_empty_single_20261103.json`
- Gate verdict: **FAILED** — 0/8 (need 0.75)
- Budget: 20000 steps; every mid-eval also 0 success
- Subtype counts: `{'near_goal_never_stable_capture': 4, 'far_from_goal_timeout': 4}`

## Interpretation

Learnability floor FAILED at 20k on empty+single maps. Not collisions (empty maps have 0 buildings). Failures dominated by approach/capture credit: agents often remain far or only partially enter the capture ball; no joint simultaneous capture on any of 8 maps. Contrast: L0 urban continues at 50k–100k did produce nonzero successes, so 20k may be under-budget OR sanity curriculum/warmup differs — but zero across all mid-evals is a strong negative floor signal. Do NOT claim L0 urban failure is only map complexity; Standard-nav capture/reward reshape (C) or a longer audited sanity protocol is required. No Attention / maneuverability / L1. No silent continue budget restack.

## Per-map snapshot (final eval)

- map 870001 (empty): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=13.50 mean_final_dist=19.39
- map 870002 (empty): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=11.94 mean_final_dist=16.65
- map 870003 (empty): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=12.07 mean_final_dist=18.02
- map 870004 (empty): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=13.43 mean_final_dist=17.95
- map 870101 (single_obstacle): subtype=`far_from_goal_timeout` ever_ball=0/3 ever_cap=0/3 lag_min_dist=29.97 mean_final_dist=28.99
- map 870102 (single_obstacle): subtype=`far_from_goal_timeout` ever_ball=0/3 ever_cap=0/3 lag_min_dist=32.41 mean_final_dist=29.38
- map 870103 (single_obstacle): subtype=`far_from_goal_timeout` ever_ball=0/3 ever_cap=0/3 lag_min_dist=23.44 mean_final_dist=34.83
- map 870104 (single_obstacle): subtype=`far_from_goal_timeout` ever_ball=0/3 ever_cap=0/3 lag_min_dist=31.45 mean_final_dist=29.64

## Stance

- No silent further +50k continue budget stack on the failed continue protocol.
- Forbidden: Attention / Predictive Maneuverability / L1 from this monitor.
- User chooses next Standard-nav change (C reshape) or a new audited protocol.
