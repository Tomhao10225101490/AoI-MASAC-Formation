# Capture-latch ON 20k gate diagnosis (2026-11-03 Asia/Shanghai)

- Run: `results/masac_v4_capture_latch_on_20261103_131315`
- Protocol: `configs/masac_v4/l0_capture_latch_on_20261103.json`
- Gate verdict: **FAILED** — 0/12 (need 0.6)
- Budget: 20000 steps; every mid-eval also 0 success
- Subtype counts: `{'near_goal_never_stable_capture': 11, 'entered_ball_but_speed_gate_blocked_some': 1}`

## Interpretation

Capture-latch ON at 20k on frozen L0 maps also FAILED 0/12. Latch alone does not unlock joint success at this budget. A/B incomplete until latch-off twin finishes. Prior continue arms needed ~50k–100k for nonzero rates; 20k may be too short to isolate latch, but zero throughout still argues capture/credit assignment needs reshape (C), not only sticky latch. No Attention/L1; no silent restack.

## Per-map snapshot (final eval)

- map 830001 (urban_canyon): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=18.30 mean_final_dist=19.07
- map 830002 (warehouse): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=17.80 mean_final_dist=19.09
- map 830003 (courtyard): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=18.14 mean_final_dist=21.82
- map 830004 (overpass): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=14.89 mean_final_dist=19.65
- map 830005 (narrow_crossing): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=17.29 mean_final_dist=19.92
- map 830006 (dynamic_crossing): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=18.55 mean_final_dist=19.55
- map 830007 (urban_canyon): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=19.30 mean_final_dist=19.43
- map 830008 (warehouse): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=18.63 mean_final_dist=17.85
- map 830009 (courtyard): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=17.44 mean_final_dist=20.23
- map 830010 (overpass): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=16.33 mean_final_dist=20.04
- map 830011 (narrow_crossing): subtype=`near_goal_never_stable_capture` ever_ball=0/3 ever_cap=0/3 lag_min_dist=17.85 mean_final_dist=18.11
- map 830012 (dynamic_crossing): subtype=`entered_ball_but_speed_gate_blocked_some` ever_ball=1/3 ever_cap=0/3 lag_min_dist=16.83 mean_final_dist=18.99

## Stance

- No silent further +50k continue budget stack on the failed continue protocol.
- Forbidden: Attention / Predictive Maneuverability / L1 from this monitor.
- User chooses next Standard-nav change (C reshape) or a new audited protocol.
