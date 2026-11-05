# Final saved-trajectory dwell and reward results

Descriptive, correlated, policy-dependent lengths/terminal censoring; not causal arm comparison.

All 72 trajectories and frozen inputs verified. Formula-derived reward has no logged-return reference.

| Arm | Seed | Agent-frames | Near-stall frames (%) | >=25-frame agent-maps /36 | Never-ball with run /total | Capture entries/exits |
|---|---:|---:|---:|---:|---:|---:|
| control | 1511 | 10932 | 5248 (48.01) | 19 | 7/7 | 45/33 |
| control | 1522 | 11688 | 4328 (37.03) | 17 | 14/20 | 23/13 |
| control | 1533 | 9123 | 3899 (42.74) | 15 | 8/11 | 47/30 |
| b | 1511 | 8172 | 2925 (35.79) | 7 | 2/6 | 52/31 |
| b | 1522 | 12600 | 2531 (20.09) | 6 | 4/17 | 58/53 |
| b | 1533 | 8415 | 3007 (35.73) | 11 | 4/4 | 29/5 |

Near-stall means 4<=d<10m and speed<0.5m/s on the sampled grid. 25 samples represent 5s of occupancy count and a 4.8s endpoint span.

Never-ball population: {'agent_maps': 65, 'agent_frames': 22750, 'near_stall_frames': 11491, 'with_25plus_run': 39, 'longest_run_frames': 299, 'final_near_stall': 48}. This conditional population is not an independent arm comparison.

| Arm | Seed | Progress | Near | Acceleration | Capture delta | Terminal | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 1511 | 25.97172 | 13.68246 | -1.19050 | 1.59367 | 4.08757 | 44.98660 |
| control | 1522 | 28.02343 | 15.11548 | -1.08556 | 1.57870 | 2.65077 | 46.63337 |
| control | 1533 | 29.29665 | 13.64550 | -1.31849 | 2.56322 | 9.21172 | 54.17509 |
| b | 1511 | 28.44343 | 12.67087 | -1.36618 | 2.89537 | 10.45541 | 54.08951 |
| b | 1522 | 24.95754 | 13.95341 | -1.16506 | 0.78675 | 0.00000 | 38.79112 |
| b | 1533 | 29.48878 | 14.27614 | -0.74632 | 3.27318 | 10.33929 | 57.92065 |

Reward table: mean raw gamma=.99 discounted base component totals over all 12 saved episodes/history. Living/first-ball/collision and all undiscounted components remain in JSON. No entropy or bootstrap is included; reference validation is UNAVAILABLE.
