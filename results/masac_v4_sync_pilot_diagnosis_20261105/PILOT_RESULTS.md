# B small pilot results — 2026-11-05

Two arms × two seeds × 2,000 steps = 8,000 transitions and 7,488 SAC updates.
Both arms cold-start on CUDA. These results validate execution only.

| Arm | Seed | Success | Any/all ball | Failures | Last entropy | Last alpha |
|---|---:|---:|---:|---|---:|---:|
| control | 1411 | 0/12 | 0/0 | `{"static_collision": 7, "boundary_collision": 5}` | -0.0322 | 0.0597 |
| control | 1422 | 0/12 | 0/0 | `{"static_collision": 5, "boundary_collision": 7}` | 0.3897 | 0.0595 |
| B | 1411 | 0/12 | 0/0 | `{"static_collision": 11, "boundary_collision": 1}` | 0.3003 | 0.0600 |
| B | 1422 | 0/12 | 1/0 | `{"boundary_collision": 10, "static_collision": 2}` | 0.0536 | 0.0597 |

Every update used the finite-diagnostic stop guard; all sampled logged diagnostics are finite. Critic clipping is frequent, but this does not identify its causal role. Episode shaping/base totals exclude the unfinished final episode and can cancel internally; they are not per-step magnitude estimates.

No evidence of improved joint success at this small budget. Preserve lambda=.5 and tau=.1; a separately frozen 40k study tests B without tuning on these results.

All 12 archived L0 development maps have zero moving objects and conflict_degree=0. The early replay probe audit found moving-clearance full=1 versus static=0, and mostly saturated peer clearance; see l0_content_audit.json. This cannot isolate moving-obstacle prediction and does not characterize all historical final-policy trajectories.
