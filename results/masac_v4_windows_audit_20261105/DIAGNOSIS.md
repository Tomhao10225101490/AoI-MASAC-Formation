# Suite diagnosis — 2026-11-05 (Asia/Shanghai)

Three independent training seeds per arm; 12 reused development maps per seed. Counts are descriptive.

| Arm | Seed | Success | Any ever-ball | All ever-ball | Failure/status counts |
|---|---:|---:|---:|---:|---|
| static | 1211 | 11/12 | 12/12 | 11/12 | `{"success": 11, "timeout_after_any_ball": 1}` |
| static | 1222 | 10/12 | 12/12 | 10/12 | `{"success": 10, "timeout_after_any_ball": 2}` |
| static | 1233 | 5/12 | 12/12 | 5/12 | `{"success": 5, "timeout_after_any_ball": 7}` |
| full | 1211 | 3/12 | 12/12 | 5/12 | `{"success": 3, "timeout_after_all_ball": 2, "timeout_after_any_ball": 7}` |
| full | 1222 | 4/12 | 12/12 | 4/12 | `{"success": 4, "timeout_after_any_ball": 8}` |
| full | 1233 | 8/12 | 12/12 | 8/12 | `{"success": 8, "timeout_after_any_ball": 4}` |
| zero | 1211 | 1/12 | 12/12 | 1/12 | `{"success": 1, "timeout_after_any_ball": 11}` |
| zero | 1222 | 0/12 | 12/12 | 1/12 | `{"timeout_after_all_ball": 1, "timeout_after_any_ball": 11}` |
| zero | 1233 | 1/12 | 12/12 | 2/12 | `{"success": 1, "timeout_after_all_ball": 1, "timeout_after_any_ball": 10}` |

`Any ever-ball` does not mean all agents approach their goals. Even `all ever-ball` is historical distance-only entry, not simultaneous capture with speed <2 m/s. B sync-capture is a hypothesis; these counts do not identify a causal failure mechanism.

Standard-C is a separate warm-started learnability floor (different seeds, encoder and budget), excluded from the within-suite ranking.
