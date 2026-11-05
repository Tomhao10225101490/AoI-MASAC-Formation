# Near-objective fixed-budget development study

**INVALID_OR_INCOMPLETE**

Three independent cold training seeds and 12 repeated development maps. Correlated observations are not independent binomial trials; no pooled confidence interval, significance, superiority, convergence, safety, dynamic-prediction or novelty claim. No sealed maps, model loading, training or new policy evaluation.

Only evaluation_40000.json is read; no intermediate/best checkpoint selection and no automatic extension.

| Training seed | Original /12 | Potential /12 | Potential − original maps |
|---|---:|---:|---:|
| 1611 | 2 | 1 | -1 |
| 1622 | 0 | 0 | 0 |
| 1633 | 5 | 6 | 1 |

No advancement decision is calculated. All six final completions, finite diagnostics, frozen source/config bytes, actual final artifact hashes, reward/replay checks and matched final map signatures are required.

## All histories and intervention coverage

| Arm | Seed | Final success | Any ball | All ball | Reward rows | Nonzero corrections | Timeouts | Finite final |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| control | 1611 | 2 | 12 | 5 | 40000 | 0 | 88 | True |
| control | 1622 | 0 | 9 | 0 | 40000 | 0 | 81 | True |
| control | 1633 | 5 | 12 | 9 | 40000 | 0 | 72 | True |
| treatment | 1611 | 1 | 11 | 3 | 40000 | 25829 | 92 | True |
| treatment | 1622 | 0 | 11 | 0 | 40000 | 21035 | 80 | True |
| treatment | 1633 | 6 | 12 | 7 | 40000 | 27142 | 86 | True |

Intervention/timeout coverage is descriptive and does not change the efficacy rule. Finite values are sampled top-level learning logs; they are not a retained value for every update. Completed-episode shaped returns are not common cross-arm performance scores.

Per-map outcomes, all negative pairs, any/all ever-ball and nonexclusive collision flags are retained in analysis.json. Exact float32 replay reward/mask checks access only those two arrays, and never deserialize model weights.

## Integrity issues

- `C:\research\masac-maneuverability\results\masac_v4_near_objective_40k_sharded_cuda_20261105\error.json`: training error marker exists; completion cannot override it
- `C:\research\masac-maneuverability\results\masac_v4_near_objective_40k_sharded_cuda_20261105`: coordinator status is failed/error

## Missing evidence

- `C:\research\masac-maneuverability\results\masac_v4_near_objective_40k_sharded_cuda_20261105\completion.json`: required input not yet present
