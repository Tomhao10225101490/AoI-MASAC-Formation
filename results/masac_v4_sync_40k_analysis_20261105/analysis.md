# Static B 40k development study

**COMPLETE**

Three independent training seeds; 12 development maps are repeated measurements, not 36 independent training trials. No significance, safety, predictive-content or novelty claim. No sealed test or weights loaded.

**ADVANCEMENT_RULE_MET**; engineering rule met: **True**.

Mean paired success-fraction difference: 0.111111.

| Training seed | Control /12 | B /12 | B − control maps |
|---|---:|---:|---:|
| 1511 | 2 | 6 | +4 |
| 1522 | 1 | 0 | -1 |
| 1533 | 4 | 5 | +1 |

This rule supports planning confirmation only. It launches no training, extension or sealed test.

## Latest observed development evaluations

| Arm | Seed | Step | Success /12 | Any ball /12 | All ball /12 | Mean max occupancy |
|---|---:|---:|---:|---:|---:|---:|
| control | 1511 | 40000 | 2 | 12 | 5 | 2.0 |
| control | 1522 | 40000 | 1 | 9 | 1 | 1.25 |
| control | 1533 | 40000 | 4 | 12 | 5 | 2.0 |
| treatment | 1511 | 40000 | 6 | 12 | 8 | 2.25 |
| treatment | 1522 | 40000 | 0 | 11 | 2 | 1.1666666666666667 |
| treatment | 1533 | 40000 | 5 | 12 | 8 | 2.25 |

## Sampled learning diagnostics and completed-episode reward scale

Clip fractions and entropy/alpha below are sampled progress logs, not all updates. “pending final40k” means the final evidence is not due yet; it does not indicate nonfinite diagnostics. JSON finite_diagnostics_observed separately reports whether available learning samples are finite (null means no loss sample yet). Reward totals include completed episodes only; shaped training return is not a common cross-arm performance metric.

| Arm | Seed | Final 40k finite evidence | Critic clip fraction | Latest entropy | Latest alpha | Completed episodes | Base total | Shaping total |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| control | 1511 | verified | 1.0 | -3.030709743499756 | 0.019608404487371445 | 233 | -5216.075309475593 | 0.0 |
| control | 1522 | verified | 1.0 | -3.036823034286499 | 0.01660999469459057 | 219 | 1052.1226143696529 | 0.0 |
| control | 1533 | verified | 1.0 | -3.1283469200134277 | 0.019662726670503616 | 242 | 2490.029910208664 | 0.0 |
| treatment | 1511 | verified | 1.0 | -3.268517017364502 | 0.017173416912555695 | 235 | 499.4065907900791 | -40.826149674592294 |
| treatment | 1522 | verified | 1.0 | -2.972724676132202 | 0.017590995877981186 | 235 | -507.85500732188564 | -34.683415081988805 |
| treatment | 1533 | verified | 1.0 | -3.068627119064331 | 0.017366591840982437 | 212 | 1599.6355059670211 | -41.17271236440713 |

Per-map exclusive failures, nonexclusive collision flags, occupancy, final distance/speed availability, common-grid AUC and peak-to-last drops are retained in analysis.json. Missing trajectory velocity is null.

![Development curves](curves.png)
