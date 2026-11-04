# Static B 40k development study

**IN_PROGRESS**

Three independent training seeds; 12 development maps are repeated measurements, not 36 independent training trials. No significance, safety, predictive-content or novelty claim. No sealed test or weights loaded.

No advancement decision is calculated. All six final completions, finite diagnostics and integrity checks are required.

## Latest observed development evaluations

| Arm | Seed | Step | Success /12 | Any ball /12 | All ball /12 | Mean max occupancy |
|---|---:|---:|---:|---:|---:|---:|
| control | 1511 | 40000 | 2 | 12 | 5 | 2.0 |
| control | 1522 | 2000 | 0 | 0 | 0 | 0.0 |
| control | 1533 | None | None | None | None | None |
| treatment | 1511 | 40000 | 6 | 12 | 8 | 2.25 |
| treatment | 1522 | 2000 | 0 | 0 | 0 | 0.0 |
| treatment | 1533 | None | None | None | None | None |

## Sampled learning diagnostics and completed-episode reward scale

Clip fractions and entropy/alpha below are sampled progress logs, not all updates. “pending final40k” means the final evidence is not due yet; it does not indicate nonfinite diagnostics. JSON finite_diagnostics_observed separately reports whether available learning samples are finite (null means no loss sample yet). Reward totals include completed episodes only; shaped training return is not a common cross-arm performance metric.

| Arm | Seed | Final 40k finite evidence | Critic clip fraction | Latest entropy | Latest alpha | Completed episodes | Base total | Shaping total |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| control | 1511 | verified | 1.0 | -3.030709743499756 | 0.019608404487371445 | 233 | -5216.075309475593 | 0.0 |
| control | 1522 | pending final40k | 1.0 | -0.5639123916625977 | 0.05926528573036194 | 45 | -2314.0903247941433 | 0.0 |
| control | 1533 | pending final40k | None | None | None | 0 | 0 | 0 |
| treatment | 1511 | verified | 1.0 | -3.268517017364502 | 0.017173416912555695 | 235 | 499.4065907900791 | -40.826149674592294 |
| treatment | 1522 | pending final40k | 1.0 | 0.6511070728302002 | 0.06787844747304916 | 35 | -1729.6171949623802 | -2.6304203385418528 |
| treatment | 1533 | pending final40k | None | None | None | 0 | 0 | 0 |

Per-map exclusive failures, nonexclusive collision flags, occupancy, final distance/speed availability, common-grid AUC and peak-to-last drops are retained in analysis.json. Missing trajectory velocity is null.

![Development curves](curves.png)
