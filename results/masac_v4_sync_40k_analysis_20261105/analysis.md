# Static B 40k development study

**IN_PROGRESS**

Three independent training seeds; 12 development maps are repeated measurements, not 36 independent training trials. No significance, safety, predictive-content or novelty claim. No sealed test or weights loaded.

No advancement decision is calculated. All six final completions, finite diagnostics and integrity checks are required.

## Latest observed development evaluations

| Arm | Seed | Step | Success /12 | Any ball /12 | All ball /12 | Mean max occupancy |
|---|---:|---:|---:|---:|---:|---:|
| control | 1511 | 40000 | 2 | 12 | 5 | 2.0 |
| control | 1522 | 40000 | 1 | 9 | 1 | 1.25 |
| control | 1533 | 6000 | 0 | 0 | 0 | 0.0 |
| treatment | 1511 | 40000 | 6 | 12 | 8 | 2.25 |
| treatment | 1522 | 40000 | 0 | 11 | 2 | 1.1666666666666667 |
| treatment | 1533 | 6000 | 0 | 0 | 0 | 0.0 |

## Sampled learning diagnostics and completed-episode reward scale

Clip fractions and entropy/alpha below are sampled progress logs, not all updates. “pending final40k” means the final evidence is not due yet; it does not indicate nonfinite diagnostics. JSON finite_diagnostics_observed separately reports whether available learning samples are finite (null means no loss sample yet). Reward totals include completed episodes only; shaped training return is not a common cross-arm performance metric.

| Arm | Seed | Final 40k finite evidence | Critic clip fraction | Latest entropy | Latest alpha | Completed episodes | Base total | Shaping total |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| control | 1511 | verified | 1.0 | -3.030709743499756 | 0.019608404487371445 | 233 | -5216.075309475593 | 0.0 |
| control | 1522 | verified | 1.0 | -3.036823034286499 | 0.01660999469459057 | 219 | 1052.1226143696529 | 0.0 |
| control | 1533 | pending final40k | 1.0 | -2.5313851833343506 | 0.028101546689867973 | 118 | -4720.671082291988 | 0.0 |
| treatment | 1511 | verified | 1.0 | -3.268517017364502 | 0.017173416912555695 | 235 | 499.4065907900791 | -40.826149674592294 |
| treatment | 1522 | verified | 1.0 | -2.972724676132202 | 0.017590995877981186 | 235 | -507.85500732188564 | -34.683415081988805 |
| treatment | 1533 | pending final40k | 1.0 | -2.541816234588623 | 0.024868259206414223 | 100 | -5505.913782796434 | -8.127840599574682 |

Per-map exclusive failures, nonexclusive collision flags, occupancy, final distance/speed availability, common-grid AUC and peak-to-last drops are retained in analysis.json. Missing trajectory velocity is null.

![Development curves](curves.png)
