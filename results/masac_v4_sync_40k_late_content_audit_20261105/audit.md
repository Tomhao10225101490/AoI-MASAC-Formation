# Late-replay probe content audit

Late-training replay of the completed static-control and static+B learners, seeds 1511/1522/1533. Not final-policy on-policy observations, not historical full-trained policy replay, and not sealed-test data.

Purely offline descriptive feature content. Correlated transition/agent/direction samples do not establish prediction accuracy, feature use by learned networks, a causal explanation for old full-vs-static results, treatment efficacy, safety or novelty.

Sampling is fixed before feature computation: 1,024 unique linspace indices in transitions [32000,40000), from each of six final replays. Observations are pre-action states; the sampled transitions are not independent trials.

static_fill1 copies the static tensor and sets only output columns 4/5 (moving/peer) to one. TTC, braking, reserve, positive flag, direction, static and boundary columns remain bitwise static. This is a feature-content contrast, not a trained arm or an accuracy test.

Raw entity presence, mode inclusion, geometric pre-clipping minima and provider sentinel minima are traced separately. A valid far peer can coexist with a smaller invalid-slot sentinel; an output value of one alone is neither absence nor a physical saturation proof. Source ties are nonexclusive.

| Arm | Seed | Valid moving slots | Valid peer slots | Full peer exact-1 with valid peer | Full/fill1 whole-vector equality |
|---|---:|---:|---:|---:|---:|
| control | 1511 | 0 | 6144 | 1.0 | 1.000000 |
| control | 1522 | 0 | 6144 | 0.9938493695175439 | 0.993849 |
| control | 1533 | 0 | 6144 | 1.0 | 1.000000 |
| b | 1511 | 0 | 6144 | 1.0 | 1.000000 |
| b | 1522 | 0 | 6144 | 1.0 | 1.000000 |
| b | 1533 | 0 | 6144 | 1.0 | 1.000000 |

## Pooled channel distributions

Equal sample sizes give every history equal weight. Standard deviation is the population standard deviation of the observed feature values, not uncertainty over independent training runs.

| Channel | Full mean / std | Static mean / std | Fill1 mean / std | Full exact-1 | Full−static MAE | Full−fill1 MAE | Fill1−static MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ux | 0 / 0.561951 | 0 / 0.561951 | 0 / 0.561951 | 0.052632 | 0 | 0 | 0 |
| uy | 0 / 0.561951 | 0 / 0.561951 | 0 / 0.561951 | 0.052632 | 0 | 0 | 0 |
| uz | 0 / 0.561951 | 0 / 0.561951 | 0 / 0.561951 | 0.052632 | 0 | 0 | 0 |
| static | 0.999999 / 0.00023236 | 0.999999 / 0.00023236 | 0.999999 / 0.00023236 | 0.999977 | 0 | 0 | 0 |
| moving | 1 / 0 | 0 / 0 | 1 / 0 | 1.000000 | 1 | 0 | 1 |
| peer | 0.999869 / 0.00521077 | 0 / 0 | 1 / 0 | 0.998975 | 0.999869 | 0.000130956 | 1 |
| boundary | 0.632637 / 0.127867 | 0.632637 / 0.127867 | 0.632637 / 0.127867 | 0.002664 | 0 | 0 | 0 |
| ttc | 0.999961 / 0.00290699 | 0.999961 / 0.00290699 | 0.999961 / 0.00290699 | 0.999769 | 0 | 0 | 0 |
| brake | 0.0172554 / 0.0385176 | 0.0172554 / 0.0385176 | 0.0172554 / 0.0385176 | 0.000000 | 0 | 0 | 0 |
| reserve | 0.672089 / 0.111872 | 0.672089 / 0.111872 | 0.672089 / 0.111872 | 0.000000 | 0 | 0 | 0 |
| positive | 0.999569 / 0.0207602 | 0.999569 / 0.0207602 | 0.999569 / 0.0207602 | 0.999569 | 0 | 0 | 0 |

## Frozen-provider trace cross-check

All chunks passed absolute tolerance 1e-6, rtol=0. Maximum errors: {'full': 0.0, 'static': 0.0}. Bitwise equality on every chunk: {'full': True, 'static': True}.

The trace is audit-only code, hashed separately from frozen training sources. It reconstructs the same clearance tensor before normalization, including cleaned invalid observation slots, and independently cross-checks all 11 full/static outputs. Global object identities absent from observations remain unknown.

Fixed quantiles are [0, 0.25, 0.5, 0.75, 1], with NumPy linear interpolation. Exact equality/zero/one counts are distinct from the declared absolute 1e-6 tolerance. Nonfinite outputs fail and retain an error record.

Every channel distribution and all three pairwise differences are in audit.json and features.csv. Family preclip/sentinel/inclusion counts, typed-slot contributor histograms, current/preview/first-hit source attribution and changed derived-channel attribution are in audit.json and provenance_counts.csv. Source ties are nonexclusive.

Fixed indices and their byte SHA256 are in sample_indices.json. Each selected-input sidecar is saved before its feature computation and separately hashes canonical dtype/shape/order plus selected observation bytes. Replays/PT files are neither copied nor uploaded by this script.
