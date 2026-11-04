# Predictive-content identification plan — 2026-11-05

**Status:** analysis and proposed experiments only. This document starts no training and changes no configuration. The separately preregistered 40k-step, three-seed static B-off/B-on experiment must remain separate from the representation interventions below. No Attention, GNN, L1, dynamic-obstacle task expansion, or full paper-suite rerun is proposed for automatic launch.

## Evidence that changes the research question

The [L0 scene/content audit](../results/masac_v4_sync_pilot_diagnosis_20261105/l0_content_audit.json), produced by `scripts/audit_l0_feature_content.py`, reports that **all 12 L0 development maps have `dynamic_count = 0` and `conflict = 0`**. A map name such as `dynamic_crossing` does not establish the presence of a moving obstacle. Static obstacles are present; their actual positions and relevance to executed trajectories must be assessed from geometry, rather than inferred from scenario labels. This suite does not identify the benefit of **moving-obstacle prediction**. Other agents can still move, so the absence of external moving obstacles does not by itself eliminate the peer-prediction question.

The same audit compares recomputed `full` and `static` features on **128 replay states from the new static-control seed 1411 at 2k steps**. The reported tensor shape is **[128, 3, 19, 11]**, giving **7,296 values per channel**. In this sample:

| Channel group | `full` | `static` | What is established |
|---|---|---|---|
| Moving-obstacle clearance | Minimum = mean = maximum = 1 | 0 | This difference supplies no state-dependent moving-obstacle information in the sampled states. |
| Peer clearance | Minimum approximately 0.5683, mean approximately 0.9978, maximum 1 | 0 | There is some variation, but the mean is near the upper endpoint. The mean does not determine the exact fraction of nonconstant entries. |
| Other compared TTC/reserve/positive channels | Pointwise equal between the two variants | Same as `full` | The comparison does not establish additional predictive information in those channels for this sample. |

Use the audit's exact-one fractions and underlying counts when reporting saturation; do not infer them from a rounded mean. An endpoint of 1 may arise from a missing-entity sentinel, normalization/clipping of a valid distant entity, or both. These cases must be distinguished in the feature provider. “Padding” is therefore a confound to investigate, not a blanket label for every value of 1.

**Coverage limit:** these are early replay states from a new policy, not the historical final `full`/`static` trajectories. Feature distributions throughout the old suite, at later checkpoints, and near close peer interactions remain **to be verified**. The pilot outcome source is [summary.json](../results/masac_v4_sync_pilot_diagnosis_20261105/summary.json); short runs and early all-failure gates cannot establish a learning improvement or identify the cause of the old final ordering.

The separate [final-episode audit](../results/masac_v4_windows_audit_20261105/analysis.json) found 43 successes and 65 failures among 108 final evaluations: 61 failures lack all-agent eventual goal-ball entry, while 4 have all agents enter at some time but still fail capture. This supports diagnosing incomplete team-wide arrival. It does not identify synchronization alone as the dominant causal mechanism.

## Primary question and possible contribution

The next question is **whether the added channels contain useful state-dependent predictive information after their neutral-value convention is controlled**, and, if they do, whether their assumed peer motion is accurate enough over the probe horizon. A measured gap between `full` and zero-filled `static` cannot distinguish useful predictions from a change in constant input values, feature scale, optimization, or initialization.

An eventual paper could contribute a precise account of these confounds, a model-based error analysis, and controlled evidence showing when predictive content helps or fails. That requires evidence beyond the existing label-based ablation. PBRS and a fixed bank of geometric probes alone do not establish a frontier method contribution. A successful B run would improve the empirical story but would not settle predictive-content identification.

This positioning is constrained by [SCOPE](https://ijcai.org/proceedings/2026/7), which already combines learned previews with multi-UAV encirclement and action enforcement; [CC-HGCBF](https://arxiv.org/abs/2610.41214), which encodes finite-horizon motion capability; and [multi-agent agile racing](https://arxiv.org/abs/2605.22748), which studies anticipatory physical interactions. Our prospective claim concerns an explicitly defined surrogate representation and its limits under a frozen capture contract. We claim neither their safety properties nor their physical-control capabilities.

## First priority: constant-fill controls and channel validity

The following is a future experiment card, not permission to run its training arms. Keep the physical world, obstacle set, capture predicate, reward, B setting, policy/critic capacity, replay budget, seed convention, and evaluation distribution identical within each comparison.

| Priority / control | Exact intervention to preregister | Question it can answer | Limit |
|---|---|---|---|
| **1. Constant-fill control** | Starting from `static`, set only the added moving/peer clearance slots to the fixed neutral value 1, using the same slots as `full`. Leave every other slot untouched. Call this `constant-fill-1`; record its tensor hash/schema. | Does an information-free offset reproduce a material part of the `full` versus zero-filled `static` behavior? | A learning comparison requires matched new runs. Replacing inputs only at evaluation tests sensitivity of an already trained policy, not the effect of training on those inputs. |
| **2. Validity audit** | Log raw entity counts, inclusion masks, nearest contributing entity, pre-clipping values, clipping flags, sentinel flags, and final values for each feature family. | Which channels are absent, saturated, or informative, and in which states? | A feature equal to 1 is not enough to infer absence. Validity metadata is initially an analysis artifact; appending new policy inputs would change the experiment and capacity contract. |
| **3. Nominal `full` versus `constant-fill-1`** | Use the existing full computation for one arm and fixed neutral filling for the other. Their only difference must be state-dependent values in the added slots. | Does the varying residual carry useful information beyond neutral-value shifts? | A null result can reflect weak variation or insufficient training, not proof that dynamic information is useless. |
| **4. Centered/masked preprocessing, conditional follow-up** | Only after the provider audit, propose a fully specified subtraction/mask rule for absent or neutral channels, with fixed dimensions. | Does aligning neutral values remove an avoidable optimization artifact? | This is a representation change to preregister separately. It is not already validated and must not be inserted into the B run. |

Before any training comparison, establish feature identities on the same saved states: which slots differ, exact counts, means/quantiles, maximum absolute differences, and the nearest-entity provenance. Repeat the audit at several training stages and on terminal-neighborhood states once those trajectories exist. Report the sampling policy, checkpoint, replay indices, map, and episode rather than presenting 7,296 correlated probe values as independent observations.

If a channel is exactly constant and the encoder first applies an affine layer, its contribution can be absorbed into a bias: `W_x x + W_c c + b = W_x x + b'`. This is an elementary functional equivalence, **conditional on the actual preprocessing and first layer**. It does not make two training runs equivalent: parameter redundancy, initialization, normalization, and gradient updates can differ. Audit the implementation before using this argument. It motivates the constant-fill control; it is not a new neural-network theorem.

A shuffled-channel negative control is lower priority. It is informative only when the channel has enough state-dependent variation; shuffling almost-all-neutral values has little identifying power. If later used, preserve the declared marginal construction and state clearly that it breaks semantic alignment and may create out-of-distribution inputs.

## Second priority: peer-CV error, using the existing world

Do not introduce moving obstacles or change true acceleration limits to manufacture a positive result. On saved states, clone the existing simulator and evaluate candidate ego probes over a short horizon while peers respond under the specified checkpoint policy and the original dynamics. Use the same initial state and common random numbers for paired branches. Record precisely whether the ego holds a normalized action or a physical acceleration, including clipping and speed-limit handling.

For probe `p`, compare its predicted sampled clearance with the minimum peer clearance observed in that cloned rollout at **the same sample times**. The resulting comparator is a simulator-based diagnostic; it is not deployment-available oracle information. A peer trajectory recorded under a different ego action is not automatically a valid counterfactual, because peers may react to that ego action.

| Required error table | Report by horizon and interaction stratum |
|---|---|
| Content coverage | Included-peer count; fraction absent, saturated, and unsaturated; sample/episode/checkpoint counts. |
| Clearance error | Signed error, absolute error, quantiles, and worst observed error in physical units before feature clipping. |
| False-safe classification | Fraction of predicted-safe probes whose cloned rollout reaches nonpositive clearance; report the predicted-safe denominator. This is an empirical diagnostic rate, not a calibrated collision probability. |
| Missed-safe classification | Fraction of predicted-unsafe probes whose sampled rollout remains positive; report its denominator. |
| Probe ranking | Agreement of predicted and rollout clearance rankings and clearance regret of the predicted-best probe. |
| Task relevance | Separate near-peer interactions from long-range saturated states; associate coverage with team-wide arrival and collision/timeout outcomes without inferring causality from correlation. |

The first sensitivity test changes **only the representation's assumed horizon or acceleration scale**, with the actual world and policy rollout held fixed. Suggested levels are fractions/multiples of the current nominal setting, to be fixed after the content audit rather than selected from success results. This can test surrogate sensitivity; it is not a new environmental benchmark. A distorted direction index is useful only if the inspected encoder gives that index an action-frame meaning. Do not assume a permutation changes semantics in a permutation-invariant encoder.

Historical final-policy diagnosis requires the corresponding checkpoints or exported states. Missing Mac checkpoints cannot be reconstructed from metrics, and new 2k static policies cannot substitute for them. If only new states are available, report that scope explicitly.

## A candidate mathematical analysis with explicit limits

For a peer with known collision radius, let the predicted clearance of probe `p` be

\[
\hat c_{ij,p}=\min_{h\in\mathcal T_H}
\|\hat p_i^p(h)-[p_j(0)+h v_j(0)]\|-r_{ij},
\]

and let `c*` use the cloned ego/peer trajectories at the same finite time set. Under an ideal continuous double-integrator model, identical initial position/velocity, peer acceleration bound `A_j`, fixed radii, and a uniform ego prediction-error bound `E_i(H)`, the reverse triangle inequality gives

\[
|c^*_{ij,p}-\hat c_{ij,p}|\le E_i(H)+\tfrac12 A_jH^2.
\]

The bound follows by integrating bounded acceleration and applying the fact that taking a minimum over the same time set does not increase a uniform error bound. Without saturation, an intentional ego acceleration mismatch `Delta a` contributes at most `0.5 * ||Delta a|| * H^2` to `E_i(H)`. These are **candidate surrogate-model statements**, not yet a theorem about this repository. The formal version must follow the actual discrete update order, velocity clipping, observation errors, and candidate action semantics. Continuous collision clearance needs an additional inter-sample bound. A finite probe bank is not a complete reachable set.

This suggests measuring the prediction-error scale relative to available geometric margin. It does not yet justify an uncertainty gate, a calibrated risk probability, a safe executed policy, or improved learning. Any gate based on the bound needs a separate preregistered comparison with a scale/normalization control; otherwise an apparent gain can again be a preprocessing effect.

## What B can and cannot establish

The B pilot uses a bounded potential formed from a log-mean-exp aggregation of normalized agent distance and speed, with `lambda = 0.5`, `gamma = 0.99`, true-terminal next potential zero, and truncation next potential retained consistently with bootstrapping. This is [standard potential-based reward shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf), not a theoretical novelty. The usual telescoping argument requires the matching discount and boundary treatment. It does not guarantee faster SAC learning, successful capture, or a specific failure mechanism.

If there is no scheduled potential/weight/reset progression, describe the intervention as **fixed reward shaping**, even if B retains its historical curriculum name. A curriculum claim requires an explicit progression rule. The three-seed 40k static B-off/B-on comparison can assess the specified shaping intervention under its frozen budget. It must report per-seed team-wide arrival, joint position/speed capture, failure counts, and wall-clock cost. It provides no test of moving-obstacle prediction because the L0 maps contain no moving obstacles and the static representation masks the added channels.

## Decision rules and remaining gaps

1. **First verify information content.** If `constant-fill-1` accounts for the observed behavior and the residual has little task-relevant variation, retire a dynamic-prediction benefit claim for this suite. Publish the confound and limits faithfully.
2. **Then verify the surrogate.** If nominal peer-CV errors rise with horizon or actuation mismatch on informative states, quantify this with matched cloned rollouts. Do not infer learning harm from error alone.
3. **Only then propose a targeted learning comparison.** Keep B fixed across arms and preregister a small constant-fill/static/full comparison, followed by a validity-aware variant only if the diagnostic supports it. Existing complete-suite numbers remain historical evidence; no unapproved full rerun is implied.
4. **Reserve confirmation data.** Development maps/seeds repeatedly used for decisions are not a final holdout. Use training seeds as replicates and paired map-level summaries; do not treat probe entries or the 36 gate cells as independent Bernoulli trials.

The current evidence lacks verified feature-content coverage over the historical full trajectories, a dynamic-obstacle treatment, held-out confirmation, controlled frontier-baseline comparisons, and physical/communication-system validation. SCOPE, capability-conditioned barriers, and agile racing address stronger or different claims; our code does not reproduce their settings. A paper must explain its concrete contribution to mobile/networked systems without adding unmeasured AoI or safety claims. At present, B plus the existing ablation is not enough to promise a mature regular-paper method contribution. An identified mechanism, a justified correction, and reproducible confirmation would make that case substantially stronger; a mixed/negative result must remain an acceptable outcome.
