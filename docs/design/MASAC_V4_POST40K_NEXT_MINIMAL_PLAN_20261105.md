# Post-40k next minimal step: offline feature-content audit

**Date:** 2026-11-05 Asia/Shanghai.
**Status:** analysis plan only. This document launches no training, loads no policy, produces no policy actions, and opens no sealed-test maps or results. It does not change frozen source snapshots, configurations, capture predicates, or gates. The completed 40k study supports planning a confirmation study; it does not authorize an automatic 40k-to-80k continuation or any additional training.

## 1. Decision that motivates this audit

The canonical [40k analysis](../../results/masac_v4_sync_40k_analysis_20261105/analysis.md), with [machine-readable results](../../results/masac_v4_sync_40k_analysis_20261105/analysis.json), reports:

| Paired training seed | Static control final success | Static B final success | Paired count difference |
|---|---:|---:|---:|
| 1511 | 2/12 | 6/12 | +4 |
| 1522 | 1/12 | 0/12 | -1 |
| 1533 | 4/12 | 5/12 | +1 |
| Descriptive total | 7/36 | 11/36 | +4/36 = +1/9 success fraction |

The preregistered engineering result is **`ADVANCEMENT_RULE_MET`**. This means planning a confirmation study is warranted under that rule. The separate absolute 8/12 development gate is **FAIL, 0/3 seeds in each arm**. One paired seed regresses. Neither a statistical-significance claim nor a reliable-capture claim follows.

Across the 72 final map episodes there are 18 successes, 54 failures and zero observed collisions. The final-trajectory audit partitions the failures into **43 without all-agent eventual ball entry**, **1 with all-agent ball entry but without all-agent eventual individual capture**, and **10 with all-agent eventual individual capture but no joint capture**. These categories separate unfinished approach, individual position/speed capture, and simultaneity; zero observed collisions is not a safety guarantee. Per-seed sources are the final trajectory summaries for [1511](../../results/masac_v4_sync_40k_progress_20261105/trajectory_40000_seed1511/summary.json), [1522](../../results/masac_v4_sync_40k_progress_20261105/trajectory_40000_seed1522/summary.json), and [1533](../../results/masac_v4_sync_40k_progress_20261105/trajectory_40000_seed1533/summary.json).

B changes only fixed potential shaping in these static arms. It does not test the utility of dynamic probe information. The earlier [L0 feature-content audit](../../results/masac_v4_sync_pilot_diagnosis_20261105/l0_content_audit.json) found no moving obstacles or conflict setting in any of the 12 development maps, and found near-neutral added channels in a new 2k replay sample. That early sample cannot characterize later training. The next step addresses this narrow evidence gap without collecting new policy rollouts.

## 2. Population, allowed inputs and prohibited inference

Use exactly the **six final-40k replay artifacts** belonging to the static control/B arms and seeds 1511, 1522 and 1533. Resolve their paths from the completed run manifests, not by recursively exploring unrelated result directories. Read replay data and the public probe/configuration metadata needed to recompute features. Do not load a model checkpoint, instantiate an actor/critic, sample policy actions, call evaluation, reset a simulator, alter world parameters, or inspect a sealed test.

The population is **late-training replay collected by static control and static B policies**. It is not a fresh rollout of the final policies, a sample of the old full-policy distribution, or a sample of the historical final-80k trajectories. The two arms induce different state visitation and episode lengths; their sampled states are not matched across arms merely because the training seed matches. Only the three feature computations on each identical saved observation are directly paired.

This audit can measure available feature content and preprocessing differences on these saved states. It cannot establish that the actor used a feature, that a channel caused the old full arm's weakness, that B corrected a predictive representation, or that a dynamic-obstacle method is effective.

## 3. Freeze the last-window sample before computing features

For each replay independently:

1. Record the replay file's absolute path, byte length and SHA-256, associated run/arm/seed, completed training step, source/config/manifest hashes, tensor schema, retained size, write pointer and serialization format. Verify that it belongs to a completed 40k run.
2. Reconstruct chronological transition order correctly if storage is circular. Identify the **last 8,000 retained transitions** ending at the final training snapshot. Do not treat physical array order as chronology after wraparound. Record global environment-step ids if present; otherwise label positions as replay chronological ranks and do not invent step ids.
3. If fewer than 8,000 valid chronological transitions are recoverable, stop that file with an explicit insufficiency record. Do not silently use an earlier window, fewer transitions, or a replacement replay.
4. For `k = 0,...,1023`, choose the deterministic window-relative index

   `j_k = floor(k * 7999 / 1023)`.

   This selects exactly **1,024 distinct entries**, spans both window endpoints, uses no sampling RNG, and includes no hand-selection based on failures, feature values or outcomes. Record the mapping from window rank to physical replay index.
5. Freeze an index manifest and its SHA-256 before feature computation. Hash the selected **raw observations** in a documented canonical serialization, including dtype, shape, byte order and array order. The selected observation bytes must be identical for all feature variants. Keep replay SHA, index-manifest SHA and selected-input SHA distinct.

There are 6,144 selected transitions if all six files meet the contract. Verify the actual observation and output shapes. With the expected three agents and 19 probes, a single channel has 58,368 values per replay (`1024 * 3 * 19`), and 350,208 across six replays. These are correlated values, not independent statistical trials. Entity-validity denominators at the agent-observation level must also be reported separately.

## 4. Three feature counterfactuals on the same observation

Use the frozen probe implementation and each run's recorded public parameters. Recompute from raw observations; do not substitute the cached training-static tensor for the nominal full computation. Preserve observation masks, agent order, units, acceleration/speed limits, probe horizon, sampling grid and all 19 direction vectors.

| Variant | Definition |
|---|---|
| `full` | Existing full feature computation using all included visible object families. |
| `static` | Existing static computation, including its exclusion of moving/peer objects before derived features and zero output in their two clearance slots. |
| `static-constant-fill-1` | Copy the **already computed static tensor**, then set only its moving-clearance and peer-clearance output slots to 1. Leave all other channels bitwise unchanged. Do not insert normalized 1 into a physical clearance calculation or recompute static derived aggregates after the fill. |

The expected 11-channel schema is: direction coordinates 0–2; static/moving/peer/boundary normalized clearance at 3–6; sampled first-hit time at 7; braking distance at 8; braking reserve at 9; positive-clearance flag at 10. Verify this against the frozen provider and fail loudly on a mismatch. Do not rely on labels from an older feature version.

`static-constant-fill-1` isolates a neutral input offset. It is not an additional trained policy or a deployment evaluation. In particular, static and constant-fill must have identical directions, static/boundary features, TTC, braking distance, reserve and flag; this is a required equality check.

## 5. Separate missing entities from valid saturation

For each object family, preserve enough provider provenance to distinguish:

- No included object: the raw observation has no valid entity after its declared visibility/inclusion mask; the provider uses a neutral/sentinel value.
- A valid included entity with a pre-clipping clearance at or beyond the normalization limit: the normalized feature saturates at 1.
- A valid included entity with an unsaturated feature, including nonpositive clearance where applicable.

Log **present in the raw observation**, **enabled for this feature mode**, and **contributed to the minimum** separately. Static's deliberate disabling is not a claim that an object was absent from the original observation. Do not infer physical absence from a feature equal to 1. Record raw object count, valid/included count, nearest contributing entity type/id, pre-clipping value, clipping/sentinel flag and final normalized value wherever recoverable.

Provider instrumentation must be observational and leave computed features identical to the frozen routine. Cross-check recomputed outputs against the uninstrumented frozen function on the same bytes. Missing provenance should be reported as unknown; do not invent an absence/saturation classification from the output alone. Analysis metadata must not be appended to actor inputs.

## 6. Required tables and derived-channel propagation

For each replay and pooled descriptive summary, save:

| Table | Required fields |
|---|---|
| Sample provenance | All six input/index hashes, chronological window, exact indices and raw/probe tensor shapes. |
| Validity and saturation | Counts/denominators for absent, valid saturated, valid unsaturated and unknown provenance, by object family and agent-observation/probe level as appropriate. |
| All 11 channels | Minimum, mean, standard deviation, fixed quantiles, maximum, exact-zero/exact-one counts, and nonfinite counts for every variant. State the fixed near-endpoint tolerance separately from exact comparisons. |
| Pairwise differences | For full–static, full–constant-fill and constant-fill–static: signed summaries, maximum absolute difference, exact-equality count and count above a declared numerical tolerance, channel by channel. |
| Derived channels | Differences in TTC (7), braking distance (8), reserve (9), and flag (10), with numerator and denominator and source-object provenance. |

Derived changes need interpretation. A valid nearby peer can legitimately alter full TTC/reserve/flag through the full minimum. A missing/sentinel-only family changing an aggregate is a different issue and must be isolated. Record whether the changed minimum/hit came from a valid peer, moving object, static/boundary object, a sentinel, or unknown provenance. Do not call all aggregate differences an implementation defect.

The brute pattern to test is whether full–static is mostly the two clearance offsets, whether full–constant-fill retains material state-dependent variation, and whether that residual also propagates into derived channels. Near-neutral variation in the new late sample would strengthen a **content-coverage limitation**; it would not establish the cause of historical training outcomes. Genuine variation would justify planning a more targeted validity analysis; it would not itself show learning benefit.

Save compact JSON/CSV tables and a narrative report under a new analysis-only result id. Keep full replay/checkpoint files local and preserve input hashes for reproducibility. Record the audit script/source hash, numerical environment and any provider-trace implementation separately from the frozen training source hashes.

## 7. Completion and future confirmation planning

This minimal step is complete when all six files have verifiable sample provenance or explicit insufficiency records, all three variants share identical selected inputs, the static/constant-fill equality conditions pass, nonfinite values are accounted for, and channel/validity/aggregate tables are reviewable. No performance gate is computed from these feature statistics.

After reviewing the audit, a **future confirmation plan** may specify fresh paired training seeds, initialization, fixed budget, B setting, static/constant-fill/full comparisons, stopping rules and a sealed-data policy. It must first determine whether the desired contrast has informative, nonconstant content under the unchanged L0 task. Keep the completed 40k and historical 80k runs as separate evidence. Do not extend an existing run, select a favorable checkpoint, alter L0 world parameters, add Attention/GNN/L1, or open sealed evaluation data as an automatic response to `ADVANCEMENT_RULE_MET`.

The theoretical context remains standard: [entropy-regularized PBRS](https://arxiv.org/html/2212.01174v2) supplies a conditional exact-value/policy-ordering result, not a new 2026 theorem or a guarantee of neural learning improvement. [SCOPE](https://ijcai.org/proceedings/2026/7) and [CC-HGCBF](https://arxiv.org/abs/2610.41214) remain close precedents for preview-based coordination and capability representations. The next auditable contribution is a correct identification of what information this experiment actually varies.
