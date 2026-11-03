# Research gaps and novelty gates

Assessment date: 2026-10-31. These are falsifiable project hypotheses, not established novelty claims. Evidence level is recorded in LITERATURE_REVIEW.md and PAPER_MATRIX.md.

## Claims ruled out as standalone novelty

- “AoI observations plus Kalman prediction for UAV formation” overlaps [DI-MATD3](https://doi.org/10.3390/drones10100762); full-text comparison is still required.
- “Value-of-information scheduling for formation” overlaps [Chiariotti and Fabris](https://arxiv.org/abs/2507.06392).
- “Attention combined with multi-agent soft actor-critic” overlaps [MAAC](https://arxiv.org/abs/1810.02912).
- “Graph and recurrent networks for UAV communication” overlaps [Feng et al.](https://arxiv.org/abs/2401.17880).
- “Joint movement and discrete scheduling with age and memory” overlaps [Emami et al.](https://arxiv.org/abs/2405.00056).
- “AoI plus estimator uncertainty supplied to SAC” overlaps [Bai et al.](https://arxiv.org/abs/2507.14299).
- “Task-aware message ordering and retransmission” overlaps [Wu et al.](https://arxiv.org/abs/2312.13182).

## Candidate A: corrected and stabilized MASAC

Rationale: a reliable optimization reference is a prerequisite for assessing additional information. Preserve the original commit and label the corrected objective distinctly. Test reward scaling, LayerNorm or residual architecture only after controlled pilots; equal gradient-update and data budgets are necessary. This is engineering quality with low standalone novelty. A stronger optimizer cannot repair an uninformative objective or unavailable observations.

## Candidate B: shared graph-based age-aware MASAC

Rationale: fixed concatenation and separate networks grow with team size. Parameter sharing and masked aggregation can represent variable neighborhoods. Existing graph/attention literature makes its generic novelty weak. A useful test is held-out team sizes and topologies with matched information, capacity and training effort. Graph benefit must exceed a shared MLP/set-pooling model; count changes must not silently change radio density, area or collision difficulty.

## Candidate C: resource-accounted communication/control co-optimization

Preferred research hypothesis: under a fixed radio budget, timestamp-valid messages differ in their expected reduction of future formation error, and a scheduler can exploit that difference when age-only priority cannot. Hold the flight tracker fixed initially, then introduce motion optimization only after demonstrating a stable scheduling effect. This isolates the communications contribution while preserving a path to joint control.

A possible analytical score is delivery probability times predicted reduction in receiver slot covariance or task error, minus the resource opportunity cost. This is a proposed adaptation, not a new theorem. Receiver state used in the score must be a delivered, timestamped cache; perfect receiver knowledge belongs in a privileged reference. Evaluate a learned correction to an analytical score only if the latter leaves systematic residual error.

## Required evidence chain

1. Correct timestamps: queueing/relay age equals elapsed time since the original source sample; identical or older messages cannot reset age.
2. Feasible decisions: account for leader transmissions, relay data, synchronization/control summaries, retransmissions and guard times under the same budget.
3. Isolated value: compare no relay, random, round robin, freshest, receiver age-gap, uncertainty-weighted analytical policy, and learned policy. Label each information privilege.
4. Distinct interventions: remove age, uncertainty, receiver usefulness, graph relations and memory only when actually implemented. Match dimensions/capacity where possible.
5. Generalization: frozen unseen maps and leader maneuvers, burst losses, delayed/out-of-order packets, varying team counts. Separate fixed area from fixed density.
6. Statistical unit: independent training seeds first, paired evaluation scenarios within each seed. Do not treat many episodes from one checkpoint as many independent training runs.

## What would falsify the proposed contribution

If gains vanish at equal airtime, the effect is extra resources. If gains disappear without current receiver truth, the deployable information model is inadequate. If an analytical policy matches the learner across held-out conditions, retain the simpler method and frame the result accordingly. If uncertainty only helps because its scale reveals a hidden scenario label, the uncertainty claim fails. A positive graph ablation on training maps alone does not establish scalable generalization.

## Outstanding search boundaries

This registry is targeted, not exhaustive. More recent direct competitors from IEEE and Elsevier require full-text access, and implementation-level comparison is incomplete. The phrase “first” is not justified. No acceptance probability or publication guarantee follows from these findings.
