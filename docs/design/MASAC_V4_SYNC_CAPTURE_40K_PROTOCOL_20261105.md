# B sync-capture 40k: preregistered matched development study

**Frozen before launch:** 2026-11-05, Asia/Shanghai.
**Experiment family:** masac_v4_sync_capture_40k_20261105.
This is a new control/B experiment, not a rerun or extension of the old 80k
full/static/zero suite. The launch plan must be visible to the user first.
No Attention, L1, reward search or automatic budget extension is included.

## 1. Evidence and the one hypothesis

The two-arm, two-seed 2k Windows pilot completed with finite losses, no NaN and
zero final joint successes/all-agent eventual ball entries on all four runs.
Only B/1422 had one any-agent-entry map. This establishes implementation/runtime
sanity, not evidence for or against capture improvement. Its 128-step warmup and
small replay are not the scientific comparison for the study below.

The old final suite's 65 failures include 61 partial-agent-entry failures and
four all-agent-entry failures; none collide. Its 12 L0 development maps have
zero moving obstacles. The new early control/1411 replay audit also finds a
constant moving channel and mostly saturated peer clearances. This early sample
does not characterize the old final replay. These facts preclude treating L0
full-versus-static as an identified moving-obstacle prediction experiment.
See the root audit L0_CONTENT_IDENTIFIABILITY_AUDIT_20261105.md.

**Hypothesis H_B:** at a fixed 40k learning budget, adding the already defined
bounded team potential to the static learner improves joint capture by helping
worst-agent approach and speed control. The causal mechanism is not established;
this study tests only the total effect of enabling B under this configuration.

## 2. Arms and constants

| Item | Frozen value |
|---|---|
| Arms | static control (B disabled) versus static+B (B enabled) |
| Initialization | Cold start for every seed/arm; no pilot or Mac checkpoint load |
| Training seeds | 1511, 1522, 1533, paired by seed id across arms |
| Budget | Exactly 40,000 environment transitions per seed/arm; six runs, 240,000 total |
| Warmup / batch / replay | 1,000 / 128 / 100,000, identical in both arms |
| Learner / architecture | Shared MASAC, pool SetEncoder, static 19×11 probe inputs; unchanged optimizer/update schedule |
| B parameters | Fixed coefficient lambda=0.5, temperature tau=0.1, scales 4 m and 2 m/s |
| Discount / entropy target | Existing gamma=0.99 and implemented target differential entropy −3 |
| Reward / terminal semantics | Existing Path-C; true terminals mask bootstrap, timeout truncations do not |
| Compute | Same benchmark-supported CUDA device/build, risk-cache setting and one torch thread in both arms; record audit/config/source hashes |
| Evaluation | Original Path-C environment; B shaping disabled; strict simultaneous 4 m / 2 m/s, latch=False |
| Development maps | The same existing 12 frozen L0 map entries, including their explicit family assignments |
| Evaluation schedule | Every 2,000 steps and at 40,000; identical across arms |
| Checkpoint rule | Retain aligned checkpoints; **40k final** is the only efficacy endpoint |

JSONs, command lines, source hashes, device build, cache flag and maps manifest
must be saved before launch. The two arm JSONs differ only in identifiers and
the B enabled flag. Thread/concurrency changes for thermal or memory stability
must be recorded; do not change learning parameters after seeing outcomes.

Use unique output directories and independent replay/optimizer/RNG state.
Same seed and episode-index map-generation rules do not guarantee identical
training-map exposure at the same transition index: policy-dependent episode
length changes when the next map is reached. Report this limitation rather than
claiming transition-wise paired training maps. Evaluation maps are exactly paired.

The new CUDA arm is explicitly benchmark-supported; it does not silently change
the historical CPU protocol or establish cross-device bitwise equivalence.

## 3. Endpoints and a fixed advancement decision

For training seed s and arm a, let S_a,s be final deterministic joint successes
out of the same 12 development maps at exactly 40k. Define

    Delta_s = (S_B,s - S_control,s) / 12;
    Delta_mean = (Delta_1511 + Delta_1522 + Delta_1533) / 3.

The **primary development endpoint** is Delta_mean, with all three paired
Delta_s and both arms' raw counts shown. No best-checkpoint replacement and no
pooling of 36 rollouts as 36 independent training trials is permitted.

The preregistered **engineering advancement rule** for B_v1 is all of:

1. Delta_mean is at least 1/12 (one additional successful map per seed on average).
2. At least two of the three Delta_s are strictly positive.
3. No seed loses more than 1/12 relative to its paired control.
4. All six runs complete the frozen budget with finite diagnostics and
   unmodified evaluation/learning contracts.

This is a practical rule for deciding whether to spend on confirmation, not
a significance test or a paper claim of superiority. Three training seeds have
low inferential resolution; maps and checkpoints are repeated measurements.
Also report the unchanged official gate (at least 8/12 in **every** training seed)
for each arm. The advancement rule does not replace that gate.

If the rule is not met, **do not advance or extend B_v1 automatically**.
If both arms remain at zero joint success, or the control has not produced a
capture signal and B also fails to produce a material signal, classify the
40k comparison as **inconclusive at an unestablished capture floor**, not proof
that potential shaping cannot help. If B produces the preregistered signal while
control remains zero, label it a preliminary fixed-budget learning gain, still
requiring confirmation. If control learns and B does not improve it, report the
negative/mixed fixed-budget result. Preserve every seed and failure.

Secondary, descriptive endpoints, in this fixed order:

1. Any-UAV eventual entry, all-UAV eventual entry and maximum simultaneous
   strict capture occupancy, with exclusive failure categories and nonexclusive
   collision flags.
2. Final worst-agent goal distance and speed, from saved evaluation trajectories;
   distinguish approach, speed and timing rather than infer them from any-entry.
3. Joint-success development-curve area over the common 2k evaluation grid and
   peak-to-final drop on that grid, solely as learning/instability diagnostics.
4. Base reward versus B shaping scale, update count, samples/second, wall time,
   RAM/VRAM and failures. Training shaped return is not a common performance metric.

No secondary result overrides the primary decision. Do not choose a coefficient,
seed, checkpoint or new endpoint after inspecting the development curves.

## 4. Prespecified unseen-map namespace, initially sealed

Reserve exactly these 12 map seeds, with family assignment
FAMILIES[(k-1) modulo len(FAMILIES)] in the frozen source order:

    2026110501, 2026110502, 2026110503, 2026110504,
    2026110505, 2026110506, 2026110507, 2026110508,
    2026110509, 2026110510, 2026110511, 2026110512.

Save the explicit seed/family manifest before launch. Verify no overlap with
training/development namespaces. Geometry/reference feasibility checks may be
performed without learned policies; record any invalid scene and its reason.
Do not replace a valid map because a learned policy fails on it.

**Do not run learned policies on these maps during training or development.**
First finish all six runs, freeze the 40k policy artifacts and record the
development advancement decision. Any later one-time test evaluates the same
final checkpoints from both arms on the complete sealed set and is reported
separately. Test outcomes must not select checkpoints or retune B. This protocol
reserves the test; it does not add an automatic test launch or a code change.
Twelve maps and three training seeds remain limited evidence, including when
their result is positive. L0 test maps do not supply moving-obstacle evidence.

## 5. Stops, failures and prohibited adaptation

- No efficacy-based early stopping of a slow seed, no favorable-seed replacement,
  no best-of-many checkpoint selection, and no automatic 40k-to-80k extension.
- Stop an affected run for NaN/Inf, an invalid scene/reference, corrupted artifact,
  crash, OOM or unstable hardware. Print the stack/state, retain its artifacts,
  report the failure, and diagnose before any restart.
- A hardware/concurrency-only recovery must preserve the frozen learning protocol
  and use a verified exact checkpoint/replay/optimizer/RNG restore, or restart
  the affected pair from cold with the same seed and a documented attempt id.
  Do not relabel a weights-only load as exact continuation.
- A semantic code repair invalidates affected comparisons; register a new source
  version/experiment rather than silently mixing histories.
- No lambda/tau sweep, entropy-target change, reward rescaling, capture relaxation,
  Attention, L1, new replay mechanism or old-suite rerun is part of this study.

An inconclusive or negative outcome ends this registered budget. Any later
mechanistic change requires a recorded hypothesis and a new visible plan.

## 6. Entropy/late-collapse diagnostics without changing SAC

The implemented entropy target −3 is conventional in sign and is not itself a
bug. Existing alpha, sampled tanh-space entropy, TD residual, critic/actor
clipping and terminal-sample-fraction traces should be retained at aligned
update indices. Correlate them with fixed-grid performance drops, without
asserting that exploration decay causes a drop.

If fixed-observation log-standard-deviation or action-saturation diagnostics are
needed, compute them offline from saved checkpoints on a frozen common observation
batch. Additional stochastic diagnostic calls must not consume the training RNG.
Tanh entropy is not executed-acceleration entropy or a direct measure of state
coverage. Do not retarget entropy or modify SAC within this experiment.

## 7. Contribution boundary

The standard identity

    sum_t gamma^t F_t = -lambda Phi(s_0) + gamma^T lambda Phi(s_T)

with absorbing-zero true terminals and a consistently transformed timeout
bootstrap is the mathematical basis, not a new theorem. It preserves the
relevant **Path-C return** under the stated assumptions; Path-C itself is not
equivalent to maximizing sparse joint success. No identity implies a finite-data
SAC learning improvement, safety, moving-obstacle prediction benefit or novelty.

A positive B result would be an engineering learning response to an audited
failure mode. A paper contribution would require the controlled evidence,
transparent negative predictive-feature result, reliable failure taxonomy and
a demonstrated distinction from close preview/risk/MARL prior art. Any later
claim that B unlocks predictive information needs a separately registered
identifiable representation/mechanism contrast; static+B alone cannot establish it.
