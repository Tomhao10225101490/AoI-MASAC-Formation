# PAPER_OUTLINE_DRAFT — Predictive Maneuverability-Aware MASAC

**2026-11-05 interpretation boundary:** L0's twelve archived development maps
contain no external moving objects. The existing suite therefore does not
identify a moving-obstacle prediction effect. New early static replay also
shows largely neutral/saturated added channels; do not generalize that sample
to historical final trajectories. The2k B pilot has finite updates but0/12
final successes in all four seed/arm runs. The registered40k static B-off/on
comparison is now COMPLETE: control [2,1,4]/12, B [6,0,5]/12. Its original
advancement rule permits planning confirmation only; both absolute gates FAIL 0/3.
It tests fixed shaping only. Include these limitations in the abstract,
ablation interpretation and discussion; see `L0_CONTENT_IDENTIFIABILITY_AUDIT_20261105.md`.

**Draft date:** 2026-11-04 15:20 Asia/Shanghai  
**Target shape:** IEEE two-column (WoWMoM regular or WIP)  
**Updated:** 2026-11-05 Asia/Shanghai. Separate completed cohorts are the historical Mac 80k suite, legacy warm-started Standard-C floor and Windows 40k control/B study. No significance, predictive superiority, safety, convergence, unseen-map generalization or established novelty claim.

Working title (EN):  
**Predictive Maneuverability Probes for Multi-UAV Soft Actor-Critic: Ablating Static Geometry vs Dynamic Clearance Features under Joint Capture**

Alternate (shorter):  
**When Do Predictive Maneuverability Features Help Multi-UAV MASAC?**

---

## Abstract (evidence-constrained prose draft)

Simultaneous goal settling requires three UAVs to reach independent fixed goal balls at low speed at the same time. We evaluate an observation-only representation comprising 18 acceleration directions plus a zero command, sampled clearance features and a pooled encoder. A historical 80k-step, three-seed content-mask suite on 12 fixed L0 development maps yields descriptive totals of 26/36 successes for static-geometry preview, 15/36 for full channels and 2/36 for zero inputs; no arm passes the all-seed gate. The maps have no external moving obstacles, so these results do not identify a moving-obstacle prediction benefit. In a separate preregistered 40k-step static control/B study, fixed potential shaping gives final successes 6/12, 0/12, 5/12 versus control 2/12, 1/12, 4/12. The mean paired difference is 11.111 percentage points, with one negative seed. The original engineering advancement rule is met for planning confirmation only; both arms fail all three absolute 8/12 gates. Of 54 failures across the 72 new final rollouts, 43 lack all-agent ball entry, one lacks all-agent individual strict capture despite all-agent entry, and 10 lack simultaneous capture despite all-agent individual capture. There are no collisions. These descriptive studies motivate identifiable input-content and learning-mechanism diagnostics, without significance, generalization, safety or novelty claims.

Keep the two cohorts separate. The sealed test set remains unevaluated, and the 40k engineering rule does not establish a confirmed B effect.

---

## I. Introduction

1. Multi-UAV independent-goal navigation needs all-agent approach, low speed and cooperative timing.
2. Progress and any-agent entry do not establish an all-agent capture capability.
3. Question: what do static-geometry preview and added CV peer/moving channels contribute under the same learner, reward, capacity and budget? Existing preview/risk/agile MARL is substantive prior art; do not claim an unsearched absence.
4. Proposal: observation-only probes + RiskEncoder; ablations `full` / `static` / `zero`.
5. C1–C3 (see `PAPER_PLAN_WOWMOM2027.md`): explicit representation, controlled ablation and auditable failure taxonomy. Distinctiveness remains under review.
6. State both completed studies: historical static 1233's late drop, and 40k B's positive mean with the 1522 negative pair and failed absolute gates. The planning rule does not supply confirmatory inference.

---

## II. Related Work

Use `docs/RELATED_WORK_FRONTIER_20261104.md` and `docs/FRONTIER_GAP_UPDATE_20261105.md`. Prefer primary sources and distinguish preprint from peer-reviewed evidence.

- II-A Multi-UAV formation & obstacle avoidance RL (Xie; Huang; STAAC; DTPPO)
- II-B Agile flight, flatness, MPC+RL (AC-MPC; DATT)
- II-C Predictive / risk-aware RL (OPEN; CPA risk; CBF-Attention)
- II-D Set / attention encoders (AAC; swarm attention) — explain pool freeze
- II-E Continuous MARL / MASAC for capture/pursuit
- II-F Brief AoI adjacency (optional)

End with mechanism comparisons to SCOPE, CC-HGCBF, CPA risk and OPEN. Agile RL now includes multi-agent racing; do not describe it as exclusively single-UAV. Sampled features are distinct from action correction or certified-safe control.

---

## III. Problem Formulation

- III-A Partially observed team control: latent augmented state \(S_t\), local \(o_i=h_i(S_t)\in\mathbb{R}^{162}\), raw \(a_i\in(-1,1)^3\), \(N=3\). Central critics use joint observations; this does not prove a Markov observation state.
- III-B Dynamics & maps: L0 development set `830001`–`830012`.
- III-C Success: joint capture ball 4 m, speed < 2 m·s⁻¹, non-sticky.
- III-D Metrics: joint success, any-UAV eventual ball entry, all-UAV eventual entry, collisions and success-conditional path efficiency. Historical `ever_ball_maps` and `timeout_after_ball` use **any-UAV** entry; state that definition.

Tie to MATH E1.

---

## IV. Method

Paper-ready English prose and source-matched Equations (1)–(8) are now in
[PAPER_METHOD_DRAFT_20261105.md](PAPER_METHOD_DRAFT_20261105.md). The B policy-ordering
and neural-learning distinction is derived in
[MATH_PBRS_SCOPE_NOTE_20261105.md](MATH_PBRS_SCOPE_NOTE_20261105.md).

### IV-A Shared MASAC learner (not novelty)
- Twin team Q, observation-based target and actor/temperature losses — MATH E1–E3. Exact temperature variable is log-alpha; the implemented entropy target is \(-3\).
- Historical replay stores true terminal only and bootstraps on timeout; no unconditional finite-horizon Bellman or neural convergence theorem.
- Path-C reward — MATH E4. Its discounted progress and undifferenced near-goal bonus are not policy-invariant potential shaping.

### IV-B Predictive maneuverability probes (candidate)
- Control bank \(U\): **18 unit directions plus zero**, not a subset of the unit sphere; \(\Delta=0.2\) s, \(H=2.0\) s, 11 sample times.
- CV extrapolation for moving/peers from observation only.
- Feature \(f_j\in\mathbb{R}^{11}\) — MATH E5.
- Disclaimer: not certificates — MATH E7.

### IV-C RiskEncoder + fusion
- \(\phi:11\to64\); mean/max → 128; SetEncoder pool 384; fuse 512 — MATH E6.

### IV-D Ablation modes
| Mode | Probe content |
|------|----------------|
| full | all clearance channels |
| static | Ego preview retained; exclude mv+peer before first-hit/reserve/flag calculation, then zero those two output channels |
| zero | All \(f_j=0\), including direction coordinates; encoder can still learn a constant |

Static is not prediction-free. A valid unrun direction-shuffle holds directions fixed while remapping the eight scalars; full-token permutations are vacuous under invariant pooling.

### IV-E B fixed potential shaping (40k study COMPLETE; confirmation unestablished)

Follow `docs/design/MASAC_V4_SYNC_CAPTURE_20261105.md` for the fixed potential coefficient and terminal/bootstrap convention. B is **fixed potential shaping**, with no scheduled reset/reward progression and no explicit arrival-time-gap term. Distinguish pathwise algebra, exact soft-objective policy ordering and finite-sample learning; cite established entropy-regularized PBRS rather than claiming a new invariance theorem. Target worst-agent approach/capture; the failure counts do not identify a pure synchronization cause. Evaluation remains 4 m / 2 m·s⁻¹, latch off.

---

## V. Experimental Protocol

- Historical hardware: Apple Mac mini, CPU suite. Windows audit/device timings are separate in `docs/HARDWARE_AUDIT_WIN.md`. CPU remains the historical default; the new explicit CUDA protocols have measured throughput evidence (8.75x microbenchmark, 6.58x complete smoke) and do not retroactively change the Mac protocol.
- Historical Mac suite: 80 000 steps/seed; warmup 1000; eval every 2000.
- Seeds: 1211 / 1222 / 1233 paired across arms.
- Gate: ≥0.60 = **at least 8/12** per seed; an arm passes only if all three seeds pass.
- Within-suite full/static/zero match architecture, capture/reward, paired seed ids and 80k budget. Three seeds on repeatedly observed development maps limit inference; 36 rollouts are not 36 independent training trials.
- Standard-C is a separate floor: seeds 911/922/933, 50k L0, legacy encoder; each warm-started from the same empty-floor seed-811 40k checkpoint (weights+optimizers; replay/RNG reset). These confounds preclude fair performance comparison to cold-start V4.
- Cold start justification: legacy checkpoints are architecture-incompatible. The new Windows study is separately cold-started and must not be compared as a fair 40k-versus-80k or legacy performance contest.
- Separate Windows B protocol: static off/on, seeds 1511/1522/1533, exactly 40k, fixed coefficient 0.5 and temperature 0.1, identical source/configuration except the B flag; all six histories are COMPLETE and finite. Final 40k checkpoints and the original four-item rule remain fixed.
- The reserved 12-map test namespace is unevaluated. Freeze confirmation/inference plans before any sealed evaluation; development tuning is not generalization evidence.
- Official [WoWMoM CFP](https://wowmom2027.dii.unipi.it/call-for-papers.html): regular ≤10 pages including references, ≥10 pt, double blind, in person. Deadline **2026-12-01 23:59 AoE = 2026-12-02 19:59 Asia/Shanghai**.

---

## VI. Results

### VI-A Standard-C learnability floor
- Seeds 911/922/933 @50k: **11/12, 11/12, 10/12** PASS.

### VI-B Zero ablation (capacity control)
- Finals: **1/12, 0/12, 1/12**; bests 1/12@56k, 2/12@60k, 5/12@48k.
- All-UAV eventual-entry counts are **1/12, 1/12, 2/12**. Saturated any-UAV entry does not prove all agents approached or that timing alone failed.

### VI-C Static ablation
- Finals: **11/12 PASS, 10/12 PASS, 5/12 FAIL**. Arm **FAIL 2/3**; seed 1233 drops from 9/12 at 70k to 5/12 at 80k. All-UAV eventual-entry counts equal these success counts.

### VI-D Full probes
- Finals: **3/12 FAIL, 4/12 FAIL, 8/12 PASS**. Arm **FAIL 1/3**; seed 1233 has mid-run best 9/12 at 72k. All-UAV eventual-entry counts are **5/12, 4/12, 8/12**.

### VI-E Failure analysis
- Of **65** final failures, **61** have only some UAVs ever entering their balls; **four** have all UAVs entering at some times without joint success. There are **zero** collisions. Historical `timeout_after_ball` covers both classes because its entry test is any-UAV.
- Source: `results/masac_v4_windows_audit_20261105/analysis.json` derived from saved final evaluation records. All-entry alone does not distinguish speed from timing; inspect per-UAV trajectories.
- B targets worst-agent approach/capture as a hypothesis. Report the completed Windows 40k study separately in VI-G/H; do not merge it with these 65 historical Mac failures.
- New offline diagnosis separates \(A=\forall i\exists t\,b_i(t)\), \(C=\forall i\exists t\,c_i(t)\), and \(J=\exists t\forall i\,c_i(t)\), where \(b\) is distance-only and \(c\) adds the speed threshold. Then \(J\Rightarrow C\Rightarrow A\). For collision-free failures report \(\neg A\), \(A\land\neg C\), and \(C\land\neg J\); retain collision flags separately. Historical [interim trajectory diagnosis](SYNC_CAPTURE_INTERIM_DIAGNOSIS_20261105.md) remains post-hoc context; VI-H uses all three verified final40k trajectory summaries.

### VI-F Final within-suite ablation table

| Arm | 1211 | 1222 | 1233 | Gate |
|-----|------|------|------|------|
| zero | 1/12 | 0/12 | 1/12 | FAIL 0/3 |
| static | 11/12 | 10/12 | 5/12 | FAIL 2/3 |
| full | 3/12 | 4/12 | 8/12 | FAIL 1/3 |

Descriptive totals: zero 2/36, static 26/36, full 15/36. Mean final ordering is static > full > zero, but static/full reverses on seed 1233. A pooled total cannot replace the three-seed gate or establish significance. Standard-C belongs in its separate floor subsection with all warm-start confounds.

---

### VI-G Separate Windows 40k control/B endpoint — COMPLETE

| Arm / contrast | 1511 | 1522 | 1533 | Absolute 8/12 seed gate |
|---|---|---|---|---|
| static control | 2/12 | 1/12 | 4/12 | FAIL 0/3 |
| static+B | 6/12 | 0/12 | 5/12 | FAIL 0/3 |
| Paired success-count difference | +4 | −1 | +1 | Not a task-gate replacement |

Mean paired difference is **1/9 = 11.111 percentage points**; descriptive totals are **7/36** for control and **11/36** for B. All six histories completed with finite diagnostics and unchanged source/configuration/decision contracts. The original rule is met (mean at least 1/12, at least two positive pairs, minimum pair at least −1/12, six finite unmodified histories). The negative 1522 pair is exactly at the allowed lower bound.

**ADVANCEMENT_RULE_MET supports planning confirmation only.** Both absolute gates fail; the result does not establish significance, unseen-map generalization, a dynamic-prediction benefit, a new theorem or a confirmed B advantage. Do not compare these new cold-start 40k CUDA results fairly against the different Mac 80k cohort or warm-started legacy Standard-C floor. Source: [canonical final analysis](../results/masac_v4_sync_40k_analysis_20261105/analysis.md).

### VI-H Final 40k settling-event partition

Use the event definitions in VI-E, naming all-individual strict capture **Cevt** to avoid confusion with the control arm.

| Seed | Control (A, Cevt, J), out of 12 | B (A, Cevt, J), out of 12 |
|---|---|---|
| 1511 | (5, 5, 2) | (8, 8, 6) |
| 1522 | (1, 1, 1) | (2, 2, 0) |
| 1533 | (5, 5, 4) | (8, 7, 5) |

Of 72 final rollouts, **18 succeed and 54 fail**, with **zero collisions**. The collision-free failure partition is **43 not-A**, **one A but not-Cevt**, and **10 Cevt but not-J**. All-individual capture without joint capture identifies nonoverlap of attained strict-capture windows on the sampled grid, not its causal origin. These counts must not be merged with the 80k Mac diagnosis.

The three verified final trajectory summaries are under results/masac_v4_sync_40k_progress_20261105/trajectory_40000_seed1511, seed1522 and seed1533. Keep every seed, including B/1522=0/12; do not replace final results by a best intermediate checkpoint. Frozen test maps remain unevaluated.

## VII. Discussion

- What static preview might expose; observed results do not certify feasibility or prove encoder-use causality.
- CV mismatch, constant-fill/saturation, feature scale and optimization are candidate explanations for full's weaker historical mean, each requiring an identifiable test. Standard affine constant-offset/bias equivalence concerns representational function classes, not SAC/Adam/clipping trajectories; see [constant-fill identity](MATH_CONSTANT_FILL_IDENTITY_20261105.md).
- New [late-replay audit](../results/masac_v4_sync_40k_late_content_audit_20261105/audit.md), bounded to 6,144 joint observations from the six completed **40k** histories: full=static_fill1 in **349,849/350,208 probe vectors (99.89748949%)**. The 359 exceptions are peer-column differences, all control/1522; the other nine columns are full/static bitwise equal, and moving outputs are full=1/static=0 throughout. All 18,432 agent observations have two valid peers, so peer saturation mainly means valid far peers plus the 40 m padding sentinel, not missing peers. Five of six histories have complete identity in this audited sample. Do not extend this to all replay, Mac 80k trajectories, policy/optimizer equivalence or an identified performance cause.
- Relation to Xie/OPEN prediction: we ablate **ego maneuver bank**, they predict external targets differently.
- Limitations: three training seeds, reused development maps, sampled CV assumptions, no safety guarantee, observation aliasing, timeout bootstrapping, no sim-to-real evidence, pool-only encoder and no radio.
- WoWMoM angle: reproducible mobile multi-agent learning artifact.

---

## VIII. Conclusion

- Restate measured ablations.
- Windows audit, CPU/CUDA smokes, the two-seed 2k pilot and the three-seed/two-arm 40k study are COMPLETE. The original advancement rule is met for planning confirmation only; absolute gates fail in both 40k arms. Keep the negative pair and fixed endpoints. No automatic extension, new training or sealed-map evaluation follows; Attention, L1 and a historical complete-suite rerun remain outside authorization.

---

## Appendix plan (repo artifact)

- Config JSONs + auth protocols.
- Gate JSON/MD for each arm.
- MATH_ARGUMENT equation list.
- Training scripts under `scripts/train_masac_v4_maneuver.py`.

---

## Writing checklist

- [x] Replace all unfinished suite entries with the three final seeds.
- [x] Choose the mixed/negative story and separate the warm-started Standard-C floor.
- [x] Match equations to source, including temperature sign, zero control, Markov assumptions, discount and timeout conventions.
- [x] Separate any-agent entry, all-agent eventual entry and joint capture.
- [x] Reserve the sealed map namespace before the40k launch; no policy has been evaluated on it.
- [ ] Freeze the confirmation/inference plan before any sealed-map evaluation.
- [ ] Draw Fig.1 system diagram; Fig.2 probe geometry; Fig.3 curves; Fig.4 bars.
- [ ] Anonymize GitHub URL for double-blind if submitting WoWMoM.
- [ ] Sync EN abstract with CN one-pager for collaborators (optional).
