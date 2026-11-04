# PAPER_OUTLINE_DRAFT — Predictive Maneuverability-Aware MASAC

**2026-11-05 interpretation boundary:** L0's twelve archived development maps
contain no external moving objects. The existing suite therefore does not
identify a moving-obstacle prediction effect. New early static replay also
shows largely neutral/saturated added channels; do not generalize that sample
to historical final trajectories. The2k B pilot has finite updates but0/12
final successes in all four seed/arm runs. The registered40k static B-off/on
comparison tests fixed shaping only. Include these limitations in the abstract,
ablation interpretation and discussion; see `L0_CONTENT_IDENTIFIABILITY_AUDIT_20261105.md`.

**Draft date:** 2026-11-04 15:20 Asia/Shanghai  
**Target shape:** IEEE two-column (WoWMoM regular or WIP)  
**Updated:** 2026-11-05 Asia/Shanghai. All suite numbers below are final TomMac-mini development measurements. Windows B validation is a separate experiment. No predictive superiority, safety, unseen-map generalization or established novelty claim.

Working title (EN):  
**Predictive Maneuverability Probes for Multi-UAV Soft Actor-Critic: Ablating Static Geometry vs Dynamic Clearance Features under Joint Capture**

Alternate (shorter):  
**When Do Predictive Maneuverability Features Help Multi-UAV MASAC?**

---

## Abstract (evidence-constrained prose draft)

Joint capture by multiple UAVs requires every agent to approach its independent goal and satisfy a position-and-speed predicate at the same time. We evaluate an observation-only preview representation for multi-agent soft actor–critic: 18 acceleration directions and a zero command produce two-second sampled clearance features, which a probe MLP pools and fuses with an object-set encoder. A matched-capacity ablation compares static-geometry preview, additional constant-velocity moving-object/peer channels, and constant zero probe inputs. Under an identical 80k-step cold-start protocol, three training seeds and 12 fixed L0 development maps, final joint successes are 11/12, 10/12, 5/12 for static; 3/12, 4/12, 8/12 for full; and 1/12, 0/12, 1/12 for zero. No arm meets the all-seed gate. Among 65 failures, 61 involve only some agents ever entering their goal balls and four involve all agents entering at some times without joint capture; none collide. The results favor static-geometry preview on average but do not establish a benefit from the added CV channels. They motivate a separately tested worst-agent approach/capture shaping hypothesis.

Do not insert B effectiveness into the abstract until matched experiments supply evidence.

---

## I. Introduction

1. Multi-UAV independent-goal navigation needs all-agent approach, low speed and cooperative timing.
2. Progress and any-agent entry do not establish an all-agent capture capability.
3. Question: what do static-geometry preview and added CV peer/moving channels contribute under the same learner, reward, capacity and budget? Existing preview/risk/agile MARL is substantive prior art; do not claim an unsearched absence.
4. Proposal: observation-only probes + RiskEncoder; ablations `full` / `static` / `zero`.
5. C1–C3 (see `PAPER_PLAN_WOWMOM2027.md`): explicit representation, controlled ablation and auditable failure taxonomy. Distinctiveness remains under review.
6. State the final mixed/negative result, including static 1233's late drop and full's sole pass on that seed.

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

### IV-E B hypothesis (new experiment; effectiveness unestablished)

Follow `docs/design/MASAC_V4_SYNC_CAPTURE_20261105.md` for the fixed potential coefficient and terminal/bootstrap convention. B is **fixed potential shaping**, with no scheduled reset/reward progression and no explicit arrival-time-gap term. Distinguish pathwise algebra, exact soft-objective policy ordering and finite-sample learning; cite established entropy-regularized PBRS rather than claiming a new invariance theorem. Target worst-agent approach/capture; the failure counts do not identify a pure synchronization cause. Evaluation remains 4 m / 2 m·s⁻¹, latch off.

---

## V. Experimental Protocol

- Historical hardware: Apple Mac mini, CPU suite. Windows audit/device timings are separate in `docs/HARDWARE_AUDIT_WIN.md`. CPU remains the historical default; the new explicit CUDA protocols have measured throughput evidence (8.75x microbenchmark, 6.58x complete smoke) and do not retroactively change the Mac protocol.
- Budget: 80 000 steps/seed; warmup 1000; eval every 2000.
- Seeds: 1211 / 1222 / 1233 paired across arms.
- Gate: ≥0.60 = **at least 8/12** per seed; an arm passes only if all three seeds pass.
- Within-suite full/static/zero match architecture, capture/reward, paired seed ids and 80k budget. Three seeds on repeatedly observed development maps limit inference; 36 rollouts are not 36 independent training trials.
- Standard-C is a separate floor: seeds 911/922/933, 50k L0, legacy encoder; each warm-started from the same empty-floor seed-811 40k checkpoint (weights+optimizers; replay/RNG reset). These confounds preclude fair performance comparison to cold-start V4.
- Cold start justification: legacy checkpoints are architecture-incompatible. Reserve a sealed unseen-map test set before confirmation; development tuning is not generalization evidence.
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
- B targets worst-agent approach/capture as a hypothesis. Report new small-validation results separately.
- New offline diagnosis separates \(A=\forall i\exists t\,b_i(t)\), \(C=\forall i\exists t\,c_i(t)\), and \(J=\exists t\forall i\,c_i(t)\), where \(b\) is distance-only and \(c\) adds the speed threshold. Then \(J\Rightarrow C\Rightarrow A\). For collision-free failures report \(\neg A\), \(A\land\neg C\), and \(C\land\neg J\); retain collision flags separately. See [interim trajectory diagnosis](SYNC_CAPTURE_INTERIM_DIAGNOSIS_20261105.md) for post-hoc seed1511@20k/22k examples, not final efficacy evidence.

### VI-F Final within-suite ablation table

| Arm | 1211 | 1222 | 1233 | Gate |
|-----|------|------|------|------|
| zero | 1/12 | 0/12 | 1/12 | FAIL 0/3 |
| static | 11/12 | 10/12 | 5/12 | FAIL 2/3 |
| full | 3/12 | 4/12 | 8/12 | FAIL 1/3 |

Descriptive totals: zero 2/36, static 26/36, full 15/36. Mean final ordering is static > full > zero, but static/full reverses on seed 1233. A pooled total cannot replace the three-seed gate or establish significance. Standard-C belongs in its separate floor subsection with all warm-start confounds.

---

## VII. Discussion

- What static preview might expose; observed results do not certify feasibility or prove encoder-use causality.
- CV mismatch, feature scale and optimization are candidate explanations for full's weaker mean, each requiring a controlled test.
- Relation to Xie/OPEN prediction: we ablate **ego maneuver bank**, they predict external targets differently.
- Limitations: three training seeds, reused development maps, sampled CV assumptions, no safety guarantee, observation aliasing, timeout bootstrapping, no sim-to-real evidence, pool-only encoder and no radio.
- WoWMoM angle: reproducible mobile multi-agent learning artifact.

---

## VIII. Conclusion

- Restate measured ablations.
- Windows audit, CPU/CUDA execution smokes and the two-seed2k paired B pilot are complete. The preregistered three-seed/two-arm40k study is running; report every final40k pair after all six histories complete, without automatic extension or interim-checkpoint replacement. Large new launches require a plan summary; Attention, L1 and a historical complete-suite rerun remain unauthorized.

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
