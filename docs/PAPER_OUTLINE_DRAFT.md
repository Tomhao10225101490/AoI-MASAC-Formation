# PAPER_OUTLINE_DRAFT — Predictive Maneuverability-Aware MASAC

**Draft date:** 2026-11-04 15:20 Asia/Shanghai  
**Target shape:** IEEE two-column (WoWMoM regular or WIP)  
**Evidence policy:** Numbers below are measured on TomMac-mini suite unless marked **TBD**. Do not promote TBD to abstract claims.

Working title (EN):  
**Predictive Maneuverability Probes for Multi-UAV Soft Actor-Critic: Ablating Static Geometry vs Dynamic Clearance Features under Joint Capture**

Alternate (shorter):  
**When Do Predictive Maneuverability Features Help Multi-UAV MASAC?**

---

## Abstract (bullets → later prose)

- Problem: 3-UAV continuous control; joint capture on cluttered L0 maps is bottlenecked by sync after ball entry.
- Method: 19 fixed accel probes, 2.0 s CV preview, 11-D features → RiskEncoder (mean/max) fused with SetEncoder pool.
- Protocol: Path-C reshape; capture 4 m / 2 m·s⁻¹ latch off; cold start; 80k×3 seeds.
- Results so far:
  - Standard-C L0 floor: **11/12, 11/12, 10/12** PASS.
  - zero: **1/12, 0/12, 1/12** FAIL (capacity without content fails).
  - static: **11/12, 10/12** PASS; seed 1233 **TBD**.
  - full: **3/12, 4/12** FAIL so far; seed 1233 **TBD**.
- Takeaway (provisional): probe **content** matters; static margins help; full dynamic channels not yet beneficial under this contract.

---

## I. Introduction

1. Multi-UAV autonomous navigation needs cooperative timing, not only collision-free paths.
2. MARL (MASAC) succeeds on progress rewards but can stall on **joint** predicates.
3. Gap: existing UAV MARL adds attention / formation / pursuit prediction, but lacks a **controlled ablation of predictive maneuverability features** for capture.
4. Proposal: observation-only probes + RiskEncoder; ablations `full` / `static` / `zero`.
5. Contributions C1–C3 (see `PAPER_PLAN_WOWMOM2027.md`) — soften language until 1233 closes.
6. Preview of negative/mixed result honesty if full stays weak.

---

## II. Related Work

Use `docs/RELATED_WORK_FRONTIER_20261104.md` clusters A–F.

- II-A Multi-UAV formation & obstacle avoidance RL (Xie; Huang; STAAC; DTPPO)
- II-B Agile flight, flatness, MPC+RL (AC-MPC; DATT)
- II-C Predictive / risk-aware RL (OPEN; CPA risk; CBF-Attention)
- II-D Set / attention encoders (AAC; swarm attention) — explain pool freeze
- II-E Continuous MARL / MASAC for capture/pursuit
- II-F Brief AoI adjacency (optional)

End with explicit differentiation paragraph.

---

## III. Problem Formulation

- III-A Team MDP: \(N=3\), obs \(o_i\in\mathbb{R}^{162}\), action \(a_i\in(-1,1)^3\).
- III-B Dynamics & maps: L0 development set `830001`–`830012`.
- III-C Success: joint capture ball 4 m, speed < 2 m·s⁻¹, non-sticky.
- III-D Metrics: joint success, ever_ball, timeout_after_ball, collisions, path efficiency.

Tie to MATH E1.

---

## IV. Method

### IV-A Shared MASAC learner (not novelty)
- Twin team Q, soft Bellman, entropy temperature — MATH E2–E3.
- Path-C reward reshape — MATH E4; call it curriculum.

### IV-B Predictive maneuverability probes (candidate)
- Direction set \(U\) with \(|U|=19\); \(\Delta=0.2\) s; \(H=2.0\) s.
- CV extrapolation for moving/peers from observation only.
- Feature \(f_j\in\mathbb{R}^{11}\) — MATH E5.
- Disclaimer: not certificates — MATH E7.

### IV-C RiskEncoder + fusion
- \(\phi:11\to64\); mean/max → 128; SetEncoder pool 384; fuse 512 — MATH E6.

### IV-D Ablation modes
| Mode | Probe content |
|------|----------------|
| full | all clearance channels |
| static | mv+peer zeroed |
| zero | \(f_j=0\) (encoder present) |

---

## V. Experimental Protocol

- Hardware: Apple Mac mini (TomMac-mini), CPU training, reproducible local suite.
- Budget: 80 000 steps/seed; warmup 1000; eval every 2000.
- Seeds: 1211 / 1222 / 1233 paired across arms.
- Gate: ≥ 0.60 per seed.
- Baselines: Standard-C L0 (task-matched, not param-matched); within-suite full/static/zero.
- Cold start justification (arch incompatibility with Legacy SharedMASAC ckpts).

---

## VI. Results

### VI-A Standard-C learnability floor
- Seeds 911/922/933 @50k: **11/12, 11/12, 10/12** PASS.

### VI-B Zero ablation (capacity control)
- Finals: **1/12, 0/12, 1/12**; bests 1/12@56k, 2/12@60k, 5/12@48k.
- Pattern: ever_ball can saturate while joint success stays near zero → sync failure.

### VI-C Static ablation
- 1211: **11/12 PASS**; 1222: **10/12 PASS**; 1233: **TBD** (mid-run ~8/12 @44k as of 15:08 Asia/Shanghai).

### VI-D Full probes
- 1211: **3/12 FAIL**; 1222: **4/12 FAIL**; 1233: **TBD** (mid ~1/12 @46–48k).

### VI-E Failure analysis
- Emphasize `timeout_after_ball` vs collisions.
- Relate to PENDING sync-capture curriculum (future work, not claimed).

### VI-F Ablation summary table (fill when suite ends)

| Arm | 1211 | 1222 | 1233 | Gate |
|-----|------|------|------|------|
| Standard-C L0* | 11/12 | 11/12 | 10/12 | PASS (*different seeds/budget) |
| zero | 1/12 | 0/12 | 1/12 | FAIL |
| static | 11/12 | 10/12 | TBD | TBD |
| full | 3/12 | 4/12 | TBD | TBD |

---

## VII. Discussion

- What static probes buy (short-horizon geometric feasibility).
- Why full may hurt (noisy peer CV; scale imbalance; cold-start capacity).
- Relation to Xie/OPEN prediction: we ablate **ego maneuver bank**, they predict external targets differently.
- Limitations: CPU sim; no sim-to-real yet; pool-only encoder; no radio.
- WoWMoM angle: reproducible mobile multi-agent learning artifact.

---

## VIII. Conclusion

- Restate measured ablations.
- Future: Attention A/B; sync-capture curriculum; feature normalization; optional sim-to-real.

---

## Appendix plan (repo artifact)

- Config JSONs + auth protocols.
- Gate JSON/MD for each arm.
- MATH_ARGUMENT equation list.
- Training scripts under `scripts/train_masac_v4_maneuver.py`.

---

## Writing checklist

- [ ] Replace all TBD after full/static 1233.
- [ ] Choose story branch A/B/C (`PAPER_PLAN` §4).
- [ ] Draw Fig.1 system diagram; Fig.2 probe geometry; Fig.3 curves; Fig.4 bars.
- [ ] Anonymize GitHub URL for double-blind if submitting WoWMoM.
- [ ] Sync EN abstract with CN one-pager for collaborators (optional).
