# RELATED_WORK_FRONTIER_20261104 — Predictive Maneuverability MASAC

**Survey date:** 2026-11-04 Asia/Shanghai  
**Scope:** 2022–2026 (plus foundational SAC/CTDE). Prefer arXiv / IEEE / conference pages opened this turn.  
**Use:** WoWMoM / IEEE-style Related Work backbone. Gaps are **relative to our claim**, not absolute novelty proofs.

---

## How we differentiate (one paragraph)

Frontier multi-UAV RL mostly optimizes **formation + obstacle avoidance**, **pursuit-evasion**, or **AoI harvesting**, often with attention encoders, curriculum rewards, or CTBR actions. Agile single-UAV work emphasizes **flatness / MPC / adaptive tracking**. Our paper targets a narrower, measurable gap: under a **frozen joint-capture MASAC contract**, does an **observation-only predictive maneuverability probe set** (19 accel directions × short-horizon CV clearances/TTC/brake) improve learning versus **static-only** and **zero** probes? We do **not** claim hard safety certificates, flatness planning, or radio/AoI scheduling.

---

## Frontier table (curated)

| # | Citation | Gap vs our claim | URL |
|---|----------|------------------|-----|
| 1 | Xie, Yu, Zang, Gao, Tang, Huang, Chen, Xu, Wu, Wang. *Multi-UAV Formation Control with Static and Dynamic Obstacle Avoidance via Reinforcement Learning.* arXiv:2410.18495, 2024 (rev. 2025). | Formation + static/dynamic obstacles with reward search, curriculum, attention encoder, CTBR, sim-to-real. **Not** joint independent-goal capture; **no** fixed accel-probe RiskEncoder ablation (`full`/`static`/`zero`). | https://arxiv.org/abs/2410.18495 |
| 2 | Chen, Yu, Li, Tang, Yang, Xu, Yang, Wang. *Multi-UAV Pursuit-Evasion with Online Planning in Unknown Environments by Deep Reinforcement Learning* (OPEN). arXiv:2409.15866, 2024/2025. | Evader **prediction** + adaptive env generator + CTBR pursuit-evasion; strong dynamics. Prediction is **target-centric**, not ego maneuverability probe bank for cooperative capture on static maps. | https://arxiv.org/abs/2409.15866 |
| 3 | Dual-Transformer PPO for multi-UAV navigation in unseen environments (DTPPO). arXiv:2410.15205, 2024; Drones 2024. | Spatial/temporal transformers for navigation transfer. Architecture ≠ our RiskEncoder; no predictive clearance/TTC probe features or capture gate. | https://doi.org/10.48550/arxiv.2410.15205 |
| 4 | Huang, Yang, Krupani, Şenbaşlar, Batra, Sukhatme. *Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning.* arXiv:2309.13285, 2023. | Attention over neighbors/obstacles; real swarm CA. Focus collision-free navigation, not joint goal capture; no maneuverability probe ablations. | https://arxiv.org/abs/2309.13285 |
| 5 | Yan, Wang, Zhou, Xiang, Wang, Shen. *MARL With Spatial–Temporal Attention for Flocking With Collision Avoidance of a Scalable Fixed-Wing UAV Fleet.* IEEE T-ITS, 2025. | STAAC population-invariant attention for flocking + intruders. Different airframe/task; no CV probe RiskEncoder. | https://doi.org/10.1109/TITS.2024.3505929 |
| 6 | AttentionSwarm: RL + Attention CBF for Crazyflie swarms. arXiv:2503.07376, 2025. | Attention-weighted CBF safety for racing/landing. **Hard safety filter** vs our soft representation probes; different metrics. | https://arxiv.org/html/2503.07376 |
| 7 | Romero, Aljalbout, Song, Scaramuzza. *Actor-Critic Model Predictive Control* for agile flight. arXiv:2306.09852. | Differentiable MPC inside actor-critic; single-agent agility. Not multi-agent capture; not set-of-probes observation design. | https://arxiv.org/abs/2306.09852 |
| 8 | DATT: Deep Adaptive Trajectory Tracking for Quadrotor Control. CoRL 2023 (Oral). arXiv:2310.09053. | RL adaptive tracking under disturbances vs flatness/MPC. Single-UAV tracking, not cooperative MASAC capture. | https://arxiv.org/abs/2310.09053 |
| 9 | Zhao et al. *Towards Robust Multi-UAV Collaboration: MARL with Noise-Resilient Communication and Attention.* arXiv:2503.02913, 2025. | COMA + attention under noisy comms for information-collection paths. Comm-robust sensing ≠ maneuverability probes for joint capture. | https://arxiv.org/html/2503.02913 |
| 10 | Su, Pu, Yu. *A collaborative siege method of multiple unmanned vehicles based on RL* (attention-SAC capture). Intell. Robot. 2024. | Attention-SAC for **UGV** encirclement with individual+cooperative rewards. Ground kinematics; no 3D accel probes / TTC bank. | https://doi.org/10.20517/ir.2024.03 |
| 11 | *Multi-UAV roundup algorithm based on hierarchical MASAC with incremental learning.* AAAC 2025. | Hierarchical MASAC + trajectory prediction for dynamic-target roundup. Closest MASAC+prediction neighbor; still pursuit-roundup, not our frozen L0 capture ablation of probe **content**. | https://doi.org/10.1109/aaac66612.2025.11427681 |
| 12 | Haarnoja et al. *Soft Actor-Critic.* ICML 2018 / arXiv:1801.01290. | Foundational off-policy max-ent continuous control. Baseline algorithm; not a contribution. | https://arxiv.org/abs/1801.01290 |
| 13 | Lowe et al. *MADDPG* (CTDE). NeurIPS 2017 / arXiv:1706.02275. | Centralized critic / decentralized actors precedent. We inherit CTDE-style team Q; not novelty. | https://arxiv.org/abs/1706.02275 |
| 14 | Iqbal & Sha. *Actor-Attention-Critic.* ICML 2019 / arXiv:1810.02912. | Attention in centralized critic. Explains why Attention is a **future** A/B, not this suite’s claim. | https://arxiv.org/abs/1810.02912 |
| 15 | Emami et al. *AoI Minimization using Multi-agent UAVs…* arXiv:2405.00056, 2024. | Mean-field AoI harvesting. **Adjacent only** — different objective; cite if WoWMoM reviewers expect freshness angle, else one sentence. | https://arxiv.org/abs/2405.00056 |
| 16 | Bai et al. *AoI Minimization in UAV-Enabled ISAC.* arXiv:2507.14299, 2025. | Single-UAV AoI + sensing/comms. Not swarm formation/capture. | https://arxiv.org/abs/2507.14299 |
| 17 | *Variations in Multi-Agent Actor-Critic for UAV Swarm Networks* (survey-style). arXiv:2410.06627, 2024. | Broad MAAC taxonomy for UAV networks. Use for positioning continuous MARL; not a method baseline. | https://arxiv.org/abs/2410.06627 |
| 18 | Anticipatory Risk-Guided RL for Safe Flight Through Dynamic Clutter (CPA risk maps). arXiv HTML listing 2607.23565. | Privileged future CPA risk distillation to vision policy — **closest “predictive risk” cousin**. Single-UAV clutter; asymmetric privileged training; not multi-agent joint capture or fixed accel probe bank. | https://arxiv.org/html/2607.23565 |

---

## Thematic clusters (for Related Work subsections)

### A. Multi-UAV cooperative navigation / formation under dynamics
Primary: Xie et al. 2024/25; DTPPO 2024; Huang swarm CA 2023; Yan STAAC T-ITS 2025.  
**Our angle:** Independent goals + joint capture predicate + frozen Path-C floor; probes as representation, not formation Laplacian rewards.

### B. Maneuverability / agility / flatness / MPC+RL
Primary: Actor-Critic MPC; DATT; fixed-wing flatness planners (cite lightly).  
**Our angle:** Discrete ego accel probes approximating short-horizon maneuver margins — **not** flat outputs or online MPC.

### C. Predictive / risk-aware collision features in RL
Primary: OPEN evader prediction; CPA anticipatory risk; AttentionSwarm CBF.  
**Our angle:** Explicit, ablatable probe tensor (static vs moving/peer vs zero) under matched capacity.

### D. Set encoders / attention over teammates & obstacles
Primary: Xie attention encoder; Huang multi-head attention; STAAC; AAC 2019.  
**Our angle:** This suite freezes **pool** SetEncoder + RiskEncoder; Attention deferred to PENDING.

### E. Continuous MARL / MASAC for UAV capture
Primary: Haarnoja SAC; hierarchical MASAC roundup 2025; attention-SAC UGV siege 2024.  
**Our angle:** Shared Soft Actor-Critic team objective with Path-C reshape; contribution is probe representation, not inventing MASAC.

### F. AoI / formation+AoI (adjacent only)
Emami 2024; Bai 2025.  
**Our angle:** Out of scope for main claims; optional one-paragraph “wireless freshness is orthogonal.”

---

## Citation hygiene

- Prefer arXiv abs links + year; add venue when verified (do not invent conference DOI).
- Mark unread full PDFs carefully in camera-ready notes.
- Reuse deeper notes in `docs/design/MASAC_V4_LITERATURE_REVIEW.md` and `research/PAPER_MATRIX.md`.
- Update this file when suite results change the “gap” sentence for full vs static.
