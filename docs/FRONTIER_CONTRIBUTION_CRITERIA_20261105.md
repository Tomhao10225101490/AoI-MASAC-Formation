# Contribution criteria against inspected 2022–2026 primary sources

**Audit date:** 2026-11-05, Asia/Shanghai. **Status:** research criteria and proposed falsification tests, not a novelty clearance, comparative-performance result, or launch authorization. The new near-objective study was registered before execution at **b4b3a1e**. Its frozen scientific sources/configurations, six histories and final endpoint are not modified by this document. No training, policy rollout, or sealed-map access was performed for this review.

The present evidence does **not** establish that our method is better than the inspected papers. A useful paper needs a specific advantage under a common task and resource contract; a list of familiar components is insufficient. Independent goals, decentralized policies, variable-sized entity encoders, finite-horizon capability representations, predictive safety and low-speed goal arrival all have close precedents. The current static-probe near-objective comparison is a foundational objective test, not a test of predictive advantage.

## 1. What “better” must mean

An admissible advantage must name the mathematical quantity, comparator, population and cost. Examples are a higher final joint-capture probability at equal environment/update budgets, or lower measured inference cost at a prespecified acceptable capture/collision tradeoff. A higher shaped return, fewer stall samples, a runtime PASS, a faster GPU than our CPU, or a different paper's weaker-looking percentage does not establish that advantage.

For any future comparison, freeze all of the following before observing its results:

- The physical dynamics and limiters; sensing/observation information; goal and collision predicates; true-terminal and time-limit handling; train/development/confirmation populations.
- Environment transitions, gradient updates, warmup, replay exposure, architecture capacity and tuning budget. Paired episode map indices do not imply identical state distributions or map exposure at a fixed step.
- A final endpoint and seed-level analysis retaining regressions. Reused maps/frames are not independent training replicates. Existing engineering planning rules are not significance tests.
- Complete success, collision and failure outcomes. Report successful-trajectory efficiency as success-conditional, alongside failure frequency. Report latency distributions and hardware separately from environment-step efficiency.

An external method can only support a superiority claim after its implementation and information/action interface are made comparable and its adaptation is documented. Rotor-thrust control, a joint safety QP and a bounded-acceleration observation augmentation are different systems. If a credible matched adaptation is unavailable, restrict the claim to the tested internal contrast and make qualitative prior-art comparisons. Do not manufacture an easy baseline by removing information or safety components essential to the published method.

## 2. Direct precedents checked, with reading limits

These are the closest inspected precedents, not a ranking of all published methods. Version dates precede this audit; first-preprint, conference and volume years are kept distinct.

1. **Batra et al., CoRL 2021 / PMLR volume 164, 2022**, *Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning*. [Official record](https://proceedings.mlr.press/v164/batra22a.html), [full proceedings paper](https://proceedings.mlr.press/v164/batra22a/batra22a.pdf). **Section 3.5** explicitly implements permutation-invariant Deep Sets with mean aggregation and an attention alternative; **Sections 4.2–4.4** study encoders, scaling and dynamic obstacles. It already connects local observations, direct thrust control and physical swarm deployment. Merely replacing a flat vector with pooled entities is not a new capability.

2. **Huang et al., ICRA 2024 / arXiv v2, 2024-05-06**, *Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning*. [Version record](https://arxiv.org/abs/2309.13285v2), [full v2](https://arxiv.org/html/2309.13285v2). **Sections III-A/B** include each robot's independent, uncorrelated goal and decentralized local-observation-to-thrust control. **Sections III-C/D** combine entity encoders/attention, SDF observations and collision-episode replay; **IV-C/V** cover scaling and hardware. Its robot-level success and distance metrics are not our simultaneous three-agent speed-capture rate. This is direct prior art against “first decentralized independent-goal multi-UAV RL.”

3. **Vinod et al., ICRA 2022**, *Safe multi-agent motion planning via filtered reinforcement learning*. [Author institution record and paper](https://www.merl.com/publications/TR2022-053). Its target-regulating RL is followed by a convex optimization safety filter. The inspected institution abstract supports that architectural distinction; this round did not independently re-prove its constraint assumptions. An observation feature has no automatic equivalent of execution-time filtering.

4. **Zhang et al., IEEE T-RO 2025 / arXiv v4, 2025-02-07**, *GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multi-Agent Control*. [Version/journal record](https://arxiv.org/abs/2401.14554v4), [full v4](https://arxiv.org/html/2401.14554v4). **Sections III and IV-A, Equation 18** connect distributed safety structure and graph attention with variable-sized neighborhoods; **VI-A and Appendix D-B** discuss scaling, reach/safety tradeoffs and input constraints. Its safety statements require the specified barrier conditions; learned approximations do not become unconditional certificates by citation. Its own discussion identifies deadlock/liveness limits.

5. **Zhang, So, Black and Fan, ICLR 2025 / arXiv v3, 2025-03-13**, *Discrete GCBF Proximal Policy Optimization for Multi-agent Safe Optimal Control*. [Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b0916c29bc32d57e2c86a63e8f2df9f2-Abstract-Conference.html), [full v3](https://arxiv.org/html/2502.03640v3). **Sections 3.1, 4.1–4.5** address local observations, unknown discrete-time dynamics, input constraints and changing neighborhoods. **Appendix C.2.1, Target/Bicycle, Equations 101–105** explicitly include pre-assigned goals and acceleration-controlled double integrators. Its cumulative cost/safety rates differ from our terminal event. Its theorem conditions are not guarantees that every trained neural policy satisfies them.

6. **Chen et al., IJCAI 2026, pages 54–62**, *SCOPE: Safety-Constrained Online Preview Enforcement for Efficient Encirclement in Multi-UAV Pursuit-Evasion*. [Official proceedings, DOI and method text](https://ijcai.org/proceedings/2026/7). **Section 4/Figure 2/Algorithm 1** joins a learned short-horizon dynamics preview, preview-fused actor/critic, joint action correction, feasibility look-ahead and a safety loss. Thus future preview plus cooperative capture/encirclement is already established. Several equations are not decoded in the HTML; no uninspected formula or unconditional proof is imported here. Its joint QP and pursuit-evasion task differ from our runtime interface and independent fixed goals.

7. **Kim et al., 2026-10-30 preprint, arXiv v1**, *Capability-Conditioned Graph Control Barrier Functions for Zero-Shot Safe Navigation of Heterogeneous Mobile Robot Teams*. [Dated record](https://arxiv.org/abs/2610.41214v1), [full v1](https://arxiv.org/html/2610.41214v1). **Section IV-B, Equations 3–8** constructs a finite-horizon reachable-velocity approximation and radial/Fourier capability encoding; **IV-C, Theorem 1** states conditional continuous-time invariance, and **IV-D** gives barrier/policy training. Capability-dependent directional representations and unseen-model transfer are already explicit. This is a preprint, with no peer-reviewed venue verified. Our finite probe bank is not this reachable-set construction.

8. **Deng et al., Chinese Journal of Aeronautics 39(7), 2026**, *Learning safe and decentralized flight for aerial swarms in dynamic complex environments*, DOI **10.1016/j.cja.2026.104113**. [Publisher record](https://www.sciopen.com/article/10.1016/j.cja.2026.104113), [DOI](https://doi.org/10.1016/j.cja.2026.104113). Publisher metadata records online publication **2026-02-10**, July issue. The publisher abstract/method overview describes local LiDAR RL, dynamic tracking, a safety-assured NMPC execution layer and physical swarm flights. Direct DOI fetching was blocked during this review; the proof and full controller implementation were not independently checked. This establishes a recent system precedent, not a safety theorem for our simulator.

9. **Vasan et al., 2024 / arXiv v2, 2024-07-08**, *Revisiting Sparse Rewards for Goal-Reaching Reinforcement Learning*. [Full v2](https://arxiv.org/html/2407.00324v2). **Section 3/Table 1** requires near-zero-velocity arrival in its minimum-time formulation; **Section 4/Figure 5** measures time to goal and time on goal; **Section 5** studies complete-interaction goal-hit frequency. Arrival versus retention is established. Its reward, termination and reset comparison does not isolate our near-term coefficient or imply that our replacement must work.

10. **Adamczyk et al., AAAI 2023 / arXiv v2, 2022-12-30**, *Utilizing Prior Solutions for Reward Shaping and Composition in Entropy-Regularized Reinforcement Learning*. [Full v2](https://arxiv.org/html/2212.01174v2), **Theorem 5/Equations 20–22 and Remark 7/Equations 23–24**. Fixed bounded potential differences and exact soft-value shifts are existing theory with matched dynamics, discount, regularization and boundary conditions. They supply no new neural-SAC convergence or finite-budget advantage.

11. **Shin, Ra and Yang, 2026-07-01 preprint, arXiv v1**, *From Prediction Uncertainty to Conformalized Distance Fields for Safe Motion Planning*. [Dated primary record and abstract](https://arxiv.org/abs/2607.00776v1). **Abstract-only verification in this review:** the authors describe field-level functional conformal prediction, offline functional-PCA/coefficient-envelope construction, online adaptive updates, and tightened constraints in FCP-MPC, with pedestrian and dense 3D quadrotor tests. This establishes a direct uncertainty-aware distance-field precedent. Its theorem assumptions, coverage conditions, implementation and comparative results have not been independently checked here; no peer-reviewed venue is asserted.

12. **Ma, Wu, Kantaros and Vorobeychik, 2026-02-03 preprint, arXiv v1**, *Conformal Reachability for Safe Control in Unknown Environments*. [Dated primary record and abstract](https://arxiv.org/abs/2602.03799v1). **Abstract-only verification in this review:** the authors combine conformal dynamics-uncertainty intervals, reachability verification and policy training that considers nominal reward and a verifiable planning horizon. Its probability/coverage assumptions, horizon guarantee and theorem conditions have not been independently checked here; no peer-reviewed venue is asserted. A generic combination of prediction uncertainty and reachability is not an unexplored idea.

## 3. Candidate A: objective alignment for simultaneous low-speed completion

**Mathematical object.** For each agent define

\[
B_i(t)=\mathbbm 1\{\|p_i(t)-g_i\|<4{\rm m}\},\qquad
C_i(t)=B_i(t)\mathbbm 1\{\|v_i(t)\|<2{\rm m/s}\}.
\]

The distinct events are

\[
\mathsf A=\bigcap_i\bigcup_{t\leq H}\{B_i(t)=1\},\quad
\mathsf I=\bigcap_i\bigcup_{t\leq H}\{C_i(t)=1\},\quad
\mathsf J=\bigcup_{t\leq H}\bigcap_i\{C_i(t)=1\}.
\]

Then \(\mathsf J\subseteq\mathsf I\subseteq\mathsf A\). Their empirical frequencies and censored first-entry/exit/overlap times are diagnostic objects. The learning objective \(J_{r,\alpha}\) and the benchmark probability \(\Pr_\pi(\mathsf J)\) are different quantities unless an equivalence is established.

The active intervention uses

\[
\Phi(S)=0.4\,{\rm mean}_i\,\operatorname{clip}(1-d_i(S)/15,0,1),\quad
r_0=r_{\rm PathC}-\Phi(S_{\rm next}^{\rm physical}),\quad
r_{\rm new}=r_0+\gamma\Phi(S_{\rm next}^{\rm effective})-\Phi(S).
\]

This is mean-distance shaping, not a worst-agent/speed/synchronization representation. Its standard PBRS identity is relative to \(r_0\), not Path-C. True termination absorbs the added successor; a timeout retains the boundary/bootstrap. See [the exact mathematical audit](MATH_NEAR_OBJECTIVE_REPLACEMENT_20261105.md).

**What prior work already does.** Vasan supplies low-speed reaching/retention distinctions; Huang, GCBF+ and DGPPO already combine goal approach with collision concerns. A stricter endpoint is an evaluation choice, not a new solution or proof that prior policies cannot satisfy it.

**Current evidence/gap.** Both new 2k pilot arms finish at 0/12. The registered fresh-seed 40k comparison has no final outcome asserted here. Historical B control/treatment finals [2,1,4]/12 and [6,0,5]/12 are a different cohort and intervention; reduced near-stall in one B seed did not ensure capture. Those observations motivate an objective hypothesis without identifying its cause.

**Smallest falsifiable same-budget test.** Complete the already registered original-versus-replacement six-history study without changing its formulas, seeds, budget or final endpoint. Its sum-count gain at least 3, at least two positive seed pairs and minimum pair gain at least -1 are engineering planning criteria only. Keep the separate at-least-8/12-every-seed gate. Use \(\mathsf A,\mathsf I,\mathsf J\), collisions and retention diagnostics to explain the result after the primary endpoint, not replace it. A separately registered removal-only/PBRS factorial would be needed to distinguish those two mechanisms; none is launched by this note.

**Failure/allowed claim.** Mixed or negative final success, or lower stall without higher success, does not establish an advantage. Even a positive result supports the **complete replacement under this contract/budget**, not a new PBRS method, sparse-success equivalence, universal reward pathology, stabilization or prediction utility. On its own this is an incremental learning-objective result, not sufficient evidence for a frontier method paper.

## 4. Candidate B: physically calibrated directional probes that earn their cost

**Mathematical object.** Write the deterministic observation augmentation as

\[
\mathcal R_\eta(o)\in\mathbb R^{19\times11},\qquad
\eta=(a_{\max},H_{\rm probe},\Delta t,\text{limiter/model conventions}).
\]

A directional token indexes a sampled acceleration/continuation model, not every admissible feedback control. A candidate modeled clearance is

\[
m_k(o;\eta)=\min_{\ell\in\mathcal T,\ j\in\mathcal V(o)}
  {\rm sd}\!\left(\hat p_i^{(k)}(\ell;\eta),\hat{\mathcal O}_j(\ell)\right).
\]

This equation defines the diagnostic object to align with the actual implementation before any experiment; it does not assert a new implementation here. Finite sampled clearance, constant-velocity peer assumptions and clipped derived channels are not joint viability or CBF conditions. Grid-sampled noncollision alone also gives no continuous-time certificate.

If \(\mathcal R_\eta=f_\eta(O)\) is computed from the same observation and fixed known parameters, it adds no information conditional on \(O\): \(I(Y;\mathcal R_\eta\mid O)=0\). The possible advantage is an inductive bias or finite-computation/sample-efficiency effect. If it instead supplies a previously hidden capability parameter, give that parameter to comparator policies too; otherwise information and representation effects are confounded.

**What prior work already does.** SCOPE uses model previews and enforcement; CC-HGCBF explicitly conditions on finite-response directional capability; DGPPO handles constrained discrete dynamics. We cannot claim the first capability-aware/predictive safe multi-agent representation. A possible narrower strength is an inexpensive deterministic probe bias with a carefully identified calibration benefit, without a learned dynamics model or runtime optimizer. That strength remains hypothetical.

The two 2026 abstract-verified preprints above further rule out treating generic uncertainty-aware clearance fields or conformal prediction plus reachability as a new direction by themselves. Our proposed CV-error/support diagnostics are descriptive checks, not conformal coverage, uncertainty certification or a reachability guarantee. Adding a generic uncertainty wrapper would need a separately specified method, full-theorem comparison and matched evidence; it is not authorized or justified by this review.

**Current evidence/gap.** Historical static/full/zero totals are 26/36, 15/36 and 2/36 on reused L0 maps. All 12 maps have no moving obstacles. On the audited late-training static-control/B replay, full equals static-constant-fill-1 on 349,849/350,208 tokens; only 359 peer-clearance tokens differ and the derived TTC/reserve/flag columns do not. This population is not historical full-policy data or fresh final on-policy data. It cannot establish prediction accuracy, actor use or training causality.

**Smallest falsifiable same-budget sequence, proposed only.**

1. First freeze an eligible **unsealed development** state population with valid close peers/obstacles, hashes and sampling rules. Measure raw support and CV peer-position error by horizon; do not infer good prediction from neutral normalized channels. Construct matched states with equal current distance but different closing velocity or available acceleration and verify the probe changes in the intended direction.
2. Before any large comparison, freeze three capacity-matched arms: static, static-constant-fill-1 and physically matched full. Use the same input dimension, network, reward, fresh paired seeds and environment/update budgets. This asks whether temporal content helps beyond constant offsets. Required close-interaction exposure must be justified separately; the current L0 comparison cannot identify moving-obstacle prediction.
3. Only if matched full has a reproducible advantage, preregister **one** calibration falsifier: wrong probe horizon/acceleration limit with the real environment unchanged, or a direction-token correspondence permutation preserving tensor size. Do not alter actual physics to hide a representation mismatch. Preserve direction labels as specified by the falsifier; jointly permuting a set and its labels is a symmetry test, not a directional-content destruction test. A cheaper frozen-policy counterfactual can diagnose sensitivity, but cannot replace new training for a learning-effect claim.

**Failure/allowed claim.** If full matches constant-fill, if calibration controls work equally well, or if the useful channels are rarely active, retract the calibrated-prediction explanation. If only offsets help, call it an offset/optimization effect. Any supported advantage remains task/model/budget specific; it does not certify safety, prove prediction superiority over learned models, or explain the old full arm retrospectively.

## 5. Candidate C: representation identification that separates absence from saturation

**Mathematical object.** Preserve object validity \(v_j\), raw valid-object minimum \(d_{\min}^{\rm valid}\), normalized/clipped clearance \(\bar d\), and the derived TTC/braking/reserve/flag tuple separately. For a clipping scale \(C\), valid-far \(d_{\min}^{\rm valid}\geq C\) and no-valid-object can share \(\bar d=1\); that aliasing must be named, not interpreted as the same physical scene.

A masked minimum with invalid entries \(+\infty\), followed by clipping, can agree exactly with a finite sentinel plus clipping when all valid distances exceed the same cap. Agreement is not automatically a padding bug. Conversely, claims about object absence cannot be made from the clipped value alone. Test validity semantics against an explicit provider specification.

**What prior work already does.** Batra already uses set aggregation; Huang already compares SDF and nearest-entity representations; GCBF+ handles variable-sized graph neighborhoods without fixed padding. The novel part cannot be masking or saturation in general. A publishable empirical result would require a consequential, reproducible identification failure and a demonstrated remedy under the same task, not merely a code-level observation.

**Current evidence/gap.** The existing audit has two valid peers in every sampled agent observation; neutral peer values do not mean absent peers. It documents sampled content, including support/cap provenance and exact provider agreement, but not that repairing a mask or adding validity channels improves learned behavior. The current feature encoding may intentionally discard far-object distinctions.

**Smallest falsifiable same-budget test.** Without training first, verify permutation, invalid-slot insertion, valid-near, valid-far and no-object witnesses against the specification; report all 11 channel differences. A later capacity-matched training contrast may replace/encode validity semantics only after a separately registered method and informative exposure are specified. Hold the actual world, provider arithmetic and unrelated feature columns fixed; a change that simultaneously modifies normalization, mask semantics and channel dimension would not isolate validity.

**Failure/allowed claim.** If alternative valid implementations agree after the intended cap, there is no demonstrated mask error. If a remedy changes tensors but not final success, there is no performance benefit. This is supporting experimental methodology and a possible useful negative result, not evidence that published set/graph policies are flawed or that our frozen static benchmark tests dynamic prediction.

## 6. Candidate D: pooled execution with a verified quality–cost envelope

**Mathematical object.** For locally available entity features \(z_{ij}\), a shared pooled actor can be expressed schematically as

\[
u_i=\rho_\theta\!\left(z_i,\operatorname{Pool}_{j:v_{ij}=1}\phi_\theta(z_{ij}),
                     \operatorname{Pool}_{b:w_{ib}=1}\psi_\theta(z_{ib}),
                     \operatorname{Enc}(\mathcal R_\eta(o_i))\right).
\]

The necessary interface properties are local-neighbor permutation invariance, joint agent-label equivariance when goals/states are relabeled consistently, and invariance to adding invalid padding. These properties alone do not imply fleet-size generalization. The empirical object is a vector of final joint capture, collision rate, latency distribution, peak execution memory and supported agent/obstacle counts, under a fixed sensor-information contract.

**What prior work already does.** Batra compares Deep Sets/attention; Huang demonstrates deployed compact attention and scaling; GCBF+ and DGPPO have scalable graph constructions. “No attention” is not an intrinsic strength. Deng's 2026 system adds local perception and physical execution evidence we do not possess.

**Current evidence/gap.** Current registered training/evaluation uses three agents. The pooled interface and measured local CPU/CUDA training throughput do not establish cross-size transfer, embedded inference efficiency, or a superior quality–latency frontier. No hardware flight result is present.

The read-only source audit also limits the sensing claim: [L0 map configuration](../uav_planning/v4_maps.py) rejects an explicitly supplied range below the world diagonal and otherwise defaults to the diagonal plus 1; the active training configurations use 400 m. The [environment visibility rule](../uav_planning/v4_environment.py) filters by distance and then applies slot caps. Thus these L0 experiments do not test severe distance-limited local sensing or realistic occlusion/LiDAR perception. Slot truncation, feature omissions and hidden episode/reward memory must still be discussed separately: the long nominal range does not prove full observability or a Markov actor input. Decentralized action computation is distinct from a demanding limited-information evaluation.

**Smallest falsifiable same-budget test, proposed only.** First run algebraic/interface witnesses without new training. If the existing architecture actually supports changed entity counts, a later registered evaluation should separate fixed density from fixed workspace while changing team size, with no extra training for a zero-shot claim. Report the harder joint event and agent-level outcomes together, since joint success naturally changes with population size. A same-budget retrained comparison is a separate experiment. Measure provider plus encoder plus actor latency on the target hardware, including preprocessing and device transfer, instead of reporting only neural forward time. Attention/GNN additions remain unauthorized; if a credible published comparator cannot be reproduced, do not claim superiority to it.

**Failure/allowed claim.** A faster but low-success policy is not an efficiency win without an accepted quality threshold. More input slots are not demonstrated generalization. If only three agents work or probe computation dominates latency, retract broad scalability/lightweight claims. Any favorable small-team result must state its tested range and dynamics.

## 7. A paper-worthy evidence chain, and the stopping rule for claims

The strongest plausible path in the frozen research direction is **Candidate A as a prerequisite, Candidate B as the potential method claim, and C/D as correctness and system evidence**. That is a proposed prioritization, not a prediction of success.

First establish that a common baseline can reliably complete the stated goal/speed contract at the available budget. Then show that matched temporal/capability content improves a meaningful final endpoint beyond static and constant-fill controls, and that a physical/directional mismatch removes the claimed benefit in the predicted way. Finally quantify the quality–cost/generalization range without changing the endpoint or information model. Use fresh confirmation seeds/maps only under a separate approved protocol; do not open existing sealed tests for this planning document.

If the registered near study is negative, record that result and diagnose within scope; do not relabel it as a prediction experiment. If it is positive but predictive controls fail, the honest outcome is an objective/representation diagnosis with limited claims. If only implementation witnesses pass, this is engineering validation, not a frontier algorithm result. A WoWMoM submission still requires a concrete mobile/networked-systems relevance argument; adding networking citations supplies no evaluated networking contribution.

There is currently no basis to claim “SOTA,” first independent-goal decentralized RL, first predictive safe capture, a new PBRS theorem, general safety or asymptotic stabilization, arbitrary swarm scalability, real-world agility, or that our method already surpasses the inspected papers. The criteria above specify what evidence would turn a candidate advantage into a defensible, reviewable statement, and what results would force us to abandon it.

## 8. Existing local evidence referenced, not rerun

A separate [joint-recourse hypothesis](design/MASAC_V4_JOINT_RECOURSE_HYPOTHESIS_20261105.md)
formalizes shared versus per-constraint recovery and an elementary strict-gap
witness. It is unregistered/unimplemented, uses established reachability ideas,
and supplies neither a new theorem nor permission to change the active study.
It distinguishes finite backup-library lower approximations from per-constraint
upper relaxations and explicitly limits sampled-disturbance safety claims.

- [Historical and 40k evidence boundaries](FRONTIER_GAP_UPDATE_20261105.md).
- [Canonical completed B 40k analysis](../results/masac_v4_sync_40k_analysis_20261105/analysis.md).
- [Late-training content audit](../results/masac_v4_sync_40k_late_content_audit_20261105/audit.md).
- [Offline near-goal actor audit](../results/masac_v4_sync_40k_near_goal_actor_audit_20261105/audit.md).
- [Closed near-objective 2k results](NEAR_OBJECTIVE_2K_RESULTS_20261105.md).
- [Near-objective mathematical boundaries](MATH_NEAR_OBJECTIVE_REPLACEMENT_20261105.md).
- [Registered near-objective 40k protocol](design/MASAC_V4_NEAR_OBJECTIVE_40K_PROTOCOL_20261105.md). Its frozen source/config manifests and actual completion records govern execution and outcomes; this criteria document cannot supersede them.
