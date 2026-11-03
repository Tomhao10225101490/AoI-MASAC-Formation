# Research progress

> **2026-11-02 — Single-UAV DC-RRT* and supervised sampling.** Implemented
> native 3-D curvature-aware RRT* reconstruction, numerical spatial CSC,
> swept-body collision and independent curve/tree validation. Six focused
> tests pass. Trained a 7,817-parameter CNN for 60 epochs on 384/64 train/val
> maps, saving datasets, epoch metrics, optimizer and best/last weights.
> Local CPU training took 104.62s; validation selected epoch 56. Frozen model
> and protocol preceded 1,560 final cases: 1200 reconstructed-layout cases
> and 360 unseen-map cases, exactly three arms. All succeeded and passed
> independent path/tree checks; final provenance audit has no errors.
> Learned sampling did not improve efficiency; some scattered-scene paths
> were shorter. No test-dependent tuning or positive efficiency claim.
> Generated actual 3-D paths, search trees, training curves, metric and paired
> effect figures in PNG/vector PDF, with deterministic case selection.
> **Complete original-experiment replication remains blocked** by missing
> exact maps/body radius and ambiguous spatial connector/rewiring details.
> Current entry: `docs/DC_RRT_SINGLE_UAV_20261102.md`; protocol:
> `experiments/dc_rrt_single_uav_20261102.md`; data/model:
> `results/dc_rrt_sampler_20261102/`; evaluation:
> `results/dc_rrt_20261102_evaluation/`. Previous research is preserved.

> **Host GCOPTER execution — protocol 2026-11-01.** User authorized immediate
> local reproduction. Built the unchanged upstream GCOPTER/MINCO core with
> Apple clang 16 arm64 and a ROS-free adapter. Pinned/materialized official
> source and Eigen 3.3.7; 323 files pass Git blob verification. Native analytic
> MINCO self-test and 5 adapter tests pass. Smoke 3/3 and fixed buildings/floating
> batch 40/40 complete on the Mac CPU (four workers), passing predeclared 1%
> engineering tolerance and conservative curve checks. All 40 nominal speed
> maxima slightly exceed 4m/s, maximum 4.020629m/s; raw nominal-limit audit retained.
> No full-paper, strict safety, learned or multi-UAV result claimed. Evidence:
> `results/gcopter_native_smoke_20261101/`, `results/gcopter_native_batch_20261101/`,
> `results/gcopter_local_setup_20261101/`; scope and commands:
> `docs/GCOPTER_LOCAL_REPRODUCTION_20261101.md`. Saved raw trajectories,
> coefficients, maps, timings, logs, 3D PNG/PDF, and planned-trajectory GIF.
> No unbounded background workload remains. Existing historical work preserved.

> **Latest correction — 2026-11-01 (Asia/Shanghai).** The user explicitly
> declined the narrow-passage/bottleneck task and asked to examine 3D path
> planning first. Current scope is `docs/THREE_D_PLANNING_SCOPE_20261101.md`:
> compare 3D route search, feasible trajectory generation, and multi-UAV
> coordination; no scene or new method frozen. Added source inspection of
> GCOPTER's global-planning entry and MINCO interfaces, SUPER's planning
> entry, and official README checks. Official publications were verified as
> T-RO 2022 (GCOPTER) and Science Robotics 2025 (SUPER), not conference papers.
> No planner compiled/run or new experiment performed. Prior bottleneck
> proposal and priority order below are superseded; records remain preserved.

> **Active direction update — 2026-11-01 (Asia/Shanghai).** The user narrowed
> new work to multi-UAV path planning and requested top-conference code review.
> Recorded the scoped problem, candidate occupancy-dependency local repair,
> literature evidence levels, baselines and design-only experiments in
> `docs/PATH_PLANNING_FOCUS_20261101.md`. Statically inspected four official
> implementations (EGO-Swarm, ICRA 2023 LSC deadlock planner, Robust MADER,
> ICLR 2025 MMD), 15 files; review and immutable source identities are in
> `research/TOP_CONFERENCE_PLANNING_CODE_REVIEW_20261101.md` and
> `research/PLANNING_SOURCE_MANIFEST_20261101.json`. Checked actual Mac code
> and project Python dependency visibility. No external planner built/run,
> no training or new planning experiment; novelty and performance unverified.
> Updated entry points with targeted additions; previous code, results and
> manuscripts remain preserved. The chronology below concerns the old project.

Updated 2026-10-31. Status: IN PROGRESS; no submission-readiness claim.

## Provenance
- Baseline: 3bffd89a5f084446729f826c06df19e8e3baa6b7.
- Branch: research/wowmom2027-upgrade.
- Existing modified files: algos/masac.py, envs/formation_env.py, tests/test_env.py, tests/test_relay.py.
- Existing untracked work: graph_policy.py, ablation_belief_no_aoi.yaml, server_protocol.yaml, plan_server_study.py and four test files. These predate this execution; preserved, not attributed as newly implemented work.
- Tracked diff backup: results/upgrade_audit/preexisting.patch. Original commit remains intact.

## Executed
- Hardware detection: M4 Mac mini; 10 CPU cores (4P+6E), 10 GPU cores, 16 GiB, macOS 15.1; available disk 121 GiB; memory_pressure reported 41% free.
- Python 3.9.6 arm64; torch 2.8.0; numpy 2.0.2; MPS available; CUDA unavailable.
- Current working-tree pytest: 83 tests passed, exit 0.
- CPU/MPS project benchmark launched, execution session 99516; output results/upgrade_audit/hardware-benchmark.json; completion must be checked.
- Same process tests original git archive in a temporary directory; output baseline-tests.txt.
- Environment lock snapshot: results/upgrade_audit/pip-freeze.txt.

## Findings and decisions
- Existing Q-centering fix is mathematically consequential: mean(Q-mean(Q)) is zero, including its gradient. Re-evaluate old residual-policy negative results before interpreting them scientifically.
- Existing graph/GRU module explicitly has no learner or environment adapter. Do not report graph MARL training as implemented.
- Server protocol is design-only; no server access or expensive server job launched.
- Benchmark uses unique output root to preserve prior results/study artifacts.

## Next
1. Inspect benchmark and original tests, persist resource recommendations and raw logs.
2. Complete code/formula/timestamp/checkpoint audit and verify inherited modifications.
3. Read primary literature, especially control-value scheduling overlap; build verified bibliography before selecting main algorithm.
4. Implement and test the chosen design, then run paired baselines and ablations with frozen evaluation splits.
5. Figures, full survey, paper, reviewer audit remain Pending.

All PHASE 2–10 deliverables remain incomplete. Hardware audit is preliminary pending benchmark review. No performance improvement is claimed.

## Completed execution update
Session 99516 finished successfully. CPU and MPS six-episode smoke training completed; mean update times 0.0223 s and 0.0564 s respectively. Original untouched baseline: 55 tests passed. No training from this session remains running. Checkpoints and metrics are in results/upgrade_audit/bench_runs. Hardware report saved; profiling limitations explicitly listed. First audit documentation commit contains only new audit files; inherited implementation changes remain uncommitted and preserved.

## Candidate-C integration checkpoint 1
- Added `CallbackLink` for geometry-dependent, traffic-independent per-slot link traces and an explicit `SlottedRadio.seed_cache` path for the documented free initial synchronization.
- Added tests for callback sampling, idle-slot advancement, and source-timestamp age after initial cache insertion.
- Result: `tests/test_slotted_radio.py` passes 14/14. This is infrastructure only; the UAV environment and learner are not yet connected.
- Next: instantiate this radio inside `UAVFormationEnv`, route timestamped source/relay packets through it, and expose delivered-cache/eligibility features to the actor.

## Candidate-C integration checkpoint 2
- `UAVFormationEnv(radio_model="slotted")` now adds one policy-controlled transmit/idle dimension for a cyclically eligible follower.
- Source and forwarded packets use immutable source timestamps, one-slot delivery latency, local caches, charged airtime, and a fixed slot budget.
- Legacy direct/relay behavior remains the default; the slotted path is opt-in.
- Actor observations now expose only local delivered-cache age/uncertainty, cyclic eligibility, and local transmit history. The privileged critic receives cache and eligibility summaries.
- Radio airtime, utilization, broadcast count, forward requests, rejections, transmit-time decodes, and accepted relay updates are carried into episode CSV/summary plumbing.
- Result: focused radio, adapter, accounting, metrics, and algorithm tests pass 39/39.
- Next: train one tiny MASAC checkpoint on the fourth communication dimension, verify replay/checkpoint/evaluation round-trip, then build the paired pilot runner and live terminal board.

## Candidate-C integration checkpoint 3
- Added `communication_only` stage-C1 mode: the follower tracks its delivered belief with zero movement residual while the actor controls only transmit/idle.
- A one-episode MASAC smoke run successfully trained, buffered, checkpointed, reloaded, and evaluated with `action_dim=4`, `obs_dim=69`, and `global_dim=626`. After pending-packet state was added, the final Candidate-C critic dimension is 630.
- The evaluation CSV contains charged airtime, budget utilization, broadcast count, forward requests, decode-at-transmit, and accepted relay updates.
- The smoke result is plumbing evidence only; two evaluation episodes are far too few for any research claim.
- Next: add budget-matched scripted communication arms and a live pilot runner with fixed train/evaluation splits.

## Candidate-C integration checkpoint 4
- Added a staged pilot runner with a live board, training/evaluation holdout seeds, paired episode records, summary JSON, report, and PNG/PDF figures.
- Budget-matched arms are learned, idle, random-50%, always-forward, local-age, and a clearly labeled privileged oracle-fresh diagnostic.
- Added pending-packet state to the privileged critic and verified communication-only action masking.
- Smoke pilot completed end to end and produced readable figures, but one episode and one training seed are explicitly non-inferential.
- Full current test suite passes (119 passed, 1 skipped).
- Next: run the first multi-seed short pilot, inspect whether the learned communication action changes usefully, then diagnose or extend only from that evidence.

## Candidate-C pilot result
- The first run exposed an evaluation-wrapper defect that zeroed the learned communication dimension. Its directory is explicitly marked `INVALID.md`; those rows are not used.
- Fixed pilot: 2 training seeds x 24 episodes, 8 fixed holdout episodes per seed and policy, CPU, communication-only fixed belief tracker.
- All six arms share cyclic eligibility, a 48-byte packet, one-slot delivery latency, and a 0.008 s single-packet slot budget.
- Held-out mean formation error: learned `18.54 ± 2.88 m`; idle `16.37 m`; random-50 `18.96 m`; always-forward `38.48 m`; local-age `15.46 m`; privileged oracle-fresh `51.45 m`.
- Learned issued a mean `8.62` accepted relay updates per episode but did not beat source-only at this budget. Local-age was descriptively `0.91 m` better with `6.75` accepted updates. These are engineering diagnostics, not inferential or novelty claims.
- Artifacts: `results/candidate_c_pilot_20261031_v2/REPORT.md`, `summary.json`, `heldout_episodes.csv`, and PNG/PDF figures under `figures/`. The live board was updated throughout training/evaluation.
- Current full test suite passes; `git diff --check` is clean.
- Scientific next gate: if Candidate C continues, compare learned and local-age policies over more independent training seeds and unseen canyon layouts, then add a value-of-information analytical scheduler. The present source-vs-forward single-slot evidence does not justify a paper claim.

## Successor takeover, 2026-10-31

- Read the full working tree, audit notes, research notes, configs, tests, stored pilot artifacts and git history. No destructive git operation was run. Codex's uncommitted changes and untracked files are preserved.
- The sandbox cannot import PyTorch: the project `.venv` is macOS arm64 and package egress is blocked. NumPy-only suite (9 files) passes 80/80; torch-dependent tests and training remain unverified in this session.
- Current narrative conflict recorded: README still describes the altitude-for-measurement story, `docs/RESULTS.md` records that it failed, and `research/PROPOSED_METHOD.md` selects Candidate C. The successor decision is to make value-of-information/opportunity-cost relay scheduling the paper line and keep the altitude story as a documented negative result.
- New findings to address before interpreting any learned result: (i) communication-only training keeps four action dimensions with three masked and `target_entropy=-4`; (ii) with `leader_mode: route`, replay stores the raw actor action for the leader even though the executed leader action comes from the tracker; (iii) scripted arms in the Candidate-C pilot are duplicated across training seeds, so their cross-seed standard deviation is zero by construction.
- Next: implement a one-step expected-variance-reduction VoI selector for the neighbour beacon and the slotted decision, test it with NumPy, and run paired unseen-layout experiments. Details: `docs/TAKEOVER_STATUS.md`.

## Successor execution checkpoint, 2026-10-31

- Implemented `communication/voi.py` and environment adapters for `beacon_select` in {`voi`, `voi_nounc`, `voi_nogeo`, `voi_gap`, `voi_local`} and `radio_tx_policy` in {`action`, `voi`, `voi_local`}. Added `tests/test_voi_scheduler.py`; NumPy-only suite now passes 90/90.
- Frozen protocols: `experiments/voi_beacon_study.yaml` (discovery, seeds 910000–910079) and `experiments/voi_gap_confirm.yaml` (replication, seeds 930000–930079, with a recorded deviation).
- Discovery block, 8 conditions × 80 paired episodes: unit (receiver age + link geometry) minus freshest = −1.00 m [−1.56, −0.47]; unit minus no relay = −6.87 m. The hard cell jink=1, nlos=0.01, p0=0.6 improves from 18.20 m to 14.32 m, CVaR90 42.2→19.8 m.
- Replication block, 8 conditions × 80 paired episodes: unit minus freshest = −0.83 m [−1.27, −0.40]; combined stratified estimate over 16 strata = −0.91 m [−1.26, −0.57]; unit minus none = −7.10 m.
- Covariance weighting adds nothing; age-gap weighting failed its first replication cell (27.1 m vs freshest 25.2 m, SD 34.4) and is rejected; the local no-remote-age proxy is 11.5 m worse than freshest in the hard cell. Remote age feedback is a prerequisite, not a detail.
- In the slotted one-packet-per-slot model, the one-step value says not to forward: the source packet is fresher and serves every follower. The learned Candidate-C pilot's failure to beat idle is consistent with this opportunity cost.
- Fixed: route-leader replay now stores the executed tracker action; probe `d0` default now matches `EnvConfig` (200 m); Candidate-C pilot summary no longer treats checkpoint-independent scripted arms as independent training seeds.
- New results page: `docs/VOI_SELECTION_RESULTS.md`. Analysis artifacts: `results/voi_beacon_study/analysis.md`, `results/voi_gap_confirm/analysis.md`, `results/voi_combined.md`.
- Still unverified: torch-dependent tests and any learned-policy claim; communication-only entropy accounting; graph/MAPPO; event-based peak AoI.

## Night continuation — 2026-11-01 Asia/Shanghai

- Read all AGENTS.md takeover sources and inspected current selector implementation and original CSVs. Linked Codex thread could not be read (tool failure); no claim of access to that conversation.
- Host full test rerun: **132 passed, 1 skipped**; `results/overnight_20261101/preflight-tests.txt`. MPS unavailable to this process; CPU selected.
- Corrected local-proxy prose: raw `results/voi_beacon_study/j1_n0.01_p0.3_C/voi.csv` records 1.0 transmission/step and 0.05869 accepted updates/step, not 0.059 transmissions/step.
- Added resumable per-episode runner and deadline-bounded supervisor without editing the inherited environment/learner. End-to-end smoke: 66/66 jobs; resume preserves 66 unique IDs, zero duplicates. Smoke is plumbing only.
- Protocol frozen in `experiments/overnight_voi_20261101.md`. Ordered independent replication, cadence sensitivity, then channel/manoeuvre extension; 5280 jobs per seed block, four workers, new seeds starting 960000. Later blocks advance by 10000, with no effect-dependent selection.
- Source/config snapshots, SHA256 hashes, inherited diff/status, dependency versions, exact commands, per-episode JSONL, logs and automatic paired analyses are retained in `results/overnight_20261101/`. No manuscript numbers changed.
- Background supervisor launched with caffeinate idle-sleep inhibition; stop new dispatch at **2026-11-01 08:00 +08:00**, drain active episodes. State verified running; inspect top-level status.json then current block/status.json.
- Codex heartbeat creation failed, so no model-level scheduled review was established. Local supervisor performs experiments and automatic numeric analysis; open-ended scientific reasoning is not autonomous overnight. Remaining feedback/MAC, scalability, learning and literature gaps remain explicit.
- Reproduction plumbing check: original seed 910000 at jink=1, nlos=.01, p0=.6 exactly matches stored rounded per-episode errors for none/freshest/unit; `results/overnight_20261101/reproduction-check.json`. This checks three episodes, not the full published mean.

## Rigour continuation — 2026-11-01

- Overnight raw audit:140544 recorded episodes,26 complete blocks,2080 seeds per fixed cell; incomplete final block retained but excluded as a whole. Joint whole-seed bootstrap across conditions gives unit−freshest −0.954 m [−1.077,−0.829] over the original eight cells. Per-cell22-family simultaneous bootstrap intervals and raw hashes stored in results/rigor_20261101/; analysis choices are explicitly post-run.
- Added formal surrogate derivation and corrected manuscript main score/failed age-gap sign and unsupported feedback-necessity inference.
- Implemented centralized paid report/grant/lease adapter in a separate module. Auditing the first version found inherited full covariance snapshots counted as48-byte payloads without encoding. V1 pilot and confirmation are retained and marked NOT_VALID_FOR_BYTE_FAIRNESS.md, not used for byte-fairness evidence.
- Wire v2 serializes24-byte status,8-byte grant,32-byte raw measurement plus original generation; receiver reconstructs the estimator, with identical data semantics in oracle and paid arms. Independent protocol experiments/priced_feedback_wire_v2_20261101.md;80 seeds×two cells×nine arms=1440 jobs, source snapshot saved before execution. Confirmation underway; results must be read from the completed v2 report, not v1.
- Full suite:144 passed,1 skipped (MPS unavailable). Raw tests:results/rigor_20261101/tests.txt. No inherited environment/learner changes, no destructive git actions.
- Wire v2 confirmation completed and audited:1440/1440 rows,80 seeds per arm/cell, no identity/budget/source-hash errors. All six simultaneous95% intervals for paid-unit minus paid-freshest include zero; superiority not established. Five point estimates favor freshest; no equivalence/impossibility claim. Final report, raw rows, audit, diagnostic counts and plot are in results/priced_feedback_wire_v2_20261101/. V1 byte-invalid outputs excluded. Need separately frozen transport/staleness ablations to identify mechanism; no learned/decentralized/deployment-readiness claim.

## Mechanism ablation and feedback abstraction — 2026-11-01 (successor)

- Ran Codex's frozen mechanism ablation, which had a protocol and runner but no results. Pilot: seeds 1950000–1950011, 12 episodes/arm/cell, 408/408 jobs. Confirmation: seeds 1960000–1960079, 80 episodes/arm/cell, 17 arms/cell, **2720/2720 jobs**. Raw records: `results/mechanism_ablation_20261101_confirm/episodes.jsonl`; report, summary and completion status in the same directory.
- Primary single-removal contrasts (unit, mechanism removed minus base): only **grant loss in p0=0.3** is isolated, −6.884 m with simultaneous 95% CI [−13.259, −0.510]. R and S at p0=0.3, and R/G/S at p0=0.6, all cross zero. Conclusion: `results/mechanism_ablation_20261101_confirm/CONCLUSION.md`. This identifies grant delivery as one hard-cell bottleneck; it does not establish paid unit superiority and does not show that fixing grants alone recovers the oracle gain.
- Diagnostic: base arms deliver only ~8% of status reports and ~8% of grants; removing grant loss raises relay transmissions from 0.052 to 0.612/step at p0=0.3, while removing report loss alone leaves relay at 0.063/step because grants are still lost.
- Added an optimistic separate-channel feedback abstraction (`beacon_select="voi_feedback"`, 24-byte reports, one-tick delay, bytes counted but control slots not charged). On seeds 960000–960079, 80 episodes: p0=0.6 K=5 gives 15.79 m vs 15.23 oracle and 18.21 freshest; p0=0.3 K=1 gives 21.55 m vs 21.27 oracle and 25.38 freshest. This is an upper bound, not deployment evidence; wire-v2 remains authoritative and negative. Protocols/results: `experiments/voi_feedback_confirm.yaml`, `results/voi_feedback_confirm/`.
- Compatibility fix: `scripts/voi_probe._run` now uses `.get` defaults for fields added by the successor, so Codex's frozen mechanism job builder runs unchanged.
- Paper: added Section 5.6 (mechanism ablation) and the feedback-abstraction caveat; updated abstract/conclusion and data availability. `docs/REVIEWER_RISKS.md`, `AGENTS.md`, `paper/README.md` now state that paid-feedback superiority is not established.
- Go/no-go recorded in `docs/GO_NOGO_20261101.md`: **no-go for a learned scheduler or same-family paid tuning.** Even with grant loss removed, paid unit minus paid freshest is +2.451 m at p0=0.3 and +0.214 m at p0=0.6; removing both report and grant loss gives +0.074 m at p0=0.3 and −1.828 m [−3.747, +0.062] at p0=0.6. The paper is positioned as an oracle analysis plus a negative deployment result. A new architecture must be separately priced and confirmed on fresh seeds; do not train a larger model to rescue a transport-level loss.

## Decentralized protocol test — 2026-11-01

- Recovered the inherited state, re-read the takeover, go/no-go, wire-v2,
  mechanism-ablation, reviewer-risk and draft documents. No destructive git
  operation. The sandbox still has no torch; NumPy-only experiments run.
- Independently recomputed the wire-v2 and ablation primary contrasts from
  `episodes.jsonl`. Point estimates match the stored reports; the stored
  intervals are joint/simultaneous and are wider than naive marginal
  intervals. A new observation: paid freshest itself degrades to 43.9 m at
  p0=0.3 versus 23.0 m for oracle freshest on the same seeds, so the paid
  control plane is a first-order failure for both rules.
- Designed and implemented two decentralized alternatives with separate
  frozen protocols:
  - **OSE** (`self_elect`): each follower scores itself from its own cache,
    overheard peer generations, and believed geometry; a value-proportional
    backoff resolves one sender. No status report, no grant.
  - **RSR** (`solicited_broadcast`): a token receiver piggybacks an 8-byte
    request on its data beacon; peers that hear it and hold a newer source
    generation self-select; the freshest eligible helper answers on the next
    tick; a scheduled fallback sender keeps one data packet per tick.
  - Ablations: `self_elect_nogeo` (degenerates to fixed; not informative),
    `self_elect_oracle` (matches oracle unit exactly), `solicited` (unicast),
    `solicited_random`, `solicited_oracle`.
- Design pilots (seeds 2190000–2190007 and 2190100–2190119) were inconclusive
  and are excluded from confirmation. The 8-episode OSE pilot's 37.8 m did
  not replicate; the 80-seed value is 29.7 m.
- Confirmation seeds 2200000–2200079, 80 episodes/arm/cell, three cells:
  - RSR − best(fixed,round) combined = **+3.602 m [+1.195, +6.100]**;
    RSR − round = +0.033 m [−2.724, +2.774]; RSR − random = +0.849 m
    [−1.494, +3.323]; RSR − unit = +5.666 m [+3.334, +8.103].
  - OSE − round combined = +0.743 m [−2.761, +4.146]; OSE − best = +4.313 m
    [+1.881, +6.905]; OSE − unit = +6.377 m [+3.535, +9.199].
  - `voi_status` is at round-robin performance (C1 30.6 vs 30.7; C2 17.9 vs
    18.0; C3 16.2 vs 16.1).
  - Accepted relay updates/step for RSR equal round-robin; the 8-byte request
    is a 12.5% byte cost with no measured benefit.
- Decision: **no-go for both decentralized architectures**; recorded in
  `docs/GO_NOGO_20261101.md` and
  `results/rsr_20261101_confirm/CONCLUSION.md`. Do not tune on the
  confirmation seeds. The paper is now an oracle analysis plus three
  independent negative deployment tests; the organizing claim is the
  information-opportunity-cost boundary.
- New code: `envs/formation_env.py` selectors `fixed`, `self_elect*`,
  `solicited*`; `scripts/voi_probe.py` metrics; `tests/test_solicited_relay.py`
  (5 direct-harness functions pass in the sandbox); `scripts/analyze_solicited.py`.
- New evidence: `results/rsr_20261101_confirm/` (`voi_all.csv`, `analysis.md`,
  `summary.json`, `CONCLUSION.md`), design-pilot roots, and the two protocols.
- Deviations: confirmation was split into fourteen output roots because the
  sandbox caps a call at ~180 s; the broadcast decline counter was populated
  after the run (derived in the report); OSE was run in two extra roots after
  the RSR block. No seed, cell, arm definition, or score changed after the
  confirmation started.
- Still unverified: torch tests, any learned policy, host pytest count after
  the edits, and the full text of the three adjacent 2026 records.

## 2026-11-02 — DC-RRT fidelity audit follow-up

## 复现忠实性专项更新（2026-11-02，后续轮次）

本轮已在Mac CPU完成 **6,190次主批/敏感性运行 + 400次终止定义诊断**，11项测试通过，模型未重训、旧记录保留。最新结论和全部原始数据入口见 [DC_RRT_REPRODUCTION_RESULTS.md](docs/DC_RRT_REPRODUCTION_RESULTS.md)；逐项来源及差异见 [DC_RRT_REPRODUCTION_AUDIT.md](docs/DC_RRT_REPRODUCTION_AUDIT.md)。

追溯[31]补充了空间CSC几何依据；[24]场景不能直接当作DC原图。Algorithm1未明确最近邻、ChooseParent/Rewire，且无碰撞目标连接分支可能无法推进；本轮独立保存literal、正文补全text及标准RRT*补全star解释，不再把上一轮添加的选父/重连称为明确原文机制。主对照是RRT*、DC-text、冻结Learning-Guided DC-text。其成功率在四图依次为RRT* 100/100/100/100%，DC-text 100/6/27/1%，学习版本100/11/30/1%；不能据此断言作者真实程序失败。

**仍不能称为Faithful Reproduction。** 正式PDF图像核查未完成，II–IV精确地图、机体半径、部分连接/碰撞与终止细节仍未知。以下1,560次记录为上一轮标准RRT*补全重建的历史结果，不能替代本轮审计。


Logs: `results/dc_rrt_fidelity_setup_20261102/`. Full job IDs/source hashes/model and failure-aware geometry audit: `results/dc_rrt_fidelity_full_20261102/final_audit.json` (errors=[]). Email draft prepared, not sent.

## 2026-11-02 — Structured mission scenes

User ended author-contact/original-map recovery requirement. Added procedural campus/industrial/urban scenes, 9 maps × 5 seeds × 3 arms = 135 runs. All 135 successes independently path/tree validated; 2 map-contract tests passed. No retraining. Independent standard-RRT* completion remains explicitly labelled. Results: `results/dc_rrt_operational_pilot_20261102/`; report `docs/DC_RRT_OPERATIONAL_SCENES_20261102.md`.

## MASAC-only execution, 2026-11-02
New mission environment, 9 tests passed, corrected four-arm/three-seed pilot completed with saved weights/replay. 240 evaluations: zero full-team successes. See docs/MASAC_EXECUTION_20261102.md. Longer standard-only budget diagnosis launched separately; no novelty claim.

## Structured V3 design and execution, 2026-11-02
User requested stronger maps and network design before experiments. Design frozen in docs/design/MASAC_V3_DESIGN.md before code/tests and new training. 14 tests passed; two512-step smoke arms completed384 updates each; three-seed20000-step standard training launched with live stdout/status/checkpoints. All current success claims remain negative/unconfirmed. Terminal application launch failed; viewer command and script provided.

## Structured V3 diagnosis and controlled continuation, 2026-11-02

All three 20,000-step parent runs completed, 19,000 updates each, final validation 0/6 each. The saved episode audit records 622/610/410 episodes, dominated by static and boundary collisions. No convergence or learned-mechanism claim. Original source snapshots and failures are preserved.

New reproducible audit: 90 maps / 390 tasks; 248 direct paths intersect buildings, all 90 admit static overhead geometric witnesses (not joint dynamic/time feasibility). A paired acceleration intervention keeps current clearance at 9m and CV preview at −7m, while best tested control margin changes −2.60/+1.80/+4.44m for amax2/4/6. This validates a diagnostic signal only. Raw data and actual failed 3D trajectories: results/masac_v3_diagnosis_20261102/; report docs/MASAC_V3_DIAGNOSIS_20261102.md.

Protocol docs/design/MASAC_V3_DIAGNOSTIC_PLAN.md frozen before launching results/masac_v3_budget_20261102: same three saved seeds resume 20,000→100,000 total steps, same environment/reward/network/update, sequential CPU job. Continuation restores optimizers, Replay, environment, return and RNG; new exact next-action/transition/update tests pass, 16 relevant tests total. Viewer follows an active-run pointer, prints actual validation step/completion/error and stale heartbeat. Terminal auto-open remains blocked. Candidate arms are not queued while baseline navigation is unestablished.

Exact method specification added at docs/design/MASAC_V3_METHOD_SPEC.md, including Actor+Critic mechanism inputs, raw-action entropy convention and preview/termination geometry mismatch. Local fixed-state 100-sample pipeline profile (background training active) measures candidate observation+Actor medians 1.787ms/3.483ms for3/6 UAVs; no learning claim. Original inference_ms excludes feature computation. Raw per-call timings saved with source hashes.

Parent Replay coverage audit: 180,000 correlated pre-action agent states across the three seeds, none within50m of their goals; minimum distances80.29/69.51/96.02m. This identifies an experience-coverage gap, not a unique cause or a capacity verdict. Saved counts/quantiles and source Replay hashes: results/masac_v3_diagnosis_20261102/replay_coverage.json.

First budget-extension seed411 completed100,000 total steps /99,000 updates, final validation0/6 (2 boundary,4 episode limit). Development mean remaining distance248.02→121.97m; no success/convergence claim. New Replay has2,026/300,000 pre-action agent states within50m,35 within10m, none within4m/capture; closest6.38m. This is single-seed development coverage evidence. Seed422 now continues automatically; seed433 is queued. Numeric analyses saved in seed411_budget_diagnosis.json and budget_replay_coverage_seed411.json. Candidate arms remain deferred.

## 2026-11-03 16:26 Asia/Shanghai — Predictive Maneuverability paper suite LAUNCHED

Suite `masac_v4_maneuver_paper_suite_20261103` (stamp 20261103_162345): parallel arms **full / zero / static** on frozen L0 12-map, seeds 1211/1222/1233 × 80k, Path-C reshape + capture 4/2 latch=False, V4 pool+RiskEncoder, **cold start** (Legacy Path-C incompatible). Baseline path: Path-C L0 `..._145043`. `do_not_auto_launch_further`. PIDs 45262/45327/45356.

