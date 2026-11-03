# Agent coordination

> **Newest result — 2026-11-02, bound audit and confirmation.** The projection
> lower bound passed an independent certificate: 11 600 integrated
> curvature-bounded curves and 6 692 CSC paths, worst margins
> `6.5e-10` m and `2.9e-6` m (`results/dc_rrt_bound_audit_20261102/audit.json`).
> The heading bound `r∠(h_a,h_b)` is also admissible; the gate uses the maximum.
> The 600-pair `dc_order` arm sorted rewires ascending and skipped improving
> rewires (urban rewires 22.3→1.3). That column is historical, not the method.
> Corrected confirmation, budgets not pooled: 240 fresh structured pairs at
> 1000 nodes, length-gated 1.60–1.83× and corrected order 1.60–1.84×, path
> length change 0, strict arms byte-identical, same 40 failures at a separate
> 5000-node cap. Those four maps reverse-search 0/5 with one node; the other
> 20 maps reverse-search 5/5. This is not a non-existence proof, and the maps
> stay in the tables. Fixed-footprint density: ratio rises from ~1.02 (one
> start-goal call) to 1.80× [1.66, 1.98] at 80 m spacing. Read
> [docs/DC_RRT_BOUND_CONFIRM_20261102.md](docs/DC_RRT_BOUND_CONFIRM_20261102.md)
> and [paper/DC_RRT_PRUNING_DRAFT.md](paper/DC_RRT_PRUNING_DRAFT.md).

> **Newest result — 2026-11-02, admissible pruning for DC-RRT***. The dense-scene
> bottleneck is now measured and addressed: 91.6 % of choose-parent, 96.6 % of rewire
> and all but one goal Dubins call per run do not change the cost, and Dubins solves
> plus their collision checks are ~90 % of urban planning time. A closed-form
> admissible lower bound plus pre-solve pruning, length-gated collision checks and
> bound-ordered early stopping is implemented in `planning/dc_rrt/pruning.cpp`.
> Final run: 600 structured scene/seed pairs × 5 arms × 3 timing repeats, fresh seeds
> 8100000/8200000. Solved-case speed-up 1.49–1.94× (byte-identical output) and
> 1.62–2.37× (ordered, mean path-length change ≤ 0.13 %). Success sets identical.
> Read [docs/DC_RRT_PRUNING_20261102.md](docs/DC_RRT_PRUNING_20261102.md), the frozen
> protocol [experiments/dc_rrt_pruning_20261102.md](experiments/dc_rrt_pruning_20261102.md),
> and the draft [paper/DC_RRT_PRUNING_DRAFT.md](paper/DC_RRT_PRUNING_DRAFT.md).
> This is still an independent reconstruction, not a faithful original reproduction;
> two urban maps remain unsolved even at 5000 nodes. **Success audit (2026-11-02):**
> those two maps place the goal 14.9/19.7 m from 104.9/116.8 m buildings; a reverse
> search adds a single node and never succeeds, while the other 28 maps reverse-solve
> 5/5. The remaining 2 of 42 failed pairs are budget-limited (1531/1077 nodes at a
> 5000-node cap). Branch-aware CSC, reverse CSC, approach sampling and
> stagnation-triggered steering were all tested and none robustly improved success;
> the fix is a goal-validity check in the benchmark, not a new planner. Read
> [docs/DC_RRT_SUCCESS_AUDIT_20261102.md](docs/DC_RRT_SUCCESS_AUDIT_20261102.md).
> Pilot data at seeds 7300000/7400000 is retained separately.

> **最新方向更新（2026-11-02）**：用户不再要求联系作者或精确恢复原图；按公开资料自主实现，重点转向有实际任务结构的单无人机三维场景。邮件草稿归档，不发送。独立标准RRT*补全版本已完成135次结构化场景试跑，冻结模型未重训；不是作者源码复现或真实测绘/实飞验证。见 [结构化场景结果](docs/DC_RRT_OPERATIONAL_SCENES_20261102.md)。历史审计和数据保留。


## Newest fidelity checkpoint — 2026-11-02 follow-up

Read `docs/DC_RRT_REPRODUCTION_AUDIT.md` and `docs/DC_RRT_REPRODUCTION_RESULTS.md`
before any claim. Completed 6190 main/sensitivity cases plus 400 endpoint-policy
diagnostics on Mac CPU; 11 tests pass. Frozen CNN unchanged. Literal Algorithm1
can stall and omits nearest/parent/rewire; dc_text and dc_star are separately
labelled completions. Missing exact maps/body/collision/author semantics and
formal-PDF visual verification still preclude Faithful Reproduction. Source
snapshots are per batch; the optional endpoint macro was added after main freeze.
Do not overwrite prior 1560 cases or retune to these test outcomes.


## Latest user override — 2026-11-02

This round is **single-UAV 3D path planning only**: DC-RRT* (IEEE Access
2025, DOI 10.1109/ACCESS.2025.3600283), plus a genuinely trained small neural
sampling guide inspired by MPN-RRT* (Sensors 2025, 25(13):4142). Evaluate only
RRT*, DC-RRT*, Learning-Guided DC-RRT*. Do not add multi-UAV or unrelated
models. Preserve existing code/records per the user's final instruction.
Start with `experiments/dc_rrt_single_uav_20261102.md`; explicit missing
source details must remain fidelity limitations, never invented originals.
All older direction blocks below are historical and superseded for this round.

Execution checkpoint: `docs/DC_RRT_SINGLE_UAV_20261102.md` is the current
result entry. Native independent reconstruction and actual 60-epoch CNN
training completed on the Mac CPU. Final evaluation contains 1560 unique
three-arm cases, all successful and independently curve/tree-validated;
`results/dc_rrt_20261102_evaluation/final_audit.json` has no errors.
The frozen learned sampler did not improve efficiency; some paths shortened.
No final-test-dependent retuning. Full original-paper experiment replication
is unfinished: exact maps, numeric body radius and some spatial connector
details remain missing. Keep the reconstruction label and read
`research/DC_RRT_FIDELITY_20261102.md` before any reproduction claim.

This repository is shared by more than one coding agent. Read this file before
editing. The human also runs Codex on the same working tree.

## Active direction override — 2026-11-01, user instruction

New research focuses on **3D UAV path planning**, with the multi-UAV context
retained. The user explicitly rejected the narrow-passage/bottleneck focus.
Read `docs/THREE_D_PLANNING_SCOPE_20261101.md` first. Compare 3D route search,
dynamically feasible trajectory generation, and multi-UAV coordination using
official high-quality implementations. GCOPTER/MINCO and the EGO family are
current reading priorities; no special scene or new method is frozen yet.
Do not resume the earlier bottleneck/local-repair proposal by default. The
older planning focus and code review are historical records, not current
task selection. Do not default to relay/AoI tuning or larger MARL networks.
Preserve historical code, results and manuscripts; their measurements do not
establish a new path-planning result. Distinguish source reading from runs.

Host execution now exists: `docs/GCOPTER_LOCAL_REPRODUCTION_20261101.md`.
Unchanged official GCOPTER/MINCO core was built natively, with a documented
ROS-free local A*/box-corridor adapter; smoke 3 cases and a fixed 40-case
batch completed. This is a single-UAV numerical-core adaptation, not the full
paper benchmark or a multi-UAV result. Validation uses a declared 1% engineering
tolerance: all 40 nominal speed maxima slightly exceed 4m/s (largest 4.020629).
Read the saved audit before describing constraint satisfaction or performance.

## Read first

1. `docs/ACTIVE_RESEARCH_HANDOFF.md` — current verified state, exact
   reproduction commands, next task, blockers.
2. `docs/TAKEOVER_STATUS.md` — takeover state, verified vs unverified work,
   environment limits, file inventory.
3. `docs/VOI_SELECTION_RESULTS.md` — the current paper line and all numbers.
4. `RESEARCH_PROGRESS.md` — chronological execution log.
5. `paper/DRAFT.md` and `paper/README.md` — manuscript status.
6. `docs/REVIEWER_RISKS.md` — known objections and evidence gaps.

## Current paper line (2026-11-01)

**Who Should Beacon? Age- and Geometry-Aware Relay Selection for UAV Formation
behind Roofs.** The paper is a communication/scheduling study, not a MASAC
novelty paper. The original “pay altitude for a measurement” line is a
documented negative result; do not revive it as the main claim.

Key result: geometry-aware relay selection (`b1:voi_nounc`, called “unit”)
beats freshest-sender by 1.00 m in the 910000 block and 0.83 m in the 930000
block; combined −0.91 m (95% CI −1.26 to −0.57). Against no relay: −7.10 m.
The hard p0=0.6 cell improves from 18.20 to 14.32 m, CVaR90 42.2 to 19.8 m.
Covariance weighting adds nothing; age-gap and local/status variants failed.
Relaying that pre-empts the leader's slot is usually not worth its opportunity
cost.

Deployability is a separate, negative result. The paid wire-v2 study (seeds
1820000–1820079, 1440 jobs) charges status/grant slots and serializes 32-byte
raw measurements; all six simultaneous paid-unit minus paid-freshest intervals
cross zero. The pre-registered mechanism ablation (seeds 1960000–1960079,
2720 jobs) isolates grant loss only in the harder cell (p0=0.3: −6.88 m,
simultaneous CI [−13.26, −0.51]); report loss and stale selection state are
not isolated. Do not turn the oracle gain into a deployment claim. The
optimistic separate-channel feedback abstraction (24-byte reports, no slot
charge) recovers the oracle gain at K=5 in p0=0.6 and K=1 in p0=0.3, but it is
an upper bound, not deployment evidence.

Two decentralized alternatives were then frozen and confirmed on fresh seeds
(2200000–2200079, 80 episodes per arm/cell, three cells). Sender-side
self-election (OSE) is at round-robin performance (combined OSE − round
+0.743 m, CI [−2.761, +4.146]); receiver-solicited relay (RSR) is
indistinguishable from round-robin (+0.033 m, CI [−2.724, +2.774]) and does
not beat a cyclic-helper ablation (`solicited_random`; +0.849 m, CI [−1.494, +3.323]). Accepted
relay updates per step are unchanged. The decision is **no-go for both**;
evidence in `results/rsr_20261101_confirm/CONCLUSION.md`, protocols in
`experiments/self_election_20261101.md` and
`experiments/solicited_relay_20261101.md`. The paper now has three independent
negative deployment tests; round-robin is a strong deployable baseline.

## Non-negotiable rules

- Do **not** run `git reset --hard`, `git clean -fd`, force-push, or delete
  results. The working tree contains inherited uncommitted work.
- Do not overwrite another agent's uncommitted changes. Add new files or make
  targeted edits; record deviations in the relevant protocol file.
- Every number in a paper or report must point to a stored CSV/log. Do not
  invent literature, runs, or test results. Mark metadata-only papers as such.
- Do not claim a learned-policy, convergence, decentralized, energy, or safety
  result without a matching experiment.

## Environment

- **Local-first policy (2026-11-01).** Run tests, training and experiments on
  the user's Mac host with `~/Desktop/research/aoi-masac-formation/.venv/bin/python`,
  using local CPU/MPS and disk. Do not present a sandbox run as host-verified.
  The one-command host runner is `bash scripts/run_local_mac.sh` (add `--rsr`
  for the frozen receiver-solicited replication).
- Host: macOS arm64, project `.venv` with PyTorch 2.8 and MPS. Use
  `.venv/bin/python` for torch tests and training.
- The successor agent's Linux sandbox cannot import the macOS `.venv` and has
  no package egress or pytest. It can run NumPy-only code. Do not assume its
  `/tmp/pyextra` path exists on the host.
- Last sandbox full NumPy suite: 91 passed (2026-10-31). The 2026-11-01
  session ran only the new 5-function protocol harness in the sandbox; the
  full host suite has not been rerun after the edits. Torch tests and
  training are unverified; run them on the host before making learned claims.

## Files added by the successor session

New files (safe to read; avoid overwriting without checking):

- `communication/voi.py`
- `tests/test_voi_scheduler.py`
- `scripts/analyze_voi_beacon.py`, `scripts/combine_voi_blocks.py`,
  `scripts/run_voi_study.sh`
- `experiments/voi_beacon_study.yaml`, `experiments/voi_gap_confirm.yaml`
- `docs/VOI_SELECTION_RESULTS.md`, `docs/REVIEWER_RISKS.md`,
  `docs/TAKEOVER_STATUS.md`, `research/SEARCH_LOG_SUCCESSOR.md`
- `paper/DRAFT.md`, `paper/README.md`
- `scripts/analyze_voi_feedback.py`, `experiments/voi_feedback_confirm.yaml`
- `results/voi_beacon_study/`, `results/voi_gap_confirm/`,
  `results/voi_validation/`, `results/voi_design_checks/`,
  `results/voi_feedback_design/`, `results/voi_feedback_confirm/`,
  `results/voi_combined.md`
- `results/mechanism_ablation_20261101_pilot/`,
  `results/mechanism_ablation_20261101_confirm/` (run by the successor from
  Codex's frozen protocol/script; `CONCLUSION.md` added)
- Codex-owned additions: `communication/reported_feedback.py`,
  `communication/mechanism_ablation.py`, `scripts/run_priced_feedback.py`,
  `scripts/run_mechanism_ablation.py`,
  `results/priced_feedback_wire_v2_20261101/`,
  `experiments/priced_feedback_wire_v2_20261101.md`,
  `experiments/mechanism_ablation_20261101.md`

Materially edited existing files: `envs/formation_env.py` (VoI selectors,
`radio_tx_policy`, peer status table, route-leader replay fix),
`scripts/voi_probe.py`, `train.py` (probe `d0`), `tests/test_env.py`,
`scripts/run_candidate_c_pilot.py` (scripted-arm summary), `README.md`,
`RESEARCH_PROGRESS.md`, `docs/RESULTS.md`.

## Reproduce the current result

```bash
python3 scripts/run_voi_study.sh          # full 2-block scripted study, CPU only
python3 scripts/analyze_voi_beacon.py --root results/voi_beacon_study
python3 scripts/analyze_voi_beacon.py --root results/voi_gap_confirm
python3 scripts/combine_voi_blocks.py
```

## Open decisions

1. Deployability: the paid wire-v2 protocol has not beaten paid freshest, and
   both tested decentralized protocols are at round-robin performance. The
   decision record `docs/GO_NOGO_20261101.md` says **no-go for a learned
   scheduler, same-family paid tuning, and the tested decentralized
   protocols**; position the paper as an oracle analysis plus three
   independent negative deployment tests, unless a separately priced new
   architecture is justified.
2. Add a team-size/city-density experiment, or keep scalability as a stated
   limitation.
3. Read the three 2026 adjacent records in full text before finalizing related
   work; they are currently metadata-only in
   `research/SEARCH_LOG_SUCCESSOR.md`.
4. Convert `paper/DRAFT.md` to the IEEE two-column template and fit the page
   limit.
