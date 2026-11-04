# Persistent Windows research state

Date: 2026-11-05, Asia/Shanghai. Root: `C:\research\masac-maneuverability`.
Branch: `research/wowmom2027-upgrade`. Starting tip: `f81a588`.

## Standing user instruction

Continue the research loop; if account usage is exhausted, wait and resume when
usage becomes available. An ACTIVE 15-minute heartbeat named **MASAC 研究持续续接**,
automation id `masac`, returns to the current chat. It cannot bypass usage limits
or guarantee the exact reset instant. Local scheduling needs the computer/app on.
Do not create duplicate automations. Keep unchanged/non-actionable checks quiet.

The user explicitly authorizes Windows takeover, B sync-capture implementation
and small validation; present a concrete plan before large B training. Do not run
Attention/L1, a historical complete paper-suite rerun or unannounced continuation.
Each shell/Git/Python/training command must be displayed in full in commentary
before executing. Training/evaluation must stream through tee to screen and disk.
Do not kill unrelated processes, force-push, upload credentials or checkpoint/replay
binaries. Meaningful stages must be committed and pushed to both active branches.

## Completed and durable evidence

- Hardware audit **committed/pushed `59fec6b`** to main and research branch before
  training: i5-12400F 6C/12T, 31.84GiB RAM (~17.79GiB initially available),
  RTX4060 8188MiB, driver595.79, torch2.5.1+cu121, CUDA12.1/cuDNN9.1, CUDA works.
- `.venv\Scripts\python.exe` uses system-site-packages; requirements satisfied.
- Four metrics archives restored to ignored `local_metrics_restore/results/`;
  preserve original tracked `results/` because archive snapshots can differ.
  Reassembly/extraction and SHA256s are logged in
  `results/masac_v4_windows_audit_20261105/analysis_console_snapshot.log`.
- Independent finals match results_key: static11/10/5, full3/4/8, zero1/0/1
  successes out of12; arm all-seed gates fail2/3,1/3,0/3 respectively.
- **Important corrected diagnosis:** final108 evals:43success65fail;61 failures
  have only some agents ever in their distance ball,4 all agents ever in ball but
  fail capture. Historical ever_ball_maps uses **any**, not **all**. No collision
  failures in these finals. Do not claim a purely synchronization bottleneck.
- `results/masac_v4_windows_audit_20261105/{analysis.json,DIAGNOSIS.md,learning_curves.png}`.
- Device benchmark completed: median CPU threads1/2/3=1.216/1.939/2.549 steps/s,
  CUDA1=22.318steps/s (~8.75x). Initial action/probe error <=1e-4. Includes full
  act/environment/replay/update, excludes eval/reference/checkpoint I/O. Old CPU
  default unchanged. Complete CPU/CUDA trainer smoke also passed (see below).
- B/source Windows lock tests + existing maneuver tests: **9 passed**.
  `results/masac_v4_windows_audit_20261105/tests_sync_console.log`.
- After CUDA guard and RNG-save changes, targeted tests again **9 passed**.
- Original CPU protocol static seed1211 smoke:256steps/128updates,54.075s;
  explicit CUDA control same smoke seed/budget:8.219s (~6.58x faster), both
  finite and0/2success (boundary collision), both saved checkpoints. CUDA RNG
  stream presence verified. This is execution validation only.

## Current work / next ordered actions

1. CPU/CUDA smokes and two-arm two-seed2k pilot **COMPLETE**. All four final
   evals0/12, all-agent-ever-ball0; B/1422 has any-ball1/12, others0. All7,488
   updates passed the per-update finite guard. No method gain claimed. Reports:
   `results/masac_v4_sync_pilot_diagnosis_20261105/{summary.json,PILOT_RESULTS.md}`.
   Two concurrent processes:~15–16steps/s each, GPU84%,~1.5GiB VRAM total,
   CPU27%,~15.6GiB RAM available at sample. Each process RSS~1.22GB.
2. **New identification finding:** all12 archived L0 maps have no moving
   objects and conflict_degree0. On new control/1411@2k sampled128states,
   moving clearance is constant1 for full versus0 static; peer7204/7296 values
   equal1. Other channels match in this sample. Not an audit of old final-policy
   trajectories. See `docs/L0_CONTENT_IDENTIFIABILITY_AUDIT_20261105.md`.
3. Next registered study **40k ×3seeds ×2arms**, static control/B with fixed
   lambda.5/tau.1, cold seeds1511/1522/1533, warmup1000, batch128, capacity100000,
   eval2k, checkpoint10k, CUDA1Torch thread per process,2processes. Plan was
   summarized visibly before launch. Protocol:
   `docs/design/MASAC_V4_SYNC_CAPTURE_40K_PROTOCOL_20261105.md` and
   `configs/masac_v4/sync_capture_40k_{control,b}_cuda_20261105.json`.
   Intended outputs:
   `results/masac_v4_sync_40k_control_cuda_20261105` and
   `results/masac_v4_sync_40k_b_cuda_20261105`. Check their status before launching.
   Final40k primary; no lambda search/best-checkpoint replacement/auto extension.
   Reserved test manifest exists, **do not evaluate policies on it yet**.
   **LAUNCHED 2026-11-05 04:49 Asia/Shanghai** after protocol/pilot evidence push
   `eaf83e2`. Control PID25212, B PID2940 at launch. Current tool sessions
   control27303/B5353 can be polled while still valid; verify OS process command
   lines and output-local files on resumption. Both first2k evals0/12; control
   any-ball2/12, B0/12. Intermediate values are not efficacy conclusions.
   Independent analyzer:
   `.venv\Scripts\python.exe -u scripts/analyze_sync_study.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --out results/masac_v4_sync_40k_analysis_20261105`.
   Its14 decision/integrity tests passed; current output is IN_PROGRESS with
   decision_ready=False. Rerun on later evidence; never infer completion from
   absence of a recent log line. Five independent discrete-dynamics checks also
   passed; see `docs/MATH_DISCRETE_PROBE_ERROR_20261105.md` (8.8m ideal discrete
   peer-CV bound, not a safety certificate). Both running source hash sets were
   independently rechecked unchanged.
   **Latest completed pair:** both1511 histories finished their registered40k,
   then the same original workers cold-started1522. Two of six histories are
   complete;1533 remains queued. Control/B final success=2/12 versus6/12;
   any-ball12/12 both, all-ball5/12 versus8/12; collisions0 both. Each completed
   39000 learner updates. Both per-seed gates fail (<8/12); do not early-stop
   or change the remaining histories. This single+4/12 pair is not the full
   advancement decision. The canonical analyzer still reports IN_PROGRESS,
   decision_ready=False. See `docs/SYNC_CAPTURE_FIRST_PAIR_20261105.md`.
   Both40k checkpoint/replay byte hashes were independently verified against
   manifests; all14 declared source hashes and both protocol hashes match.
   All195 recorded post-warmup progress diagnostics per arm are finite.
   Proof record: `results/masac_v4_sync_40k_progress_20261105/first_pair_integrity.json`.
   This is not exact-CUDA-resume validation. Final progress.last_eval_step38000
   is expected because that line precedes final evaluation; use completion
   plus evaluation_40000.json for endpoints. Never overwrite old metrics.
   Resource sample at05:37: GPU54%,1528MiB,59C; freeRAM~14.5GiB. Four python
   PIDs are two venv launchers plus two workers, not four training jobs.
   The offline trajectory analyzer ran on **all48 saved trajectories** across
   both arms at1511@20k/22k and passed metric/map-pair cross-checks:
   `scripts/diagnose_sync_capture_trajectories.py`. See
   `docs/SYNC_CAPTURE_INTERIM_DIAGNOSIS_20261105.md`. Distinguish J=>C=>A:
   simultaneous capture, all-agent individual capture, all-agent ball entry.
   B20k has two C-but-not-J episodes with maximum occupancy1; most episodes
   still lack A. This is post-hoc diagnosis, not a B effect or causal claim.
   Offline read mistakes (wrong root manifest filename; numeric glob initially
   included evaluation_summary.json) were corrected. They did not affect any
   training process/source/config. The failed integrity attempt and corrected
   success are both retained in integrity_progress_console.log.
4. Commit/push protocol and completed pilot evidence before40k launch. Freeze
   every file in trainer SOURCES for all six histories; checkpoint hashing reads
   disk and would otherwise misidentify the loaded code. Analyze/log without
   editing these sources. Keep output-local status/completion authoritative;
   legacy shared registry can lose a display pointer under concurrent writes.
5. After40k, record all seed-paired differences and the preregistered engineering
   decision (not significance). If no capture floor/signal, call it inconclusive;
   no automatic budget restack. Proposed constant-fill/peer-CV identification is
   separate and not queued. Do not bulk-add partially extracted historical
   untracked metrics or assembled archive duplicates.

Paper edits pushed: final suite numbers, Standard-C common empty811@40k
warm-start confound, POMDP/terminal/bootstrap assumptions, entropy target **-3**,
19 probes =18 unit directions + zero, static is static-geometry preview, no
unsupported safety/novelty/superiority claim. Frontier updated from primary sources
including SCOPE and actuator-aware safe MARL. Official WoWMoM deadline: Dec1 2026
23:59AoE = Dec2 19:59 China time; see FRONTIER_GAP_UPDATE_20261105.md.

Paper-ready English Methods now lives in `docs/PAPER_METHOD_DRAFT_20261105.md`.
`docs/MATH_PBRS_SCOPE_NOTE_20261105.md` connects B to established entropy-regularized
PBRS (AAAI2023), states common-information/initial-distribution/boundary conditions,
and explicitly separates exact objective ordering from approximate critic,
target-network and automatic-temperature learning paths. B is fixed shaping,
not a scheduled curriculum; it has no explicit arrival-time-gap penalty.

## Resumption safety

Inspect per-output `status.json`, `completion.json`, `error.json`, manifest and
live process command lines before launching. Output-local files are authoritative;
shared legacy registry pointers may lag. Never reuse an existing output directory.
Historical Mac .pt/replay absent: cold pilots are not resumed histories. Match
checkpoint/replay/RNG/optimizers if later copying a named continuation from Mac.
