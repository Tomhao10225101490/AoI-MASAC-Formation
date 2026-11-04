# Persistent Windows research state

Date: 2026-11-05, Asia/Shanghai. Root: `C:\research\masac-maneuverability`.
Branch: `research/wowmom2027-upgrade`. Starting tip: `f81a588`.

## Standing user instruction

Continue the research loop; if account usage is exhausted, wait and resume when
usage becomes available. An ACTIVE hourly heartbeat named **MASAC 研究持续续接**,
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
  default unchanged; CPU smoke and CUDA trainer timing still required.
- B/source Windows lock tests + existing maneuver tests: **9 passed**.
  `results/masac_v4_windows_audit_20261105/tests_sync_console.log`.

## Current work / next ordered actions

1. Finish source review and freeze B/CUDA pilot configs. New opt-in B uses bounded
   worst-agent normalized distance/speed potential; true termination zeros next
   potential, time-limit truncation retains it to match existing bootstrap mask.
   Evaluation uses original unshaped env and frozen4/2 latchFalse.
2. Run original-protocol **CPU static smoke** first (256 steps/128 updates), with
   `--torch-threads 3`. Label execution reproduction, not a scientific gate.
3. Run an explicitly named CUDA control smoke and compare complete trainer wall
   time. Then paired small B/control (two seeds, equal budgets/configs), visible
   logs and all/any arrival, max simultaneous occupancy, failure counts.
4. Summarize results, finite losses and compute use. Update paper/math/frontier
   docs. Push only intentional source/docs and compact new result evidence; do
   not bulk-add extracted historical metrics or assembled archive duplicates.
5. If small pilot supports a next experiment, state its hypothesis/formulas,
   controls/seeds/budget/gate and resource plan before launching larger work.

Paper edits under review: final suite numbers, Standard-C common empty811@40k
warm-start confound, POMDP/terminal/bootstrap assumptions, entropy target **-3**,
19 probes =18 unit directions + zero, static is static-geometry preview, no
unsupported safety/novelty/superiority claim. Frontier updated from primary sources
including SCOPE and actuator-aware safe MARL. Official WoWMoM deadline: Dec1 2026
23:59AoE = Dec2 19:59 China time; see FRONTIER_GAP_UPDATE_20261105.md.

## Resumption safety

Inspect per-output `status.json`, `completion.json`, `error.json`, manifest and
live process command lines before launching. Output-local files are authoritative;
shared legacy registry pointers may lag. Never reuse an existing output directory.
Historical Mac .pt/replay absent: cold pilots are not resumed histories. Match
checkpoint/replay/RNG/optimizers if later copying a named continuation from Mac.
