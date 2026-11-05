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

## Current authoritative status: completed40k, next offline diagnosis

Updated after final verification, 2026-11-05. **All six histories COMPLETE**.
Both workers finished normally at07:22/07:23; no Python training process remained
at the post-completion check. Do not restart or extend these completed runs.
The launch PIDs/sessions and pending statements in the historical snapshot below
are obsolete. Per-output manifest/completion/status remain authoritative.

- Control final successes[2,1,4]/12; B[6,0,5]/12; paired differences[+4,-1,+1].
  Mean fraction difference1/9. Canonical analyzer COMPLETE/decision_ready=true/
  ADVANCEMENT_RULE_MET. This means **planning confirmation only**, not a launch,
  significance or a paper superiority claim. Both arms FAIL the absolute gate,
  zero of three seeds reaching8/12. Retain the negative1522 seed.
- All six histories reached40,000 transitions/39,000 updates. SOURCE/config
  freeze was honored through all completions and final integrity/analysis.
  All12 final checkpoint/replay SHA256s,14 declared source hashes and both
  config hashes match. Each history has195 finite sampled post-warmup diagnostic
  rows plus the trainer's per-update finite guard, not39,000 retained rows.
  This does not validate exact CUDA resume. Preserve snapshots and original bytes.
- All72 final trajectories checked:54 failures =43 incomplete all-agent ball
  arrivals +1 all-ball without all-individual capture +10 individual capture
  without overlap; no final collision. Not a purely synchronization bottleneck.
- Full report: `docs/SYNC_CAPTURE_40K_FINAL_RESULTS_20261105.md`.
  Canonical decision/curves/failure_taxonomy.json:
  `results/masac_v4_sync_40k_analysis_20261105`.
  Integrity proof added: progress root's `seed1533_integrity/verification.json`;
  trajectory summaries `trajectory_40000_seed{1511,1522,1533}/summary.{json,md}`.
- Sealed seeds2026110501..2026110512 remain unopened. No automatic confirmation,
  40k-to80k continuation, best-checkpoint substitution or test launch is queued.

Next ordered actions:

1. Complete results, paper/math/frontier and the separate offline plan were
   pushed to both branches as d8ff5ad before the next diagnostic. Final failure
   tables/figure are reproduced by scripts/summarize_sync_final_failures.py.
   Post-hoc never-ball localization:65agent-map records,29closest4–5m and36
   closest5–10m, all finalspeed<2. This is near-goal descriptive evidence, not
   proof of hovering/reward causality or a replacement efficacy endpoint.
2. Offline plan above is **COMPLETE**, script reviewed and run successfully on
   CUDA/chunk128/Torch1thread: six final replays,6144jointobs,48chunks. Every
   full/static trace bitwise-equals the frozen provider, maxerror0;3 independent
   geometry witnesses pass. No model/actor/rollout/test loaded or run.
   Report: `docs/LATE_REPLAY_CONTENT_AUDIT_20261105.md`; raw results in
   `results/masac_v4_sync_40k_late_content_audit_20261105`.
   IndicesSHA eca0cfd9c12108af9627b89910fcf6a657c168a9c1bb54622eb628d15d3b3fc7.
   Full==static-fill1 in349849/350208tokens=99.89748949%;359peer-only residuals
   all incontrol1522; no TTC/brake/reserve/flag or other-column residual.
   Every18432agentobs has2validpeers; most peer1values are valid-far geometry
   capped by40m padding sentinel, not absence. Moving0valid slots. Population
   remains late static-trained replay, not historical full/finalonpolicy.
   Do not rerun into the same output (the script rejects overwrite).
3. Integrate/push this audit and reward mathematical diagnosis. Path-C ball
   bonus is once-ever; capture is+8/-8 state difference. Positive holding
   reward alone does not establish hover preference; a feasible near-goal
   finish yields54.06949 and beats waiting in the documented example.
   See `docs/MATH_PATHC_OBJECTIVE_DIAGNOSTIC_20261105.md`; no root cause claimed.
4. Separately frozen **offline actor-response** diagnostic is **COMPLETE**.
   Protocol/code pushed3615a66 before any checkpoint load; actual CUDA run exit0.
   Protocol: `docs/design/MASAC_V4_NEAR_GOAL_ACTOR_OFFLINE_PROTOCOL_20261105.md`;
   script: `scripts/audit_sync_near_goal_actor.py`; intended new output:
   `results/masac_v4_sync_40k_near_goal_actor_audit_20261105`.
   completion.json written last is authoritative; all six histories complete,
   actor hashes/global Torch RNG unchanged; four limiter witnesses and48chunks
   pass. Output hash/bin/paired-count integrity independently PASS. Do not rerun.
   The same6144
   observations/hashes; compare tanh(mu), not E[action], with64fixed isolated
   Gaussian draws; only analytic one-step distance/speed, no simulator/rollout,
   new training or sealed maps were used. Primarynear subset8000agentobs,7359
   fail a necessary distance bound for one-step capture. Mean inwarddetaccel
   1.90513, dchange-.0712755m; deterministic168/8000 versus marginalMC10440/512000,
   MC-minus-det=-.0609375percentagepoints, three positive/three negativehistories.
   Not uniform MCrescue, control failure, causal evidence or actualfinaloccupancy.
   Report: `docs/NEAR_GOAL_ACTOR_OFFLINE_RESULTS_20261105.md`.
   It is separate from completed phaseA's no-model contract.
5. Next minimum under preparation: **saved final trajectory dwell/reward audit**,
   exactly72 original40000NPZs/JSONs, no new rollout/model/training/test. Fixed
   near-stall description4<=d<10 andspeed<.5,>=25consecutivepoststepframes=5s,
   plus all distance/speed bins and A/C/J/event cross-checks. Protocol/code need
   review/push before run. Actual evalJSON lacks cumulative return, so source-
   derived reward components must mark reference-return validation UNAVAILABLE;
   do not invent it. Reward/dwell remain diagnostic, not a new primary endpoint.
   Any substantial new training still needs a visible plan; relative40k rule
   supports planning confirmation only. No training or sealed evaluation queued.

Before this final stage both remote branches were e3425c2; obtain the actual new
tip from Git. Preserve uncommitted intended edits and historical restoration
leftovers; never bulk-add results/ or reset/clean the repository.

## Historical pilot and identification context

The CPU/CUDA smokes and two-arm/two-seed2k pilot are complete. All four final
pilot evaluations had0/12joint and0all-agent-ball; only B1422 had1any-ball.
All7488updates passed finite guards. This was runtime sanity, not gain evidence.
Pilot reports: results/masac_v4_sync_pilot_diagnosis_20261105.

The early control1411@2k128-observation audit found moving clearance constant1
in full versus0static and peer7204/7296exact1, with other channels equal only
on that sample. All12development scenes have no external movers/conflict0.
That early sample does not describe late replay or historical full-policy
trajectories; the separately pushed next plan addresses only late-replay content.

Original40k launch was04:49 after protocol push eaf83e2; runtime two concurrent
workers used approximately1.5GiBVRAM and15–16steps/s each in the pilot. Old
worker/session identifiers are retired. Source/config remained frozen through
the final six-history verification. Full completed-study evidence and offline
plan pushed to both branches as d8ff5ad; later analysis has separate artifacts.

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
