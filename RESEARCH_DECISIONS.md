# Research decisions

## 2026-11-02 — explicit user-authorized reset
The user froze MASAC multi-UAV 3D path planning and one maneuverability-aware candidate.
Earlier single-UAV DC-RRT direction is superseded, its files/results retained.
Old AGENTS/README copied into archive/scope_before_masac_20261102 before changes.
This is a targeted document backup, not a complete repository/data backup. No old files deleted.

Local directory remains aoi-masac-formation: parent directory is outside allowed write roots.
GitHub rename not completed: available connector lacks rename capability; gh not installed.
Remote unchanged to avoid pointing at an unverified repository. Branch protection not configured.
CI exists locally only until these files are committed/pushed; it cannot police scientific meaning.

Audit: algos/masac.py provides decentralized actors, centralized per-agent twin critics,
automatic temperature, replay and gradient updates. Existing formation environment must not
be relabelled as independent-goal planning. New environment and matched experiments remain pending.

## Execution continuation
New independent-goal environment and matched observation-feature pilot implemented.
A goal-freezing acceleration violation was found and fixed before the v2 comparison.
Pre-fix data retained and excluded. This is feature-level instrumentation, not a validated
novel safety mechanism. No change to the frozen MASAC research direction.

## V3 design before training

User rejected the simplistic sphere-map/unstructured-network pilot. Frozen V3 config and design precede new runs. Parameter sharing/set pooling addresses 3–6 agent interfaces; it is not a novelty claim. Changed baseline from per-agent scalar Q to centralized twin team Q and mean-team entropy; do not pool V2/V3 results. No reward added specifically for the candidate. Preview probes observe only the same nearby buildings as the policy; no full-map privileged information is slipped into the candidate.
14 tests pass (5 V3, 9 previous actor/dynamics tests). V3 smoke standard and candidate each performed384 real update rounds, saved checkpoints/replay; final 2-map validation success0/2 each. A new baseline-only 3-seed20000-step batch is running, with live log and status JSON. No candidate advantage claimed.

## V3 controlled budget diagnosis — 2026-11-02 continuation

The original three-seed 20,000-step batch is now complete; each final development validation is 0/6 and all failures are retained. Final parent audits show 19,000 real updates per seed. Raw audit: `results/masac_v3_diagnosis_20261102/diagnosis.json`.

Freeze `docs/design/MASAC_V3_DIAGNOSTIC_PLAN.md` before further training. Extend only Standard MASAC from each final checkpoint/replay to 100,000 total steps, seeds 411/422/433 sequentially. Preserve the same environment, rewards, architecture, optimizer state and RNG streams; no new warmup. This isolates training budget and is exploratory development, not confirmation or a capacity change. No four-arm training until baseline navigation is established. Exact restore is checked against an uninterrupted next action, transition and SAC update, including bitwise-equal Actor/Critic weights and all saved RNG streams.

The paired amax probe distinguishes control reserves with unchanged current distance and CV prediction; this validates the constructed diagnostic signal only. Static overhead witnesses check geometry only. Neither is learned-policy effectiveness, multi-UAV joint feasibility, safety certification or novelty evidence. The common-grid map-family limitation and the two-second preview being shorter than full-speed stopping time are explicitly retained in the design.

## V3 paired longer-budget continuation — 2026-11-03

The fixed goal-distance curriculum completed at 100,000 environment steps for each of seeds 411/422/433. Final full-task development validation was 0/6 for every seed, as for the original Standard MASAC history. The final replay contains individual goal-capture states, mostly from the short-task stage; the full-task stage has only 17/0/1 correlated pre-action capture states for the three seeds. See `docs/MASAC_V3_CURRICULUM_RESULTS_20261103.md` and saved audits; these observations do not establish navigation or a curriculum benefit.

The user requested increased training. Freeze `docs/design/MASAC_V3_PAIRED_300K_20261103.md` and `configs/masac_v3/paired_300k_protocol.json`: continue both Standard MASAC histories, each with all three seeds, from their own complete 100,000-step checkpoint and replay to 300,000 total steps, without new warmup or changing the network, reward, physics, optimizer or update rule. This is a matched-budget development diagnosis, not a claim that more training fixes the problem. The original 100,000-transition FIFO will replace the short-stage data during continued full-task sampling; document this distribution change. Source/parent preflight and restoration tests precede launch. No candidate mechanism arm is queued before baseline navigation is established.

The new run is `results/masac_v3_paired_300k_20261103`; six runs proceed sequentially with 50,000-step aligned checkpoint/replay snapshots. The active pointer and run's own status, error and completion markers are authoritative for its live state. A single recovery shard has a separate completion marker and must not be represented as a completed six-run batch.

### Paired 300k result and offline diagnosis

The paired batch subsequently completed all six histories; each reached 300,000 environment steps and 299,000 SAC updates. Every final six-map development evaluation remained 0/6; all 100k–300k periodic development evaluations were also 0/6. Completed training episodes in the extension total 6,155, with no team successes. See `docs/MASAC_V3_PAIRED_300K_RESULTS_20261103.md` and the independent evaluation, replay, collision, timeout, learning and fixed-sample TD audit JSONs in the paired run directory. These are development results and do not establish convergence, safety, novelty or a curriculum advantage.

On the saved 300k replay, the fixed offline TD audit found terminal transitions rare and associated with larger one-step TD residuals than the sampled nonterminal transitions; progress-log samples also show frequent critic gradient clipping. These are descriptive findings, not proven failure causes. A future critic-sampling diagnosis, if separately frozen and justified, must preserve matched budgets and control the actor/temperature batch and the nonlinear effect of clipping. The current paired run is complete; no maneuverability feature arm has been queued.
