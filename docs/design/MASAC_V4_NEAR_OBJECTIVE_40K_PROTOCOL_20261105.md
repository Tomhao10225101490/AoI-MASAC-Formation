# New fixed40k near-objective matched development study

Date2026-11-05. Final pre-execution specification; launch only after root review,
required tests, a visible concrete plan and push of protocol/config/source/analyzer
to both branches. This registers a new cold-start study under the user's continuing
research authorization; it does not extend the completed B or2k histories.
The previous proposal/pilot are known development context, not efficacy evidence.

## Question, formula and limits

Does replacing Path-C's undifferenced near occupancy reward by the fixed
discounted near-potential difference improve final simultaneous capture at a
matched40k budget? This is a training-objective hypothesis, not predictive-channel
utility or a new PBRS theorem. Both arms use static probes and disable B.

For N=3, g(d)=clip(1-d/15,0,1), Phi(s)=.4mean_i g(||g_i-p_i||), gamma=.99.
The original scalar Path-C reward includes Phi(s_next physical). The two modes:

    original:  r_train = r_PathC                         (scalar object unchanged)
    potential: r0 = r_PathC - Phi(s_next physical)
               r_train = r0 + gamma*Phi_eff(s_next) - Phi(s)

Phi_eff=0 at true termination (success or collision), otherwise observed Phi.
True termination takes precedence if both terminated/truncated flags are true.
Timeout alone retains Phi_next and the original done=False bootstrap mask.
Only the added potential absorbs; the removed near term always uses physical
post-step geometry, including terminal transitions. Keep the exact existing
multiply-per-agent-then-mean near arithmetic. Analytic potential range[0,.4],
correction relative to Path-C[-.404,0]nonterminal/[-.8,0]terminal, allowing1e-12
roundoff in checks. No clipping of original scalar reward in control.

This intentionally changes the old Path-C soft objective. PBRS policy-order
conditions apply relative to r0, not Path-C. At a finite timeout the added
potential sum retains gamma^T Phi_T; only a correctly transformed continuation
value cancels it. Partial observations do not make Phi a function of a local
observation or guarantee an exact neural critic shift. Equal entropy/information
classes and valid boundary conditions matter. Fixed-position near flow changes
from+Phi to-.01Phi, which does not order full policies or prove faster success.
The whole intervention combines removing occupancy reward and adding potential
credit assignment; gains would not separate those mechanisms. Ordinary progress,
capture changes, living/acceleration/terminal terms remain empirical objectives.
No sparse-success equivalence, safety, asymptotic-stability or novelty claim.
See `../MATH_NEAR_OBJECTIVE_REPLACEMENT_20261105.md` and primary-source gap notes.

## Fixed execution and population

Two arms, mode original/potential. New training seeds1611,1622,1633, cold each;
40,000transitions/seed/arm,1,000warmup,39,000updates, batch128, replay100,000.
Total240,000transitions/234,000updates. CUDA explicit with the previously measured
full-workload evidence; Torch1thread per worker, at most four independent history
workers. Queue order is original1611,potential1611,original1622,potential1622,
original1633,potential1633. Each child owns one fresh seed and one output.
Fill an empty slot with the next queued job; no extra seed or continuation.
The old CPU protocol/default is untouched. Estimate3–4hours/6–8GBnew local
snapshots, subject to measured throughput and prelaunch resource check.
Before registration the user explicitly requested fuller compute utilization.
The matched static act/environment/cached-replay/update benchmark measured
aggregate30.9038/32.6480/33.9149steps/s for2/3/4workers:four improved9.7%over two,
with0nonfinite values. GPU point samples81/89/93%and1615/2048/2422MiB are not
sustained-utilization guarantees. Engineering timing excludes eval/reference
audits/snapshots and uses early replay; it does not establish training outcomes.
Freeze max4 on this evidence; do not adapt concurrency based on science results.

Static probe/pool encoder, same architecture/optimizer/action limits, frozen
Path-C environment and4m/2mps unlatched strict capture. Evaluate deterministically
on exactly830001..830012 L0 development maps every2k; checkpoints10k, log200.
Evaluation always uses original Path-C; the near callback never runs in evaluation.
Old14 core sources, pure adapter and pilot entry remain unchanged. No checkpoint
or replay load. Episode-index training map seeds are paired, but policy-dependent
episode lengths may change exposure at a given transition; samples are not
identical paired transitions. CUDA bitwise training/resume equivalence is unproved.

Entry `scripts/train_masac_v4_near_objective_study.py`; runtime
`uav_planning/v4_near_study_runtime.py`; pure adapter `v4_near_objective.py`.
Configs `configs/masac_v4/near_objective_40k_{original,potential}_cuda_20261105.json`.
Coordinator `scripts/launch_near_objective_study.py` owns new root
`results/masac_v4_near_objective_40k_sharded_cuda_20261105`. Each child output is
`runs/{original,potential}_{1611,1622,1633}` beneath it, with its own truthful
single-seed run manifest/status/completion/error and inner maneuver_static_seed
directory. Root study_manifest.json fixes all six identities/relative paths,
both full3seed configs,20source hashes, benchmark evidence and max4 before
spawning. Root status tracks exact child PIDs and own statuses; root completion
requires all six child returncodes0 and own completions true. It never fabricates
a three-seed trainer completion from a single-seed worker. CLI seed selection
only selects a registered cold seed; raw paired configs keep all three seeds.
Entry validates both configs differ only by intervention mode and nominal IDs;
both config hashes are recorded. Freeze20 declared sources including this
protocol, coordinator and independent decision analyzer, copied in each child's source_snapshot and
hashed in run/snapshot manifests. No source/config changes during these runs.

Training runtime identifies the actual seed from its owned directory, resets
per-seed counters and reinitializes potential/component memory each episode.
Each seed stores exactly40k records in its own near_reward_transitions.jsonl,
schema_version1, training_seed/training_step(1..40000), episode/map_seed,
original base reward, physical/effective potentials, removed/potential/correction
terms, replay reward/done and flags. base_return remains original Path-C;
shaping_return means near-replacement correction, not B shaping. Keep all logs.

Each evaluation checks training state isolation and records a durable per-seed
evaluation_state_checks.jsonl plus streamed PASS marker,20checks at2k..40k.
Serialize global-registry read/modify/write with the shared near_study_registry
process lock (maximum wait30s). Own output manifest/status/completion/error remain
authority; unrelated legacy writers are not made lock-aware. Do not launch them.
Per-output run locks and create-fresh behavior prevent overwrite. Checkpoint
and replay hashes are retained locally; neither binary is uploaded to Git.
Runtime/callback state is not an exact CUDA resume implementation.

## Frozen endpoints, decisions and failure handling

Primary endpoint: final40k successes/12 on every history and paired difference
D_k=S_potential,k-S_original,k. All three pairs are reported, including negative.
Mean fraction gain=(sum_k D_k)/36. Original engineering planning rule freezes:

1. sum_k D_k >=3 (equivalent to mean fraction gain>=1/12);
2. at least two D_k>0;
3. min_k D_k>=-1;
4. all six histories complete, finite, unmodified, with audited source/config,
   final checkpoint/replay and full reward-transition integrity.

ADVANCEMENT_RULE_MET supports planning only, not significance/launch permission.
Separate absolute task gate: each seed at least8/12, all three required for an
arm pass. It is not replaced by the relative rule. No peak/intermediate endpoint
selection, model/evaluation-policy changes, effect-based stopping, seed replacement
or automatic additional budget. Three training-seed pairs are replication units;
reused maps/frames are correlated. No independent-binomial CI on36maps.

Secondary descriptions: success, any/all eventual ball, all-individual strict
capture and simultaneous capture, nonexclusive collision/failure counts, capture
entries/exits and sampled near-stall occupancy with actual denominators/censoring.
Path efficiency remains success-conditional. Reduced stall without higher final
success is not improvement. Report reward-intervention/timeout coverage honestly;
coverage does not change the primary decision threshold. Mixed/negative results
are retained. Do not compare fairly to the old B cohort, Mac80k or warm-start
Standard-C floor as if budget/initialization/exposure matched.

Independent analyzer `scripts/analyze_near_objective_study.py --study <root>` requires all six
final histories and full source/config/snapshot/replay SHA validation before
decision_ready. It checks40k ordered reward conversions/done masks per seed,
195finite sampled progress rows plus the per-update guard,20state-isolation
records and final12map records. A skip-heavy-hashes diagnostic cannot decide.
Inconsistent data/source/hash produces explicit failure, not omission or imputation.
Fresh analysis outputs stream per-history checks; completion written last with
artifact hashes. Any analysis bug is archived and corrected separately, with
rules frozen; do not change training or overwrite the first output.

Stop/save/diagnose training numerical errors, stale state, lock timeout, hash
mismatch or incomplete artifact. No silent restart/restack or best checkpoint
substitution. On child failure the coordinator stops submitting queued jobs,
waits for already-running registered siblings to save/finish and records failure;
it neither kills unrelated processes nor retries. Interrupted coordinator state
lists owned PIDs; inspect live processes before any recovery decision. Existing
root/child directories are never overwritten or automatically resumed.
On normal completion independently analyze all six histories,
update paper/math/frontier/state/ledger and push selected files to both branches.
Sealed test seeds2026110501..2026110512 remain unopened; no Attention/L1,
historical full-suite rerun, cloud job or weights/replay/secret upload.
