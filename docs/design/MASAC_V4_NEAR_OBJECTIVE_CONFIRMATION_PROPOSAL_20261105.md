# Next near-objective development study proposal — not registered/launched

This is the next ordered implementation/review task after the closed2k pilot.
It is a complete planning proposal, not permission to reuse the pilot runner's
fixed2k guard or to continue any weights. Final executable protocol/config/code
must be reviewed and pushed, and the concrete plan shown before launch.

Question: does replacing the old undifferenced near occupancy term by the fixed
discounted near potential improve final joint capture at a matched training
budget? Current evidence describes failures and validates execution, not cause
or gain. Both2k modes fail0/12; do not select seeds or parameters from that result.

Proposed new cold seeds1611/1622/1633; two arms, static/B-off, original near vs
the exact potential formula of the2k protocol. c=.4,R15,gamma.99 fixed; no
coefficient/temperature/evaluation-policy search.40k transitions/seed/arm,
warmup1k,batch128,replay100k,eval2k,checkpoint10k,log200,CUDA/Torch1thread,
two concurrent jobs, each advancing its three seeds sequentially.240k total
new transitions,234k updates. Estimate3–4hours subject to measured throughput,
roughly6–8GBnew local artifacts; audit disk/RAM/processes before launch.
No weights/replay upload. Do not increase concurrency beyond these two arms.

Unique scientific change: the training near-reward formula. Reuse all14 frozen
core sources; create a separate reviewed formal entry point (the pilot entry
remains frozen). Preserve ordinary Path-C environment/reward for deterministic
evaluation, simultaneous4m/2mps capture/unlatched, architecture/optimizer and
the same12development maps. Keep explicit per-seed reward-state reset, complete
transition audit, source/config hashes, evaluation-state guards and checkpoint
hashes. The formal entry must fix the pilot's single-seed logging assumption:
record the actual current training seed and reset per-seed counters. Tests must
exercise multiple seeds/episode resets and nonzero true-terminal/timeout paths
without spending the formal budget. No old run source/config may be overwritten.
The pilot also exposed an unlocked global-registry read/modify/write race;
the new entry must wrap registry updates in a shared local process lock and
test simultaneous index updates. Each output remains the recovery authority.

Primary endpoint: final40k successes/12 for every arm/seed, paired differences
and their mean; no best intermediate checkpoint or early stopping by effect.
Keep repeated maps as development evidence, training seeds as replicates. No
independent-binomial CI or significance conclusion from36correlated maps.
Separate absolute gate: at least8/12on every seed. Engineering planning rule,
to freeze before launch: mean paired fraction >=1/12, at least2strict positive
pairs, minimum pair >=-1/12, six complete finite unmodified histories. It is
planning confirmation only and must not replace the absolute gate.

Secondary: any/all eventual ball, all-individual capture and simultaneouscapture,
collision flags/nonexclusive failure counts, capture-entry/exit and sampled
near-stall occupancy with actual denominators/censoring. Reward components and
nonzero intervention/timeout coverage are implementation checks. Reduced stall
without higher success is not success; positive endpoint supports the whole
formula replacement but not separate removal/PBRS mechanisms. Mixed/negative
seeds remain in all tables. No old40k B cohort or warm-start Standard-C as fair
same-budget comparator.

Stop/diagnose numerical error, stale state, hash mismatch or incomplete artifact;
no silent restart, source/config edit, new seed, horizon/budget change or automatic
extension. On normal completion run an independent analyzer of all six histories,
then update paper/math/frontier/state and push both branches. Sealed tests,
Attention/L1, historical whole-suite reruns and cloud training remain excluded.
