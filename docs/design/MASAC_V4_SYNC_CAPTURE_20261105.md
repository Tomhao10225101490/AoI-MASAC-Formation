# B sync-capture: isolated potential-shaping pilot (2026-11-05)

Experiment family: `masac_v4_sync_capture_pilot_20261105`; one scientific change-set,
`sync_potential_v1`. This is an opt-in training mechanism, not a demonstrated
improvement, safety certificate, or novelty claim. The user explicitly authorized
B implementation and small validation on Windows on 2026-11-05. Large training,
Attention/L1, old paper-suite reruns, and checkpoint continuation are not queued.

## Diagnosis and hypothesis

The completed 80k-step paper suite ranks static (2/3 gates), full (1/3), zero
(0/3), with finals static 11/12,10/12,5/12; full 3/12,4/12,8/12; zero 1/12,0/12,1/12.
High **any-agent** ever-ball is not all agents reaching their balls, and even
all-agent ever-ball need not occur simultaneously or at speed below 2 m/s.
The new archive audit (`results/masac_v4_windows_audit_20261105/analysis.json`)
finds 108 final map episodes, 43 successes and 65 failures: 61 failures have
only some agents ever reaching a ball; only 4 have all agents ever reaching
their balls and still fail. Static's all-ball counts [11,10,5] equal its final
success counts. Thus the earlier pure-synchronization diagnosis is too strong.
The hypothesis is that a dense team signal emphasizing the worst normalized
distance/speed may improve worst-agent approach and joint capture. A short
pilot only tests implementation, finiteness and runtime, not this hypothesis.

## State, potential and exact reward change

For UAV i=1,...,N, let p_i,g_i,v_i be position, fixed goal and world velocity.
Let d_i=sqrt(||g_i-p_i||²+epsilon²)-epsilon and
w_i=sqrt(||v_i||²+epsilon²)-epsilon, epsilon=1e-8. Form the 2N dimensional
nonnegative vector q=(d_1/4,...,d_N/4,w_1/2,...,w_N/2). Define

```
M_tau(s) = tau log[(1/(2N)) sum_j exp(q_j(s)/tau)], tau=0.1;
Phi(s) = 1 / (1 + M_tau(s));  0 < Phi <= 1;
F_t = lambda [gamma * (1-done_t) * Phi(s_{t+1}) - Phi(s_t)];
r'_t = r_Path-C,t + F_t, lambda=0.5 fixed, gamma=0.99 (the existing SAC target).
```

The stable log-mean-exp emphasizes the largest normalized distance/speed,
with max(q)-tau*log(2N) <= M_tau <= max(q); it has no hard success decision.
Its derivatives are dM/dq_j=softmax(q/tau)_j and
dPhi/dq_j=-softmax(q/tau)_j/(1+M)^2, emphasizing the largest burdens.
The strict frozen capture predicate remains ||g_i-p_i||<4 AND ||v_i||<2
for every agent at the same step, latch=False and no collision.
Potential scales align with this predicate; Phi is neither an exact occupancy
fraction nor a guarantee that all strict thresholds hold. Training uses state
only to add one team scalar reward, without changing observation or dynamics.

This first B stage uses one fixed coefficient. No occupancy bonus, arrival-gap
term, living-cost change, new observation, map change, architecture change,
or time-varying curriculum is combined with it. A future time-dependent
coefficient must use F_t=gamma*lambda_{t+1}*Phi_{t+1}-lambda_t*Phi_t on an
explicit time-augmented state; lambda_t[gamma*Phi_next-Phi] does not telescope
when lambda changes. That further curriculum requires a separate change-set.

## Terminal, time limit, partial observation and limits of the argument

The frozen runner stores `model.push(..., done)`, not `done or truncated`.
Collision or success is true termination: its absorbing next potential is 0.
A horizon timeout is `truncated=True, done=False`: replay bootstraps the saved
next observation, so its next potential is retained. This baseline semantic is
preserved in both arms. It is a continuing-surrogate learning objective, and
its correctness as a finite-horizon task model is not established here.

For T transitions, the discounted sum of shaping is
`sum_t gamma^t F_t = -lambda Phi(s_0) + gamma^T lambda Phi(s_T)`.
The endpoint is zero at a true absorbing termination. At a timeout the endpoint
is not zero; the bootstrap transform `V'(s_T)=V(s_T)-lambda Phi(s_T)` cancels it.
Thus a raw truncated return without bootstrap is not generally shifted by an
action-independent constant. Changing timeout masks in only the B arm would
confound this experiment and is forbidden.

At true termination (or with the stated consistent bootstrap), trajectory
telescoping holds for arbitrary history-dependent joint policies and any fixed
initial-state distribution; partial observation does not invalidate this
algebra. Exact Markov value/Bellman equivalence additionally requires a suitable
state (including relevant history/time for that model). Observation-based
approximate critics, state aliasing, off-policy replay, clipping and optimization
prevent any claim of convergence, faster learning, finite-sample superiority or
safety. The legacy Path-C reward is heuristic; adding PBRS does not make all its
terms potential-based. No probe-quality causal claim follows from a B effect.

## Frozen evaluation and paired small protocol

Files `configs/masac_v4/sync_capture_{control,pilot}_20261105.json` use the same
static probes, cold initialization, seeds 1411/1422, maps, 2,000 steps/seed,
128 warmup, 128 batch, 4,096 replay, CPU, and frozen 4/2 latch=False. They differ
only in experiment identifiers and `sync_capture.enabled`. Capacity/cache mode
is identical in both arms. No Mac checkpoint is needed for cold pilots.
The pairing is by training seed, episode-index map-generation mechanism and
fixed development maps. It is not transition-by-transition map matching:
policy-dependent episode lengths can change map exposure at a fixed step.
All 12 development maps are evaluated at 1k and 2k; they are development maps,
not a sealed final test. `--smoke` instead uses 256 steps and two maps and must
be labelled smoke. Larger-budget claims require separately frozen budgets,
matched static/full/zero controls and adequate independent training seeds.

The separate `sync_capture_{control,pilot}_cuda_20261105.json` files change only
the execution device relative to their CPU counterparts, plus identifiers and
compute provenance. They require the measured report
`results/masac_v4_windows_audit_20261105/device_benchmark.json` to exist, a finite
CUDA throughput exceeding the fastest measured CPU thread setting, action and
risk-feature max error below 1e-4, and `torch.cuda.is_available()`. The runner
hashes this evidence into its manifest. The measured 3-replicate, 32-step
act/environment/cache/update benchmark reports 22.318 steps/s CUDA versus
2.549 steps/s best measured CPU (8.75x); it excludes full evaluation/checkpoint
overhead, so full-run throughput is reported separately after smoke. Original
CPU protocols and their defaults remain CPU. This is explicit pilot execution
on a benchmark-supported device, not a silent historical-protocol default change.

`evaluate()` constructs the original Path-C environment directly, never calls
`shape_reward`, and records `evaluation_shaping_enabled=False`. Each map/eval
streams success, any/all ever-ball, maximum simultaneous strict capture
occupancy, and explicit failure counts. `timeout_after_ball_any_only` and
`timeout_after_ball_all` prevent conflating any and all agents. Episodes record
base/shaped return separately; checkpoints include source/config hashes.

## Required validation and decisions

Unit checks cover disabled exact reward-object parity and identical seeded
environment rollouts, bounded/finite potential and worst-agent distance/speed
sensitivity, true-terminal telescoping, timeout boundary plus bootstrap
cancellation, strict capture thresholds, any/all failure labels, and a real
child-process lock conflict/release on the current operating system.
The parent runner executes tests and finite-loss training sanity before
claiming implementation validation. At this writing no pilot outcome is claimed.

Report both seeds, both arms, every eval and failure. Do not select the best
checkpoint to stand in for final performance, infer a gate from two-map smoke,
or rebrand a 2k pilot as evidence of a method improvement. Inspect sparse
all-ball and max-occupancy logs before proposing the next formula or budget.
