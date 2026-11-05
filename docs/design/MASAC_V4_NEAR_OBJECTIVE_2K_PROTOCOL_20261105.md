# New near-objective replacement: fixed two-arm2k runtime pilot

2026-11-05. Register and push reviewed code/config/protocol before execution.
This is a new cold-start pilot after the final40k study and three offline audits.
It does not reopen or extend any completed history. No gain decision, sealed-map
test, Attention/L1, old-suite rerun, warm-start, cloud job or subsequent40k launch
is authorized by this protocol. Two concurrent local CUDA jobs are the complete
runtime scope. The user's persistent research instruction authorizes this
problem-hypothesis-minimal-experiment cycle; the concrete plan was shown first.

## Question and exact intervention

Saved final traces reveal both approach shortfall and nonabsorbing capture.
Near-stall occupancy alone is insufficient: B1522has0/12success despite the
smallest occupancy fraction and58capture entries/53exits. This motivates a
controlled training-objective intervention, without treating diagnosis as cause.

Let g(d)=clip(1-d/R,0,1), R=15m, c=.4, gamma=.99 and
Phi(s)=c/N sum_i g(||g_i-p_i||), N=3, 0<=Phi<=c. The old Path-C scalar
team-mean reward includes exactly Phi(s_next) as its near term. Control mode
`original` returns that scalar object unchanged. Treatment mode `potential` is

    r0 = r_PathC - Phi(s_next physical)
    r_new = r0 + gamma*(0 if terminated else Phi(s_next physical)) - Phi(s).

True termination (collision or success) absorbs the added potential only; the
removed original near term always uses actual post-step geometry. If terminal
and truncated are both true, terminal takes precedence. A timeout alone retains
the next potential and the existing done=False bootstrap mask. The raw finite
sum has boundary -Phi(s0)+gamma^T Phi(sT) at a timeout; cancellation requires
the corresponding transformed continuation value. No exact neural SAC/Adam
trajectory equivalence is claimed. At a nonterminal fixed state the added term
is -(1-gamma)Phi, whereas the old near term was +Phi. This removes that local
occupancy stream, not all empirical reward biases or the retention problem.

**This deliberately changes the old Path-C soft objective.** PBRS policy-order
scope is relative to r0, not the old full reward. Ordinary progress, capture
entry/exit, living, acceleration and terminal/collision terms remain empirical;
neither reward is proven equivalent to sparse joint-success maximization.
The intervention combines removal of the occupancy reward and addition of
potential credit assignment. A later gain could support the whole replacement,
not separate these two effects or establish a new PBRS theorem.

## Fixed population and execution

Both arms: static probes, pool backbone, B disabled, new cold seed1644,
2,000transitions,1,000warmup, batch128, replay10,000, gamma.99, CUDA,
torch_threads1, log100, eval/checkpoint2000. Same architecture/optimizer/action
transform, original environment/capture4m/2mps/unlatched, same training-map
generator/families and12original development maps. Episode lengths may diverge,
so map exposure by transition is not identical. No checkpoint/replay is loaded.

Configs: `configs/masac_v4/near_objective_2k_{original,potential}_cuda_20261105.json`.
Outputs: `results/masac_v4_near_objective_2k_{original,potential}_cuda_20261105`.
Entry point: `scripts/train_masac_v4_near_objective.py`; pure adapter:
`uav_planning/v4_near_objective.py`. Original14 sources stay byte-for-byte
unchanged and are compared to the completed40k manifest before launch. The new
entry point explicitly hooks only training episode creation/reward logging;
the original evaluation constructs its own environment and never calls the
training reward callback. Evaluation remains deterministic original Path-C.

Manifests hash all17sources (14old +entry/adapter/this protocol), configs and
state capture. They explicitly name the intervention and logging semantics:
`base_return` is original Path-C; `shaping_return` is near replacement correction,
not B shaping. Episode metrics also retain removed-near and potential-difference
totals. Progress reports the current base/removed/potential/correction/replay
reward; all original seed/progress/eval/any/all/failures/checkpoint lines remain.
All2,000training reward transitions are additionally recorded in a separate
JSONL, with analytic correction bounds[-.404,0] for nonterminals and[-.8,0]
for true terminals (roundoff tolerance1e-12). An evaluation wrapper checks
that training environment position/velocity, previous potential and callback
counter remain unchanged by evaluation. It does not modify the evaluated policy.
Per-output status/manifest/completion/error is authoritative; create-fresh and
locks prevent overwrite. Exact CUDA resume remains unsupported/unvalidated.

## Runtime decision and stopping

Before launch: review formula/callback boundary, run algebraic tests and syntax
checks, then push. Required numerical properties include original object/value
preservation, true-terminal telescope, timeout boundary/bootstrap transformation,
stationary-state flow and finite/shape/type rejection. During each2k run require
finite per-update diagnostics, exactly1,000updates, no stale reward-state across
episode resets/evaluation, final12map metrics and final checkpoint/replay hashes.
Stop/diagnose any error; do not relaunch into the same directory or change rules.

Report every finalsuccess/any/all/collision/failure count and reward component,
including zero/negative results. Performance is **not** a pass criterion: one
seed/2k is numerical/runtime sanity only. Both arms completing finite unmodified
execution is the runtime milestone; no statistical/generalization/causal gain
claim follows. No intermediate selection, automatic extra seed, budget extension
or sealed evaluation. A future new40k×3study would require its own visible
concrete plan and pushed protocol, with fresh seeds and final endpoint.

Primary literature boundaries are in RELATED_WORK/FRONTIER_GAP: Vasan2024 already
distinguishes low-speed arrival/time on goal; reward/termination/reset changes
are not isolated coefficient evidence. Continuing SAC/stability sources do not
justify increasing entropy, switching evaluation or claiming stabilization here.
