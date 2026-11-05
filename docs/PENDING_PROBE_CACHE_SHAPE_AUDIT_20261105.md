# Pending optimization: action-time probe reuse shape guard

Date2026-11-05. Read-only source finding during the registered near-objective
40k study. **Do not modify its20frozen sources or configurations.** This note
does not authorize a rerun, budget extension, new probe arm or current-source fix.

## Established source facts

`ManeuverMASAC.act` in `uav_planning/v4_maneuver_masac.py` adds a batch axis to
the current observation `(N,162)`, obtains probes `(1,N,19,11)`, and stores
`risk[0]` in `_last_risk`, shape `(N,19,11)` when `cache_risk=True`.

The implicit-risk branch of `push` tests:

```python
np.asarray(o).shape == self._last_risk.shape
```

In the inspected actor/trainer path these shapes are `(N,162)` and `(N,19,11)`.
They differ in rank for everyN, so the reuse branch cannot be selected.
The trainer calls `model.push(obs,action,reward,next_obs,done)` without an
explicit risk argument, after `model.act(obs)` outside warmup. Thus this path
recomputes the current probes in `_risk_numpy(o)`, then computes next probes.
This finding is from source, not a new model/environment/profile execution.

**The replay cache itself remains active.** `risk_feat` and `next_risk_feat`
are stored by `push` and supplied to actor/critic minibatch updates. The problem
is specifically reusing the action-time current probe, not the batch cache.
The three provider calls along post-warmup `act`/implicit `push` are current
at act, current again at push, and next at push; logging/evaluation can have
additional calls. No wall-time cost or speedup has been measured for a fix.

The active study uses a deterministic static provider in every arm. The repeated
calculation uses the same specified function and input, so this does not identify
a scientific outcome cause or establish that saved rewards/actions are wrong.
No new bitwise equality, performance or resume validation is claimed by this note.
Existing CPU/CUDA/concurrency measurements describe the implementation actually
run, including its recomputation; their execution provenance is not replaced.

## Future change requirements

After the current six-history final analysis, a separately reviewed implementation
may reuse action-time probes, but shape compatibility alone cannot prove that
they belong to the same observation. Preserve an observation identity/explicit
act-to-push contract, warmup/no-act fallback, explicit-risk override, clearing,
dtype/device behavior and evaluation isolation. Caching learned latent encoder
outputs would be a different and generally invalid optimization as weights change;
only the deterministic raw19x11provider output is at issue here.

For stochastic shuffle, recomputation can draw a different direction permutation
from the one used to choose the behavior action. A later shuffle protocol must
declare whether it stores the action-time augmentation or intentionally resamples
it; those are different algorithm contracts. No shuffle study is launched here.

Before changing a future default, test observation ownership, warmup, evaluation
interleaving, explicit arguments and random-generator accounting; then benchmark
the same real end-to-end act/environment/replay/update workload and report
uncertainty across repeated timings. Preserve this completed study's original
sources and snapshots. Do not apply an unreviewed hot fix or silently relabel its
timing as a different implementation's performance.
