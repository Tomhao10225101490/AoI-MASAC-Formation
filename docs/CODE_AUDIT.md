# Research and code audit

Audit date: 2026-10-31. Baseline commit: `3bffd89a5f084446729f826c06df19e8e3baa6b7`. This report covers the legacy environment and off-policy learners, plus inherited relay/graph changes. It is not a certification of the proposed future algorithm. Source names below are stable evidence anchors; line numbers may move during implementation.

## Provenance and verification

The working tree initially contained uncommitted changes to MASAC, relay/AoI handling and tests, and untracked graph-policy/server-protocol files. These predate this audit; `results/upgrade_audit/preexisting.patch` preserves the original tracked diff. The graph module must not be attributed to this audit. The untouched baseline passed 55 tests; the inherited working tree passed 83 tests. This audit added seven mathematical/regression tests in `tests/test_audit_invariants.py`; all seven passed after the parent engineer fixed the covariance and reward defects reported below. Command: `.venv/bin/python -m pytest tests/test_audit_invariants.py -q` (exit 0).

Sources actually inspected include `algos/{masac,networks,buffer,common}.py`, `envs/{formation_env,belief,channel,scenario}.py`, `train.py`, `evaluate.py`, `utils/rollout.py`, `scripts/voi_probe.py`, existing tests and experiment configurations. Resume implementation is being developed and verified separately; its acceptance test is not replaced by this report.

## Findings by priority

### P0 — task-value actor gradient canceled in the original regularized objective

Original `algos/masac.py` centers Q by its differentiable batch mean, divides by the batch standard deviation, then averages the centered values in the policy loss when `action_l2 > 0`. The batch average and its actor gradient are zero. Entropy and action regularization remain, so old zero-residual experiments are not conclusive evidence against useful learned correction.

**Status:** fixed in the inherited working tree before this execution. `sac_actor_loss` now retains the uncentered Q term. `tests/test_actor_gradient.py` checks constant and heterogeneous Q slopes, with and without L2. **Required experimental action:** rerun old negative-result comparisons under corrected code, with equal environment steps and fixed evaluation conditions. Preserve the old experiments as legacy evidence; do not overwrite them or silently reinterpret them as corrected results.

### P1 — covariance clipping violated positive semidefiniteness

`LeaderBelief._clip_covariance` originally clipped matrix entries independently. A reachable outage with `dt=.2`, measurement variances `(1,.25,.01)`, process variances `(.25,1,.01)`, and cap 22500 produced a negative eigenvalue at the 132nd prediction: minimum eigenvalue −1.651356, with position/velocity block `[[22500,1735.8],[1735.8,132.25]]`. A matrix with a negative eigenvalue is not a covariance; downstream standard-deviation features then lack their stated interpretation.

**Status:** reported to parent and fixed during this audit with diagonal congruence scaling `P <- D P D`, which preserves PSD for PSD input. The new 200-step outage test checks eigenvalues and diagonal cap on every step. This preserves admissibility, not calibration: bounded covariance remains a heuristic approximation to uncertainty.

### P1 — successful-completion bonus disagreed with success metric after a collision

With `end_on_collision=False`, `step()` rejects success after any earlier collision, but `_rewards()` formerly checked only collision on the current step. A previously crashed team reaching the final formation could receive the success bonus repeatedly until timeout. An isolated reward reproduction gave `[100,100,100]` despite `_hit_obstacle_ever=True`.

**Status:** reported and fixed by parent; `_rewards()` now includes both collision-history flags. `test_prior_collision_cannot_receive_success_bonus` passes. No old completion-rate claim should be inferred from reward alone.

### P1 — perfect-trained observation could leak truth on a lossy distance channel

Parent independently found `evaluate.resolve_test_observation` treated `p0=1` as lossless, even though distance attenuation and blockage still reduce delivery. Auto evaluation could therefore expose the true leader pose to a perfect-trained baseline.

**Status:** parent fixed the auto condition so every distance-channel evaluation is treated as potentially lossy. Explicit `--obs-mode perfect` remains an intentional oracle. This must be clearly labeled in reports and never pooled with deployable policies.

### P1 — resource and information fairness are not yet established for a new relay method

`beacon` adds a neighbor transmission without suppressing the leader; `tdma` replaces a leader transmission; `free` grants multiple orthogonal relay transmissions. These use different communication resources. Transmission counts alone omit packet length, PHY rate, control exchange, guard time and retransmissions, so they are not measured airtime or energy. The code now says so in `communication_counters()`.

`freshest` selects using all agents' current ages; `_leader_slot()` with an age gate also queries all links. This is a centralized/oracle-age reference unless an explicit, charged age exchange protocol supplies those values. `oracle_team_aoi=False` correctly hides team-min age in actor observations by default. Do not call an oracle scheduler decentralized merely because the flight actors are local.

**Open:** total-airtime gate and physically specified resource accounting. Free-all relay is a resource-privileged reference, not a mathematically established performance upper bound. Its observed mean can be worse due to trajectory-dependent stochasticity and estimator effects.

### P1 — graph modules are not an implemented learning system

`algos/graph_policy.py` provides masked local attention/GRU forward computation and a centralized set-value module. It has no delivered-message adapter, recurrent rollout storage, learner, or communication action integration. `experiments/server_protocol.yaml` correctly labels the prospective graph/PPO methods planned. Forward-pass tests establish shape/mask behavior, not training correctness, decentralized information access or scalability of the physical environment.

**Open:** method selection after overlap analysis, then complete integration. Do not report the planned graph method as trained.

### P1 — critic input is privileged but not a complete Markov state

`global_state()` includes physical poses, geometry and summary belief information. It omits in-flight relay packets, transmission phase/counters, current leader maneuver process state and full filter covariance. Two internal states can share the same critic tensor and have different next-state laws, especially under delayed relay. This is not direct actor leakage, but the description “full global Markov state” would be inaccurate.

**Open:** describe this as a privileged state representation, or add the relevant latent state for a new formulation. A recurrent critic is an approximation, not proof that the state is sufficient. No single-agent SAC convergence theorem transfers automatically to this partially observed, concurrently learned multi-agent setting.

### P2 — current uncertainty is a model feature, not a calibrated posterior

`_write_measurement()` copies exact leader position, velocity and heading; it draws no measurement noise. `LeaderBelief.reset_from_measurement()` then assigns configured nonzero measurement covariance. There is no Kalman gain or measurement fusion step. The implementation is a constant-velocity predictor with covariance propagation and reset. NLOS inflates process noise using a known building map, without a fitted stochastic dynamics model.

**Open:** call it a model-based belief predictor and evaluate calibration separately (prediction error versus uncertainty, coverage). Noise injection would change the environment and requires matched reruns; do not introduce it silently during a numerical fix. `slot_std()` is first-order propagation through heading rotation, whose approximation becomes weaker for large heading variance.

### P2 — evaluation random streams and data split need a frozen protocol

`reset(seed)` creates separate leader-maneuver and relay streams, so the leader's maneuver draws are protected from follower radio choices. However, the main generator still serves map generation, wind, initialization, wind-estimate noise and direct reception draws. TDMA skips some direct draws, changing the subsequent random stream relative to beacon/off. Random relay selection and decoding also share one relay generator. Equal seed does not mean equal exogenous link uniforms across policies.

Paired episode comparisons remain legitimate paired samples when maps/initial conditions share seeds, but should not be described as common full channel traces. For variance-controlled counterfactuals, key channel draws by episode, tick, sender and receiver, and separate physical exogenous streams. Method-dependent distances must still change success probabilities.

Training seeds are `seed*100000+episode`; training probes use `10000+seed*1000+episode`; standalone evaluation defaults to 1000. These ranges are not enforced disjoint for arbitrary long runs. Fixed test manifests and a separate validation split are required before tuning. A seed overlap can be checked mechanically; none is asserted here for the short benchmark runs.

### P2 — scalability is currently bounded by layout and network construction

The canyon generator supplies exactly four follower offsets and `EnvConfig` enforces their length. Changing only `n_followers` in a canyon configuration is not a valid 10/20-UAV experiment. Transit geometry supports generated formation offsets, but does not represent the same street problem. MASAC constructs one actor and twin critic per agent with joint-action input; checkpoint dimensions are fixed. Graph forward support for variable masks does not remove these environment or checkpoint restrictions.

### P2 — replay/resume, checkpoint and performance limitations

At initial audit, checkpoints saved weights, optimizers, episode and step numbers, but no replay contents or RNG states, and training had no resume path. Deterministic evaluation roundtrip did not imply resumable training. A separate agent is implementing episode-boundary resume; accept it only after uninterrupted-versus-resumed trajectory/update checks. Mid-episode exact resume is a separate capability.

MASAC computes critic parameter gradients during the actor backward pass without stepping the critic a second time. These gradients are cleared at the next critic update, so this is avoidable work rather than observed update corruption. Each scalar loss converted to Python can synchronize accelerator execution; actor action transfer is already batched in `stack_to_numpy`. The device benchmark's assumed 1.6× throughput for two CPU workers is a heuristic, not a concurrent measurement.

## Formula-to-code checks

The SAC reference uses a clipped twin-Q bootstrap, reparameterized squashed policy and entropy adjustment. The audited implementation uses, per agent, `y_i = r_i + gamma*(1-terminal)*(min Qbar_i(s',a') - alpha_i*log pi_i(a_i'|o_i'))`; `MASAC.update()` matches this stated individual-entropy convention. Its actor objective is `mean(alpha_i*log pi_i - min Q_i) + lambda*mean(a_i²)`. This is a chosen multi-agent adaptation, not a claim that there is one canonical MASAC formula. The log-alpha surrogate detaches log probability; its gradient sign adjusts entropy toward the target. [SAC, Sections 4–5](https://arxiv.org/html/1812.05905v2).

`Normal.rsample()` provides reparameterization; the stable tanh Jacobian is summed over action coordinates. The new density test compares it to a direct change-of-variables computation and checks nonzero actor gradients. Double Q targets are frozen; next actions are sampled under no-grad from current policies. The Polyak update is checked numerically. The deterministic target regression verifies terminal reward-only targets and nonterminal bootstrap, including the entropy sign.

For the multi-agent actor update, only agent i's replay action slice is replaced by its reparameterized action. Other agents' replay actions are constants, so no teammate-actor gradient is introduced through that slice. Central critics use privileged state and joint actions; actor inference uses only local observation. This follows the separation motivated by CTDE, while stochastic entropy regularization remains this implementation's adaptation. [MADDPG, Section 4.1](https://arxiv.org/html/1706.02275v4).

Replay sampling is uniform with replacement over filled entries. Arrays are copied at insertion. The terminal mask passed by `train.py` is `terminated`, not `truncated`, preserving bootstrap on time-limit truncation. Residual policy actions are retained in replay while realized tracker-plus-residual actions drive dynamics; this is appropriate if Q is defined over the policy's residual command. Fixed route/slot controllers make some action dimensions irrelevant to physical transitions; their actors still consume compute and receive entropy terms. Remove irrelevant dimensions only in a versioned new model.

## AoI and relay semantics

With tick `k` measured after motion integration, a newly generated direct measurement received at that instant has age zero. Without an update, age increments by one. A relayed snapshot carries source generation tick `g`, predictor reference tick and delivery tick; accepted age is `k-g`, not zero. Older/equal source timestamps are rejected, and same-tick direct updates outrank old relays. This agrees with the source-timestamp definition of freshness; direct zero age relies on the explicit zero-delay direct-link abstraction. [AoI survey, Sections I–II](https://user.eng.umd.edu/~ulukus/papers/journal/aoi-survey.pdf).

Internal ages are uncapped; actor/critic normalized age features are capped. Relay payloads are copies, so later sender updates cannot mutate packets in flight. Incoming relays are delivered before new relay snapshots are scheduled, allowing store-and-forward across successive ticks. All successful relay payloads have one control-tick delay. Actual decoding and fresher-information adoption are different counters. Radio success is sampled at transmission geometry; receiver motion during propagation does not trigger a second channel draw.

`mean_aoi` is a per-link, per-control-tick mean after updates. Seconds are explicitly `dt * age`. `max_aoi` is the maximum sampled age over the episode, **not average peak AoI**. Arrival-event peak AoI is not yet implemented. `mean_outage_length` tracks direct-source outages, can include skipped TDMA opportunities, and includes right-censored final streaks; it is not a pure physical packet-loss burst statistic. Report it with this definition.

## Physical assumptions and required reporting boundaries

- Outer-loop kinematics bound speed, yaw rate and climb acceleration; no six-degree-of-freedom rotorcraft or flight-energy model is present.
- Buildings are finite vertical cylinders; other aircraft are collision spheres. Radio visibility uses line intersection, not diffraction or a calibrated ray-tracing PHY.
- Distance/exponential delivery is a stylized probability law with an NLOS multiplier; no explicit fading time process, interference, queueing, retransmission or MAC contention is implemented.
- Collision is tested at discrete positions. Do not claim continuous-time collision avoidance or formal safety guarantees.
- Nearby teammate/obstacle sensing is idealized and range-limited; leader truth is excluded from follower teammate slots. Building maps used for prediction are assumed known.
- All agents receive free initial leader synchronization. Cold-start operation requires a separate experiment.
- No energy comparison is scientifically available from action penalties alone. A physical energy model and units must precede such a figure.

## Acceptance gates still open

Complete learned graph/communication integration; fair radio budget and observable information protocol; independent train/validation/test manifests; corrected multi-seed learning evidence; physical/noise/generalization tests; event-based peak AoI; validated energy/delay/fading assumptions if claimed; resume acceptance and full post-integration suite. Passing current tests does not establish novelty, convergence, fairness or submission readiness.
