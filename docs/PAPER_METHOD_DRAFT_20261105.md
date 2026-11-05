# Method: observation-only control probes and team potential shaping

*Paper draft, 2026-11-05. The B 40k study (seeds1511/1522/1533) is COMPLETE: all six final histories and artifacts were verified. Its original advancement rule permits planning confirmation only; both arms fail the absolute task gate. A separate near-objective 40k study (fresh1611/1622/1633) was registered/pushed b4b3a1e before launch and is RUNNING. No final outcome for that study is asserted here.*

## Task, information and constrained motion

We consider three UAVs with independent, fixed goals \(g_i\). “Capture” denotes operational **simultaneous goal settling**: at some common step, every UAV must satisfy \(d_i=\|p_i-g_i\|_2<4\) m and \(\|v_i\|_2<2\) m/s, without collision. The predicate is unlatched; an agent that leaves its ball or accelerates above the threshold loses its captured status. There is no moving target, pursuit–evasion objective or prey-encirclement geometry.

The latent state \(s_t\) includes scene geometry, vehicle states, time and reward memory. Agent \(i\) receives a masked current observation \(o_{i,t}=h_i(s_t)\in\mathbb R^{162}\), containing ego variables and visible object slots. A parameter-shared actor executes locally; centralized critics receive \(O_t=(o_{1,t},\ldots,o_{N,t})\) and joint actions. Joint current observations do not necessarily recover the Markov state, particularly first-entry memory and time. We therefore describe the implemented observation-based critic without asserting an observation-space Bellman convergence result.

The L0 configuration requires nominal sensor range at least the world diagonal;
the evaluated configurations use400m. Visibility filters by distance and then
retains capped object lists. These experiments therefore do not establish
performance under severe range-limited sensing, occlusion or physical LiDAR.
Slot truncation and omitted episode memory still prevent an inference of full
observability from the long nominal range. Decentralized actor execution and a
demanding limited-information benchmark are distinct claims.

The actor uses a reparameterized diagonal Gaussian
\(z_i\sim\mathcal N(\mu_\theta(o_i),\operatorname{diag}\sigma_\theta^2(o_i))\).
For \(a_i=\tanh z_i\), the idealized dynamics are
\[
u_i=\frac{a_i}{\max(1,\|a_i\|_2)},\quad
v_{i,t+1}=\Pi_V(v_{i,t}+hAu_i),\quad
p_{i,t+1}=p_{i,t}+hv_{i,t+1},
\tag{1}
\]
where \(\Pi_V\) projects onto the speed ball, \(A=4\) m/s², \(V=10\) m/s and \(h=0.2\) s. Episodes terminate on success/collision or truncate at 350 steps. Equation (1) idealizes exact projection; the simulator uses a denominator regularizer, probe code uses a clamped denominator, and observation/integration arithmetic has finite precision. Mathematical bounds below exclude unquantified numerical residuals.

## Shared soft actor–critic and Path-C reward

We use the existing shared soft actor–critic learner [Haarnoja et al.](https://arxiv.org/abs/1801.01290), with the approximate learner and automatic-temperature context of [SAC applications](https://arxiv.org/abs/1812.05905). Let \(\bar\ell=N^{-1}\sum_i\log\pi_\theta(a_i\mid o_i)\), with coordinate log densities summed within each UAV. Both the scalar reward \(r=N^{-1}\sum_i r_i\) and entropy contribution use agent means. Twin critics minimize summed squared residuals against
\[
y=r+\gamma(1-\delta)
 [\min_k Q_{\bar\psi_k}(O',A')-\alpha\bar\ell'],\quad
 A'\sim\pi_\theta,\quad \gamma=0.99.
\tag{2}
\]
Replay stores the true-terminal flag \(\delta\), not the truncation flag. Timeout transitions therefore bootstrap even though their episodes reset, retaining the historical continuing-value approximation.

With \(\beta=\log\alpha\), the source-matched losses are
\[
J_\pi=\mathbb E[\operatorname{sg}(e^\beta)\bar\ell-\min_kQ_{\psi_k}(O,A)],
\quad
J_\beta=-\mathbb E[\beta(\operatorname{sg}(\bar\ell)-3)].
\tag{3}
\]
Here \(\operatorname{sg}\) stops gradients. The implemented differential-entropy target is **−3**: the temperature stationary condition is \(\mathbb E[\bar\ell]=3\). Exact tanh Jacobian correction defines this density before radial command projection and speed clipping. Actor/critic gradient norms are clipped at five; target updates use coefficient 0.005.

Let \(b_{i,t}=\mathbf1[d_{i,t}<4]\), \(c_{i,t}=b_{i,t}\mathbf1[\|v_{i,t}\|<2]\), and let \(e_{i,t}\) track previous ball entry. Write \(G_t\) for collision-free joint success and \(K_t\) for collision on transition \(t\). Path-C assigns
\[
\begin{aligned}
r_{i,t}={}&d_{i,t}-d_{i,t+1}-0.005-0.002\|\Delta v_{i,t}/h\|_2^2\\
&+0.4\,\mathrm{clip}(1-d_{i,t+1}/15,0,1)
 +3b_{i,t+1}(1-e_{i,t})\\
&+8(c_{i,t+1}-c_{i,t})+50G_t-20K_t .
\end{aligned}
\tag{4}
\]
Thus ball entry is rewarded once, whereas losing unlatched capture incurs −8. Path-C is an engineering reward, not policy-invariant shaping of sparse success. For example,
\[
\sum_{t=0}^{T-1}\gamma^t(d_t-d_{t+1})
=d_0-\gamma^{T-1}d_T-(1-\gamma)\sum_{t=1}^{T-1}\gamma^{t-1}d_t,
\]
and the near-goal term is added undifferenced.

## Observation-only control probes

Our representation uses zero, six axial and twelve face-diagonal commands: 18 unit directions plus zero. For each \(u_j\), Equation (1) generates a constant-command ego preview over \(H=2\) s, with ten updates and eleven sampled positions. Zero holds velocity; it is not braking. Visible peers and moving objects are extrapolated at their decoded current world velocities. No simulator future, reference path or peer-policy rollout enters the actor.

At each sample, signed clearances are computed to inflated static boxes, moving spheres, peer spheres and world bounds. Let \(m_j^q\) be the minimum across sample times for clearance family \(q\), \(c_{j,0}\) the minimum current clearance, and \(D=\|v\|^2/(2A)\) a stopping-distance surrogate. The 11-component feature is
\[
f_j=[u_j;\ C(m_j^{\rm st}/L),C(m_j^{\rm mv}/L),
 C(m_j^{\rm peer}/L),C(m_j^{\rm bd}/L);\
 T_j/H;\ \mathrm{clip}(D/L,0,1);\
 C((c_{j,0}-D)/L);\ \mathbf1[\min_qm_j^q>0]],
\tag{5}
\]
where \(L=40\) m and \(C(x)=\mathrm{clip}(x,-1,1)\). The first sampled nonpositive-clearance time \(T_j\) defaults to \(H\), conflating no hit with a hit exactly at the horizon.

A shared 11→64→64 probe MLP followed by mean/max pooling produces 128 features. Concatenation with the 384-dimensional pool SetEncoder gives 512 features; the actor and each critic have separate encoder parameters. Full/static/zero arms retain identical architecture. The object encoder consumes typed geometry/velocity tokens with explicit validity masks; padding contributes neither to masked means nor maxima. In the pool backbone, token-wise residual MLP blocks retain permutation invariance without attention. Probe pooling is likewise invariant to whole-token permutation, so a direction-alignment intervention must retain canonical direction coordinates while permuting scalar contents; permuting complete tokens would be vacuous. Static replaces moving/peer clearances by the neutral sentinel \(L\) before computing derived features, then zeros those two output channels; ego/static/boundary preview remains. Zero replaces all features, including directions, by constants. Neither contrast proves encoder-use causality.

The evaluated L0 maps contain no external moving obstacles. Missing moving channels are therefore constant full-versus-static fills, while peer channels can saturate. Consequently this mask contrast does not identify moving-obstacle prediction quality. Unlike preview coupled to action correction in [SCOPE](https://ijcai.org/proceedings/2026/7), these features do not replace or filter executed actions.

For fixed public model parameters \(\eta\), the provider is a deterministic map
\(\mathcal R_\eta=f_\eta(O)\). It supplies a physical inductive bias rather than
additional information conditional on its complete input:
\(I(Y;\mathcal R_\eta\mid O)=0\). A finite-budget benefit must therefore be
identified against information-, capacity- and budget-matched controls. If
capability or history is supplied only to the provider, the comparator must
also receive it to isolate representation from information. Static, neutral
constant fills and physically/directionally mismatched probes answer different
questions; neither encoder capacity nor training trajectories can be assumed
equivalent merely from equal tensor dimensions. The existing L0 suite has not
established this predictive-content advantage. See the
[contribution criteria](FRONTIER_CONTRIBUTION_CRITERIA_20261105.md).

## Fixed team potential shaping

B adds one training-only scalar to the static learner. Define \(\nu_\epsilon(x)=\sqrt{\|x\|^2+\epsilon^2}-\epsilon\), \(\epsilon=10^{-8}\), and
\[
q=[\nu_\epsilon(g_i-p_i)/4;\nu_\epsilon(v_i)/2]_{i=1}^N,\quad
M_\tau=\tau\log\!\left(\frac1{2N}\sum_j e^{q_j/\tau}\right),\quad
\Phi(s)=\frac1{1+M_\tau},\quad \tau=0.1.
\tag{6}
\]
Since \(q_j\ge0\), \(0<\Phi\le1\). For \(q_{\max}=\max_jq_j\), the standard log-mean-exp bounds give \(q_{\max}-\tau\log(2N)\le M_\tau\le q_{\max}\). Thus this potential softens the worst normalized distance/speed; it is not an exact capture predicate and contains no explicit arrival-time-gap term. With fixed \(\lambda=0.5\),
\[
F_t=\lambda[\gamma(1-\delta_t)\Phi(s_{t+1})-\Phi(s_t)],
\qquad r'_t=r_t+F_t.
\tag{7}
\]
B changes neither observations nor evaluation reward and is not a time-varying curriculum. Current team states compute this centralized training reward only; they provide no additional actor input or future information.

The replay reward in Equation (2) is \(r'_t\) when B is enabled and \(r_t\) otherwise.

For absorbing-zero true terminals, discounted shaping telescopes to \(-\lambda\Phi(s_0)\). At a timeout it retains \(\gamma^T\lambda\Phi(s_T)\), canceled only by the consistent bootstrap transformation \(V'=V-\lambda\Phi\). Standard entropy-regularized PBRS theory, including [Adamczyk et al., Theorem 5 and Remark 7](https://arxiv.org/html/2212.01174v2), supplies context rather than a new theorem. Exact policy-order preservation requires a common policy information class, initial distribution, entropy convention and valid terminal/continuation boundary. It concerns the Path-C soft objective, not sparse success. It does not imply equivalent neural critic, target-network or automatic-temperature learning trajectories; see the [scope derivation](MATH_PBRS_SCOPE_NOTE_20261105.md).

## Discrete model error and evidence scope

For a feasible initial velocity and a common timestep/speed ball, projection nonexpansiveness gives a bounded-acceleration peer's error relative to its initial-velocity CV model:
\[
E_{\rm peer}(k)\le\epsilon_p+kh\epsilon_v+
 \tfrac12Ah^2k(k+1).
\tag{8}
\]
With the same projection/direction, ego acceleration-magnitude error \(\delta A\) adds at most \(\tfrac12\delta A h^2k(k+1)\), plus initial errors. For a changing executed policy, a vector-control discrepancy bound is required; equal acceleration magnitudes do not imply equal directions. Reverse triangle inequality bounds peer-sphere clearance error by the two position errors. On the same peer set/grid, minima and normalized clipping preserve the corresponding Lipschitz bound; TTC and the positive flag do not.

At \(H=2\), \(h=0.2\), \(A=4\), the discrete term is 8.8 m; continuous \(AH^2/2=8\) m underestimates the semi-implicit witness. These standard perturbation bounds exclude unseen peers, mask changes, numerical residuals, early termination and inter-sample contact. They do not constitute joint safety certificates or the reachable-set/safety-filter construction studied in the [CC-HGCBF preprint](https://arxiv.org/abs/2610.41214).


## Training and evaluation contract

The separate post-hoc [saved-trajectory audit](FINAL_TRAJECTORY_DWELL_RESULTS_20261105.md)
describes all72 final episodes without executing a policy. It uses post-step
distance/speed occupancy, strict capture entries/exits and realized acceleration;
25samples at h=.2s represent5s occupancy count but4.8s endpoint span. Among65
never-ball agent-map records,39have such a near-stall run (4<=d<10,speed<.5),
and48end in that predicate. B/1522 nevertheless has the lowest near-stall frame
fraction,0/12joint successes and58capture entries/53exits. Approach and retention
must both remain visible. These correlated, policy-selected/censored traces do
not identify a causal failure mechanism. Formula-derived base reward components
have no saved cumulative evaluation-return reference, so validation is explicitly
UNAVAILABLE; raw discounted sums omit entropy and bootstrap.

The registered B study cold-starts both static arms at training seeds 1511, 1522 and 1533, with 40,000 transitions per run, 1,000 warmup steps, batch size 128 and replay capacity 100,000. Both arms use the same frozen optimizer, cache and benchmark-supported execution configuration; their scientific difference is the B enabled flag. Episode-index map generators are paired, but policy-dependent episode lengths can change training-map exposure at a given transition index. We do not describe training transitions as identical paired samples.

Deterministic evaluation uses the same 12 L0 development maps every 2,000 steps with shaping disabled. The fixed 40k checkpoint, rather than a selected best checkpoint, supplies each seed's endpoint. We separately report any-agent eventual ball entry, all-agent eventual entry and simultaneous strict capture: their quantifiers are respectively \(\exists i\exists t\), \(\forall i\exists t\) and \(\exists t\forall i\), with the speed predicate added for capture. Failures and collisions remain explicit; path efficiency is conditional on success. A separate 12-map namespace is reserved but remains sealed until the final artifacts and development decision are frozen. Training seeds, rather than repeatedly evaluated maps or checkpoints, are the replication units. The completed analysis additionally distinguishes all agents ever satisfying individual strict capture from simultaneous team capture; these events have different temporal quantifiers.

Equations (5)–(7) define the representation and finite-budget learning hypothesis. At the fixed 40k endpoint, control successes for seeds 1511/1522/1533 are 2/12, 1/12, 4/12; B successes are 6/12, 0/12, 5/12. The paired count differences are +4, −1, +1, with mean success-fraction difference 1/9 (11.111 percentage points). The original preregistered engineering rule is met, supporting planning of confirmation rather than an efficacy conclusion. Both arms pass zero of three absolute 8/12 seed gates. These are three training-seed pairs on repeatedly evaluated development maps; no significance, unseen-map, moving-obstacle benefit or new-theorem claim follows. Static+B does not identify predictive-content utility. Historical Mac 80k arms and the warm-started legacy Standard-C floor are separate cohorts, not fair 40k performance comparators. The [canonical final analysis](../results/masac_v4_sync_40k_analysis_20261105/analysis.md) records the completed endpoints; the sealed test set remains unevaluated.

## Separate near-objective replacement study (registered, running)

This experiment disables B and fixes static probes in both arms. To avoid
confusing it with the worst-distance/speed potential in Equation(6), denote
its **mean-distance** potential by
\[
\Psi(s)=\frac{0.4}{N}\sum_{i=1}^{N}
 \operatorname{clip}(1-\|p_i-g_i\|/15,0,1),\qquad 0\le\Psi\le0.4.
\tag{N1}
\]
The actual scalar implementation preserves the multiply-per-agent-then-mean
arithmetic of Equation(4), subject to floating-point roundoff. Define
\(\Psi^+_{t+1}\) from physical post-step geometry even at termination. The
original arm keeps the Path-C scalar reward unchanged. The replacement arm uses
\[
r^0_t=r^{\rm PathC}_t-\Psi^+_{t+1},\qquad
r^{\rm new}_t=r^0_t+\gamma(1-\delta_t)\Psi^+_{t+1}-\Psi(s_t).
\tag{N2}
\]
True termination sets only the added successor potential to zero. A timeout
alone retains it and retains Equation(2)'s original true-terminal replay mask.
If both flags are true, termination takes precedence. Thus the correction
relative to Path-C is \(-\Psi(s_t)-0.01\Psi^+_{t+1}\) on nonterminal transitions
and \(-\Psi(s_t)-\Psi^+_{t+1}\) on true terminals. Their analytic ranges are
\([-0.404,0]\) and \([-0.8,0]\), respectively, with numerical tolerance in the
runtime checks. This is not the coefficient0.5 B intervention.

For a complete sampled episode of lengthT, writingG for discounted **reward
sums without entropy or bootstrap**, direct telescoping gives
\[
G_{\rm new}-G_{\rm PathC}
=-\sum_{t=0}^{T-1}\gamma^t\Psi^+_{t+1}-\Psi(s_0)
 +\gamma^T(1-\delta_{T-1})\Psi^+_T.
\tag{N3}
\]
Equation(N3) makes the changed objective explicit. Standard PBRS reasoning
applies relative to \(r^0\), under the previously stated information, entropy
and boundary conditions; it does not preserve the old Path-C objective or imply
equivalence to sparse simultaneous success. At a timeout, a consistent
continuation-value transformation is needed to cancel the retained boundary.
The identity supplies neither a finite-sample neural-SAC advantage nor identical
critic, target or automatic-temperature updates. Removing occupancy reward and
adding potential credit assignment are one combined intervention; this two-arm
study cannot isolate their individual causal contributions.

Fresh training seeds1611/1622/1633 each receive40,000transitions/39,000updates
in both arms, with1,000warmup, batch128, replay100,000, CUDA/Torch1thread and at
most4independent history workers. The coordinator fixes six owned single-seed
outputs and source/config hashes before execution; no checkpoint/replay is
loaded. Each episode reinitializes the previous potential from its actual reset
geometry. Each transition records the original scalar, removed physical near
term, added potential difference, replay reward/done and termination flags.
Evaluation uses original Path-C with B and the replacement callback disabled;
state-isolation checks preserve the training environment and reward memory.

Only each history's final40k success count on the12fixed development maps is
primary. All three paired differences are retained. The frozen integer rule
requires total gain at least3, at least two positive pairs, minimum pair gain
at least-1, and complete finite histories with full source/config/checkpoint/
replay/reward integrity. It is an engineering planning rule, not significance.
The absolute at-least8/12-every-seed task gate is reported separately. Mid-run
evaluations, peaks, returns, ball entry or reduced stall cannot replace the
final endpoint. No final results or confirmed objective/prediction advantage
are asserted while these histories are running. The
[registered protocol](design/MASAC_V4_NEAR_OBJECTIVE_40K_PROTOCOL_20261105.md)
and [source-matched derivation](MATH_NEAR_OBJECTIVE_REPLACEMENT_20261105.md)
govern this separate comparison; its outcomes must not be merged with the
completed B cohort.
