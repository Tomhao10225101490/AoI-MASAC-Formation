# Scope of the B potential-shaping argument — 2026-11-05

**Read-only mathematical/literature audit.** No running source, configuration, gate, optimizer, or evaluation protocol is changed. Implementation descriptions below come from the existing [B design](design/MASAC_V4_SYNC_CAPTURE_20261105.md), [math audit](METHOD_MATH_AUDIT_20261105.md), and [canonical equations](MATH_ARGUMENT_IMPROVEMENTS_20261103.md); this note does not independently re-audit executable source.

The defensible distinction is between **policy-independent changes to an exact objective** and **the finite learning trajectory of neural SAC**. B has a standard PBRS identity. It has no new policy-invariance theorem and no guarantee of unchanged actor, temperature, target-network, or replay evolution.

## 1. Primary references and their actual scope

| Primary source | Relevant result | Scope that must accompany a citation |
|---|---|---|
| Ng, Harada, Russell, ICML 1999. [Author-hosted paper](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) | Theorem 1 gives policy-preserving potential differences; its sufficiency argument shifts action values by a state-only quantity. | Main rigorous setting is finite, fully observed MDPs. Discounted use assumes the ordinary value problem is well defined; infinite-state extensions require regularity, including bounded potentials. Its converse is a guarantee over unknown transition/reward problems, not a claim that every preserving transformation for one fixed task must be PBRS. |
| Adamczyk et al., AAAI 2023, first preprint December 2022. [Official proceedings PDF](https://ojs.aaai.org/index.php/AAAI/article/download/25817/25589), [author preprint/full HTML](https://arxiv.org/html/2212.01174v2) | Theorem 5 extends PBRS sufficiency to entropy-regularized tasks; Remark 7 covers fixed, possibly nonoptimal policies. | Same dynamics, discount, inverse temperature and reference policy; bounded rewards/potential and well-defined regularized values. Proofs are stated for discrete settings. The paper discusses continuous extensions, rather than proving neural continuous-action SAC convergence. Do not confuse its inverse temperature with this repository's `beta = log(alpha)`. |
| Skalse et al., ICML 2023, first preprint March 2022. [PMLR](https://proceedings.mlr.press/v202/skalse23a.html), [full text](https://arxiv.org/html/2203.07475v2) | Lemma B.1 records the trajectory/value shifts; Lemma B.2 gives the soft-Q shift; Theorem 3.3 describes maximum-causal-entropy policy invariances. | Explicitly finite, observable, infinite-horizon MDPs, with fixed discount and regularization convention. These are value/policy results, not claims about a replay-based neural optimizer's path. |
| Forbes et al., AAMAS 2024. [Official proceedings](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p589.pdf) | Theorem 1 supplies an episodic boundary condition for more general potential variables; the expected residual boundary must be independent of the current action. | Useful context for history-dependent potentials and terminal treatment. A generic nonzero finite-horizon endpoint is not harmless merely because a shaping term is written as a difference. This work does not establish this MASAC implementation's convergence. |
| Haarnoja et al., 2018/2019. [SAC applications paper](https://arxiv.org/abs/1812.05905), [full text](https://arxiv.org/html/1812.05905v2) | Actor, approximate critic/target, and automatic temperature objectives supply the learner context. | Its exact soft-policy-iteration statement has a finite-action assumption; practical continuous control uses neural and stochastic approximations. Automatic temperature adjustment is a separate optimization step. |
| Pardo et al., ICML 2018. [PMLR](https://proceedings.mlr.press/v80/pardo18a.html) | Distinguishes intrinsic task horizons from artificial training cutoffs. | Bootstrapping across an artificial cutoff and terminating at a genuine task horizon solve different modeled problems. This distinction does not determine which objective our experiment should adopt; it requires that the chosen one be stated. |

The 2022–2024 references above directly address the requested issue. No artificial 2025/2026 citation is needed to make standard PBRS appear new.

## 2. A pathwise statement that does not require Markov observations

Write `psi_t = lambda * Phi_t`, with fixed `lambda = 0.5`, `gamma = 0.99`, and `0 <= psi_t <= 0.5`. At true termination, define the absorbing successor potential to be zero. At a time-limit truncation, retain the saved successor potential. Along a segment without a preceding true termination, B therefore uses

\[
F_t=\gamma\psi_{t+1}-\psi_t,
\qquad
\sum_{t=0}^{T-1}\gamma^tF_t=-\psi_0+\gamma^T\psi_T. \tag{1}
\]

Equation (1) is an algebraic identity for each realized sequence. `Phi_t` may be a fixed measurable function of a sufficient state, observations, or available history; the identity itself does not require those observations to be Markov. For policy ordering, additional conditions are essential:

- Compare the **same class of admissible policies**, with the same information available to each actor, dynamics, reset distribution, discount, and action/entropy convention. The initial potential's distribution must be independent of the policy being compared.
- Use a true absorbing endpoint with zero successor potential, an infinite discounted continuation with bounded potential, or an exact, consistently shifted continuation value. A raw finite segment with a nonzero endpoint is a different case.
- The potential is fixed and action-independent at the decision instant. Its successor value can depend on the executed action; this is already accounted for by telescoping. Do not differentiate a shaping potential through a learned actor or silently refresh a learned potential during this argument.
- For the entropy-regularized comparison, keep the same entropy term. For a fixed `alpha`, integrability of the return/entropy objective must be sufficient to take expectations and limits.

Under the true-terminal/infinite-continuation alternatives, for any policy in that common class,

\[
J'_{\alpha}(\pi)=J_{\alpha}(\pi)-\mathbb E[\psi_0]. \tag{2}
\]

The unchanged entropy terms cancel in the comparison. Thus all policy rankings for that exact objective, including rankings inside a shared memoryless/decentralized policy class, are preserved. This is our direct consequence of (1), consistent with the cited PBRS results. It does not require the actor class to contain an optimal full-state controller.

The objective in (2) is the baseline **Path-C plus the specified entropy regularization**. Path-C already contains non-PBRS reward terms. Preserving its ordering does not prove preservation or maximization of the probability of capture, safety, or any different sparse-success objective.

**Information caveat:** changing a reward signal can change what a general history-based POMDP controller can infer. Keep the information class fixed; do not compare an unshaped controller denied a hidden signal with a shaped controller allowed to condition on it. The current design says the reward changes but actor observations do not. The exact reconstructibility of the simulator-computed distance/speed potential from joint observations should be checked before writing `Phi = f(O)` as an implementation fact. Even if that reconstructibility holds, joint observations still omit episode time and first-entry reward memory.

## 3. Timeout bootstrapping is a conditional continuation argument

At the 350-step cutoff, `done=False` keeps a nonzero boundary in (1). Exact cancellation occurs only when the continuation quantity also obeys the matching shift:

\[
\sum_{t<T}\gamma^t(r_t+F_t)+\gamma^T V'_{\alpha}(X_T)
=\sum_{t<T}\gamma^t r_t+\gamma^T V_{\alpha}(X_T)-\psi_0,
\quad V'_{\alpha}=V_{\alpha}-\psi. \tag{3}
\]

Identical entropy terms may be included on both sides. Equation (3) assumes a sufficient state/history `X` and the exact continuing values. A bootstrapping flag and a neural prediction alone do not establish these assumptions. Current training resets after the cutoff while bootstrapping from the saved pre-reset next observation; the documents correctly describe this as a **continuing-value surrogate**. It is not an exact finite-350-step terminal model or a proof that its optimal policy equals the benchmark's finite-horizon optimum.

For a raw discounted cutoff sum, the remaining boundary is at most `0.5 * 0.99^350`, approximately 0.015, because the potential is bounded. That numerical bound is not an exact invariance statement. Logged **undiscounted** episode returns have no policy-independent-shift guarantee from (1), even at true termination. Compare common base rewards and success/failure metrics across arms, not raw shaped-return magnitudes as if they were the same metric.

No timeout-mask or time-input change is proposed here; either would require a separately matched experiment.

## 4. Exact soft values and actor objectives

On a suitable Markov state or history `X`, for a fixed policy and fixed temperature the ideal values satisfy

\[
Q'^{\pi}_{\alpha}(X,A)=Q^{\pi}_{\alpha}(X,A)-\psi(X),
\quad V'^{\pi}_{\alpha}(X)=V^{\pi}_{\alpha}(X)-\psi(X). \tag{4}
\]

The corresponding optimal-value identities are the established entropy-regularized PBRS result. An action-independent shift leaves the normalized soft action preference unchanged. Mean per-agent entropy, used here, is compatible with the same argument when kept identical in both tasks.

Equation (4) is **not** an unconditional identity for learned `Q(O,A)`. The observation projection can alias distinct sufficient states/histories, and the critic may not represent the shift. If the potential itself is not measurable from the critic's input, there need not even exist a single output offset for every aliased input. Measurability of `Phi(O)` is helpful but still does not turn an observation-space projected Bellman update into exact Markov policy evaluation.

There is also a useful purely algebraic statement for the implemented actor loss, independent of whether the critics are exact values. Hold a replay batch, actor, sampled actions and temperature fixed. **If both supplied critics satisfy a common pointwise shift**, `Q'_k(O,A) = Q_k(O,A) - psi(O)`, then

\[
\min_k Q'_k=\min_k Q_k-\psi(O),
\quad L'_\pi=L_\pi+\mathbb E_{O\sim\mathcal D}[\psi(O)]. \tag{5}
\]

If `psi` is independent of actor parameters/current sampled action and the batch is held fixed during the update, the extra term has zero actor gradient. This includes reparameterized action gradients. It is a **conditional gradient identity**, not a statement that independently trained B-on/off critics acquire or retain that shift. If state visitation or replay differs, the conditioning no longer describes the two training runs.

## 5. Temperature and target-network learning dynamics

The source-matched temperature variable is `beta = log(alpha)`, with

\[
L_\beta=-\mathbb E_{\mathcal D}
[\beta(\operatorname{sg}(\bar\ell)-3)],
\qquad \bar\ell=N^{-1}\sum_i\log\pi(a_i\mid o_i). \tag{6}
\]

Its differential-entropy target is **-3**. There is no reward, critic value, or potential directly in (6). Consequently, an exact critic-only common shift with the same policy, batch and action sampling leaves this temperature update unchanged. It does not imply equal learned `alpha`: changes to critics can change actors, visitation and replay, which change the log-probability samples used by (6).

For every fixed temperature, exact objective invariance can also be used inside an ideal entropy-constrained problem with an unchanged feasible policy set. This concerns exact optima and does not establish the path of a stochastic primal/dual or log-temperature optimizer. No entropy retargeting follows from PBRS.

For the twin-target TD update, define `m = 1 - done` and consider the same transition and next-action sample. If online **and target** critics all have the matching shift, then

\[
\begin{aligned}
y'&=r+\gamma m\psi(X')-\psi(X)
 +\gamma m[\min_k(Q_{\bar\theta_k}(X',A')-\psi(X'))-\alpha\bar\ell']\\
&=y-\psi(X),
\qquad Q'_k-y'=Q_k-y. \tag{7}
\end{aligned}
\]

True-terminal successors use zero potential and zero bootstrap; truncations retain both. Equation (7) is function-level algebra. Its sufficient conditions are not automatically maintained by the frozen learner:

- Same random neural weights in the two arms initially give the **same** critic function, not `Q'_0 = Q_0 - psi`.
- A nonlinear state-only potential may not have an exact, parameter-preserving representation as a shift of the given network. Equal TD residuals alone also do not imply equal parameter gradients unless the parameterization/Jacobians correspond.
- Adam states, finite batches, clipping, approximate fits and changed replay can generate different update paths.
- Polyak averaging is performed on **parameters**, not exact value functions. A general nonlinear parameter transformation representing a function shift need not commute with parameter averaging. Hence maintaining an online shift alone does not guarantee a target-network shift.

An explicit frozen residual wrapper `Q'_theta = Q_theta - psi`, applied consistently to online and target outputs, with matched initialization, optimizer state and data, could preserve these algebraic identities under further implementation assumptions. The running protocol does not claim such a construction. This is an explanatory sufficient condition, **not a proposed source change**.

## 6. Recommended paper wording and unresolved checks

Use: “B adds a fixed bounded potential difference using the learner's discount and true-terminal convention. The standard discounted telescoping identity preserves the exact baseline soft-objective ordering under a common policy information class, initial distribution and consistent terminal/continuation assumptions. Our time-limit bootstrap is a continuing surrogate, and our observation-based neural learner need not preserve exact value shifts or learning dynamics. Learning effects are evaluated empirically.”

Avoid: “PBRS leaves SAC training unchanged,” “automatic temperature is invariant throughout training,” “the target networks preserve the potential shift,” “partial observations satisfy the state Bellman theorem,” or “a bootstrap flag proves policy invariance for the 350-step capture benchmark.” Also avoid treating the base Path-C objective as a sparse-success objective.

Before a stronger implementation-specific statement, verify potential reconstructibility/units from critic inputs, reward additivity without subsequent clipping or rescaling, the saved next-observation/reset convention, and consistency of every true-terminal mask. Scalar telescoping and terminal unit checks validate those identities on tested cases; they do not validate a neural value-shift, learning-dynamics, convergence or performance theorem. This audit proposes no additional experiment while the preregistered runs execute.
