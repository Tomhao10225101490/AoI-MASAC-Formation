# Research design decision record — 2026-10-31

Status: staged proposal, not an implemented or validated main method. Architecture stabilization (Candidate A) and legacy pilot are executable. Radio primitives and graph actor are separate building blocks; neither constitutes end-to-end Candidate C.

> **Update 2026-10-31 (successor execution).** The evidence moved the main line
> from “learned slotted communication” to “who should beacon on a separate
> neighbour channel”. A one-step score using receiver age and link geometry
> beats freshest-sender by 0.91 m (95% CI 0.57–1.26) across two seed blocks and
> 3–4 m in the hard low-penetration cell, with no extra beacons. Covariance
> weighting adds nothing; the local proxy and a local status table both lose;
> relaying that pre-empts the leader's slot is usually not worth its
> opportunity cost. Details: `docs/VOI_SELECTION_RESULTS.md`. The Candidate C
> staged plan below is retained as the record of the earlier design, not as the
> current main claim.

## Candidate comparison and decision

A. Enhanced MASAC: retain independent local Gaussian actors and centralized twin critics; optional LayerNorm, correct actor gradient, reliable checkpoint continuation. Motivation is optimization and reproducibility, not a new policy-gradient contribution. Complexity and information access remain those of the baseline. Low implementation cost; new local pilot tests LayerNorm and removal of uncertainty. A win would support an engineering choice, not research novelty.

B. Graph AoI-aware MASAC: parameter sharing, local delivered-neighbor features, pooled centralized critic and optional recurrence could improve team-size transfer. Requires variable-size observations, replay and environment geometry; fixed-width concatenation cannot claim zero-shot transfer. Graph/attention alone overlaps broad prior MARL and UAV formation work. Existing algos/graph_policy.py is a forward-only graph/GRU prototype with a discrete head, not MASAC or MAPPO. Moderate/high implementation cost; variable-count and masking ablations essential.

C. Budgeted communication/control: measure the downstream control benefit of forwarding an available timestamped leader estimate under a common airtime budget. The original relay observations motivate this, but free neighbor radio cannot establish a shared-spectrum gain. Need explicit source timestamps, delivery delay, broadcast accounting, transmitter eligibility, and local information constraints. Greater systems relevance and highest implementation/validation cost; novelty is an unproven hypothesis.

**Selected main direction: C, with A as a diagnostic baseline and B conditional on measurable benefit.** First hold the flight controller fixed to identify the communication mechanism; then add learned motion only if an equal-budget communication-only comparison establishes room for benefit. Do not force the final paper title to contain graph or MASAC if experiments do not justify them.

Nearest verified overlap: Chiariotti and Fabris, *VoI-aware Scheduling Schemes for Multi-Agent Formation Control*, arXiv:2507.06392 (2025), https://arxiv.org/html/2507.06392v1. They already study formation-oriented information scheduling. Our possible distinction must be demonstrated for lossy, delayed **relay of existing estimates**, including overhead and redundant packets, rather than claimed from combining AoI and formation. This statement is a proposed research boundary, not proof of novelty.

## 1. System model and code status

Control ticks have dt=0.2 seconds. One leader plus N followers; bounded speed, yaw rate and vertical acceleration are implemented by UAVFormationEnv._integrate. Desired follower slot is s_i=p_L+R(psi_L)d_i; actual tracking loss is ||p_i-s_i||_2. Actor references use its allowed belief, not true leader pose. LeaderBelief implements constant-velocity propagation and heading covariance, with exact state reset on packet reception; it currently does not sample measurement noise.

A source message has generation tick g, transmission tick t and delivery tick t+d, d>=1 for the proposed radio. Receiving a forwarded estimate does not generate a fresh source observation. Current direct leader updates are zero-latency within the control tick; current relay payloads arrive one tick later. These distinct conventions must remain explicit, or both must be moved into the new radio for comparisons.

## 2. AoI and constrained objective

Let u_i(t) be the newest delivered source-generation tick. After all deliveries, Delta_i(t)=t-u_i(t), in ticks; report dt*Delta_i(t) in seconds. Accept a payload iff its g>u_i. Thus a delayed update resets age to its delivery delay, not zero. True age is uncapped; clipping is permitted only in network input normalization.

Proposed objective is min_pi E[sum_t gamma^t (mean_i ||p_i-s_i|| + lambda_c collision_t)] subject to per-slot airtime sum_k tau_k <= B_t. Air time tau_k = 8(payload_bytes+header_bytes)/bitrate + guard_seconds + explicitly charged control seconds. Count a broadcast once regardless of receiver count; reception metrics are separate. Energy must remain Pending until transmit power and receive/flight models are specified; do not rename action magnitude as physical energy.

AoI is an observation and diagnostic metric. Adding an AoI reward is a separate hypothesis and ablation, not automatically part of the task objective.

## 3. Decision process / information access

The full simulator state includes physical poses, radio channel states, in-flight messages, newest source timestamps, belief means/covariances and resource ledger. It is privileged during centralized training only. A decentralized actor sees its own state, own leader belief and a cache populated **only by delivered neighbor packets**. Neighbor age/uncertainty is estimated from the cached header and its timestamp; current remote age is not available for free.

A practical initial decentralized MAC option is cyclic transmitter eligibility with local transmit/idle gating. It needs no global freshest election and guarantees one contender per slot. All methods share eligibility and airtime budget. It optimizes *whether* to forward, not arbitrary sender selection. A centralized election variant must be labeled separately and include report/grant overhead; it cannot be called decentralized execution.

## 4. Actions and reward

Stage C1 holds slot tracking fixed and uses transmit/idle for the eligible agent. Stage C2 may add bounded motion residuals, but would require a hybrid-action learner and matched communication-only/motion-only ablations. Neither stage has yet been integrated. Reward uses the true simulated control task during training, with no true state included in deployment observations. Hard airtime feasibility is enforced by the radio, not merely encouraged by a reward coefficient.

## 5. Graph representation and models

Candidate node features: own belief age, predicted slot covariance, own velocity/relative slot error and cache age; edge features: locally observed neighbor displacement and delivered metadata age. Graph edges are available cached messages, not simulator-truth radio-success edges. Parameter-shared actor aggregates K cached neighbors; a GRU is optional for history. Existing LocalGraphGRUActor requires a real adapter and learner. A centralized set critic may read privileged nodes only during training. It currently outputs state values, not SAC action values.

No architecture equation is labeled implemented beyond the current network forward modules. The first executable comparison uses standard MASAC:

L_Qi=E[(Q_i1(s,a)-y_i)^2+(Q_i2(s,a)-y_i)^2],
y_i=r_i+gamma*(1-terminal)*(min_j Q_ij_target(s',a')-alpha_i log pi_i(a_i'|o_i')).

L_pi_i=E[alpha_i log pi_i(a_i|o_i)-min_j Q_ij(s,(a_i,a_-i,replay))]+lambda_a E[||a_i||²].

L_alpha_i=-E[log(alpha_i)*(log pi_i + H_target)].

Mapping: algos/masac.py update, sac_actor_loss; algos/networks.py sample uses reparameterized Gaussian with stable tanh Jacobian; algos/common.py soft_update. These are standard baseline objectives, not a new algorithm. Default network layout/checkpoint keys are retained; train.layer_norm enables Candidate A.

## 6. Candidate-C pseudocode (planned integration)

1. Reset independent map, motion, channel and policy random streams; assign immutable split/seed metadata.
2. Deliver due packets and accept only strictly newer source timestamps.
3. Construct local cache observations and eligibility mask.
4. Eligible actor samples transmit/idle; enforce airtime including control overhead.
5. Broadcast one immutable timestamped payload; queue successful receptions with latency.
6. Execute fixed slot controller; accumulate control loss, collisions, AoI and radio counters.
7. Save transitions with masks and terminal/truncation distinction; update shared policy and training-only critic.
8. Evaluate frozen checkpoints on unseen maps with identical resource constraints and predefined seed pairs.

Steps 2 and 5 partly exist in legacy relay code; steps 3–4 and 7 need new integration. No C learning results exist yet.

## 7. Complexity and tests

Legacy MASAC has N independent actors and critics; each critic input concatenates joint actions and global state, so cost and parameter count grow with N. A shared local graph encoder with width H and K neighbors costs roughly O(N*K*H²) for dense projections and O(N*K*H) for one-head attention scores, plus recurrent updates O(N*H²); exact parameters and runtime must be measured. Radio potential pair delivery is O(N²) for full broadcasts; spatial restriction can reduce this only if modeled consistently.

Test gates: exact source timestamps; delayed/out-of-order rejection; no same-tick forwarding; budget conservation; broadcast charging; deterministic seed replay; no privileged actor inputs; mask/permutation equivariance; variable agent count; checkpoint equality. Then budget-matched analytic, freshest-oracle, random, round-robin and learned policies; ablate uncertainty/graph/memory only when implemented.

## 8. Falsifiable hypotheses and limitations

H1: A control-value relay policy improves formation error over age-only and analytic baselines at the same total airtime, across multiple seeds and unseen maps. H2: cached uncertainty predicts useful deliveries beyond age alone. H3: shared graph policy transfers to larger teams better than capacity-matched masked pooling. Each remains unverified.

Failure of H1 means the learned scheduler is not justified. Failure of H2/H3 means omit the associated innovation claim. Current street geometry, exact measurements, independent distance attenuation and fixed flight tracker limit external validity. Burst channels are sensitivity models, not calibrated RF measurements. No convergence, safety or real-world performance guarantee is claimed.
