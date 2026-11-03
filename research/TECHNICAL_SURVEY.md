# Information freshness, wireless decisions and UAV formation: technical synthesis

This survey synthesizes a targeted primary-source review conducted on 2026-10-31. It separates three quantities that the project must keep distinct: message age, estimation uncertainty, and control loss. Metadata, access levels, and detailed technical reading notes are in PAPER_MATRIX.md and LITERATURE_REVIEW.md. The associated BibTeX deliberately cites verified preprints when publication metadata has not been checked.

## 1. AoI-aware UAV networks

Age measures how old the newest source observation available at a receiver is; it is not simply latency of the last transmission. Buffering and failed receptions change the time average, and a relayed observation retains its original sampling timestamp. [Yates et al.](https://arxiv.org/abs/2007.08564) provide the relevant analytical foundation. Our consequence is operational: record generation and arrival events separately, and report age units explicitly.

UAV freshness optimization spans distinct tasks. [Emami et al.](https://arxiv.org/abs/2405.00056) study sensor collection, whereas [Bai et al.](https://arxiv.org/abs/2507.14299) couple sensing and communications. [Sarathchandra et al.](https://arxiv.org/abs/2501.14603) consider changing age–power tradeoffs. These studies motivate communication-aware decisions, but none of their abstract performance claims can serve as a baseline number in a different formation environment.

## 2. Formation control

Formation introduces a task cost that need not rank messages like mean age. A turn can make the same age more harmful than during constant motion; geometry changes which followers benefit. This is the project's hypothesis, requiring controlled experiments rather than assumption. [Chiariotti and Fabris](https://arxiv.org/abs/2507.06392) already establish a relevant age/value scheduling connection. [DI-MATD3](https://doi.org/10.3390/drones10100762) is also a direct comparison for delayed-state reconstruction; it remains abstract-level in this review.

Robotic realism and networking realism are separate axes. [Xie et al.](https://arxiv.org/abs/2410.18495) provide a formation/obstacle benchmark with physical deployment. [Park et al.](https://arxiv.org/abs/2501.02453) explicitly connect urban geometry to radio blockage. A simplified outer-loop model can isolate communications, but must not be described as a full flight or propagation simulator.

## 3. Multi-agent reinforcement learning

CTDE permits a training critic to use information unavailable to executed actors. [MADDPG](https://arxiv.org/abs/1706.02275) formalizes this separation; [SAC](https://arxiv.org/abs/1812.05905) supplies entropy-regularized continuous control; [TD3](https://arxiv.org/abs/1802.09477) supplies a relevant deterministic reference. The multi-agent adaptation, entropy convention and terminal bootstrap rules must be documented precisely rather than inferred from an algorithm name.

[MAPPO](https://arxiv.org/abs/2103.01955) motivates a strong uncomplicated scheduling baseline. Optimizer changes, representation changes and information changes should be tested separately. An invalid actor objective can make any architecture appear ineffective, so correctness tests must precede performance interpretation.

## 4. Graph-based and temporal approaches

[MAAC](https://arxiv.org/abs/1810.02912) shows that selective attention in centralized critics is established. [DGN](https://arxiv.org/abs/1810.09202) addresses changing neighborhoods, while [GAT](https://arxiv.org/abs/1710.10903) and [Deep Sets](https://arxiv.org/abs/1703.06114) supply generic relational and invariant representations. A graph encoder is consequently an inductive bias, not by itself a scientific contribution.

Temporal state can help when observations are incomplete, as motivated by [DRQN](https://arxiv.org/abs/1507.06527). However, recurrence requires sequence-consistent training and reset handling. [MAT](https://arxiv.org/abs/2205.14953) has different inference information requirements from independent local actors. [Feng et al.](https://arxiv.org/abs/2401.17880) further limits the novelty of merely combining graph recurrence and UAV resource allocation.

## 5. Joint communications and control

Communication influences control through delivered content, not the transmission attempt alone. [Wu et al.](https://arxiv.org/abs/2312.13182) provides direct evidence that task-aware handling of commands is established. [ET-MATD3](https://doi.org/10.1016/j.jfranklin.2026.108729), reviewed only through its publisher abstract, is another necessary competitor for claims about learned triggering. Neither establishes the result of our relay design.

Our proposed distinction is narrower: evaluate receiver-specific usefulness of timestamped relay information with an enforceable radio-resource account. This places implementable observation access and resource fairness at the center of the experiment. It remains a hypothesis until an explicit protocol and matched comparisons demonstrate it.

## 6. Existing limitations and comparison boundaries

Methods optimizing age, throughput, buffered collection and formation error solve different objectives. For example, [queue-aware UAV graph learning](https://arxiv.org/abs/2607.00324) addresses buffered utility, and [federated vehicular graph MARL](https://arxiv.org/abs/2407.02342) concerns offloading. They are architectural and networking context, not directly comparable formation baselines. The absence of a feature in one paper is not proof that the feature is absent from the entire literature.

## 7. Evidence needed to establish the research gap

The experiment must distinguish additional bandwidth from better scheduling, local caches from privileged state, and geometric generalization from retraining. [Agarwal et al.](https://arxiv.org/abs/2108.13264) motivates uncertainty-aware evaluation. A useful protocol reports independent training-seed variation and paired scenario differences, with frozen selection and test sets. Effect sizes should accompany uncertainty; significance alone does not establish practical value.

## 8. Positioning and current status

A defensible working position is “resource-accounted, task-aware information relay for formation under stale and unreliable observations.” Graph learning is a candidate implementation. A corrected baseline, valid timestamps, equal resource accounts and reproducible negative results are necessary foundations. The current literature does not justify a first-of-kind claim or guaranteed advantage. Full-text verification of the direct 2026 competitors, reproduction of adapted analytical schedulers, and multi-seed experiments remain necessary before the final paper can make substantive novelty claims.
