# Technical reading notes

Primary sources retrieved on 2026-10-31. These notes accompany the metadata and compact evidence fields in PAPER_MATRIX.md. Fourteen entries below received substantive full-text inspection; the long AoI survey was read selectively. This is a targeted technical review, not a systematic review or independent reproduction. Formula labels refer to the linked versions; arXiv HTML can contain conversion errors. No external performance percentage is imported into our results.

## 1. Chiariotti and Fabris — VoI-aware formation scheduling

[Full text, §§2–4, equations 11–18](https://arxiv.org/html/2507.06392v1).
Position uncertainty grows as dσ²Δ; the scheduler optionally weights it by formation centrality. Updates localize one node perfectly; synchronization makes schedules precomputable. Eight-agent, 100-episode simulations compare age, expected error, weighted value and an oracle. This already establishes task-aware freshness scheduling for formations. Our research distinction must concern lossy information relay and decision information costs, not the label “VoI-aware formation.” Adapt its age/error/centrality priorities as baselines; adaptations must be identified.

## 2. Haarnoja et al. — SAC with automatic temperature

[Full text, §§4–7 and appendix C](https://arxiv.org/html/1812.05905v2).
Equations 5 and 9 give squared soft-Bellman residual and E[α log π−Q]. Reparameterization preserves the action-to-Q gradient; entropy tuning enforces an expected entropy constraint. Reported robot and simulation successes do not prove MARL convergence. A batch-centered differentiable Q term averaged in the actor loss cancels its task gradient. Use the canonical objective before studying architecture effects; distinguish numerical safeguards from an algorithmic contribution.

## 3. Yu et al. — MAPPO

[Full text, §§3–5](https://arxiv.org/html/2103.01955v4).
Centralized value inputs supplement local actors; PPO clips likelihood ratios. The ablations isolate value normalization, critic features, data reuse, clipping and batch size. GAE must use denormalized values when targets are normalized. MPE, SMAC, football and Hanabi support a strong plain baseline, not a guarantee on wireless problems. For scheduling, compare a simple shared policy under exactly the same observation and action constraints before adding graph memory.

## 4. Lowe et al. — MADDPG

[Full text, §4 equations 4–6 and §5](https://arxiv.org/html/1706.02275v4).
The critic conditions on joint actions while each executed actor receives only its local observation. The actor differentiates its own action coordinate and the target evaluates target joint actions. Particle-world experiments test communication and physical coordination. Privileged training inputs are allowed; privileged execution inputs are not. Preserve this distinction when adapting the project's centralized critic, and disclose dimensions that grow with team size.

## 5. Iqbal and Sha — MAAC

[Full text, §3.2 and §4](https://arxiv.org/html/1810.02912v2).
A queried weighted sum of other agents' embeddings enters each critic. Soft policy learning and a counterfactual baseline are already combined with attention. Their experiments match approximate parameter counts, include uniform attention, and use six seeds. A graph-attention critic alone is therefore not novel MASAC research. The uniform aggregation control is especially useful: it tests selective relations beyond capacity and access to additional agents.

## 6. Jiang et al. — DGN

[Full text, §3 and §4](https://arxiv.org/html/1810.09202v5).
Shared encoders aggregate changing neighborhoods; relation attention and temporal regularization encourage consistent cooperation. Multiple layers enlarge the receptive field through neighbor communication. Battle uses three training runs; the displayed min–max band is not a confidence interval. Transferable lessons concern parameter sharing and permutation-compatible inputs. On a radio system, each message-passing round needs a time and byte budget; graph locality alone does not establish feasible distributed execution.

## 7. Wen et al. — MAT

[Full text, §§2.2–5](https://arxiv.org/html/2205.14953v3).
The advantage decomposition motivates sequential conditional actions. Training computes stored-action likelihoods in parallel, while inference autoregressively feeds prior actions into later decisions. Full encoder attention accesses joint observations. Benchmark gains and variable-agent transfer do not remove those information requirements. Its stated theoretical guarantees concern the specified update assumptions; they should not be transferred to a modified finite-step neural implementation. We should avoid a MAT baseline unless its communication assumptions can be matched fairly.

## 8. Emami et al. — mean-field hybrid PPO

[Full text, §§IV–VI](https://arxiv.org/html/2405.00056v2).
The model combines continuous flight with categorical sensor selection, population interaction approximation and LSTM state. Its cost averages sensor AoI. Training uses 3000 episodes of 40 steps. This preempts novelty from simply combining recurrent MARL with movement and scheduling. Equation 20 and surrounding entropy description need care: the text describes multiplying component entropies; independent joint-policy entropy is normally additive. Do not copy this implementation choice without derivation. The collection objective differs from formation-control loss.

## 9. Bai et al. — AoI-centric UAV ISAC

[Full text, §§II–IV, equations 35–37](https://arxiv.org/html/2507.14299v1).
The SAC state includes AoI, SINR and Kalman covariance, with continuous movement and power-allocation intentions. Negative mean age is the reward. Six users and 100 shared layouts test a fixed-altitude UAV. It already combines age and estimator uncertainty in control observations. Our distinction requires demonstrating why formation task loss changes optimal decisions relative to this freshness objective, with matched channel information and physical resources.

## 10. Yates et al. — age foundations

[Selected full-text sections II and VI–VII](https://arxiv.org/html/2007.08564v1).
Age at reception resets to delivery time minus source generation time, not automatically zero. Mean age integrates the age process; peak age samples pre-update peaks. Delay, buffering and relaying therefore require explicit timestamps. This is an analytical survey, with no single experiment to reproduce. Application-aware penalties connect age to estimation/control but do not make age sufficient for every plant. Preserve uncapped age internally; any observation clipping is a separate representation decision.

## 11. Xie et al. — formation with obstacle avoidance

[Full text, §§III–VI](https://arxiv.org/html/2410.18495v2).
Actions are collective thrust and body rates. A two-stage search/curriculum selects reward tradeoffs, while entity attention handles drone and obstacle observations. The three-drone simulation uses substantial training (15M/150M/150M curriculum steps), followed by physical deployment. It is a valuable realism reference but not a matched radio benchmark. Copying its compute scale before validating the communication hypothesis would confound control learning, reward search and network contribution.

## 12. Park et al. — blockage-aware trajectory optimization

[Full text, §§II–V](https://arxiv.org/html/2501.02453v1).
The problem maximizes minimum uplink spectral efficiency with building avoidance and geometric LoS determination. A segment–building condition links trajectory to radio state; SCA and coordinate updates solve a nonconvex formulation. Flight-time and power sweeps show setting-dependent rankings of baselines. This supports explicitly modeled urban geometry, not arbitrary blockage multipliers as a calibrated channel. Our simulator must label simplified attenuation and test sensitivity before claiming propagation realism.

## 13. Wu et al. — age/value robotic command queues

[Full text, §§II and IV–V](https://arxiv.org/html/2312.13182v2).
A DQN command generator, ACK-limited proactive repetition and receiver-side age/value queue ranking couple communications to tracking error. AoI uses command generation time; usefulness estimates future target distance. The simulation averages 1000 repetitions and separates components. This is close task-oriented communication prior art. A stale message need not be valueless, but repeated delivery must not manufacture freshness. A relay study should distinguish state-estimate replacement from execution of queued commands.

## 14. Feng et al. — graph-attention trust-region UAV communication

[Full text, §§II–IV](https://arxiv.org/html/2401.17880v1).
Graph recurrent features and attention critics support joint trajectory/resource allocation. Two-to-four UAV experiments compare graph-only and attention-only variants. Trust-region arguments precede finite neural training; convergence in a plot is not evidence of exact equilibrium. This is direct evidence that graph, recurrence and wireless co-optimization are an established combination. New claims should rest on explicit freshness semantics, budget feasibility, or measured generalization, not architecture vocabulary.

## Not yet deeply reviewed

Twelve registry entries are abstract-level or publisher-summary records. In particular DI-MATD3's publisher page was returned in search, but direct full-text retrieval produced HTTP 429. ET-MATD3 remains publisher-abstract only. Their formulas must not be reconstructed from the abstracts or the old repository narrative. The registry records first arXiv years rather than guessing publication years; final paper citations should update verified journal metadata where available.
