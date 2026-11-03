# Verified paper registry and evidence matrix

Retrieved 2026-10-31. 26 primary-source records. `full` means selected substantive method and evaluation sections of the full HTML were read; it does not mean independent reproduction or formal proof verification. `full-selected` is the relevant survey sections. Abstract-only entries are discovery records, not deep reviews. Preprint years below are first arXiv submission years; later revisions are discussed separately. Publisher venue is stated only when verified. No result below belongs to our project.

## chiariotti2025voi
- **Title:** [VoI-aware Scheduling Schemes for Multi-Agent Formation Control](https://arxiv.org/abs/2507.06392)
- **Authors:** Federico Chiariotti; Marco Fabris
- **Year / venue:** 2025; arXiv preprint (consulted version; venue not asserted)
- **Access:** full
- **Problem / method:** Formation localization scheduling; AoI, error and centrality priorities.
- **Formulation:** First-order 3D formation; perfect positioning updates.
- **Experiment / finding:** 100 simulated episodes, eight agents; Control value beats purely chronological priority in tested cases.
- **Limit / relevance:** No stochastic relay MAC; strong local connectivity assumption; Closest conceptual overlap.

## haarnoja2018applications
- **Title:** [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
- **Authors:** Tuomas Haarnoja; Aurick Zhou; Kristian Hartikainen; George Tucker; Sehoon Ha; Jie Tan; Vikash Kumar; Henry Zhu; Abhishek Gupta; Pieter Abbeel; Sergey Levine
- **Year / venue:** 2018; arXiv preprint (consulted version; venue not asserted)
- **Access:** full
- **Problem / method:** Sample-efficient continuous control; Maximum entropy off-policy actor critic.
- **Formulation:** Soft Bellman target and reparameterized actor.
- **Experiment / finding:** Simulation and physical robot tasks; Strong task performance reported.
- **Limit / relevance:** Tabular proof does not certify nonlinear MARL convergence; Audit SAC objectives.

## yu2021mappo
- **Title:** [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- **Authors:** Chao Yu; Akash Velu; Eugene Vinitsky; Jiaxuan Gao; Yu Wang; Alexandre Bayen; Yi Wu
- **Year / venue:** 2021; arXiv preprint (consulted version; venue not asserted)
- **Access:** full
- **Problem / method:** Cooperative MARL baseline quality; PPO with centralized value.
- **Formulation:** Clipped surrogate and GAE.
- **Experiment / finding:** MPE, SMAC, football, Hanabi; Competitive benchmark performance.
- **Limit / relevance:** Wireless protocol not modeled; Strong discrete scheduling baseline.

## lowe2017maddpg
- **Title:** [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
- **Authors:** Ryan Lowe; Yi Wu; Aviv Tamar; Jean Harb; Pieter Abbeel; Igor Mordatch
- **Year / venue:** 2017; arXiv preprint (consulted version; venue not asserted)
- **Access:** full
- **Problem / method:** Nonstationary interacting agents; CTDE deterministic policy gradients.
- **Formulation:** Q_i(global state,joint action).
- **Experiment / finding:** Particle-world cooperation and competition; Improved coordination versus independent learners.
- **Limit / relevance:** Fixed joint inputs do not imply scalable execution; Preserve continuous-control comparator.

## iqbal2018maac
- **Title:** [Actor-Attention-Critic for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1810.02912)
- **Authors:** Shariq Iqbal; Fei Sha
- **Year / venue:** 2018; arXiv preprint (consulted version; venue not asserted)
- **Access:** full
- **Problem / method:** Selective centralized credit information; Attention critics with soft actors.
- **Formulation:** Q_i=f_i(g_i(o_i,a_i),sum_j alpha_ij v_j).
- **Experiment / finding:** Treasure collection and rover-tower; six seeds; Competitive returns and scaling.
- **Limit / relevance:** Critic attention is not a deployable radio protocol; Attention-MASAC combination already exists.

## jiang2018dgn
- **Title:** [Graph Convolutional Reinforcement Learning](https://arxiv.org/abs/1810.09202)
- **Authors:** Jiechuan Jiang; Chen Dun; Tiejun Huang; Zongqing Lu
- **Year / venue:** 2018; arXiv preprint (consulted version; venue not asserted)
- **Access:** full
- **Problem / method:** Learning on changing agent graphs; Shared graph Q-network.
- **Formulation:** Attention relation kernels and temporal regularization.
- **Experiment / finding:** Battle, jungle, routing; Improved reported cooperation.
- **Limit / relevance:** Feature exchange assumptions need radio accounting; Graph representation baseline.

## wen2022mat
- **Title:** [Multi-Agent Reinforcement Learning is a Sequence Modeling Problem](https://arxiv.org/abs/2205.14953)
- **Authors:** Muning Wen; Jakub Grudzien Kuba; Runji Lin; Weinan Zhang; Ying Wen; Jun Wang; Yaodong Yang
- **Year / venue:** 2022; arXiv preprint (consulted version; venue not asserted)
- **Access:** full
- **Problem / method:** Structured joint decisions; Autoregressive transformer.
- **Formulation:** Joint advantage decomposition.
- **Experiment / finding:** SMAC, MuJoCo, hands, football; Reported performance and task transfer gains.
- **Limit / relevance:** Execution needs predecessors actions and encoded joint observations; Do not call unrestricted MAT decentralized.

## emami2024meanfield
- **Title:** [Age of Information Minimization using Multi-agent UAVs based on AI-Enhanced Mean Field Resource Allocation](https://arxiv.org/abs/2405.00056)
- **Authors:** Yousef Emami; Hao Gao; Kai Li; Luis Almeida; Eduardo Tovar; Zhu Han
- **Year / venue:** 2024; arXiv preprint (consulted version; venue not asserted)
- **Access:** full
- **Problem / method:** Fresh sensor collection by UAVs; Mean-field hybrid PPO plus LSTM.
- **Formulation:** Mixed discrete scheduling and continuous movement; mean AoI cost.
- **Experiment / finding:** 3000 episodes of 40 steps; Lower simulated AoI reported.
- **Limit / relevance:** Data harvesting differs from closed-loop formation; Hybrid actions plus memory not new.

## bai2025isac
- **Title:** [Age of Information Minimization in UAV-Enabled Integrated Sensing and Communication Systems](https://arxiv.org/abs/2507.14299)
- **Authors:** Yu Bai; Yifan Zhang; Boxuan Xie; Zheng Chang; Yanru Zhang; Riku Jantti; Zhu Han
- **Year / venue:** 2025; arXiv preprint (consulted version; venue not asserted)
- **Access:** full
- **Problem / method:** Joint sensing and communication freshness; SAC, Kalman filter, regularized zero forcing.
- **Formulation:** Reward negative average AoI; trajectory and beam control.
- **Experiment / finding:** Six users, 60 slots, 100 common layouts; Lower simulated mean AoI reported.
- **Limit / relevance:** Single UAV, fixed altitude; no swarm formation; AoI and covariance observations already used.

## yates2020survey
- **Title:** [Age of Information: An Introduction and Survey](https://arxiv.org/abs/2007.08564)
- **Authors:** Roy D. Yates; Yin Sun; D. Richard Brown III; Sanjit K. Kaul; Eytan Modiano; Sennur Ulukus
- **Year / venue:** 2020; arXiv preprint (consulted version; venue not asserted)
- **Access:** full-selected
- **Problem / method:** Timestamp-based freshness; Queueing and control synthesis.
- **Formulation:** Age=current time-latest delivered generation time.
- **Experiment / finding:** Survey, not one benchmark; Relates freshness, estimation and control.
- **Limit / relevance:** Cannot substitute AoI for arbitrary task loss; Timestamp and peak-age definitions.

## xie2024formation
- **Title:** [Multi-UAV Formation Control with Static and Dynamic Obstacle Avoidance via Reinforcement Learning](https://arxiv.org/abs/2410.18495)
- **Authors:** Yuqing Xie; Chao Yu; Hongzhi Zang; Feng Gao; Wenhao Tang; Jingyi Huang; Jiayu Chen; Botian Xu; Yi Wu; Yu Wang
- **Year / venue:** 2024; arXiv preprint (consulted version; venue not asserted)
- **Access:** full
- **Problem / method:** Formation amid static/dynamic obstacles; Reward search, curriculum, attention encoder.
- **Formulation:** Multi-objective Dec-POMDP; thrust/body-rate actions.
- **Experiment / finding:** Three drones; simulation and real deployment; Improved collision-free formation reported.
- **Limit / relevance:** Large training cost; not a radio scheduling study; Robotics fidelity and training-budget reference.

## park2025blockage
- **Title:** [Blockage-Aware UAV-Assisted Wireless Data Harvesting With Building Avoidance](https://arxiv.org/abs/2501.02453)
- **Authors:** Gitae Park; Kanghyun Heo; Kisong Lee
- **Year / venue:** 2025; arXiv preprint (consulted version; venue not asserted)
- **Access:** full
- **Problem / method:** Urban data harvesting; SCA and block coordinate descent.
- **Formulation:** Max-min uplink efficiency; geometric LoS constraints.
- **Experiment / finding:** Geometry and flight-time/power sweeps; Improved minimum spectral efficiency reported.
- **Limit / relevance:** Known cuboid geometry; no MARL freshness control; Separate collision geometry from link blockage.

## wu2023semantic
- **Title:** [Goal-oriented Semantic Communications for Robotic Waypoint Transmission: The Value and Age of Information Approach](https://arxiv.org/abs/2312.13182)
- **Authors:** Wenchao Wu; Yuanqing Yang; Yansha Deng; A. Hamid Aghvami
- **Year / venue:** 2023; arXiv preprint (consulted version; venue not asserted)
- **Access:** full
- **Problem / method:** Communication-sensitive waypoint tracking; DQN, proactive repetition, age/value queue ordering.
- **Formulation:** Timestamp age and predicted task-distance value.
- **Experiment / finding:** 1000 simulation repetitions, component comparisons; Reduced tracking MSE reported.
- **Limit / relevance:** Single UAV command queue; not distributed relay; Task-value prioritization already established.

## feng2024graph
- **Title:** [Graph Attention-based Reinforcement Learning for Trajectory Design and Resource Assignment in Multi-UAV Assisted Communication](https://arxiv.org/abs/2401.17880)
- **Authors:** Zikai Feng; Di Wu; Mengxing Huang; Chau Yuen
- **Year / venue:** 2024; arXiv preprint (consulted version; venue not asserted)
- **Access:** full
- **Problem / method:** UAV base-station cooperation; Graph recurrent representation and attention trust-region MARL.
- **Formulation:** Joint trajectory and resource Markov game.
- **Experiment / finding:** Two, three and four UAV base stations; Higher rewards in most reported comparisons.
- **Limit / relevance:** Finite training is not proof of Nash convergence; Graph plus memory plus radio control already studied.

## sarathchandra2025meta
- **Title:** [Age and Power Minimization via Meta-Deep Reinforcement Learning in UAV Networks](https://arxiv.org/abs/2501.14603)
- **Authors:** Sankani Sarathchandra; Eslam Eldeeb; Mohammad Shehab; Hirley Alves; Konstantin Mikhaylov; Mohamed-Slim Alouini
- **Year / venue:** 2025; arXiv preprint (consulted version; venue not asserted)
- **Access:** abstract
- **Problem / method:** Freshness-power tradeoff; DQN with MAML.
- **Formulation:** Varying weighted AoI/power objective.
- **Experiment / finding:** Simulation; detailed setup not reviewed; Faster adaptation reported in abstract.
- **Limit / relevance:** Full experimental details pending; Test changed objective weights.

## wang2024fgnn
- **Title:** [Optimizing Age of Information in Vehicular Edge Computing with Federated Graph Neural Network Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2407.02342)
- **Authors:** Wenhua Wang; Qiong Wu; Pingyi Fan; Nan Cheng; Wen Chen; Jiangzhou Wang; Khaled B. Letaief
- **Year / venue:** 2024; arXiv preprint (consulted version; venue not asserted)
- **Access:** abstract
- **Problem / method:** Fresh vehicular offloading; Federated graph MARL.
- **Formulation:** AoI under RSU resource constraints.
- **Experiment / finding:** Simulation; detailed setup pending; Improvement reported in abstract.
- **Limit / relevance:** Vehicles/offloading differ from UAV control; AoI+GNN+MARL combination is prior art.

## fujimoto2018td3
- **Title:** [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
- **Authors:** Scott Fujimoto; Herke van Hoof; David Meger
- **Year / venue:** 2018; arXiv preprint (consulted version; venue not asserted)
- **Access:** abstract
- **Problem / method:** Critic overestimation; Twin critics and delayed actor.
- **Formulation:** Minimum double-Q target.
- **Experiment / finding:** Continuous control benchmarks; Improved benchmark performance reported.
- **Limit / relevance:** Single-agent method; MATD3 adaptation must be specified; Audit twin-Q and delayed update.

## velickovic2017gat
- **Title:** [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- **Authors:** Petar Velickovic; Guillem Cucurull; Arantxa Casanova; Adriana Romero; Pietro Lio; Yoshua Bengio
- **Year / venue:** 2017; arXiv preprint (consulted version; venue not asserted)
- **Access:** abstract
- **Problem / method:** Graph representation; Masked neighborhood attention.
- **Formulation:** Weighted neighbor aggregation.
- **Experiment / finding:** Citation and protein graphs; Strong benchmark classification reported.
- **Limit / relevance:** No communication protocol or MARL guarantee; Architecture attribution.

## zaheer2017sets
- **Title:** [Deep Sets](https://arxiv.org/abs/1703.06114)
- **Authors:** Manzil Zaheer; Satwik Kottur; Siamak Ravanbakhsh; Barnabas Poczos; Ruslan Salakhutdinov; Alexander Smola
- **Year / venue:** 2017; arXiv preprint (consulted version; venue not asserted)
- **Access:** abstract
- **Problem / method:** Permutation-invariant inputs; Shared embedding and aggregation.
- **Formulation:** Invariant/equivariant set functions.
- **Experiment / finding:** Set-based learning benchmarks; Useful invariant models demonstrated.
- **Limit / relevance:** Not proof of UAV scale generalization; Pooling control for graph ablation.

## agarwal2021statistics
- **Title:** [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)
- **Authors:** Rishabh Agarwal; Max Schwarzer; Pablo Samuel Castro; Aaron Courville; Marc G. Bellemare
- **Year / venue:** 2021; arXiv preprint (consulted version; venue not asserted)
- **Access:** abstract
- **Problem / method:** Uncertain few-run RL conclusions; Interval estimates and robust aggregates.
- **Formulation:** Bootstrap profiles and interquartile mean.
- **Experiment / finding:** Atari, Procgen, control reevaluation; Point-estimate conclusions can change.
- **Limit / relevance:** Task/seed resampling structure matters; Avoid pseudo-replication.

## yang2018meanfield
- **Title:** [Mean Field Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1802.05438)
- **Authors:** Yaodong Yang; Rui Luo; Minne Li; Ming Zhou; Weinan Zhang; Jun Wang
- **Year / venue:** 2018; arXiv preprint (consulted version; venue not asserted)
- **Access:** abstract
- **Problem / method:** Many-agent interaction explosion; Population-average interaction.
- **Formulation:** Mean-field Q and actor-critic.
- **Experiment / finding:** Gaussian squeeze, Ising, battle; Learning effectiveness reported.
- **Limit / relevance:** Averages can suppress heterogeneous link differences; Scalability comparator.

## hausknecht2015drqn
- **Title:** [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)
- **Authors:** Matthew Hausknecht; Peter Stone
- **Year / venue:** 2015; arXiv preprint (consulted version; venue not asserted)
- **Access:** abstract
- **Problem / method:** Partial observation; LSTM Q-network.
- **Formulation:** History summarized in recurrent state.
- **Experiment / finding:** Standard/flickering Atari; Improved robustness to observation changes.
- **Limit / relevance:** No universal recurrent advantage; History ablation motivation.

## schulman2017ppo
- **Title:** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **Authors:** John Schulman; Filip Wolski; Prafulla Dhariwal; Alec Radford; Oleg Klimov
- **Year / venue:** 2017; arXiv preprint (consulted version; venue not asserted)
- **Access:** abstract
- **Problem / method:** Stable policy improvement; Clipped surrogate optimization.
- **Formulation:** Multiple minibatch passes over rollout.
- **Experiment / finding:** Locomotion and Atari; Useful sample/walltime balance.
- **Limit / relevance:** On-policy update constraints remain; Discrete scheduler optimizer foundation.

## li2026queuegraph
- **Title:** [Queue-Aware Graph Reinforcement Learning for UAV-ISAC-Assisted Maritime Data Collection](https://arxiv.org/abs/2607.00324)
- **Authors:** Bohan Li; Min Ye; Haochen Liu; Yongkang Gong; Ning Gao; Jie Nie; Pei Xiao; Xiuzhen Cheng
- **Year / venue:** 2026; arXiv preprint (consulted version; venue not asserted)
- **Access:** abstract
- **Problem / method:** Buffered maritime collection; Heterogeneous graph and masked b-matching.
- **Formulation:** Queue utility with energy/association constraints.
- **Experiment / finding:** Congested simulation and transfer, abstract only; Large utility gain reported, not independently verified.
- **Limit / relevance:** Preprint; full assumptions pending; Feasible scheduling decoder overlap.

## yu2026dimatd3
- **Title:** [DI-MATD3-Based Multi-UAV Formation Trajectory Planning Under Asynchronous Communication Delays](https://doi.org/10.3390/drones10100762)
- **Authors:** Jingzhao Yu; Hongdong Fan; Le Wang; Jiuan Gao; Yulin Yi
- **Year / venue:** 2026; Drones 10(10), 762
- **Access:** abstract-publisher
- **Problem / method:** Delay-sensitive formation; MATD3 with per-link AoI and Kalman prediction.
- **Formulation:** Timestamped caches propagated by age.
- **Experiment / finding:** Three seeds, dual-offset gate scenario; Improved success reported.
- **Limit / relevance:** Publisher full-text fetch returned HTTP429; Closest AoI-observation baseline.

## li2026etmatd3
- **Title:** [ET-MATD3-based UAV formation control in dynamic obstacle environments](https://doi.org/10.1016/j.jfranklin.2026.108729)
- **Authors:** Junze Li; Lei Liu; Chengqing Liang; Dongmei Yan; Lei Li
- **Year / venue:** 2026; Journal of the Franklin Institute 363(9), 108729
- **Access:** abstract-publisher
- **Problem / method:** Resource-limited formation; Event-triggered MATD3 and switching.
- **Formulation:** Augmented state includes previous action.
- **Experiment / finding:** Full experimental details pending; Abstract reports method; no numeric claim adopted.
- **Limit / relevance:** Full-text technical verification pending; Learning when to communicate is prior art.
