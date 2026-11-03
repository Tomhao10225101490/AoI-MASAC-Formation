# MASAC V4 research specification

Frozen initial specification, 2026-11-03. User explicitly authorized V4 implementation, richer scenes, an independently controlled attention comparison and a critic-sampling diagnosis. Core remains learning-based multi-UAV 3D path planning with MASAC. Predictive Maneuverability-Aware MASAC is the single unverified candidate contribution.

## Evidence and progression

Read `V4_FAILURE_ROOT_CAUSE_AUDIT.md` and the V3 paired 300k report first. All six histories finished with 0/6 final joint development successes. Terminal replay samples were rare, sampled critic gradients frequently clipped, and low-speed near-goal failures persisted. These observations motivate tests and are not identified causes. V3 source/results remain historical and immutable during new runs.

1. Complete V3 implementation audit and evidence classification.
2. Freeze V4 map, architecture and experiment specifications before implementation/training.
3. Test geometry, dynamics, masks, observations and three-agent reference trajectories under the actual time limit. Failed feasibility checks remain visible.
4. Run isolated V3 Critic A/B diagnosis from matching original-history checkpoints. Only Critic sampling changes; the control uses identical dual-batch infrastructure.
5. Run the unchanged legacy Pooling MASAC learner on V4 L0 with fixed training/development seeds and a 20k–50k initial budget. This changes the task distribution in a separately named experiment, not inside the sampling A/B diagnosis.
6. A minimum 60% joint success on the fixed L0 development set with all three training seeds and finite diagnostics is an engineering gate. Report every seed; a pooled average must not conceal a failed seed. Then freeze L1 protocol. If the gate fails, diagnose rather than silently extending budgets or selecting a best checkpoint.
7. After reproducible navigation, compare legacy Pooling, capacity-controlled Pooling and attention; select the Standard architecture on development evidence. Only then train the maneuverability feature ablations with matched reward, capacity and budget. 500k–1M studies require a new protocol.

## Interfaces and information

V4 maps retain a 240×240×90 m base world and parameterized L0–L3 scenes. Larger worlds are separate generalization cases. Three independent goals, norm-bounded acceleration, speed limits, continuous swept collision checks and explicit per-episode horizon remain central.

The initial compatibility observation is `(agents,162)` with ego20, peer5×7, building8×7, moving4×8 and a reserved19 channel. Padding masks are explicit; V4 sensing uses documented current-state sensor limits. L0 sensor coverage includes its entire task. Changing observability at harder levels is a scene condition that must be reported. A separate token adapter supplies ego20 and a variable object set with type, relative geometry and velocity to V4 representations. Actor and feature construction use only this observation and public dynamics constants. Reference paths and global scene metadata are confined to offline feasibility audits.

The centralized twin critics use joint actor-available current observations and joint raw actions. Actor parameters are shared across agents. No reference action, privileged future trajectory or oracle waypoint enters learned-policy inference.

## Candidate and controls

Nineteen fixed acceleration probes use the exact bounded velocity update for a two-second sampled preview. Per-direction features summarize static/dynamic/peer/boundary margins, sample-grid TTC and a documented braking surrogate. They are finite-probe risk estimates, not a viability certificate or safety guarantee. All ablation actors reserve the same risk encoder and parameter count: full, static-only, independently direction-mismatched and zero-risk. Reward and executed MASAC action remain identical across feature ablations.

## Reproducibility

Training seeds, map seeds, development seeds and sealed final-test seeds are separate namespaces. Repeated V3 six-map evaluations are development evidence only. New outputs use unique directories with manifest, source/config hashes, exact RNG/replay/optimizer checkpoints, raw failures, terminal/truncation labels, stage and completion/error markers. Recovery and compute-device parity are tested before long runs. The progress report must identify implementation, smoke evidence, actual learning evidence and unverified claims separately.
