# V4 execution and gates

Initial protocol, 2026-11-03, before new learning runs. Read the research spec, map benchmark, architecture and V3 failure audit together. All outputs are new directories; legacy sources and failed results remain available.

## Isolated Critic diagnostic

Use only the original Standard 300k histories, training seeds411/422/433, each restored twice from its exact same final full checkpoint/replay. A: dual-batch uniform Critic; B: terminal-stratified Critic with correct actual-population importance weights. Actor and alpha use the same uniform rule and independent saved sampling RNG in both arms. Same environment, raw162 observation, legacy Pooling, rewards, optimizer/lr, clipping5, gamma0.99, tau0.005, batch128, replay100k and continuing-task truncation semantics. No new warmup. Initial budget20,000 additional environment steps per arm/seed; no adaptive extension. Fixed six V3 development maps retain historical comparability and are explicitly not a final test. The detailed sampler protocol and correctness tests are mandatory before execution. Metrics include population/sample proportions, effective weights, pre/post-clip gradients, TD/Q/entropy, timing and total update calls. A/B source/RNG/recovery and weighted-gradient expectation tests precede training.

## L0 baseline qualification

After diagnostic implementation and environment/reference tests, train legacy Standard Pooling from fresh seeds511/522/533 on V4 L0. Freeze a first50,000-step budget with warmup1000 and evaluation every2000 steps. Use all newly frozen L0 development maps, with separate seeds from training. No reward, network or time-limit change is bundled into this scene qualification. Keep all three seeds and last checkpoints. The 60% joint-success threshold on each seed is an engineering gate, not a paper performance claim. If zero or below threshold, save and diagnose; do not queue Attention or maneuver arms or expand the budget automatically.

Maps use a frozen `configs/masac_v4/seed_sets.json` namespace. Final-test seeds are reserved and must not be evaluated by tuning/profiling scripts. Larger maps and L1–L3 use new protocols after L0 qualification. Test set content is never inspected during development selection.

## Long-term comparisons

After L0/L1 navigation is reproducible, compare old Pooling, capacity-controlled Pooling and attention with equal environment/update budgets and matching seeds/maps. Architecture selection precedes candidate feature comparisons. Full/static/shuffled/zero maneuverability modes share model capacity, reward, task generator and training budgets. Three independent training seeds minimum; five for confirmation when needed. Report inference feature construction plus actor runtime and all failures. No 500k–1M training is scheduled before these gates.

## Runtime records

Every formal run has `run_manifest.json`, `status.json`, `metrics.jsonl`, `train.log`, `evaluation.json`, source/config manifest, exact checkpoints, and `completion.json` or `error.json`. Checkpoints contain Actor/Q/targets/optimizers/alpha, Replay arrays/position, Python/NumPy/Torch/device/sampler RNG, environment and unfinished episode. Only fully aligned state pairs with a final hash/completion marker qualify for resume. A file lock prevents duplicate writers. Every100–500 steps log experiment/seed/level/step/budget/episodes, losses/alpha/Q/TD/entropy/clip rates, recent success and causes, actual evaluation step, throughput/elapsed/ETA. Closing the read-only viewer leaves the detached training alive. `caffeinate` guards idle sleep. Device choice follows measured CPU/MPS throughput and correctness.
