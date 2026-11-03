# V4 actor/critic and candidate feature architecture

Initial executable design frozen before implementation, 2026-11-03. Architecture upgrades are explicitly user-authorized hypotheses. Their training remains behind the legacy Standard navigation gate.

## Representations

The compatibility adapter reads ego20 and current peer/building/moving slots from the V3-compatible 162-vector, retaining masks. Object features are 13-dimensional: one-hot type3, relative position3, relative velocity3, half-extent3 and radius1. Missing attributes are zero. Relative distances retain the documented 240 m normalization, velocities use observed vmax. A direct token API accepts variable M and a boolean mask. No padded token contributes to aggregation.

The ego encoder is 20→128→128 and object encoder is 13→128→128. Attention uses an ego token plus object tokens, two pre-norm transformer-style blocks, four heads at width128, feedforward256 and no dropout. This is ordinary masked attention, not an innovation claim. Object order has no positional encoding. Final masked mean/max object pooling (256) and ego128 form384 features. Actor fusion is384→256→256 with separate mu/log_std3, Gaussian reparameterization, tanh action and exact tanh log-Jacobian. The physical action uses the existing radial acceleration bound; entropy is defined in raw tanh space, not claimed as executed-acceleration entropy.

Controls are (i) unchanged legacy V3 Pooling, (ii) width128 Pooling with additional per-token residual MLP capacity chosen to approximate the attention parameter count, and (iii) attention. Compare small→large Pooling for capacity and matched Pooling→attention for representation. Report exact counts and any residual mismatch; do not claim exact matching by name alone.

Each centralized Q has its own complete encoder. Per-agent encoded observation plus raw action feeds a 256-dimensional agent encoder; joint mean/max pooling feeds a scalar Q. Q1/Q2 and target storage are independent. Actor execution never reads Q/global metadata. Actor gradients pass through the frozen online critics' action input; Bellman targets use no_grad.

## Maneuverability embedding

For each of zero, six axes and twelve face diagonals, simulate bounded acceleration/velocity for 2s with dt0.2 from current allowed observation. Reconstruct only visible local AABBs, current dynamic objects and peers. Other objects follow explicitly constant-velocity extrapolation; no simulator future arrays are accepted. Public world size, body radius and configured safety margin are known constants.

Per probe: minimum clearance to inflated static boxes, moving spheres, peer spheres and world bounds; first sampled nonpositive-clearance time; current-speed stopping-distance surrogate v²/(2amax); current-clearance minus this distance; positive-preview flag. Normalize/clip distances at40m and TTC at2s. Concatenate canonical control direction3, yielding11 features. Shared probe MLP11→64→64, mean/max pooling→128 risk embedding. All compared actors have this encoder and the same fusion input size. Zero control feeds zero features. Static-only zeros dynamic/peer channels and bases TTC/reserve on static/bounds. A shuffle remaps risk values among fixed canonical directions on each observation; shuffling complete tokens before invariant pooling would be an invalid ablation. A wrong-amax mode is separately available.

These discretely sampled margins use local perception and do not certify continuous or joint feasibility; constant-velocity prediction and a two-second window can miss braking and interaction effects. No hard action replacement/filter or candidate-specific reward is used.

## Engineering analysis

The implementation profile must save exact trainable counts, approximate linear/attention multiply-adds (state assumptions), input tensor sizes, model/checkpoint/replay memory estimates and warmed CPU/MPS actor and update timing. Benchmark one run at a time on this Mac. CPU/MPS equivalence tolerances and stochastic-RNG differences must be stated. Profiling initialized networks is engineering evidence, not policy learning or architecture benefit.
