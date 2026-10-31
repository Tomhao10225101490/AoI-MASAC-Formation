# Code audit — initial findings, 2026-10-31

Baseline 3bffd89; inherited working-tree changes are separately identifiable in results/upgrade_audit/preexisting.patch. This is an ongoing audit, not full certification.

## Critical: actor objective under action regularization
Original algos/masac.py computes centered=Q-Q.mean(), scales centered, then averages the negative result in the actor loss when action_l2>0. Its batch mean is identically zero. The task-value gradient therefore cancels; entropy and action regularization can dominate behavior. This makes earlier zero-residual observations inconclusive about the research hypothesis.

The existing local sac_actor_loss fix uses mean(alpha*logp-Q)+lambda*mean(action²). tests/test_actor_gradient.py checks nonzero Q slopes with and without L2; all current tests passed. This fix predates this execution. Priority P0: preserve it and rerun controlled experiments; do not silently mix old and corrected results.

## High: incomplete graph implementation
algos/graph_policy.py contains local masked graph/GRU actor and centralized set value modules, but explicitly no PPO rollout/GAE/learner/MAC adapter. experiments/server_protocol.yaml correctly labels main methods planned. Forward tests are not evidence of an end-to-end learning algorithm. Priority P1: verify information access and implement a complete learner only after method selection.

## High: radio-resource fairness
Existing beacon results retain leader broadcasts and add a neighbor channel. Free relay is a resource-privileged reference, not a proven universal performance upper bound. Freshest selection needs knowledge of agents' ages, whose exchange must be accounted for in a distributed protocol. Priority P1: equal total airtime and explicit information-access controls.

## Medium: benchmark concurrency inference
scripts/bench_device.py choose() assumes 1.6x CPU throughput for two jobs; it does not measure concurrent throughput. Do not present this multiplier as hardware evidence. Actual concurrency and memory must be profiled separately.

## Pending audit
Full SAC target/temperature derivation, replay and checkpoint resume, timestamps and delayed/out-of-order messages, random stream coupling, reward/termination correctness, energy/delay/interference model validation, and fairness across train/test conditions.
