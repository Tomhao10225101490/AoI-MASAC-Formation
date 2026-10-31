# WoWMoM research upgrade progress

Updated 2026-10-31. Status: IN PROGRESS; no submission-readiness claim.

## Provenance
- Baseline: 3bffd89a5f084446729f826c06df19e8e3baa6b7.
- Branch: research/wowmom2027-upgrade.
- Existing modified files: algos/masac.py, envs/formation_env.py, tests/test_env.py, tests/test_relay.py.
- Existing untracked work: graph_policy.py, ablation_belief_no_aoi.yaml, server_protocol.yaml, plan_server_study.py and four test files. These predate this execution; preserved, not attributed as newly implemented work.
- Tracked diff backup: results/upgrade_audit/preexisting.patch. Original commit remains intact.

## Executed
- Hardware detection: M4 Mac mini; 10 CPU cores (4P+6E), 10 GPU cores, 16 GiB, macOS 15.1; available disk 121 GiB; memory_pressure reported 41% free.
- Python 3.9.6 arm64; torch 2.8.0; numpy 2.0.2; MPS available; CUDA unavailable.
- Current working-tree pytest: 83 tests passed, exit 0.
- CPU/MPS project benchmark launched, execution session 99516; output results/upgrade_audit/hardware-benchmark.json; completion must be checked.
- Same process tests original git archive in a temporary directory; output baseline-tests.txt.
- Environment lock snapshot: results/upgrade_audit/pip-freeze.txt.

## Findings and decisions
- Existing Q-centering fix is mathematically consequential: mean(Q-mean(Q)) is zero, including its gradient. Re-evaluate old residual-policy negative results before interpreting them scientifically.
- Existing graph/GRU module explicitly has no learner or environment adapter. Do not report graph MARL training as implemented.
- Server protocol is design-only; no server access or expensive server job launched.
- Benchmark uses unique output root to preserve prior results/study artifacts.

## Next
1. Inspect benchmark and original tests, persist resource recommendations and raw logs.
2. Complete code/formula/timestamp/checkpoint audit and verify inherited modifications.
3. Read primary literature, especially control-value scheduling overlap; build verified bibliography before selecting main algorithm.
4. Implement and test the chosen design, then run paired baselines and ablations with frozen evaluation splits.
5. Figures, full survey, paper, reviewer audit remain Pending.

All PHASE 2–10 deliverables remain incomplete. Hardware audit is preliminary pending benchmark review. No performance improvement is claimed.

## Completed execution update
Session 99516 finished successfully. CPU and MPS six-episode smoke training completed; mean update times 0.0223 s and 0.0564 s respectively. Original untouched baseline: 55 tests passed. No training from this session remains running. Checkpoints and metrics are in results/upgrade_audit/bench_runs. Hardware report saved; profiling limitations explicitly listed. First audit documentation commit contains only new audit files; inherited implementation changes remain uncommitted and preserved.
