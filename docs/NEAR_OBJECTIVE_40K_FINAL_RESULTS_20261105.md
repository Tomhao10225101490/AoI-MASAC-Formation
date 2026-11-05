# Near-objective final endpoints and coordinator protocol deviation

Date: 2026-11-05, Asia/Shanghai. Registered before launch: **b4b3a1e**.
Study: `results/masac_v4_near_objective_40k_sharded_cuda_20261105`.
Frozen analyzer output: `results/masac_v4_near_objective_40k_analysis_20261105`.

## Fixed final outcomes

All six owned histories saved final40000 completions and exited0. Each used
40000 transitions,1000 warmup and39000 updates; total240000/234000. The last
original1633/potential1633 workers finished15:17/15:18. No extension, restart,
best-checkpoint selection, model loading, new rollout or sealed evaluation was
performed by this analysis. This is the separate static/B-off reward study,
not the earlier worst-agent B intervention or historical80k content suite.

- Seed1611: original2/12, replacement1/12, paired difference-1.
  Any/all ever-ball: original12/5, replacement11/3.
- Seed1622: original0/12, replacement0/12, difference0.
  Any/all ever-ball: original9/0, replacement11/0.
- Seed1633: original5/12, replacement6/12, difference+1.
  Any/all ever-ball: original12/9, replacement12/7.

Each arm totals7/36 descriptive successes; mean paired success-fraction
difference0. Both absolute gates fail0/3seeds at8/12. Only one pair is positive,
and the total gain is0, below the registered planning conditions even as simple
arithmetic. **No formal advancement label is calculated or rescued:** the
coordinator failure prevents decision readiness under the frozen contract.
The repeated12development maps do not supply36independent training replicates.

Original failures:3timeout-before-any-ball,19timeout-after-any-only,
7timeout-after-all-ball. Replacement failures:2,24,3 respectively. All recorded
final collision flags are zero. These are each arm's29failures and7successes.
All-ball without success does not distinguish low-speed individual capture
from simultaneous overlap; no new dwell/trajectory inference is made here.
Any-ball totals33/36versus34/36; all-ball14/36versus10/36. Lower all-ball counts
are descriptive and do not identify a causal mechanism.

## Completed integrity audit, incomplete coordinator contract

The frozen independent analyzer finished normally (exit0), using actual final
checkpoint/replay SHA256s, all240000reward rows, replay reward/done checks,
finite sampled diagnostics, source/config bytes and final map signatures.
Its output reports actual artifact hashes, reward/replay verification, finite
final diagnostics, source agreement and map pairing all true; no input changed
during analysis. Replacement corrections were nonzero25829/21035/27142times.

Its scientific study status is **INVALID_OR_INCOMPLETE**, decision_ready=false,
decision=null. Two integrity issues are the coordinator error marker and failed
root status; missing evidence is the required root completion. Its aggregate
`all_six_training_completions=false` includes coordinator validity: it must not
be rewritten to true merely because all six owned completion markers exist.
The per-history evidence and the failed overall contract must both be retained.

At14:34:32, coordinator PID12436 raised PermissionError/WinError5 replacing
`status.json.tmp.12436` with `status.json`. It stopped submission, drained the
existing children and ended FAILED. No child training error was established.
The original error/status remain untouched. First-observed raw error snapshot:
`results/masac_v4_windows_audit_20261105/near_objective_coordinator_error_snapshot_20261105_1454.json`,
SHA256`1e4066dceec9bac195a2ad6a3407e19e85f41fa36c47caef8aec0095e580873a`.
The historical file holder was not identified; dashboard or antivirus causation
is not established. This is a monitoring I/O deviation, not demonstrated SAC
numerical divergence. It does not authorize retrospective contract changes.

## Windows reader checks and limits

The first attempted dashboard sharing improvement failed2of6temporary-file
tests. Original failing output is retained as `progress_shared_read_tests.log`.
Direct native CreateFileW and CRT-wrapped readers both still blocked replacement
of an open destination on this host, even with READ|WRITE|DELETE sharing.
This matches the limitation described in the
[CPython maintainer discussion](https://github.com/python/cpython/issues/90161).
Microsoft's [CreateFileW documentation](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)
defines share consent; it is not a blanket guarantee for replacing destinations.

The dashboard now requests these sharing flags, bounds JSON reads and closes
the handle before decoding. Final six tests PASS, including the explicit
replacement limitation, replacement after close, Unicode/binary data, append,
cleanup, document refresh and partial JSONL/tail behavior. This is **not a proven
solution to writer/reader races**. A separately reviewed future writer needs
bounded retries and concurrent-reader tests before another training launch.
No frozen coordinator/trainer/analyzer was modified to hide this incident.

The owned dashboard alone was restarted; new API PID36692/session65925 at this
check, http://127.0.0.1:8765/. Real browser shows6/6owned histories complete,
0active slots and the original coordinator error. Current process identity must
be rechecked before future service management. Source20/config2 remain frozen.

## Scientific consequence and next question

The replacement is the combined removal of physical successor near occupancy
and addition of discounted potential credit assignment. PBRS invariance is only
relative to the removal-defined base reward, not old Path-C or sparse capture.
This cohort supplies no aggregate efficacy gain and no predictive-channel test.
Do not extend or tune it from the final result. Retain the negative1611pair and
zero1622pair alongside1633. The earlier B cohort remains separate.

Next, narrow a genuine predictive-representation question against the direct
joint-feasibility and MARL conflict-resolution precedents in
[the new frontier check](FRONTIER_JOINT_FEASIBILITY_GAP_20261105.md). A shared
recourse definition or a toy conflict alone is not new. Any geometry diagnostic
needs its own reviewed/pushed protocol and real3Descape/accounting controls;
any later training needs matched information/cost and a concrete prior plan.
