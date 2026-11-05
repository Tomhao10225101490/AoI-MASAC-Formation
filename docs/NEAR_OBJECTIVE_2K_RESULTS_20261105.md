# Near-objective2k pilot: runtime pass, no performance evidence

2026-11-05. Code/config/protocol pushed **9a751c3** to both branches before
either job. Two fresh cold-start seed1644jobs, static/B-off,2k transitions and
1k updates each, CUDA/Torch1thread. No checkpoint loaded or historical run
extended. Both finished normally; no further training was queued by this pilot.

| Training near mode | Success /12 | Any-ball /12 | All-ball /12 | Final failure counts | Trainer elapsed s |
|---|---:|---:|---:|---|---:|
| original |0|0|0|boundary12|85.864|
| potential replacement |0|0|0|static9, boundary3|86.770|

The absolute task gate fails in both arms. This single-seed/2k experiment is
only numerical/runtime sanity, not improvement, equality, safety or futility
evidence. Failure types differ; no favorable-seed replacement or extra training.

## Integrity and intervention coverage

Independent analysis v2 PASS: each history has2,000ordered reward transitions,
1,000updates,17unchanged source hashes, original config hash, final checkpoint
and replay SHA256match. Every replay reward matches the logged training reward
within float32 storage tolerance, and every replay done mask matches true
termination. Every closed episode's base/removed/potential/correction totals
matches the transition records. Ten sampled post-warmup progress rows/history
are finite, alongside the trainer's per-update guard (not1,000retained loss rows).
Evaluation asserts unchanged training environment/reward state in both arms.

Original:84transitions have a nonzero pre/post near potential, but exactly0
reward corrections; original scalar rewards are preserved. Treatment:103
nonzero-potential/nonzero-correction transitions, correction range
[-0.09573271419590462,0], within the analytic bounds. Thus the nonzero intervention
was exercised after the initial all-zero near warmup. Each run has26training
true terminations and0timeouts. Timeout algebra is tested by the22math tests,
but **actual timeout execution is not covered by this pilot**. Exact CUDA resume
is unsupported/unvalidated; neither binary is uploaded to Git.

GPU sample during the two jobs:60%utilization,1570MiB/8188MiB,54C; about14.9GiB
RAM free. These are one resource sample, not a sustained-throughput benchmark.

The independent analyzer initially failed because it expected a nested losses
field; the original trainer stores flattened progress diagnostics. The failure
output/error and full traceback log are preserved. Corrected analyzer ran into
a fresh `...analysis_v2_20261105`; no training was rerun or modified.

The legacy global registry has an unlocked read/modify/write race: concurrent
updates lost the original-arm index entry. The independent per-output records
were intact. After both workers exited, root restored both completed index
entries from those authoritative manifests/completions, recording before/after
hashes in `near_pilot_registry_repair.json`. Scientific artifacts were unchanged.
The next formal entry must serialize registry updates before parallel launch.

## Interpretation and recovery

The new formula removes the observed physical near term and adds a discounted
potential difference, with absorbing-zero true terminals and bootstrapped
timeouts. It intentionally changes the old Path-C soft objective; PBRS applies
relative to Path-C-minus-near. No sparse-success equivalence or new theorem.
Any later gain would concern the complete replacement, not independently the
removal and added credit assignment. The pilot supplies no performance decision.

- Authoritative run roots: `results/masac_v4_near_objective_2k_{original,potential}_cuda_20261105`.
- Final snapshots: each `maneuver_static_1644/checkpoint_002000/{checkpoint.pt,replay.npz,manifest.json}`; binaries local only.
- Independent result: `results/masac_v4_near_objective_2k_analysis_v2_20261105/analysis.json`.
- Failed first analyzer: `results/masac_v4_near_objective_2k_analysis_20261105/error.json` and Windows audit console log.
- Full commands/logs: `WINDOWS_COMMAND_LEDGER_20261105.md`; `near_objective_2k_*_console.log`.

Next work is a separate fixed-budget, fresh-seed matched study design, with a
visible plan and pushed protocol/code before any substantial training. The
completed2k histories remain closed; sealed maps remain unopened.
