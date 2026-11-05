# Completed late-replay content audit: six static control/B histories

2026-11-05, Asia/Shanghai. The offline plan was pushed in `d8ff5ad` before
feature computation. This is a within-observation representation audit, not a
trained full/constant-fill comparison or a new performance endpoint.

## Frozen population and execution

Six final40k replay snapshots, seeds1511/1522/1533 in both arms; each file's
last8000 retained chronological transitions, exactly1024 fixed indices using
`32000 + floor(k*7999/1023)`, k=0..1023. Six actual replay SHA256s and all
source/config hashes matched before feature computation. Index manifest SHA256:
`eca0cfd9c12108af9627b89910fcf6a657c168a9c1bb54622eb628d15d3b3fc7`.
Selected observation hashes include explicit dtype/shape/byte order and bytes.

CUDA, one Torch thread,128-transition chunks;48 chunks and6144 joint observations
complete. Every full/static trace output matched the frozen provider bitwise,
maximum absolute error0. Three separate synthetic geometry checks also passed:
absent masked-NaN slots, far valid peer with padding sentinel, and near valid peer.
They are excluded from real-data denominators. No nonfinite output or error
artifact occurred. No checkpoint was deserialized, actor instantiated, policy
action sampled, simulator stepped, new scene evaluated or sealed map accessed.

## Content, absence and saturation are different

The6144 joint observations contain18432 agent observations,350208 direction
tokens/channel, with19 directions and11 channels/token. Every agent observation
contains exactly two valid peers; moving-object validity is zero across73728
moving slots. This is not a claim that peers are absent.

| History | Full equals static-fill1 tokens /58368 | Residual peer tokens |
|---|---:|---:|
| Control1511 | 58368 | 0 |
| Control1522 | 58009 | 359 |
| Control1533 | 58368 | 0 |
| B1511 | 58368 | 0 |
| B1522 | 58368 | 0 |
| B1533 | 58368 | 0 |

Overall349849/350208 = **99.89748949%** of full11-vectors equal static-fill1.
The only359 residuals are peer clearance in control1522, maximum normalized
absolute difference0.4561988711. Full moving clearance is always1, static0;
the peer channel is1 in349849 tokens and0 in static. Static-fill1 is formed
**after** static features: only output columns4/5 become1, no aggregate recompute.

For all349849 saturated peer tokens, a valid peer's geometric minimum exceeds
40m while an invalid-slot sentinel40m is strictly smaller. The other359 tokens
have a valid geometric minimum below40m. Thus a feature1 often denotes valid
far geometry limited by the provider, not missing geometry. Static buildings
also remain present:52038 valid slots/147456, but only8/350208 static-clearance
tokens are below1. Boundary, ego braking and other channels still vary.

Full and static have **zero difference** in the other nine channels, including
sampled TTC, braking distance, reserve and the positive-clearance flag. The
359 peer residuals do not propagate into those aggregates on this sample.
The preclip/mode-inclusion/source tables distinguish raw presence, disabled
families, padding minima, current/preview/first-hit attribution and nonexclusive
ties; persistent world-object IDs are not encoded and remain unknown.

## What follows, and what does not

This closes the early2k128-observation coverage gap only for the declared late
replay population. These are states collected by static control/B learners,
not fresh final-policy on-policy rollouts or the historical Mac full-policy
distribution. Most full/static feature differences here are fixed offsets;
standard first-affine bias compensation can preserve a function for an isolated
constant offset. It does not establish equivalent Adam/clipping trajectories,
actor feature use, prediction accuracy, a cause of the old full/static ranking,
or a capture benefit. Rare residual information is not zero information.

All external movers are absent in this L0 population; no external-motion
prediction claim is identified. The next diagnosis should address the observed
near-goal shortfall without automatically spending another learning budget on
a largely constant representation contrast. The separately planned offline
actor-response check will compare tanh(mu) with a fixed sampled action
distribution on the **same saved observations**, without training or rollouts.
It must distinguish a one-step kinematic predicate from task success and account
for float32 observation quantization. No confirmation or40k-to80k continuation
is queued, and test seeds remain sealed.

## Reproduction and artifacts

- Plan: `design/MASAC_V4_POST40K_NEXT_MINIMAL_PLAN_20261105.md`.
- Script: `scripts/audit_final_replay_probe_content.py`.
- Results: `results/masac_v4_sync_40k_late_content_audit_20261105/`:
  audit.json/audit.md, features.csv, provenance_counts.csv, sample_indices.json
  and six selected-input hash sidecars. Weights/replay stay local.
- Streaming console and compact extraction:
  `results/masac_v4_windows_audit_20261105/late_replay_content_console.log`,
  `late_content_summary_console.log`.
- Related mathematical scope: `MATH_CONSTANT_FILL_IDENTITY_20261105.md` and
  `MATH_PATHC_OBJECTIVE_DIAGNOSTIC_20261105.md`.
