# Path C L0 frozen42 gate — 2026-11-03 Asia/Shanghai

**Result: PASSED (3/3 seeds)** — first Standard L0 multi-seed win under Path-C reshape + frozen capture 4/2 latch=False.

| Field | Value |
|-------|-------|
| Out | `results/masac_v4_c_reshape_l0_frozen42_20261103_145043` |
| Experiment | `masac_v4_c_reshape_l0_frozen42_20261103` |
| Auth | widget **t4s11** |
| Seeds / steps | 911, 922, 933 × 50k sequential |
| Gate | ≥0.60 on 12 L0 development maps **each** seed |
| Finals | **911=11/12**, **922=11/12**, **933=10/12** |
| Capture (train+gate) | ball 4.0 m, speed 2.0 m/s, latch=False |
| Rewards | Path-C reshape magnitudes |
| Warm-start | empty C seed 811 @40k (weights+opt; replay discarded) |
| Finished | 2026-11-03 15:41 Asia/Shanghai (~50.8 min wall) |
| Stance | `do_not_auto_launch_further=true` — await user before further stages |

## Per-seed

| Seed | Final | Gate | First nonzero | First ≥0.6 | Peak |
|------|-------|------|---------------|------------|------|
| 911 | 11/12 | PASS | @16000 | @32000 | 12/12 @44k-46k |
| 922 | 11/12 | PASS | @8000 | @30000 | 11/12 @48k-50k |
| 933 | 10/12 | PASS | @16000 | @30000 | 12/12 @30k-48k |

Diagnoses: `results/masac_v4_c_reshape_l0_frozen42_seed{911,922,933}_gate_20261103/` and batch `results/masac_v4_c_reshape_l0_frozen42_gate_20261103/`.

## Forbidden without new auth

Attention / Predictive Maneuverability / L1 / silent +50k continue restack / auto further stages / Cursor cloud agents.
