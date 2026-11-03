# Pending / active — Path C L0 return frozen 4/2 (2026-11-03 Asia/Shanghai)

**Authorization:** user widget **t4s11**. Local TomMac-mini only. No cloud agents.

**Status:** LAUNCHING / RUNNING L0 12-map reshape with frozen capture (do **not** auto-launch further stages).

## Arm

| Item | Value |
|------|-------|
| experiment_id | `masac_v4_c_reshape_l0_frozen42_20261103` |
| entrypoint | `scripts/train_masac_v4_c_reshape_l0.py` |
| protocol | `configs/masac_v4/c_reshape_l0_frozen42_20261103.json` |
| seeds | 911, 922, 933 (sequential) |
| steps | 50000 each |
| warm-start | empty C `.../reshape_811/checkpoint_040000` (weights+opt; replay discarded) |
| capture | frozen 4 m / 2 m/s, latch=False (train+gate) |
| rewards | Path-C reshape magnitudes retained |
| gate | ≥0.60 on 12 L0 development maps **each** seed |
| forbidden | Attention / maneuverability / L1 / silent continue restack / auto further |

## Prior empty (complete)

- `results/masac_v4_c_reshape_empty_20261103_142623` seed 811 40k **4/4 PASS** soft 6/3.5

## Monitor

`results/masac_v4_active_run.json`, `results/masac_v4_parallel_runs.json`, run `status.json` / `train.log`.

`do_not_auto_launch_further=true`.

## Live

- out: `results/masac_v4_c_reshape_l0_frozen42_20261103_145043`
- train_pid: 38603
- caffeinate_pid: 38605
- launched_at_asia_shanghai: 2026-11-03 14:50 Asia/Shanghai
- warm_start: `results/masac_v4_c_reshape_empty_20261103_142623/reshape_811/checkpoint_040000`
- mode: sequential seeds 911→922→933 × 50k; torch_threads=4
