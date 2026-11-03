# SUPERSEDED by docs/PENDING_LAUNCHES_C_RESHAPE_L0_FROZEN42_20261103.md

Empty Path C completed PASS 4/4; user t4s11 authorized L0 return.

# Pending / active — Path C reshape empty floor (2026-11-03 Asia/Shanghai)

**Authorization:** user widget **t4s5** → C. Local TomMac-mini only. No cloud agents.

**Status:** LAUNCHING / RUNNING empty-map reshape (do **not** auto-launch L0).

## Arm

| Item | Value |
|------|-------|
| experiment_id | `masac_v4_c_reshape_empty_20261103` |
| entrypoint | `scripts/train_masac_v4_c_reshape.py` |
| protocol | `configs/masac_v4/c_reshape_empty_20261103.json` |
| seed | 811 |
| steps | 40000 |
| kinds | empty only |
| gate | ≥0.75 on 4 empty maps (soft capture 6 m / 3.5 m/s / latch) |
| forbidden | Attention / maneuverability / L1 / silent continue restack / auto L0 |

## Launch

```bash
cd /Users/a1/Desktop/research/MASAC
TS=$(date +%Y%m%d_%H%M%S)
nohup caffeinate -dims ./.venv/bin/python scripts/train_masac_v4_c_reshape.py \
  --out results/masac_v4_c_reshape_empty_${TS} \
  --protocol configs/masac_v4/c_reshape_empty_20261103.json \
  --authorization-protocol configs/masac_v4/c_reshape_empty_20261103.auth.json \
  --torch-threads 4 \
  > results/masac_v4_c_reshape_empty_${TS}.launcher.out 2>&1 &
```

Monitor: `results/masac_v4_active_run.json`, `results/masac_v4_parallel_runs.json`, run `status.json` / `train.log`.

## Live

- out: `results/masac_v4_c_reshape_empty_20261103_142623`
- launcher_pid: 33473
- launched_at_asia_shanghai: 2026-11-03 14:26 Asia/Shanghai
