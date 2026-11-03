# MASAC V4 Path C — L0 return with frozen capture 4/2 (2026-11-03 Asia/Shanghai)

## Decision

User widget **t4s11** authorized: return to **L0 12-map gate**, **keep Path-C reward reshape**, use **frozen capture 4 m / 2 m/s** for train+gate (L0-default **latch=False**). Local TomMac-mini only. No Cursor cloud agents.

## Provenance

| Prior | Result |
|-------|--------|
| Path C empty `masac_v4_c_reshape_empty_20261103` | **PASSED** 4/4 @40k soft 6/3.5+latch |
| L0 continues 511/522/533 | FAILED 1/12, 5/12, 5/12 |
| Sanity / latch on/off @20k | 0/8, 0/12, 0/12 |

## What changes vs empty Path C

| Knob | Empty Path C | This L0 return |
|------|--------------|----------------|
| Reward reshape (progress/living/near/ball/cap/team) | C magnitudes | **Same C magnitudes** |
| `train_ball_m` / `train_speed_mps` | 6.0 / 3.5 | **4.0 / 2.0 (frozen)** |
| `capture_latch` | True | **False (L0 default)** |
| Maps | 4 empty | **12 frozen L0 development** |
| Gate | ≥0.75 | **≥0.60 each seed** |
| Seeds | 811 | **911 / 922 / 933** (new namespace) |
| Budget | 40k | **50k** (audited L0-return; not silent continue restack) |

Frozen `uav_planning/v4_environment.py` / `scripts/train_masac_v4_l0.py` stay byte-stable. New entrypoint: `scripts/train_masac_v4_c_reshape_l0.py`.

## Warm-start

Prefer weights from empty C checkpoint:

`results/masac_v4_c_reshape_empty_20261103_142623/reshape_811/checkpoint_040000`

Mode: **weights + optimizers only**; soft-capture replay / env / step discarded (incompatible with frozen 4/2). Fresh RNGs per seed. If checkpoint missing → cold start documented in `warm_start.json`.

## Run

| Item | Value |
|------|-------|
| experiment_id | `masac_v4_c_reshape_l0_frozen42_20261103` |
| entrypoint | `scripts/train_masac_v4_c_reshape_l0.py` |
| protocol | `configs/masac_v4/c_reshape_l0_frozen42_20261103.json` |
| gate | ≥0.6 on 12 maps (frozen 4/2, latch off) |
| device | cpu; sequential 3 seeds (RAM ~7GB free — avoid 3×100k-replay parallel) |
| forbidden | Attention / maneuver / L1 / silent continue restack / auto further stages |

```bash
cd /Users/a1/Desktop/research/MASAC
TS=$(date +%Y%m%d_%H%M%S)
nohup caffeinate -dims ./.venv/bin/python scripts/train_masac_v4_c_reshape_l0.py \
  --out results/masac_v4_c_reshape_l0_frozen42_${TS} \
  --protocol configs/masac_v4/c_reshape_l0_frozen42_20261103.json \
  --authorization-protocol configs/masac_v4/c_reshape_l0_frozen42_20261103.auth.json \
  --torch-threads 4 \
  > results/masac_v4_c_reshape_l0_frozen42_${TS}.launcher.out 2>&1 &
```

## Gate policy

- **PASS** (all seeds ≥0.6) → declare L0 floor under Path-C reshape + frozen capture; **do not auto-launch further**.
- **FAIL** → diagnose; no Attention / L1 / silent budget restack.
