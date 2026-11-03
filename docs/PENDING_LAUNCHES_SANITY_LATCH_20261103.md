# Pending launches — sanity + capture-latch (2026-11-03 Asia/Shanghai)

**Status at write:** DEFERRED for RAM. TomMac-mini had ~16 GB unified memory with
~4.5 GB Python RSS across 3 L0 continues + critic + multiple MPS/CPU bench workers,
load ~16–17, free pages ~tens of MB. Adding two more full replay arms risked OOM-killing continues.

**Do not kill:** 533 continue ~98782; 511/522 parallel continues; critic; mps workers.

When headroom returns (continues finish or benches exit), launch **one** sanity and **one** latch arm:

```bash
cd /Users/a1/Desktop/research/MASAC
TS=$(date +%Y%m%d_%H%M%S)

# Arm 1 — sanity (empty + single-obstacle), 20k, seed 611
caffeinate -dims ./.venv/bin/python scripts/train_masac_v4_sanity_empty.py \
  --out results/masac_v4_sanity_empty_single_${TS} \
  --protocol configs/masac_v4/l0_sanity_empty_single_20261103.json \
  --authorization-protocol configs/masac_v4/l0_sanity_empty_single_20261103.auth.json \
  --torch-threads 1

# Arm 2 — capture latch ON (prefer latch over baseline twin while RAM tight)
caffeinate -dims ./.venv/bin/python scripts/train_masac_v4_capture_latch.py \
  --out results/masac_v4_capture_latch_on_${TS} \
  --protocol configs/masac_v4/l0_capture_latch_on_20261103.json \
  --authorization-protocol configs/masac_v4/l0_capture_latch_on_20261103.auth.json \
  --torch-threads 1

# Optional later — baseline twin OFF
# caffeinate -dims ./.venv/bin/python scripts/train_masac_v4_capture_latch.py \
#   --out results/masac_v4_capture_latch_off_${TS} \
#   --protocol configs/masac_v4/l0_capture_latch_off_20261103.json \
#   --torch-threads 1
```

Smoke (safe anytime; tiny replay):

```bash
./.venv/bin/python scripts/train_masac_v4_sanity_empty.py --out results/masac_v4_sanity_smoke_${TS} --smoke
./.venv/bin/python scripts/train_masac_v4_capture_latch.py --out results/masac_v4_latch_smoke_${TS} --smoke
```

Per-arm locks: `results/locks/masac_v4_sanity_*.lock`, `results/locks/masac_v4_capture_*.lock`.
