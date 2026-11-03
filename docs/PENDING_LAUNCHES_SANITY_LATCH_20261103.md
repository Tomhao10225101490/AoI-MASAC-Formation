# Pending launches — sanity + capture-latch (2026-11-03 Asia/Shanghai)

**Status:** LAUNCHED 2026-11-03 13:15 Asia/Shanghai

Continues 511/522/533 +50k all finished gate FAILED (see `docs/MASAC_V4_L0_CONTINUE_BATCH_GATE_20261103.md`).
Free pages recovered after continues exited; launched one sanity + one latch-on arm.

## Active arms

```text
# sanity empty/single — seed 611, 20k — pid 9337
results/masac_v4_sanity_empty_single_20261103_131315

# capture latch ON — seed 711, 20k — pid 9338
results/masac_v4_capture_latch_on_20261103_131315
```

Optional later — baseline twin OFF (still deferred; start when RAM allows after these finish or in parallel if headroom):

```bash
cd /Users/a1/Desktop/research/MASAC
TS=$(date +%Y%m%d_%H%M%S)
caffeinate -dims ./.venv/bin/python scripts/train_masac_v4_capture_latch.py \
  --out results/masac_v4_capture_latch_off_${TS} \
  --protocol configs/masac_v4/l0_capture_latch_off_20261103.json \
  --torch-threads 1
```

## Script fix applied

`train_masac_v4_sanity_empty.py` and `train_masac_v4_capture_latch.py` now `mkdir` `source_snapshot/configs/masac_v4` before copying the protocol (was FileNotFoundError on first launch attempts).

## Stance

- No silent further +50k continue budget stack on the failed continue protocol.
- Forbidden: Attention / Predictive Maneuverability / L1 from this monitor.
- User still chooses next Standard-nav protocol change (e.g. capture/reward reshape C).
