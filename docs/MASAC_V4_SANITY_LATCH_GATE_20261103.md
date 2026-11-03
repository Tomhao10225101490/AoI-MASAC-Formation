# MASAC V4 sanity + capture-latch gate summary (2026-11-03 Asia/Shanghai)

## Continues (prior)

511/522/533 +50k all COMPLETE gate FAILED: 1/12, 5/12, 5/12 (need ≥0.6). See `docs/MASAC_V4_L0_CONTINUE_BATCH_GATE_20261103.md`.

## Sanity empty/single (seed 611, 20k)

- Out: `results/masac_v4_sanity_empty_single_20261103_131315`
- Final: **0/8** vs gate ≥0.75 — **FAILED**
- Every eval 2k…20k also 0/8
- Diagnosis: `results/masac_v4_sanity_empty_single_gate_diagnosis_20261103`
- Dominant: near-goal never stable capture (4 empty) / far-from-goal timeout (4 single-obstacle); **0 agents ever entered capture ball** on any map

## Capture-latch ON (seed 711, 20k)

- Out: `results/masac_v4_capture_latch_on_20261103_131315`
- Final: **0/12** vs gate ≥0.6 — **FAILED**
- Diagnosis: `results/masac_v4_capture_latch_on_gate_diagnosis_20261103`
- Dominant: near-goal never stable capture (11/12); 1 partial ball/speed; latch alone did not help at 20k

## Capture-latch OFF (seed 712, 20k)

- Launched 2026-11-03 13:25 Asia/Shanghai: `results/masac_v4_capture_latch_off_20261103_132438` (pid 13638) to complete A/B

## Paper implication

Learnability floor on simplest maps is **not** demonstrated at 20k. Do not interpret L0 urban continue failures as “only map complexity.” Prefer **C** capture/reward reshape or a new audited protocol (longer sanity / denser goal shaping). No Attention / maneuverability / L1. No silent continue budget restack.
