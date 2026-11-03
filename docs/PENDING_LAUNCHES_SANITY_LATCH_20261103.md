# Pending launches — sanity + capture-latch (2026-11-03 Asia/Shanghai)

**Status:** ALL pending launches COMPLETE; **all gates FAILED**. Path C empty reshape later **PASSED** (see below). Machine idle as of ~14:40 Asia/Shanghai. Awaiting user authorization for L0 return.

## Completed (pre-C)

| Arm | Out | Final | Gate | Diagnosis |
|-----|-----|-------|------|-----------|
| sanity empty/single seed 611 20k | `results/masac_v4_sanity_empty_single_20261103_131315` | **0/8** | ≥0.75 | `results/masac_v4_sanity_empty_single_gate_diagnosis_20261103` |
| capture latch ON seed 711 20k | `results/masac_v4_capture_latch_on_20261103_131315` | **0/12** | ≥0.6 | `results/masac_v4_capture_latch_on_gate_diagnosis_20261103` |
| capture latch OFF seed 712 20k | `results/masac_v4_capture_latch_off_20261103_132438` | **0/12** | ≥0.6 | `results/masac_v4_capture_latch_off_gate_diagnosis_20261103` |

## Path C (user widget t4s5)

| Arm | Out | Final | Gate | Diagnosis |
|-----|-----|-------|------|-----------|
| c_reshape empty seed 811 40k | `results/masac_v4_c_reshape_empty_20261103_142623` | **4/4 PASSED** | ≥0.75 | `results/masac_v4_c_reshape_empty_gate_diagnosis_20261103` / `docs/MASAC_V4_C_RESHAPE_EMPTY_GATE_20261103.md` |

Soft capture for this exp: 6 m / 3.5 m/s + latch. Frozen L0 remains 4 m / 2 m/s.

## Stance

- No silent further +50k continue budget stack on the failed continue protocol.
- Forbidden: Attention / Predictive Maneuverability / L1 from this monitor.
- **`do_not_auto_launch_l0`:** true until user authorizes return to L0 12-map gate.
- **User chooses next:** authorize L0 return (with reshape rewards and/or frozen capture), or further empty/single hardening under frozen thresholds.
