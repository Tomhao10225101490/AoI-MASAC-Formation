# Pending launches — sanity + capture-latch (2026-11-03 Asia/Shanghai)

**Status:** sanity + latch-on COMPLETE (both gate FAILED); latch-off RUNNING since 2026-11-03 13:25 Asia/Shanghai

## Completed

| Arm | Out | Final | Gate | Diagnosis |
|-----|-----|-------|------|-----------|
| sanity empty/single seed 611 20k | `results/masac_v4_sanity_empty_single_20261103_131315` | **0/8** | ≥0.75 | `results/masac_v4_sanity_empty_single_gate_diagnosis_20261103` |
| capture latch ON seed 711 20k | `results/masac_v4_capture_latch_on_20261103_131315` | **0/12** | ≥0.6 | `results/masac_v4_capture_latch_on_gate_diagnosis_20261103` |

Audit highlight: almost no ball entry at 20k (empty maps lagging UAV min-dist ~12–13 m; single-obstacle ~23–32 m; latch-on mostly near-goal never capture). Learnability floor not established at this budget.

## Active

```text
# capture latch OFF baseline twin — seed 712, 20k — pid 13638
results/masac_v4_capture_latch_off_20261103_132438
```

## Stance

- No silent further +50k continue budget stack on the failed continue protocol.
- Forbidden: Attention / Predictive Maneuverability / L1 from this monitor.
- User still chooses next Standard-nav protocol change (e.g. capture/reward reshape **C**) or a new audited longer-sanity protocol.
