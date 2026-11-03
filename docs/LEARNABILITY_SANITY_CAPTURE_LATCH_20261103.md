# Learnability floor: sanity maps + capture-latch isolation (2026-11-03)

## Why this exists (paper claim path)

The WoWMoM / learning-strategy paper track needs an honest **learnability** story:

1. **SharedMASAC + Standard observation can learn navigate → brake → joint capture** on the *simplest* feasible maps (empty volume, then one box).
2. Only after that floor is positive do we interpret failures on denser L0 urban families as **coordination / capture / path-execution** problems rather than “the stack cannot learn at all.”
3. Gate diagnosis (`results/masac_v4_l0_legacy_50k_gate_diagnosis_20261103`) showed timeouts dominated by **partial ball entry + speed-gate / staggered capture**, not map infeasibility (reference 12/12). Capture-latch A/B isolates whether sticky capture credit assignment unlocks joint success without changing the observation or adding Attention / maneuverability.

Novelty (Predictive Maneuverability-Aware MASAC) stays **locked** until Standard navigation gates pass. These arms are evidence instruments, not novelty claims.

## Arm A — Empty / single-obstacle sanity

| Item | Value |
|------|-------|
| Entrypoint | `scripts/train_masac_v4_sanity_empty.py` |
| Maps module | `uav_planning/v4_sanity_maps.py` (does **not** edit frozen `v4_maps.py`) |
| Protocol | `configs/masac_v4/l0_sanity_empty_single_20261103.json` |
| Fixtures | `data/masac_v4_sanity/catalog.json` (4 empty + 4 single-obstacle) |
| Stack | SharedMASAC / InstrumentedLegacyMASAC, Standard obs |
| Budget (default) | 20k steps, seed 611, gate ≥ 0.75 on 8 maps |

Empty = `density=0` L0. Single-obstacle = same empty scene + one mid-path AABB that keeps starts/goals valid.

## Arm B — Capture-latch isolation (Standard L0)

| Item | Value |
|------|-------|
| Entrypoint | `scripts/train_masac_v4_capture_latch.py` |
| Module | `uav_planning/v4_capture_latch.py` (subclass; frozen `v4_environment.py` untouched) |
| Latch ON protocol | `configs/masac_v4/l0_capture_latch_on_20261103.json` (`capture_latch=true`, seed 711) |
| Baseline twin | `configs/masac_v4/l0_capture_latch_off_20261103.json` (`capture_latch=false`, seed 712) |
| Maps | Same 12 frozen L0 development maps / family schedule |
| Budget (default) | 20k steps each |

**Frozen rule:** `dist < 4` AND `speed < 2`, not sticky; leaving the ball can emit `delta=-1` → −5 reward.  
**Latch rule:** once satisfied, `captured` stays true; leave does not emit −5; team success when all are latched.

## Frozen-hash hygiene

Ongoing L0 continues depend on byte-identical:

- `scripts/train_masac_v4_l0.py`
- `uav_planning/v4_environment.py`
- `uav_planning/v4_maps.py`
- `uav_planning/shared_masac.py`
- (other `SOURCES` in the frozen runner)

Sanity / latch land in **new** files and entrypoints (same pattern as `train_masac_v4_l0_parallel_continue.py`).

## Authorization

User intensity `t1u` (2026-11-03) plus local-only wiring `t2u`/`t3u`. No Attention / 19-probe maneuverability training.
