# Predictive Maneuverability-Aware MASAC — paper suite protocol (2026-11-03)

Asia/Shanghai. Authorization: user widget **t4s17** + phone steering for a comprehensive paper-facing suite (not a thin smoke arm).

## Why this suite

Standard L0 Path-C already passed (`masac_v4_c_reshape_l0_frozen42_20261103`, seeds 911/922/933 → 11/12, 11/12, 10/12) with C reward reshape and frozen capture 4 m / 2 m·s⁻¹ (latch off). Design docs freeze **Predictive Maneuverability-Aware MASAC** as the single unverified candidate contribution after reproducible navigation.

## Innovation (plain numbers)

- **19** fixed acceleration probes, **2.0 s** sampled preview at **dt = 0.2 s** (10 interior steps + t0).
- Per probe: **11** features = canonical direction **3** + clearances static/moving/peer/boundary **4** + TTC + braking distance + reserve + positive-preview flag.
- Clearances clipped/normalized at **40 m**; TTC normalized by **2 s**.
- Shared RiskEncoder **11→64→64**, mean/max pool → **128**; V4 SetEncoder **pool** → **384**; fused actor input **512**.
- **No Attention** in this suite (architecture frozen to pool after Standard gate).
- Probes are observation-only CV estimates — **not** safety certificates; no hard action filter.

## Arms (capacity / reward / budget matched)

| Arm | Role | Probe mode |
|-----|------|------------|
| `full` | Innovation | full CV probes |
| `zero` | Control | zeros into same RiskEncoder |
| `static` | Ablation | dynamic/peer channels zeroed |
| `shuffle` | Optional ablation | risk scalars remapped across fixed directions (not default-launched) |

## Shared training contract

- Maps: frozen L0 12 development maps (`830001`–`830012`)
- Seeds (paired across arms): **1211 / 1222 / 1233**
- Budget: **80 000** steps/seed, warmup **1000**, eval every **2000**, ckpt every **10 000**
- Reward: Path-C reshape (progress×1.0, living −0.005, near-goal +0.4·clip(1−d/15), ball enter +3, capture +8, team +50, collision −20)
- Capture train+gate: **4 m / 2 m·s⁻¹**, **latch=False**
- Gate: ≥ **0.60** joint success (8/12) per seed
- Device: CPU; local Mac only; no CloudAgent
- Warm-start: **cold** — Legacy SharedMASAC Path-C checkpoints are architecture-incompatible with V4 pool+RiskEncoder (~506k vs ~65k actor params). Documented; not a silent partial load.

## Baseline comparison path

- **Task-matched Standard-C**: `results/masac_v4_c_reshape_l0_frozen42_20261103_145043` (same maps/rewards/capture; legacy encoder). Report as pre-innovation floor; do **not** claim parameter-matched vs Standard-C.
- **Parameter-matched contrast**: full vs zero vs static inside this suite.

## Paper metrics logged per eval map

joint success, ever-in-ball, path efficiency (direct/path on success), timeout/collision, probe clearance / TTC / positive-preview fraction, near-goal low-speed uncaptured steps.

## Non-goals / do-not-auto

No Attention arms, no L1, no silent unrelated restack, no auto-launch after gates. `do_not_auto_launch_further=true`.

## Entrypoints

- Protocol: `configs/masac_v4/maneuver_paper_suite_20261103.json`
- Auth: `configs/masac_v4/maneuver_paper_suite_20261103.auth.json`
- Train: `scripts/train_masac_v4_maneuver.py --probe-mode {full,zero,static}`
- Agent: `uav_planning/v4_maneuver_masac.py`
