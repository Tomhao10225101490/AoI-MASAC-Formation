# MASAC multi-UAV 3D path planning

Learning-based **independent-goal** multi-UAV navigation in 3D clutter with
Multi-Agent Soft Actor-Critic (MASAC). GitHub repo: **`Tomhao10225101490/masac-maneuverability`** (renamed from
`AoI-MASAC-Formation`). Local folder may still be named `MASAC` / legacy AoI;
the active line is **not** AoI / formation / relay.

**Status (2026-11-03, Asia/Shanghai):** Standard L0 is learnable under Path-C
reward reshape. A paper-facing **Predictive Maneuverability-Aware** suite is
training now. Do not silently restack Attention / L1 / extra capture changes.

---

## What we are doing now

| Stage | Result | Notes |
|-------|--------|-------|
| Path C empty-map floor | **PASSED 4/4** (seed 811 @40k) | Soft capture 6 m / 3.5 m·s⁻¹ + latch for learnability only |
| Path C → Standard L0 (12 maps) | **PASSED 3/3 seeds** | Frozen capture **4 m / 2 m·s⁻¹**, latch **off**; finals **11/12, 11/12, 10/12** |
| Predictive Maneuverability paper suite | **RUNNING** | Arms `full` / `zero` / `static`, 80k×3 seeds each |

**Active run:** `results/masac_v4_maneuver_paper_suite_20261103_{full,zero,static}_20261103_162345`  
**Task-matched Standard-C baseline:** `results/masac_v4_c_reshape_l0_frozen42_20261103_145043`

Frozen research scope: `RESEARCH_SCOPE.md`, `scope-lock.yaml`, `RESEARCH_DECISIONS.md`,
`NOVELTY_ANALYSIS.md`. Maneuverability-Aware MASAC is an **unverified candidate**,
not established novelty.

---

## Problem

- 3–6 UAVs, **independent goals**, continuous velocity/acceleration limits
- Static + moving spherical obstacles, continuous segment collision checks
- Team success = every agent reaches its goal ball at low enough speed
- Evaluation maps are frozen L0 development maps (`830001`–`830012`)

---

## Path C reshape (what made Standard L0 learnable)

After budget continues, empty sanity, and capture-latch A/B all failed to establish
a multi-seed L0 floor, Path **C** changed reward / horizon / capture shaping under a
**new experiment id** (no silent restack of the failed configs).

**Reward magnitudes (train):**

- progress ×1.0, living −0.005, near-goal dense +0.4·clip(1−d/15)
- ball enter +3, capture rise +8, team success +50, collision −20

**L0 return contract (authorized):** keep Path-C rewards; capture train+gate =
frozen **4 m / 2 m·s⁻¹**, **latch=False**. Warm-start from empty-C seed 811 @40k
(weights+optimizer; replay discarded).

Evidence:

- Protocol: `docs/MASAC_V4_C_RESHAPE_PROTOCOL_20261103.md`,
  `docs/MASAC_V4_C_RESHAPE_L0_FROZEN42_PROTOCOL_20261103.md`
- Gate write-up: `docs/MASAC_V4_C_RESHAPE_L0_FROZEN42_GATE_20261103.md`
- Code: `uav_planning/v4_reward_reshape.py`, `scripts/train_masac_v4_c_reshape_l0.py`

---

## Predictive Maneuverability paper suite

Single unverified contribution after reproducible Standard navigation:

- **19** fixed acceleration probes, **2.0 s** CV preview (`dt=0.2`)
- **11** features per probe (direction, clearances, TTC, braking, reserve, flag)
- Shared RiskEncoder + V4 SetEncoder **pool** (no Attention in this suite)
- Observation-only estimates — **not** safety certificates

| Arm | Role |
|-----|------|
| `full` | Innovation (full CV probes) |
| `zero` | Capacity-matched control (zeros into same RiskEncoder) |
| `static` | Ablation (dynamic/peer channels zeroed) |

Shared contract: L0 12 maps, seeds **1211/1222/1233**, **80k** steps/seed, Path-C
rewards, capture **4/2** latch off, gate ≥0.60. Cold start (architecture change
vs Legacy SharedMASAC). Parameter-matched contrast is **inside** the suite;
Standard-C L0 is the task-matched pre-innovation floor only.

- Protocol: `docs/MASAC_V4_MANEUVER_PAPER_SUITE_PROTOCOL_20261103.md`
- Config: `configs/masac_v4/maneuver_paper_suite_20261103.json`
- Train: `scripts/train_masac_v4_maneuver.py`
- Agent / nets: `uav_planning/v4_maneuver_masac.py`, `v4_networks.py`, `v4_prediction.py`
- Math argument upgrades: `docs/MATH_ARGUMENT_IMPROVEMENTS_20261103.md`
- Training speed notes: `docs/TRAINING_OPTIMIZATION_MANEUVER_20261103.md`

### Do not silent-restack

Without a fresh user authorization and a **new experiment id**:

- no Attention arms
- no L1 maps
- no capture/latch policy change mid-suite
- no auto-launch after gates (`do_not_auto_launch_further=true`)

---

## How to train / evaluate

Use the local venv on TomMac-mini:

```bash
cd /Users/a1/Desktop/research/MASAC
.venv/bin/python -m pytest -q tests/test_v4_maneuver_masac.py tests/test_masac_v4_c_reshape.py

# Path-C empty floor (reference)
.venv/bin/python scripts/train_masac_v4_c_reshape.py

# Path-C L0 return (frozen 4/2)
.venv/bin/python scripts/train_masac_v4_c_reshape_l0.py

# Maneuver paper suite arms (do not duplicate a live stamp)
.venv/bin/python scripts/train_masac_v4_maneuver.py --probe-mode full
.venv/bin/python scripts/train_masac_v4_maneuver.py --probe-mode zero
.venv/bin/python scripts/train_masac_v4_maneuver.py --probe-mode static
```

Inspect live status (does not start training):

```bash
cat results/masac_v4_maneuver_paper_suite_20261103_full_20261103_162345/status.json
cat results/masac_v4_active_run.json
```

---

## Key paths

| Path | Role |
|------|------|
| `uav_planning/` | Active V3/V4 planning stack (env, maps, MASAC, reshape, maneuver) |
| `configs/masac_v4/` | Frozen experiment protocols + auth JSON |
| `docs/MASAC_V4_*.md` | Gate / protocol write-ups |
| `docs/design/MASAC_V4_*.md` | Architecture / map / research specs |
| `scripts/train_masac_v4_*.py` | Launchers |
| `results/masac_v4_*` | Local runs (mostly gitignored; gate diagnoses may be tracked) |
| `archive/scope_before_masac_20261102/` | Pre-freeze AoI/formation scope backup |

Historical AoI / relay / DC-RRT / GCOPTER / VOI code and notes were pruned from the
active tree or left only as archive. They are **outside** the frozen research
direction.

---

## Branch / repo

- GitHub: [`Tomhao10225101490/masac-maneuverability`](https://github.com/Tomhao10225101490/masac-maneuverability)
  (renamed 2026-11-03 Asia/Shanghai from `AoI-MASAC-Formation`; old URL redirects).
- Clone: `git clone https://github.com/Tomhao10225101490/masac-maneuverability.git`
- Active development on `research/wowmom2027-upgrade`; **`main`** carries Path-C + maneuver.
- Local Mac only for training; do not disturb live suite PIDs under
  `results/masac_v4_maneuver_paper_suite_20261103_*_162345`.
- Theory / formula hygiene: `docs/MATH_ARGUMENT_IMPROVEMENTS_20261103.md`
- Next-launch training opts: `docs/TRAINING_OPTIMIZATION_MANEUVER_20261103.md`
