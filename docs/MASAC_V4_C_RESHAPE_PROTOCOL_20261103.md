# MASAC V4 Path C — capture / reward reshape protocol (2026-11-03 Asia/Shanghai)

## Decision

User widget **t4s5** chose **C**: reshape reward / capture / horizon with a **new experiment id**, prove nonzero success on empty maps first, then return to L0. Local TomMac-mini only. No Cursor cloud agents.

## Why C (evidence)

| Arm | Result |
|-----|--------|
| L0 continues 511/522/533 ~20k–100k | 1/12, 5/12, 5/12 — near-goal / non-simultaneous |
| Sanity empty+single seed 611 20k | **0/8** — empty lag min-dist ~12–13 m, ever_ball=0 |
| Latch ON 711 / OFF 712 20k | **0/12** / **0/12** |

Learnability floor not shown at 20k. Approach credit + capture simultaneity/speed gate dominate; not map density alone. Forbidden from this path: Attention / Predictive Maneuverability / L1 / silent +50k continue restack.

## Frozen baseline (unchanged files)

`uav_planning/v4_environment.py` stays byte-stable for L0 continues:

- progress `(before−after)/2`, living `−0.02`, capture `+5·Δ` (leave can `−5`)
- arrived = `dist < 4` ∧ `speed < 2`, not sticky; team success `+20`

## Path-C reshape (new module)

`uav_planning/v4_reward_reshape.py` — subclass; Standard obs only.

| Knob | Frozen | Path C |
|------|--------|--------|
| progress_scale | 0.5 | **1.0** |
| living_cost | 0.02 | **0.005** |
| near_goal dense | none | **+0.4·clip(1−d/15)** inside 15 m |
| ball-enter (dist only) | none | **+3** rising edge |
| capture rising edge | +5 | **+8**, latch ON |
| team success | +20 | **+50** |
| ball / speed (this exp) | 4 / 2 | **6 / 3.5** |
| latch | off | **on** |
| horizon | 350 | 350 (goals ≈40 m; travel budget ample) |

**Eval for this experiment** uses the soft train thresholds. **Frozen L0 return gate** remains `dist<4` ∧ `speed<2` simultaneous (documented; not applied here).

## Run

| Item | Value |
|------|-------|
| experiment_id | `masac_v4_c_reshape_empty_20261103` |
| entrypoint | `scripts/train_masac_v4_c_reshape.py` |
| protocol | `configs/masac_v4/c_reshape_empty_20261103.json` |
| kinds | **empty only** |
| seed | 811 |
| budget | **40k** steps (new audited empty-floor budget — not a continue restack) |
| eval maps | 870001–870004 (4 empty) |
| gate | ≥ **0.75** (3/4) joint success under soft capture |
| device | cpu (project pattern); torch threads ≥4 when machine idle |

```bash
cd /Users/a1/Desktop/research/MASAC
TS=$(date +%Y%m%d_%H%M%S)
caffeinate -dims ./.venv/bin/python scripts/train_masac_v4_c_reshape.py \
  --out results/masac_v4_c_reshape_empty_${TS} \
  --protocol configs/masac_v4/c_reshape_empty_20261103.json \
  --authorization-protocol configs/masac_v4/c_reshape_empty_20261103.auth.json \
  --torch-threads 4
```

## Gate policy

- **PASS** → declare empty learnability floor under path-C reshape; **do not auto-launch L0**; await explicit authorization to return to 12-map frozen gate (possibly with transfer / tightened capture).
- **FAIL** → diagnose; further reshape or longer audited empty protocol; still no Attention / L1 / silent continue restack.

## Status files

Updates `results/masac_v4_active_run.json` and `results/masac_v4_parallel_runs.json` with `kind=v4_c_reshape_empty` and `do_not_auto_launch_l0=true` so routine `v4-l0` monitors without escalating.
