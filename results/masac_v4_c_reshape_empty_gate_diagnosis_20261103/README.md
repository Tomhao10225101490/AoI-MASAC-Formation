# Path C empty-map reshape gate — 2026-11-03 Asia/Shanghai

**Result: PASSED** — seed 811 @ 40k final **4/4** (need ≥0.75). First nonzero team success under Path-C reshape.

| Field | Value |
|-------|-------|
| Out | `results/masac_v4_c_reshape_empty_20261103_142623` |
| Experiment | `masac_v4_c_reshape_empty_20261103` |
| Seed / steps | 811 / 40k |
| Wall | ~809s (~13.5 min) |
| Gate | ≥0.75 empty-only development maps |
| Final | **4/4 = 1.0** `gate_met=true` |
| Soft capture (this exp) | ball 6.0 m, speed 3.5 m/s, latch on, horizon 350 |
| Frozen L0 (unchanged) | ball 4.0 m, speed 2.0 m/s, latch off |

## Eval curve (4 empty maps)

| Step | Success | Ever-ball maps |
|------|---------|----------------|
| 2000 | 0/4 | 0/4 |
| 4000 | 0/4 | 0/4 |
| 6000 | 0/4 | 0/4 |
| 8000 | 0/4 | 0/4 |
| 10000 | 0/4 | 4/4 |
| 12000 | 0/4 | 4/4 |
| 14000 | 0/4 | 4/4 |
| 16000 | 0/4 | 1/4 |
| 18000 | 0/4 | 2/4 |
| 20000 | 0/4 | 1/4 |
| 22000 | 0/4 | 4/4 |
| 24000 | 0/4 | 0/4 |
| 26000 | 0/4 | 0/4 |
| 28000 | 0/4 | 4/4 |
| 30000 | 0/4 | 4/4 |
| 32000 | 0/4 | 4/4 |
| 34000 | 0/4 | 4/4 |
| 36000 | 0/4 | 4/4 |
| 38000 | 2/4 | 4/4 |
| 40000 | 4/4 | 4/4 |

First nonzero formal eval: **2/4 @ 38k**, then **4/4 @ 40k**. Ball entry appeared much earlier (ever_ball 4/4 from ~10k intermittently) while simultaneous capture lagged until late.

## Caveat

Experiment eval uses **soft train thresholds** (6 m / 3.5 m/s + latch). This proves learnability on empty maps under the reshape, **not** frozen L0 capture (4 m / 2 m/s). Closest-goal distances on final maps are often ~4–6 m — soft pass can succeed where frozen would still fail.

## Stance

- `do_not_auto_launch_l0`: **true** — monitor must **not** auto-start L0.
- Forbidden still: Attention / Predictive Maneuverability / L1 / silent +50k continue restack.
- **Next (needs user):** authorize return to L0 12-map gate with frozen capture, optionally keeping reshape rewards; or further empty/single hardening under frozen thresholds.

## Diagnosis artifact

`results/masac_v4_c_reshape_empty_gate_diagnosis_20261103/`

Authorization: widget t4s5 Path C, 2026-11-03.
