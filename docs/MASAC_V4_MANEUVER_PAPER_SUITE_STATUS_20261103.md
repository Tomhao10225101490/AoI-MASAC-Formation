# Maneuver paper suite — mid-run status (2026-11-03 21:05 Asia/Shanghai)

Standing order **t8u**: keep TomMac-mini experiments running; iterate after results.
**Do not kill** PIDs 45262 / 45327 / 45356 (+ caffeinates).

Machine snapshot JSON: `results/masac_v4_maneuver_paper_suite_20261103_status_20261103_2105.json`

## Health

| Arm | PID | Seed | Step | step/s | Last eval | Best success | Best ever_ball | Alive |
|-----|-----|------|------|--------|-----------|--------------|----------------|-------|
| full | 45262 | 1211/3 | 41600/80000 | 2.47 | 0/12 @40k | 1/12 @32k | 11/12 | yes |
| zero | 45327 | 1211/3 | 49400/80000 | 2.93 | 0/12 @48k | 0/12 | 12/12 | yes |
| static | 45356 | 1211/3 | 41600/80000 | 2.47 | 0/12 @40k | 1/12 @32k | 12/12 | yes |

- Suite stamp: `20261103_162345`
- Contract: C reshape + frozen capture 4/2 latch=False; cold start; gate ≥0.60/12
- RAM: ~16 GB, free ≈0.08 GB, compressor busy → **no fourth heavy arm**

## Failure mode (latest eval)

All three arms: almost exclusively **timeout_after_ball**.
Collisions ≈0. `simultaneously_arrived=false`. High `near_goal_low_speed_uncaptured_steps`.
Remaining distance often ~3–13 m with 0–2 UAVs in-ball at timeout.

Read: **navigation / ball-entry is learning; joint capture coordination is the bottleneck** (same qualitative lag Path-C showed before the 11/12 floor).

## Curves (seed 1211, ever_ball / success)

- **full**: ever_ball rises from 0 → 6–11 after 12k; success flicker 1/12 @32k then 0.
- **zero**: ever_ball to 12/12 by 42–48k; success still 0 (capacity control still finds goal vicinity).
- **static**: ever_ball to 12/12 @40k; success flicker 1/12 @32k.

## Stance

- Suite continues undisturbed.
- Perf opts already in tree (`cache_risk`, vectorized probes, threads=1 guidance) apply to **next** launches only — live processes keep imported modules.
- Next experiment plan is **PENDING** (not launched): `docs/PENDING_LAUNCHES_MANEUVER_NEXT_ITER_20261103.md`.
- No Attention / L1 / silent restack without explicit user auth.
