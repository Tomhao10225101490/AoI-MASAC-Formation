# Maneuver paper suite — status (2026-11-04 07:45 Asia/Shanghai)

Standing order: keep TomMac-mini experiments running; iterate after results.
**Do not kill** PIDs 45262 / 45327 / 45356 (+ caffeinates 45265/45330/45359).

Machine snapshot JSON: `results/masac_v4_maneuver_paper_suite_20261103_status_20261104_0745.json`

## Health (~15h22m wall)

| Arm | PID | Seed | Step | step/s | Last eval | Seed finals so far | Alive |
|-----|-----|------|------|--------|-----------|--------------------|-------|
| full | 45262 | 1222/3 | ~58800/80000 | ~2.60 | 1/12 @58k | 1211 FAIL 3/12 (best 4/12@78k) | yes |
| zero | 45327 | **1233**/3 | ~3400/80000 | ~4.4→~3 | 0/12 @2k | 1211 FAIL 1/12; **1222 FAIL 0/12 (best 2/12@68k)** | yes |
| static | 45356 | 1222/3 | ~56800/80000 | ~2.51 | **11/12 @56k** | 1211 PASS 11/12 | yes |

- Suite stamp: `20261103_162345`
- Contract: C reshape + frozen capture 4/2 latch=False; cold start; gate ≥0.60
- `do_not_auto_launch_further=true` (no Attention / L1 / silent restack)

## Seed 1222 — zero arm FINAL (this tick)

- `completion.json`: final_development_success=0.0, gate_met=false @80k
- Best mid-eval: **2/12 @68k** (also 2/12 @60/64/66k); final EVAL 0/12 ever_ball 12/12
- Pattern matches seed 1211 zero: strong ever_ball, weak joint success
- Arm auto-rolled to seed **1233** (healthy; no crash)

## Contrast (same budget window)

- **static** seed 1222 already hit **11/12 @56k** mid-eval (ever_ball 12/12) — on track to match seed-1211 PASS if it holds
- **full** seed 1222 best still only **2/12 @48k**; latest 1/12 @58k — predictive full probe not beating static on this seed so far
- Ablation read (preliminary): static probe ≫ full ≳ zero on success; ever_ball saturates on all arms

## Stance

- Suite continues undisturbed through remaining seeds (full/static finish 1222 + 1233; zero finish 1233).
- Next-iter plan stays **PENDING**: `docs/PENDING_LAUNCHES_MANEUVER_NEXT_ITER_20261103.md` (RAM still shared by 3 arms; no fourth heavy launch).
- After suite gate: prefer documented A/B candidates (throughput optpack / sync-capture curriculum) — ask user if claim-changing.
