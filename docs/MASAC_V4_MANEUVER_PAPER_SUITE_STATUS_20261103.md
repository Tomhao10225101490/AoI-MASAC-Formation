# Maneuver paper suite — status (2026-11-04 18:40 Asia/Shanghai)

Standing order: keep TomMac-mini experiments running; iterate after results.
**Suite stamp `20261103_162345` is COMPLETE.** No live train/caffeinate PIDs (machine idle).

Gate notes:
- zero: `docs/MASAC_V4_MANEUVER_ZERO_GATE_20261103.md` (+ `.json`)
- full: `docs/MASAC_V4_MANEUVER_FULL_GATE_20261103.md` (+ `.json`)
- static: `docs/MASAC_V4_MANEUVER_STATIC_GATE_20261103.md` (+ `.json`)

## Suite FINAL — all three arms closed

| Arm | Finished (Asia/Shanghai) | Gate | Finals | Bests |
|-----|--------------------------|------|--------|-------|
| zero | 2026-11-04 14:52 | **FAIL 0/3** | 1/12, 0/12, 1/12 | — |
| full | 2026-11-04 18:21 (~26.0 h) | **FAIL 1/3** | 3/12, 4/12, **8/12** | 4@76k, 4@80k, **9@72k** |
| static | 2026-11-04 18:34 (~26.2 h) | **FAIL 2/3** | **11/12**, **10/12**, 5/12 | **11@80k**, **11@58k**, **9@70k** |

Contract: C reshape + frozen capture 4/2 latch=False; cold start; gate ≥0.60 each seed; V4 pool+RiskEncoder.
`do_not_auto_launch_further=true` (no Attention / L1 / silent restack without new auth).

## Ablation narrative (paper-ready contrast)

1. **zero FAIL 0/3** — RiskEncoder *capacity* without probe content cannot pass joint-capture gate.
2. **static FAIL 2/3 but strongest** — 2/3 seeds PASS at high margin (11/12, 10/12); seed 1233 late-collapsed 9/12@70k → 5/12@80k.
3. **full FAIL 1/3** — only 1233 PASS 8/12; 1211/1222 weak 3–4/12. Predictive/full CV channels are **not** a reliable win vs static under this protocol.
4. Rank under matched capacity: **static ≫ full ≫ zero**.

## Stance / next

- Evidence committed under per-arm gate notes; machine idle after ~26 h three-arm run.
- Next-iter plan: `docs/PENDING_LAUNCHES_MANEUVER_NEXT_ITER_20261103.md` — prefer **A** (throughput optpack) or **B** (sync-capture curriculum); **ask user** before launch (claim/method choice).
- Paper docs: `docs/PAPER_PLAN_WOWMOM2027.md`, `docs/RELATED_WORK_FRONTIER_20261104.md`, `docs/PAPER_OUTLINE_DRAFT.md`.
- Forbidden without new auth: Attention / L1 / silent budget restack / Cursor cloud agents.
