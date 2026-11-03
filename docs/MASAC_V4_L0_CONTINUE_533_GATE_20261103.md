# V4 L0 continue 533 +50k — gate diagnosis (2026-11-03 Asia/Shanghai)

- **Run:** `results/masac_v4_l0_continue_533_plus50k_20261103_121717`
- **Result:** COMPLETE, **gate FAILED** — final development success **5/12 (0.417)** vs threshold **0.6**
- **Budget:** resumed from legacy `checkpoint_050000` → **100000** steps (authorized B, not silent expand)
- **Legacy at 50k:** 6/12; after +50k: 5/12 (no durable gate pass)

## Eval volatility
Nonzero peaks during continue: 10/12@56k, 8/12@82k, 6/12@86k, 6/12@94k, 8/12@98k, then **5/12@100k**. Policy intermittently clears ≥60% mid-eval but does not hold.

## Constraint
Do **not** silently restack budget on the same continue protocol. Forbidden until new user auth: Attention arms, Predictive Maneuverability-Aware, L1 escalation.

## Still running
Parallel continues 511/522 (~63k/100k at monitor). Full three-seed gate summary after both finish.
