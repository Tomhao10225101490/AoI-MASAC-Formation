# MASAC-3D-UAV-Path-Planning — mandatory takeover rules

Read RESEARCH_SCOPE.md, scope-lock.yaml, RESEARCH_DECISIONS.md and NOVELTY_ANALYSIS.md before edits.
RESEARCH_DIRECTION = FROZEN (user authorized 2026-11-02). Only MASAC multi-UAV 3D path planning.
Never pivot to AoI, communication, formation control or traditional-search research without explicit user approval.
Maneuverability-Aware MASAC is an unverified candidate, not established novelty.
No unsupported GNN/Diffusion/Transformer additions. Diagnose negative results within scope.
Hyperparameter/network/mechanism changes require recorded rationale, a **new experiment id**, and evidence.
Other planners are comparison baselines only. Preserve independent seeds, failures and fair budgets.
Never claim training, convergence, safety or novelty without matching saved evidence.
Preserve uncommitted work and historical results; no destructive reset/clean/force-push of live runs.
Use local .venv/bin/python; distinguish smoke training from scientific confirmation.
CI checks syntax/consistency, not scientific meaning. Scope changes require human review.

## Active line (2026-11-03)

- Path-C reshape established empty floor then Standard L0 multi-seed pass (11/12, 11/12, 10/12)
  under frozen capture 4 m / 2 m·s⁻¹, latch off. See docs/MASAC_V4_C_RESHAPE_L0_FROZEN42_GATE_20261103.md.
- Predictive Maneuverability paper suite (full/zero/static, 80k×3) is the authorized next stage.
  Protocol: docs/MASAC_V4_MANEUVER_PAPER_SUITE_PROTOCOL_20261103.md
  Code: uav_planning/v4_*.py, scripts/train_masac_v4_maneuver.py
- Do **not** stop or overwrite results/masac_v4_maneuver_paper_suite_*_162345 while PIDs are live.
- do_not_auto_launch_further: no Attention / L1 / capture restack without a new user authorization.
- V3 structured design docs under docs/design/ are historical path-to-L0 context; do not mix V2/V3/V4 result tables.
