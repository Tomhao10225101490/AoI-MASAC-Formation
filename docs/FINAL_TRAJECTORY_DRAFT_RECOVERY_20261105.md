# Recovery: unexecuted final-trajectory diagnostic draft

2026-11-05, Asia/Shanghai. Completed actor results were pushed to both branches
as83ea09073bf5461d579835fc1bffc868373c7bf0. The trajectory-analysis agent then
reported account usage unavailable. No billing/account changes were made.
The sustained objective is unfinished; keep the existing ACTIVE15-minute
heartbeat idmasac and resume when normal usage becomes available. This is not
a user pause, a completed research goal or a reason to duplicate the automation.

## Preserved state

- `design/MASAC_V4_FINAL_TRAJECTORY_DWELL_PROTOCOL_20261105.md` and
  `../scripts/audit_sync_final_trajectory_dwell.py` are **unreviewed drafts**.
  They were not executed, syntax-checked or registered as a final protocol.
  Source CLI has an explicit DRAFT_REVIEW_PENDING exit guard.
- Draft file hashes: `FINAL_TRAJECTORY_DRAFT_FILES_20261105.json` in this directory.
  Preserve existing work; do not overwrite it with a new agent's implementation.
- No Python training/analysis process remained at the recovery check. Completed
  40k training, content audit and actor audit must not be restarted.
- Never treat a draft commit or a future partial summary.json as completion.

## Intended smallest analysis

Use exactly72 existing final40000 trajectories from the six static control/B
histories, seeds1511/1522/1533 and maps830001..830012. No policy/checkpoint load,
new rollout, environment creation/step, training, sealed map or budget extension.
Inputs are original evaluation JSONs, NPZ arrays and already checked per-seed
trajectory summary JSONs. Record all input/source/config/protocol/script hashes.
Each new output needs own manifest/status/error and a final completion marker.

NPZ fields: positions/velocities(T+1,3,3)float64 include reset; actions(T,3,3)
float32; goals(3,3)float64; buildings(M,2,3)float64. Use allow_pickle=False.
Evaluation JSON includes reshape coefficients, terminal/truncation/collision
and capture information, **but has no accumulated evaluation reward/return**.
Source-derived reward components may be computed, while independent logged-return
validation must remain UNAVAILABLE. Do not substitute training-episode returns.

Prespecified descriptive bins: distance[0,4),[4,5),[5,10),>=10 crossed with
speed<2/>=2. Near-stall label:4<=distance<10 and speed<.5m/s. Report every map
and agent, frame denominators, occupancy counts, longest consecutive run and
>=25post-step-frame run count. Keep failed/empty/unavailable cases explicit.

## Review corrections that must be checked before registration

1. The shaper initializes capture/ever-ball memories to **False**, not the reset
   geometry. Reset is used for first progress/velocity differences. Validate this
   directly against current frozen source, even if all current starts are far.
2. Realized acceleration penalty uses velocity difference/dt after speed limiting;
   command acceleration is different. Saved actions allow a pure-limiter physics
   consistency check, not an environment rollout or reward-reference validation.
3. Include the last true-terminal transition. Common+50 remains50 after team
   averaging. Timeout is a truncation; never invent its bootstrap or soft value.
4. Report raw undiscounted and gamma=.99 discounted formula-derived components
   separately, first post-step reward indexed0. First-ever ball and signed
   capture entry/exit memory must follow the source; A/C/J counts must agree
   with the existing trajectory checker.
5. **Time definition:**25consecutive sampled post-step frames correspond to25h=5s
   of occupancy-equivalent step count, but their first/last sample timestamps
   span24h=4.8s. They do not prove continuous-time stillness between samples.
   Check every draft statement that currently abbreviates this as continuous5s.
6. Distinguish agent-frame, team-step, agent-map and map denominators. These are
   repeated observations from3training replicates, not72independent training runs.
7. Missing required data must produce UNAVAILABLE with a reason, not fabricated
   numbers or an overstated complete reward validation. Dwell availability and
   reference-return availability must be recorded separately.

After full review/fixes, syntax-check, remove the CLI draft guard, and **push the
final reviewed protocol/code before first execution**. Show the complete CLI,
tee per-map progress, use a new output directory and inspect for any running
copy/status/completion beforehand. The previous registered study's decision,
40k budgets, capture thresholds and sealed set remain unchanged.
