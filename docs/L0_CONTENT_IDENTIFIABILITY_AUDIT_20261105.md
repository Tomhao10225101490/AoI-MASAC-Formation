# L0 predictive-content identifiability audit

2026-11-05, Asia/Shanghai. Evidence:
`results/masac_v4_sync_pilot_diagnosis_20261105/l0_content_audit.json`;
reproducer: `scripts/audit_l0_feature_content.py`.

All twelve archived L0 development maps have **zero external moving objects**
and `conflict_degree=0`. Each has two or three static boxes. Family names such
as `dynamic_crossing` do not establish dynamic-object exposure at L0. Other
UAVs still move. The historical full/static suite therefore does not identify
the effect of predicting external moving obstacles.

On 128 uniformly indexed pre-action states from the **new static-control
seed1411 at 2k** replay, recomputed features have shape `[128,3,19,11]`:
7,296 correlated values per channel. Full moving-clearance is identically1;
static zeroes it. Full peer-clearance has min0.568308, mean0.997800, max1;
static zeroes it. Exactly7,204/7,296 peer values equal1. Other directions,
static/boundary clearance, TTC, brake, reserve and positive flags are pointwise
identical between the modes in this sample.

This exposes a **neutral-value/constant-offset confound** in interpreting
full-versus-static. A feature value1 can reflect absent-entity sentinel or
saturation of a valid distant object; it does not by itself identify padding.
The sample does not characterize late historical full-policy trajectories or
near-peer interactions, and cannot establish that a constant offset caused the
observed performance ordering. Replay/probe/model hashes and indices are saved.

The current B study uses static features in both arms and changes only known
potential shaping. It does not test the value of predictive content. Describe B
as fixed shaping, with its historical curriculum name, because it has no staged
schedule. Standard PBRS is not a new theorem or an established paper contribution.

After the registered B budget, the proposed identification work is a validity
audit followed by a **constant-fill-1** control and peer-CV model-error diagnosis.
These are separately proposed, not queued training or authorization to add L1.
See `PREDICTIVE_CONTENT_IDENTIFICATION_PLAN_20261105.md`. Close prior work and
remaining evidence gaps are recorded there and in the frontier survey.
