# Figure1: implemented pooled V4 data flow

Date2026-11-05. Source-aligned architecture illustration, not experimental
outcomes, a novel-architecture claim or a safety certificate.

Final layout: [vectorSVG](figures/masac_system_20261105_v3/figure1_system.svg),
[300dpiPNG](figures/masac_system_20261105_v3/figure1_system.png),
[source/artifact manifest](figures/masac_system_20261105_v3/illustration_manifest.json).

## Manuscript caption

**Figure1. Observation-only probe augmentation in the implemented pooled MASAC.**
Each agent's current162-dimensional observation feeds a masked typed-object
encoder (384features) and a fixed19-by11probe provider followed by a learned
probe encoder (128features). Their512-dimensional concatenation feeds a
parameter-shared local Gaussian actor. The tanh command precedes radial
acceleration limiting, speed projection and the semi-implicit position update;
the modeled action density is defined before those environment transformations.
Each centralized critic has its own feature-encoder parameters: a515-dimensional
per-agent feature/action vector passes through a256-dimensional MLP, then agent
mean/max aggregation supplies512features to its scalar head. The two critics
and their target copies are distinct networks, not statistically independent
estimates. Replay may cache observation-derived raw probe tensors for current
and next observations; it does not cache learned encoder latents. Dashed paths
denote training collection and training-only reward processing. B potential
addition and near-occupancy replacement are separate registered comparisons,
with B disabled in the latter. Evaluations retain original Path-C reward.

## Scope and source checks

- `uav_planning/v4_networks.py`: `adapt_observation`, `SetEncoder`, `RiskEncoder`,
  `Features`, `Actor`, `TeamQ`, `TwinCritics`. The critic pools agent latents;
  it does not concatenate a fixed number of agent latents into its team head.
- `uav_planning/v4_prediction.py`: deterministic visible-observation-only
  probes. The19commands are18unit directions and zero. Zero preserves initial
  velocity under this preview; it is not a braking command. Full/static/zero
  preserve network dimensions but are different content interventions.
- `uav_planning/v4_maneuver_masac.py`: cached raw probe replay and separate
  actor/critic/temperature optimization. The pending action-time reuse guard
  [audit](PENDING_PROBE_CACHE_SHAPE_AUDIT_20261105.md) does not disable the
  minibatch replay cache shown here.
- `v4_sync_capture.py`, `v4_near_objective.py` and `v4_near_study_runtime.py`:
  separate training-reward interventions and original-reward evaluation.

Parameter sharing is across agents within an actor, not across experiment arms.
Current centralized observations need not recover the latent Markov state.
The diagram neither adds future information to the actor nor implements a CBF,
MPC action filter, communication protocol, arbitrary-fleet-size validation or
sim-to-real result. L0's long nominal sensor range and fixed observation-slot
caps remain as documented in Methods.

## Reproduction and layout QA

From repository root, print then run this command with a **fresh** output path:

```powershell
.\.venv\Scripts\python.exe -u scripts\render_masac_system_figure.py --out docs/figures/masac_system_rebuild
```

The renderer uses MatplotlibAgg on CPU, no Torch/model/environment or experiment
input. It records source hashes before rendering, checks they remain unchanged,
and writes SVG/PNG artifact hashes plus manually reviewed source facts. This
manifest certifies illustration provenance, not implementation correctness or
research outcomes. Finalv3 was visually inspected for legibility and arrow/
label overlap. v1failed on mathtext braces before any image was saved; v2fixed
syntax; v3rerouted a collection arrow and clarified the motion/critic labels.
These are layout revisions, with no outcome-based selection or study changes.
The initial error stack and subsequent render logs are retained in the Windows
audit directory. Earlier layout drafts remain local; the manuscript linksv3.
The final asset directory disables Git newline conversion to preserve generated
SVG/JSON bytes. Source hashes describe the inspected working-tree bytes; a
new rendering on another platform/version is not promised bit-identical.
