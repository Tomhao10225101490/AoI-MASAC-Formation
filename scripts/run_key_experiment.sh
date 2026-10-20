#!/usr/bin/env bash
# Main paper evaluation, then the independent-loss stress test.
#
# Usage:
#   bash scripts/run_key_experiment.sh runs/<algo>_seed0_<stamp>/ckpt_final.pt 20
#
# The second argument is the number of evaluation episodes per p0 (and per
# independent loss rate). A paper figure wants something like 50-100 episodes
# and a checkpoint trained for the full paper budget (configs/paper_*.yaml).
# The default of 20 episodes only checks that the scripts run.
#
# results/geometry is the headline: distance x blockage, with formation
# error split into line-of-sight and behind-a-roof. results/sweep_independent
# forces a single loss probability and is not the training channel.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?pass a checkpoint path, e.g. runs/.../ckpt_final.pt}
EPISODES=${2:-20}
ROOT_OUT=${3:-results}

python scripts/sweep_geometry.py \
  --checkpoint "$CKPT" \
  --episodes "$EPISODES" \
  --out "$ROOT_OUT/geometry"

python scripts/plot_results.py geometry \
  --csv "$ROOT_OUT/geometry/geometry.csv" \
  --out "$ROOT_OUT/geometry/figs"

python scripts/sweep_packet_loss.py \
  --checkpoint "$CKPT" \
  --episodes "$EPISODES" \
  --out "$ROOT_OUT/sweep_independent"

python scripts/plot_results.py sweep \
  --csv "$ROOT_OUT/sweep_independent/sweep.csv" \
  --out "$ROOT_OUT/sweep_independent/figs"

echo "headline figures: $ROOT_OUT/geometry/figs"
echo "independent-loss stress test: $ROOT_OUT/sweep_independent/figs"
