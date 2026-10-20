#!/usr/bin/env bash
# Sweep nominal packet loss from 0% to 70% for one trained checkpoint and plot it.
#
# Usage:
#   bash scripts/run_key_experiment.sh runs/<algo>_seed0_<stamp>/ckpt_final.pt 20
#
# The second argument is the number of evaluation episodes per loss rate.
# A paper figure wants something like 50-100 episodes and a checkpoint that
# was trained for the full paper budget (see configs/paper_*.yaml). The
# default of 20 episodes is only enough to check that the script runs.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?pass a checkpoint path, e.g. runs/.../ckpt_final.pt}
EPISODES=${2:-20}
OUT=${3:-results/sweep}

python scripts/sweep_packet_loss.py \
  --checkpoint "$CKPT" \
  --episodes "$EPISODES" \
  --out "$OUT"

python scripts/plot_results.py sweep --csv "$OUT/sweep.csv" --out "$OUT/figs"
echo "figures: $OUT/figs"
