#!/usr/bin/env bash
# Paper-scale training. Do not launch this for a smoke test.
#
# It trains AoI-MASAC and the baselines that are actually implemented.
# MAPPO is not in the list: algos/mappo.py is still a stub.
#
# Usage:
#   DEVICE=cpu SEEDS=0,1,2 bash scripts/train_paper.sh
# Expect many hours per seed on a CPU. Measure one short run first
# (python train.py --config configs/smoke.yaml) and scale from speed.json.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-0,1,2}
DEVICE=${DEVICE:-cpu}

CONFIGS=(
  configs/paper_aoi_masac.yaml
  configs/paper_vanilla_masac.yaml
  configs/paper_masac_hold.yaml
  configs/paper_masac_extrapolate.yaml
  configs/paper_maddpg.yaml
  configs/paper_matd3.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "==== $cfg seeds=$SEEDS device=$DEVICE ===="
  python train.py --config "$cfg" --seeds "$SEEDS" --device "$DEVICE"
done
