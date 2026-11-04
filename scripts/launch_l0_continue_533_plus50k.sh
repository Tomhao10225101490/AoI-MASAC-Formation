#!/usr/bin/env bash
# Run ON TomMac-mini after desktop app reconnects.
# Auditable continuation: seed 533 +50k from checkpoint_050000.
set -euo pipefail
ROOT="${MASAC_ROOT:-/Users/a1/Desktop/research/MASAC}"
cd "$ROOT"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="results/masac_v4_l0_continue_533_plus50k_${TS}"
PROTO="configs/masac_v4/l0_continue_533_plus50k_20261103.json"
PARENT="results/masac_v4_l0_legacy_50k_20261103_102637"
CKPT=""
for c in \
  "$PARENT/standard_533/checkpoint_050000" \
  "$PARENT/standard_533/checkpoints/checkpoint_050000" \
  "$PARENT/seed_533/checkpoint_050000" \
  "$PARENT/standard_533/checkpoint_50000" \
  "$PARENT/standard_533/checkpoints/checkpoint_50000"
do
  if [[ -e "$c" ]]; then CKPT="$c"; break; fi
done

echo "== preflight =="
date
pwd
git rev-parse --abbrev-ref HEAD || true
test -f scripts/train_masac_v4_l0.py
test -f configs/masac_v4/l0_protocol.json

if pgrep -fl 'train_masac_v4|SharedMASAC' >/dev/null 2>&1; then
  echo "ERROR: live train process found; refuse to clear lock"; pgrep -fl 'train_masac|caffeinate'; exit 1
fi
if [[ -f results/masac_v4_l0_training.lock ]]; then
  echo "Removing stale lock results/masac_v4_l0_training.lock (no live trainer)"
  rm -f results/masac_v4_l0_training.lock
fi

echo "== help (resume flags) =="
./.venv/bin/python scripts/train_masac_v4_l0.py --help | sed -n '1,160p' || true

if [[ -z "$CKPT" ]]; then
  echo "ERROR: checkpoint_050000 not found under $PARENT"
  echo "FALLBACK: do NOT silently invent paths; switch to documented secondary arm fresh 100k under new experiment id."
  find "$PARENT" -maxdepth 4 \( -iname '*533*' -o -iname '*checkpoint*' \) 2>/dev/null | head -80
  exit 2
fi
echo "Using checkpoint: $CKPT"

mkdir -p "$OUT"
HELP="$(./.venv/bin/python scripts/train_masac_v4_l0.py --help 2>&1 || true)"
CMD=(./.venv/bin/python scripts/train_masac_v4_l0.py --out "$OUT")
if [[ -f "$PROTO" ]]; then
  if echo "$HELP" | grep -q -- '--protocol'; then CMD+=(--protocol "$PROTO"); fi
fi
if echo "$HELP" | grep -q -- '--resume-from'; then CMD+=(--resume-from "$CKPT")
elif echo "$HELP" | grep -q -- '--resume_from'; then CMD+=(--resume_from "$CKPT")
elif echo "$HELP" | grep -q -- '--checkpoint'; then CMD+=(--checkpoint "$CKPT")
elif echo "$HELP" | grep -q -- '--resume'; then CMD+=(--resume "$CKPT")
else
  echo "WARN: no resume flag in --help; relying on protocol resume_from field + seed override if available"
fi
if echo "$HELP" | grep -q -- '--seeds'; then CMD+=(--seeds 533)
elif echo "$HELP" | grep -q -- '--seed'; then CMD+=(--seed 533)
fi
if echo "$HELP" | grep -q -- '--extra-steps'; then CMD+=(--extra-steps 50000)
elif echo "$HELP" | grep -q -- '--steps'; then CMD+=(--steps 50000)
elif echo "$HELP" | grep -q -- '--total-steps'; then CMD+=(--total-steps 100000)
fi

printf '%s\n' "${CMD[@]}" | tee "$OUT/launch_command.txt"
echo "$CKPT" > "$OUT/resume_checkpoint.txt"
cp -f "$PROTO" "$OUT/protocol_snapshot.json" 2>/dev/null || true

python3 - <<PY
import json, time
from pathlib import Path
p = Path('results/masac_v4_active_run.json')
p.write_text(json.dumps({
  'experiment_id': 'l0_continue_533_plus50k_20261103',
  'status': 'launching',
  'training_started': True,
  'updated_at_asia_shanghai': time.strftime('%Y-%m-%d %H:%M:%S Asia/Shanghai'),
  'results_dir': '$OUT',
  'seed': 533,
  'resume_from': '$CKPT',
  'extra_steps': 50000,
  'protocol': '$PROTO',
  'authorization': 'user B 2026-11-03 widget t0s9',
  'parent_run': '$PARENT',
}, indent=2) + '\n')
print('wrote', p)
PY

nohup caffeinate -dims "${CMD[@]}" > "$OUT/launcher.out" 2>&1 &
PID=$!
echo "$PID" | tee "$OUT/launcher.pid"
sleep 3
if ps -p "$PID" >/dev/null 2>&1; then
  echo "LAUNCH OK pid=$PID out=$OUT"
  ps -p "$PID" -o pid,ppid,etime,command
  ls -la "$OUT" | head
  if [[ -f "$OUT/status.json" ]]; then python3 -m json.tool "$OUT/status.json" | head -80; else echo "status.json not yet; check launcher.out"; fi
  tail -n 40 "$OUT/launcher.out" || true
else
  echo "LAUNCH FAILED; launcher.out:"
  cat "$OUT/launcher.out" || true
  exit 3
fi
