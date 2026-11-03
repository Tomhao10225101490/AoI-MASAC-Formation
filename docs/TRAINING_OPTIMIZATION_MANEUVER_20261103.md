# Training optimization — Predictive Maneuverability suite (2026-11-03)

Asia/Shanghai. Profiled on TomMac-mini while paper suite stamp `162345` was live.
**Do not kill** full/zero/static PIDs; optimizations land for **next** launches and
documentation. Live processes keep already-imported modules in memory.

## Hardware / live suite snapshot

| Item | Value |
|------|-------|
| Machine | TomMac-mini, 10 CPU, 16 GB RAM |
| Live arms | full / zero / static @ seed 1211, ~30–35k / 80k |
| Throughput | ~2.5 step/s (full/static), ~3.0 step/s (zero) |
| Torch threads / arm | 2 (launch args) |
| Memory | ~15 GB used; **no fourth 80k arm** while suite runs |

## Bottleneck profile (venv, synthetic batch)

| Call | Cost |
|------|------|
| `predict_probes` full, batch (128,3) | ~17 ms (pre-opt); ~half after vectorized clearances |
| `ManeuverMASAC.update(128)` full | ~350–450 ms (dominates step time) |
| `act` | ~2–3 ms |
| Threads | **1 thread often faster than 2** per process when 3 arms share the Mac |

Root cause: each env step after warmup does **one** heavy CPU update; probes were a
smaller but real fraction; eval every 2k × 12 maps adds wall-time spikes.

## Implemented (safe for next launches; live suite untouched in-memory)

1. **Vectorized probe clearances** (`uav_planning/v4_prediction.py`)  
   Dynamics remain sequential; static/moving/peer/boundary clearances + TTC reduced
   over the horizon in batched tensors. Verified **bit-identical** vs prior loop on
   full/static/zero/mismatch/shuffle samples (~1.5–1.7× probe speedup).

2. **Risk feature replay cache** (`uav_planning/v4_maneuver_masac.py`, `cache_risk=True`)  
   Stores `(n,19,11)` features for `obs` / `next` at `push` time so `update` does not
   recompute `predict_probes` twice per batch. Q/actor inputs unchanged. Replay
   checkpoints in `train_masac_v4_maneuver.py` now persist `risk_feat` when enabled.

3. **Docs / next-launch policy** (this file + README): prefer `--torch-threads 1` when
   running **≥2** maneuver arms on this Mac; use 4 only for a lone arm on an idle machine.

## Recommended next-launch knobs (new experiment id if protocol numbers change)

| Knob | Current suite | Next recommendation |
|------|---------------|---------------------|
| `torch_threads` | 2 | **1** for 3-way parallel |
| `cache_risk` | n/a (old process) | **True** (default in code now) |
| `eval_interval` | 2000 | keep 2000 for paper curves; optional 4000 mid-budget only with new id |
| `batch_size` | 128 | keep 128 unless a dedicated microbench says otherwise |
| Fourth arm | — | **Do not** while RAM <~2 GB free |

## Explicitly not done mid-suite

- No live hyperparameter / reward / capture changes on stamp `162345`
- No Attention / L1 / extra seed restack
- No `renice` required (arms already CPU-bound; idle ~40%)

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_v4_maneuver_masac.py tests/test_masac_v4_networks.py
```

## Math companion

Paper-facing formula hygiene: `docs/MATH_ARGUMENT_IMPROVEMENTS_20261103.md`.

## Live suite pointer (updated)

As of 2026-11-03 21:05 Asia/Shanghai, stamp `162345` still running (~41–49k seed 1211).
See `docs/MASAC_V4_MANEUVER_PAPER_SUITE_STATUS_20261103.md`. Do not attach live processes to these opts;
next launch uses them. Fourth arm still rejected on RAM.

