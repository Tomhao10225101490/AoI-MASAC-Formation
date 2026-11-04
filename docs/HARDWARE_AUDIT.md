# Hardware audit — 2026-10-31

Measured locally: Mac mini Mac16,10, Apple M4, 4 performance + 6 efficiency CPU cores, 10-core integrated GPU, 16 GiB unified memory. macOS 15.1, arm64. Available disk 121 GiB; memory_pressure snapshot 41% free. These are point-in-time readings, not reserved capacity.

Project .venv: Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2. MPS available; CUDA unavailable. Package snapshot: results/upgrade_audit/pip-freeze.txt. PYTORCH_ENABLE_MPS_FALLBACK recorded in hardware-benchmark.json.

## Actual project benchmark
Executed scripts.bench_device._one sequentially for CPU and MPS into a new directory, preserving old benchmarks. Five agents, hidden width 256, batch 256, replay 4000, six episodes of at most 80 steps, seed 0. CPU mean update 0.0223 s; MPS 0.0564 s. Mean step including warmup: CPU 0.0078 s; MPS 0.0320 s. Both completed training and wrote checkpoints. This is a smoke/throughput result, not a convergence result. Different trajectories and update counts prevent treating wall-time ratios as a controlled kernel benchmark. MPS explicit synchronization, peak memory and operator profiling remain Pending.

Raw evidence: results/upgrade_audit/hardware-benchmark.json and bench_runs/{cpu,mps}/ (config, metrics, speed, checkpoint). Original baseline archive tests: 55 passed, exit 0, baseline-tests.txt. Current tree: 83 passed, exit 0.

## Conservative initial resource policy
Use CPU for current small MASAC runs, hidden 128–256 and batch 128–256. Start one training job with two CPU threads; increase to two jobs only after measured memory and concurrent throughput. Keep replay modest for local pilots (4000 for smoke; assess actual bytes before larger buffers). Use one update per collected joint step initially, checkpoints every 25 episodes for pilots plus final. Do not concurrently run full CPU and MPS studies. No evidence yet supports changing update ratio or widening to 512. Reserve MPS for a synchronized batched benchmark of the chosen architecture. Monitor memory pressure before launching larger jobs; stop scaling if sustained swapping or severe pressure occurs. Thermal throttling and peak process memory have not been measured.

## Synchronized update-only follow-up
Ran scripts/profile_compute.py with identical synthetic replay shapes, two CPU threads, three warmup updates and 15 timed updates per case. MPS synchronization is explicit after every update. Width/batch 128/128: CPU 9.13 ms, MPS 48.11 ms; 256/256: CPU 19.77 ms, MPS 48.41 ms. This supports CPU for current small independent-agent models. No unsupported operator was encountered in these paths with the fallback environment unset; this does not certify all future operations. Raw samples, parameter counts, cumulative process peak RSS and MPS allocation readings: results/upgrade_audit/synchronized-benchmark.json. Peak RSS is cumulative, not per-case isolated memory. Concurrent throughput remains unmeasured.
