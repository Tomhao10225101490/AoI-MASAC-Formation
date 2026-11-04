# Windows hardware audit — 2026-11-05 (Asia/Shanghai)

Audited before any training on this host. Repository cloned into
`C:\research\masac-maneuverability`, branch `research/wowmom2027-upgrade`;
initial HEAD **f81a588a24a808669f73480a4f9089e8e3dcddab**. `git pull --ff-only`
reported already up to date. This user's 2026-11-05 instruction authorizes this
Windows takeover, B sync-capture implementation and small validation. It supersedes
the old Mac-only handoff restriction for this host. Attention, L1 and a complete
paper-suite rerun remain outside this launch scope.

## Hardware and available resources

| Item | Observed value |
|---|---|
| OS | Windows 10 Home Chinese, 10.0.19045 |
| CPU | Intel Core i5-12400F, 6 physical cores / 12 logical processors, reported base clock 2500 MHz |
| Physical RAM | 34,191,208,448 bytes = 31.84 GiB |
| Available RAM at audit | 18,650,108 KiB = 17.79 GiB (other applications are running) |
| C: total / free | 999,442,837,504 / 335,026,315,264 bytes (930.8 / 312.0 GiB) |
| D: total / free | 1,000,202,039,296 / 655,404,724,224 bytes (931.5 / 610.4 GiB) |
| GPU | NVIDIA GeForce RTX 4060, 8188 MiB reported by nvidia-smi, WDDM |
| NVIDIA driver | 595.79; Windows driver version 32.0.15.9579 |
| nvidia-smi driver CUDA capability | 13.2 (not the installed PyTorch runtime) |
| Standalone CUDA compiler | `Get-Command nvcc` found no executable on PATH |
| Python | 3.10.6, 64-bit installation under the user's AppData |
| PyTorch | 2.5.1+cu121, MKL/OpenMP, AVX2 |
| PyTorch CUDA runtime | 12.1 |
| cuDNN runtime | 90100 = 9.1.0 |
| `torch.cuda.is_available()` | **True** |
| CUDA execution check | 256×256 random CUDA matrix product finite; synchronization passed |
| Initial PyTorch thread settings | intra-op 6, inter-op 6 |

`nvidia-smi` snapshot (driver utility printed host-clock 2026-11-05 00:30:34;
user-facing audit date follows Asia/Shanghai): GPU 39 °C, P8, 763/8188 MiB used,
48% sampled utilization, power cap 115 W, no training process launched by this
takeover. This is a single display/application-load sample, not training utilization.
Unrelated applications/processes were left running. No private process paths or
credentials are needed in the published report.

## Environment and compute policy

Git and Python were already installed despite the fresh-workspace assumption.
Use `.venv\Scripts\python.exe` on Windows (the repository's `.venv/bin/python`
rule has POSIX spelling). `.venv` is created with `--system-site-packages` to
reuse the installed CUDA-enabled PyTorch without reinstalling gigabytes of
packages. Install missing requirements inside this environment and record the
actual versions; this is not a fully hermetic environment.

1. **Preserve `device='cpu'` in the frozen protocol and training default.** First
   reproduce the CPU execution contract. A successful CUDA tensor operation is
   compatibility evidence, not a speed benchmark or scientific reproduction.
2. Start timing with **one intra-op thread per training process**. Benchmark
   1/2/3 threads on the actual ManeuverMASAC update+environment+probe workload,
   rather than assuming 12 logical CPUs imply 12 useful BLAS threads.
3. Begin with two visible processes for paired small experiments. Raise concurrency
   only after measuring RSS, available RAM and aggregate environment steps/s;
   leave at least 4 GiB available RAM and avoid sustained paging. The current
   available RAM, rather than the installed 32 GiB, controls admission.
4. CUDA may run an explicitly separate timing benchmark while CPU science runs.
   Only a warmed, repeated end-to-end benchmark (same architecture, batch size,
   probes, data movement and workload) can justify changing a future default.
   Record median/spread, parity tolerances and total throughput. No device-default
   change is authorized merely to increase displayed GPU utilization.
5. Small smoke/pilot runs do not establish learning, convergence or a gate PASS.
   Keep frozen evaluation capture **distance <4 m, speed <2 m/s, latch=False**.
   Large B training requires a plan summary first. Do not silently rerun the
   historical full/static/zero suite, enable Attention/L1 or extend budgets.

## Audit commands and observations

The chat displays every shell command before execution. Hardware sources:
`Win32_Processor`, `Win32_ComputerSystem`, `Win32_OperatingSystem`,
`Win32_LogicalDisk`, `Win32_VideoController`, `nvidia-smi`, and a Python probe
of `torch.__version__`, `torch.__config__.show()`, CUDA/cuDNN and a real CUDA
matrix multiplication. An initial nested-string CUDA print failed with NameError;
it was rerun with a PowerShell here-string and passed. No training preceded this
audit. Throughput results will be appended after their measurements.

## Measured device/thread benchmark (2026-11-05)

Report: `results/masac_v4_windows_audit_20261105/device_benchmark.json`;
command: `.venv\Scripts\python.exe -u scripts/benchmark_maneuver_windows.py
--out results/masac_v4_windows_audit_20261105/device_benchmark.json --steps 32 --repeats 3`.
Each case used full probes, batch128, cached replay, action + NumPy environment +
probe/replay transfer + SAC update, eight warm-up updates, then three timed blocks.
Cases ran sequentially. No other training ran concurrently. Archive extraction
and lightweight document/analysis work ran concurrently; timings describe this
desktop workload, not isolated hardware maximum.

| Device | Torch threads | Median environment steps/s |
|---|---:|---:|
| CPU | 1 | 1.216 |
| CPU | 2 | 1.939 |
| CPU | 3 | 2.549 |
| CUDA | 1 | 22.318 |

CUDA was approximately **8.75×** the best measured CPU configuration. Initial
deterministic actor and probe outputs passed maximum absolute error tolerance
`1e-4`; this is inference parity, not stochastic training trajectory equivalence.
The benchmark excludes periodic multi-map evaluation, joint-reference audits and
checkpoint/replay I/O. A complete trainer smoke must follow before a CUDA B pilot.
The historical CPU protocol/default remains unchanged. New explicitly named CUDA
pilot protocols may cite this evidence; that is not a rerun of the old suite.

## Artifact boundary

Historical metrics are in Git (`results_key/` and split archives). Historical
weights/replays (~41 GB) remain on the Mac. This checkout cannot resume those
histories until the matching checkpoint, replay, optimizer and RNG artifacts are
copied. New cold-start pilots do not pretend to be continuation runs. Never commit
`.pt` or replay arrays; publish compact metrics, protocols, logs and analysis.
