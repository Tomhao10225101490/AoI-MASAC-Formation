# Windows takeover command ledger — 2026-11-05 (Asia/Shanghai)

Major execution chain, with parallel stages identified. Every shell/Python/training command was
displayed before execution in the chat; console files retain training/eval
streams and exceptions. Commands below assume the repository working directory
unless explicitly stated. The two arms run in separate visible exec sessions;
do not paste their commands sequentially and assume they ran concurrently.

## 1. Inventory and repository

From `C:\research`:

```powershell
Get-Location
Get-ChildItem -Force
Get-Command git,python,py,nvidia-smi -ErrorAction SilentlyContinue | Select-Object Name,Source
git clone https://github.com/Tomhao10225101490/masac-maneuverability.git
```

In `C:\research\masac-maneuverability`:

```powershell
git checkout research/wowmom2027-upgrade
git pull --ff-only
git rev-parse --short HEAD
```

Output: `f81a588`. Hardware CIM queries, `nvidia-smi` and a real CUDA tensor
product confirmed i5-12400F6C/12T,31.84GiB RAM,RTX4060/8188MiB,torch2.5.1+cu121,
CUDA12.1/cuDNN9.1. Full values/methods: `HARDWARE_AUDIT_WIN.md`.
Audit was pushed before training: `59fec6b`.

## 2. Project environment, archive restoration and historical diagnosis

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -u scripts/analyze_maneuver_windows.py --restore --out results/masac_v4_windows_audit_20261105 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/analysis_console_snapshot.log
```

The restoration initially stopped at CRLF and then differing launcher snapshots;
no historical file was overwritten. Final restoration uses ignored
`local_metrics_restore/results/`. Four bundles:15941 files; SHA256s in
`analysis.json`. Independent finals:static11/10/5,full3/4/8,zero1/0/1 out of12.
Failures:61partial-agent-entry,4all-agent-entry-but-no-capture; no final collisions.
This is descriptive development evidence, not a pure synchronization diagnosis.

## 3. Repeated device benchmark and original CPU execution smoke

```powershell
.\.venv\Scripts\python.exe -u scripts/benchmark_maneuver_windows.py --out results/masac_v4_windows_audit_20261105/device_benchmark.json --steps 32 --repeats 3 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/benchmark_console.log
.\.venv\Scripts\python.exe -m pytest -s tests/test_v4_sync_capture.py tests/test_v4_maneuver_masac.py 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/tests_sync_console.log
.\.venv\Scripts\python.exe -u scripts/train_masac_v4_maneuver.py --protocol configs/masac_v4/maneuver_paper_suite_20261103.json --out results/masac_v4_windows_cpu_protocol_smoke_20261105 --probe-mode static --torch-threads 3 --smoke 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/cpu_smoke_console.log
.\.venv\Scripts\python.exe -u scripts/train_masac_v4_maneuver.py --protocol configs/masac_v4/sync_capture_control_cuda_20261105.json --out results/masac_v4_windows_cuda_control_smoke_20261105 --probe-mode static --seed 1211 --torch-threads 1 --smoke 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/cuda_smoke_console.log
```

Archive restoration/analysis and the device benchmark overlapped; device cases
themselves ran sequentially, with no other training. Tests9passed. Benchmark best measured CPU3threads2.549steps/s versus
CUDA1thread22.318steps/s (~8.75x), initial inference errors<=1e-4. Complete
256step/128update smoke:CPU54.075s,CUDA8.219s (~6.58x),both finite and0/2success.
Historical CPU protocol stayed CPU. CUDA checkpoint RNG state was verified.

## 4. Paired small validation and content audit

Two separate exec sessions:

```powershell
.\.venv\Scripts\python.exe -u scripts/train_masac_v4_maneuver.py --protocol configs/masac_v4/sync_capture_control_cuda_20261105.json --out results/masac_v4_sync_control_cuda_pilot_20261105 --probe-mode static --torch-threads 1 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/pilot_control_console.log
.\.venv\Scripts\python.exe -u scripts/train_masac_v4_maneuver.py --protocol configs/masac_v4/sync_capture_pilot_cuda_20261105.json --out results/masac_v4_sync_b_cuda_pilot_20261105 --probe-mode static --torch-threads 1 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/pilot_b_console.log
```

Then:

```powershell
.\.venv\Scripts\python.exe -u scripts/audit_l0_feature_content.py --out results/masac_v4_sync_pilot_diagnosis_20261105 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/content_and_pilot_audit_console.log
```

Seeds1411/1422,2k each:all four final evaluations0/12,finite7488updates.
No learning benefit established. All12 archived devmaps have no external moving
objects; early replay moving channel constant1 and peer7204/7296values1.
Scope/identification limits are in `L0_CONTENT_IDENTIFIABILITY_AUDIT_20261105.md`.

## 5. Fixed40k study — launched, not complete at ledger creation

Protocol/plan was pushed as `eaf83e2` **before** launch at04:49. Two separate exec
sessions; each runs1511/1522/1533 sequentially. These commands are records of
already launched jobs; inspect processes/status before any attempt to execute.

```powershell
.\.venv\Scripts\python.exe -u scripts/train_masac_v4_maneuver.py --protocol configs/masac_v4/sync_capture_40k_control_cuda_20261105.json --out results/masac_v4_sync_40k_control_cuda_20261105 --probe-mode static --torch-threads 1 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/study40k_control_console.log
.\.venv\Scripts\python.exe -u scripts/train_masac_v4_maneuver.py --protocol configs/masac_v4/sync_capture_40k_b_cuda_20261105.json --out results/masac_v4_sync_40k_b_cuda_20261105 --probe-mode static --torch-threads 1 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/study40k_b_console.log
```

Fixedlambda.5/tau.1,warmup1000,batch128,replay100000,eval2k,checkpoint10k.
Final40k is primary. No automatic extension, coefficient search, Attention/L1,
historical-suite rerun or early sealed-test policy evaluation.

## 6. Independent analysis and mathematical checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_discrete_probe_error.py 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/tests_discrete_bound_console.log
.\.venv\Scripts\python.exe -m pytest -q tests/test_sync_study_analysis.py 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/tests_study_analysis_console.log
.\.venv\Scripts\python.exe -u scripts/analyze_sync_study.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --out results/masac_v4_sync_40k_analysis_20261105 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/study_analysis_console.log
```

5discrete checks and14analysis tests passed. Current analysis is IN_PROGRESS,
decision_ready=False; it requires all six final histories and integrity checks.
The8.8m discrete CV bound is model-relative, not a safety certificate.

## 7. Git and continuation

Meaningful stages used explicit file lists, `git diff --cached --check`, commit
and the following non-force push:

```powershell
git push origin HEAD:research/wowmom2027-upgrade HEAD:main
```

Stages:`59fec6b`audit,`f64e456`paper/math/frontier,`aa93433`implementation/device
evidence,`eaf83e2`pilot/preregistration,`81e82d9`live study/analysis/discrete audit.
No model/replay binaries were included. Running logs naturally continue changing.
An ACTIVE15-minute heartbeat `masac` checks this chat and durable state, waits
when account usage is unavailable, and continues on a later eligible run. The
computer/app must stay on; this does not promise the exact quota-reset instant.
