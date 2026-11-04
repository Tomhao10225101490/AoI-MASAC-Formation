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

## 8. Later monitoring, trajectory diagnosis and Methods stage

All shell commands were printed before execution in the chat. No training job
was launched or restarted in this stage, and no frozen source/config was edited.
The first root `manifest.json` read failed because the file is named
`run_manifest.json`; discovery corrected the path. The first inline integrity
check matched `evaluation_summary.json` as a step, raised a parsing error, and
was rerun with numeric-stem filtering. Neither was a training failure. The
traceback and corrected output are preserved in `integrity_progress_console.log`.

Critical execution chain:

```powershell
.\.venv\Scripts\python.exe -u scripts/diagnose_sync_capture_trajectories.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --seed 1511 --step 20000 --out results/masac_v4_sync_40k_progress_20261105/trajectory_20000 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/trajectory_20000_console.log
.\.venv\Scripts\python.exe -u scripts/diagnose_sync_capture_trajectories.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --seed 1511 --step 22000 --out results/masac_v4_sync_40k_progress_20261105/trajectory_22000 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/trajectory_22000_console.log
```

All48 requested trajectories passed JSON/trajectory and paired-map cross-checks.
20k control/B individual never-ball/ball-only/capture counts were27/4/5 versus
22/2/12;22k counts20/9/7 versus24/0/12 (36 correlated agent-map records per arm
and snapshot). B20k had two all-ever-captured episodes with no simultaneous
pair occupancy. This is a descriptive intermediate diagnosis, not efficacy.

Successful disk-integrity snapshot command (the earlier failed attempt omitted
the numeric filter):

```powershell
@'
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

records = []
for arm in ("control", "b"):
    root = Path(f"results/masac_v4_sync_40k_{arm}_cuda_20261105")
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    changed = [name for name, expected in manifest["sources"].items()
               if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected]
    config = Path(f"configs/masac_v4/sync_capture_40k_{arm}_cuda_20261105.json")
    config_match = hashlib.sha256(config.read_bytes()).hexdigest() == manifest["config_hashes"]["protocol"]
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    errors = [str(p) for p in root.rglob("error.json")]
    checkpoints = [str(p) for p in root.glob("maneuver_static_*/checkpoint_*/manifest.json")]
    completions = [str(p) for p in root.rglob("completion.json")]
    records.append(dict(arm=arm, changed_sources=changed, config_match=config_match,
                        errors=errors, checkpoints=checkpoints, completions=completions, status=status))
    print("GUARD", arm, "changed_sources=", changed, "config_match=", config_match,
          "errors=", errors, "seed=", status.get("seed"), "step=", status.get("step"),
          "steps_per_s=", status.get("steps_per_s"), flush=True)
    print("CHECKPOINTS", *checkpoints, sep="\n", flush=True)
    directory = root / f'maneuver_static_{status["seed"]}'
    evaluations = [p for p in directory.glob("evaluation_*.json") if p.stem.split("_")[-1].isdigit()]
    for path in sorted(evaluations, key=lambda p: int(p.stem.split("_")[-1])):
        step = int(path.stem.split("_")[-1])
        if step < 24000:
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        collisions = sum(any(row[k] for k in ("uav_collision", "static_collision", "dynamic_collision", "boundary_collision")) for row in rows)
        print("EVAL", arm, "seed=", status["seed"], "step=", step,
              "success=", sum(row["success"] for row in rows),
              "any_ball=", sum(any(row["ever_in_ball_by_uav"]) for row in rows),
              "all_ball=", sum(all(row["ever_in_ball_by_uav"]) for row in rows),
              "collisions=", collisions, "maps=", len(rows), flush=True)
    if changed or not config_match or errors:
        raise RuntimeError(f"Integrity guard failed for {arm}")
out = Path("results/masac_v4_sync_40k_progress_20261105/integrity_snapshot.json")
out.write_text(json.dumps(dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), records=records), indent=2) + "\n", encoding="utf-8")
print("ARTIFACT", out.resolve(), flush=True)
'@ | .\.venv\Scripts\python.exe -u - 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/integrity_progress_console.log -Append
```

Snapshot: both14source sets unchanged, both protocol hashes matched, no error
files, both30k checkpoint manifests saved, no completion yet. GPU77%,1501MiB,
63C and ~14.7GiB RAM free at the resource sample. The two venv parents plus
two child workers are **two**, not four, training jobs.

```powershell
.\.venv\Scripts\python.exe -u scripts/analyze_sync_study.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --out results/masac_v4_sync_40k_analysis_20261105 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/study_analysis_console.log -Append
Get-Content -LiteralPath results/masac_v4_sync_40k_control_cuda_20261105/train.log -Tail 5
Get-Content -LiteralPath results/masac_v4_sync_40k_b_cuda_20261105/train.log -Tail 5
```

Output: `IN_PROGRESS decision_ready=False decision=None`; later log read reached
control35800/B35000. Both runs continue under the registered final40k protocol.
B eval24k/26k=1/12 then28k/30k=0/12; no intermediate peak substitutes for final.
The paper Methods, entropy-PBRS boundary note, failure hierarchy and frontier gap
updates describe implemented formulas and diagnostic limits, not method success.
