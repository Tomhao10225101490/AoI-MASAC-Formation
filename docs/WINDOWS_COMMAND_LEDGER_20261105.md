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

## 9. First registered final40k pair and transition to1522

The next heartbeat checked state/Git/processes/status/completion/resources before
any action. Both original workers were still active; no duplicate was launched.
The following foreground read-only stream ran180s and then exited successfully:

```powershell
@'
import time
from pathlib import Path

handles = {}
for arm in ("control", "b"):
    path = Path(f"results/masac_v4_sync_40k_{arm}_cuda_20261105/train.log")
    handle = path.open("r", encoding="utf-8")
    handle.seek(0, 2)
    handles[arm] = handle
print("READ_ONLY_MONITOR_BEGIN duration_s=180; no training launched or changed", flush=True)
deadline = time.monotonic() + 180
try:
    while time.monotonic() < deadline:
        for arm, handle in handles.items():
            while True:
                line = handle.readline()
                if not line:
                    break
                print(f"[{arm}] {line.rstrip()}", flush=True)
        time.sleep(1)
finally:
    for handle in handles.values():
        handle.close()
print("READ_ONLY_MONITOR_END", flush=True)
'@ | .\.venv\Scripts\python.exe -u - 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/study40k_first_pair_stream_20261105.log -Append
```

Observed final1511 evals:control2/12,any12,all5,failures7partial-arrival+3all-ball;
B6/12,any12,all8,failures4partial-arrival+2all-ball. Both printed40k checkpoint
paths and then **COLD-START seed1522** with the registered settings. Source review
confirmed seed completion follows final eval and snapshot; root arm completion
is written only after all three seeds. Per-seed completion files were read and
matched. A final progress record can still report last_eval_step38000 because
the final40k evaluation is later; it is not the authoritative endpoint.

```powershell
.\.venv\Scripts\python.exe -u scripts/diagnose_sync_capture_trajectories.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --seed 1511 --step 40000 --out results/masac_v4_sync_40k_progress_20261105/trajectory_40000_seed1511 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/trajectory_40000_seed1511_console.log
.\.venv\Scripts\python.exe -u scripts/analyze_sync_study.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --out results/masac_v4_sync_40k_analysis_20261105 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/study_analysis_console.log -Append
```

All24 final trajectories passed cross-checks. Individual never-ball/ball-only/
captured counts7/1/28 control and6/0/30 B. The study remains
`IN_PROGRESS / decision_ready=False / decision=None`.

Independent source/config, finite-logged-diagnostics and raw-byte snapshot audit:

```powershell
@'
import hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

records = []
fields = ("actor_loss", "critic_loss", "alpha", "actor_grad", "critic_grad",
          "actor_grad_post", "critic_grad_post", "q_mean", "target_mean", "td_abs", "entropy")
for arm in ("control", "b"):
    root = Path(f"results/masac_v4_sync_40k_{arm}_cuda_20261105")
    directory = root / "maneuver_static_1511"
    checkpoint = directory / "checkpoint_040000"
    manifest = read(root / "run_manifest.json")
    saved = read(checkpoint / "manifest.json")
    completion = read(directory / "completion.json")
    rows = read(directory / "evaluation_40000.json")
    progress = [json.loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    progress = [row for row in progress if row.get("kind") == "progress"]
    trained = [row for row in progress if row["updates"] > 0]
    finite = all(all(key in row and math.isfinite(row[key]) for key in fields) for row in trained)
    changed = [name for name, expected in manifest["sources"].items() if digest(Path(name)) != expected]
    config = Path(f"configs/masac_v4/sync_capture_40k_{arm}_cuda_20261105.json")
    config_match = digest(config) == manifest["config_hashes"]["protocol"]
    binaries = {}
    for filename, key in (("checkpoint.pt", "checkpoint_sha256"), ("replay.npz", "replay_sha256")):
        path = checkpoint / filename
        observed = digest(path)
        binaries[filename] = dict(bytes=path.stat().st_size, sha256=observed, matches=observed == saved[key])
        print("FILE_HASH", arm, path, binaries[filename], flush=True)
    success = sum(row["success"] for row in rows)
    valid = (completion["seed"] == 1511 and completion["step"] == 40000
             and len(rows) == 12 and all(row["step"] == 40000 for row in rows)
             and abs(completion["final_development_success"] - success / 12) < 1e-12
             and progress[-1]["step"] == 40000 and progress[-1]["updates"] == 39000
             and bool(trained) and finite and not changed and config_match
             and saved["complete"] is True and saved["step"] == 40000 and saved["seed"] == 1511
             and saved["sources"] == manifest["sources"]
             and all(item["matches"] for item in binaries.values()))
    record = dict(arm=arm, seed=1511, step=40000, success_maps=success,
                  updates=progress[-1]["updates"], logged_trained_progress_rows=len(trained),
                  finite_logged_diagnostics=finite, changed_sources=changed,
                  config_match=config_match, snapshot_sources_match_run=saved["sources"] == manifest["sources"],
                  binaries=binaries, seed_artifact_checks_passed=valid,
                  current_status=read(root / "status.json"))
    records.append(record)
    print("SEED_AUDIT", arm, "success=", success, "/12", "updates=", record["updates"],
          "finite_logged_diagnostics=", finite, "changed_sources=", changed,
          "config_match=", config_match, "checks_passed=", valid, flush=True)
    if not valid:
        raise RuntimeError(f"Seed artifact verification failed: {arm}")
out = Path("results/masac_v4_sync_40k_progress_20261105/first_pair_integrity.json")
out.write_text(json.dumps(dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                              scope="Raw-byte artifact integrity and logged diagnostics only; not exact CUDA resume validation or a study efficacy decision.",
                              records=records), indent=2, allow_nan=False) + "\n", encoding="utf-8")
print("ARTIFACT", out.resolve(), flush=True)
'@ | .\.venv\Scripts\python.exe -u - 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/first_pair_integrity_console.log
```

All four binary SHA256s matched their manifests; both protocols and14source
hash sets matched. Both histories have39000 updates and195 finite logged
post-warmup diagnostics. This verifies byte integrity and recorded numerical
diagnostics, not exact CUDA resumption. No binary is staged for Git.
The first-pair note explicitly separates these endpoints from the pending full
three-seed decision; all original workers and settings remain in use.

## 10. Second registered40k pair: mixed partial results

Three intervening read-only heartbeat checks confirmed live original workers,
unchanged declared source/config hashes, no error.json, finite current learning
diagnostics, and continuing checkpoint/eval output. Full printed commands remain
in the chat; their appended monitor evidence is in
`results/masac_v4_windows_audit_20261105/heartbeat_monitor_console.log`.

At the next material checkpoint, both1522 seed completions were present and the
same workers had cold-started1533. Final1522:control1/12, B0/12; first1511 was
2/12 versus6/12. Four of six histories are complete. The registered final40k
rule, budget and sealed-map restrictions remain unchanged; no efficacy decision
is made on these two pairs.

```powershell
.\.venv\Scripts\python.exe -u scripts/diagnose_sync_capture_trajectories.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --seed 1522 --step 40000 --out results/masac_v4_sync_40k_progress_20261105/trajectory_40000_seed1522 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/trajectory_40000_seed1522_console.log
.\.venv\Scripts\python.exe -u scripts/analyze_sync_study.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --out results/masac_v4_sync_40k_analysis_20261105 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/study_analysis_console.log -Append
```

All24 final1522 trajectories passed metric/map-pair checks. Control/B individual
never-ball, ball-only and capture counts:20/1/15 versus17/0/19. Team all-ever-ball
and all-ever-capture:1 versus2; simultaneous success:1 versus0. Control has11
incomplete-arrival failures; B10 incomplete-arrival and2 nonoverlapping-capture
failures. There are no final collisions. Proxy coverage is not the primary target.
The independent analyzer still gives IN_PROGRESS/decision_ready=False/decision=None.

The successful inline first-pair artifact audit was encapsulated in a new,
standard-library-only read-only recovery utility, without changing trainer
SOURCES, protocol/config, the canonical analyzer or decision rules. Root reviewed
the source and then ran it against the actual second pair:

```powershell
Get-Content -LiteralPath scripts/verify_sync_seed_artifacts.py
.\.venv\Scripts\python.exe -u scripts/verify_sync_seed_artifacts.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --seed 1522 --out results/masac_v4_sync_40k_progress_20261105/seed1522_integrity 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/seed1522_integrity_console.log
```

Result:PAIR_SEED_VERIFICATION seed=1522 status=PASS, exit0. Each arm has39000
updates and195 finite sampled progress diagnostics. Both14source sets and both
frozen config hashes match; all four binary hashes match checkpoint manifests.
The JSON keeps per-file byte sizes/observed/expected SHA256 and individual checks.
It neither loads weights nor validates exact CUDA resume, and PASS means artifact
checks, not passing the8/12 capture gate. Source/config paths resolve within the
repository and binary reads use1MiB chunks. The original training continues1533.

## 11. Third pair, complete registered endpoint and next offline plan

Both original workers finished normally at07:22/07:23 Asia/Shanghai. No Python
worker remained at the resumed check. The following commands were printed before
execution and completed without a training/source/config change:

```powershell
.\.venv\Scripts\python.exe -u scripts/verify_sync_seed_artifacts.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --seed 1533 --out results/masac_v4_sync_40k_progress_20261105/seed1533_integrity 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/seed1533_integrity_console.log
.\.venv\Scripts\python.exe -u scripts/diagnose_sync_capture_trajectories.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --seed 1533 --step 40000 --out results/masac_v4_sync_40k_progress_20261105/trajectory_40000_seed1533 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/trajectory_40000_seed1533_console.log
.\.venv\Scripts\python.exe -u scripts/analyze_sync_study.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --out results/masac_v4_sync_40k_analysis_20261105 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/study_analysis_console.log -Append
```

1533 artifact checks PASS, success control4/12/B5/12, all24 final trajectories
match JSON/map metadata. Across all six histories the12 final binary SHA256s,
14source sets and both config hashes match. All39,000 updates/history completed;
each history retains195 finite sampled post-warmup diagnostics. Exact CUDA
restore is not established by these checks.

The independent analyzer now reports COMPLETE/decision_ready=true/
ADVANCEMENT_RULE_MET. Paired counts control[2,1,4], B[6,0,5], differences[+4,-1,+1],
mean fraction1/9. All four preregistered engineering criteria hold. Both absolute
gates FAIL0/3. This supports planning confirmation only, not training/test launch
or significance. Full report: SYNC_CAPTURE_40K_FINAL_RESULTS_20261105.md.

A printed inline Python aggregation loaded the three existing final trajectory
summary JSONs, asserted step40000, paired-map checks, J=>C=>A, success consistency
and12 exhaustive outcomes/arm/seed, and wrote
`results/masac_v4_sync_40k_analysis_20261105/failure_taxonomy.json`.
Its complete stdout is `results/masac_v4_windows_audit_20261105/final_failure_taxonomy_console.log`.
Control failure classes25/0/4; B18/1/6. Total54failures=43 incomplete all-agent
ball arrivals+1 all-ball without all-individual capture+10 individual captures
without overlap; all72 final episodes have no collision flags. These are
descriptive classes, not identified causes or independent-map training trials.

Read-only recovery checks used WINDOWS_RESEARCH_STATE, Git status/HEAD, CIM Python
process command lines, both root completions and canonical analysis. One full
Git status and one oversized verification read were truncated by tool output;
subsequent reads were narrowed. Disk artifacts are complete and unchanged.
The persistent state now puts6/6complete above the superseded4/6 snapshot.

Paper/math/frontier integration retains the negative seed and separates the
40k Windows cohort, historical80k Mac ablation and legacy warm-start Standard-C.
SCOPE's official IJCAI2026 page and CC-HGCBF's October30preprint were rechecked
from primary sources. No novelty or guarantee follows from these comparisons.

The next plan is separately recorded in
`design/MASAC_V4_POST40K_NEXT_MINIMAL_PLAN_20261105.md`: offline only, six final
replays, last8000transitions, fixed1024linspace observations/history, paired
full/static/static-fill1 features. No policy load/actions, training, simulator
rollout or sealed-test access. Inputs/indices/source hashes and validity versus
sentinel/saturation are required before interpreting content. The new analysis
script is reviewed separately; it is not part of frozen trainer SOURCES.
The existing ACTIVE heartbeat idmasac was updated to this completed-study
recovery entry; frequency/thread were preserved, no duplicate created and no
account/payment setting changed. Confirmation training remains unqueued.

Explicit final-stage path selection was saved in
`results/masac_v4_sync_40k_analysis_20261105/git_intended_files_final40k.txt`.
394 intended paths,344 actually changed; cached diff check passed and the
forbidden-extension query returned no checkpoint/replay/credential paths.
Commit/push commands:

```powershell
git commit --quiet -m "research: finalize registered 40k B study and freeze offline content audit"
git rev-parse HEAD
git push --atomic origin HEAD:main HEAD:research/wowmom2027-upgrade
git ls-remote --heads origin main research/wowmom2027-upgrade
```

Both remote tips verified `d8ff5ada79be438c11ee5340e00fd3dd26c4b709`.
The new offline feature script was deliberately excluded until review/completion.

## 12. Reproducible final figure and near-goal shortfall localization

Printed inline plotting and distance aggregation commands were executed before
being encapsulated in a separate JSON-only entry point. Their tee logs are
final_figure_console.log and distance_shortfall_console.log in the Windows audit
root. All65 never-ball agent-map records have closest distance4–10m:29closest
4–5m,36closest5–10m; all65finalspeeds<2m/s. Their final distances need not be
their closest distances. This suggests a near-goal diagnosis, not a causal
claim or proof of stationary behavior. B/control populations are selected by
their own failures and must not be compared as matched independent samples.

```powershell
.\.venv\Scripts\python.exe -u scripts/summarize_sync_final_failures.py --progress results/masac_v4_sync_40k_progress_20261105 --out results/masac_v4_sync_40k_analysis_20261105 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/final_failure_reproduction_console.log
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv
Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory
```

Entry point exit0 reproduces all six failure-count vectors, both JSON tables
and final_endpoints_and_failure_classes.png/.svg. PNG visually reviewed: all
seeds, the negative pair,8/12gate and exhaustive descriptive classes visible.
No best checkpoint, confidence interval from pooled maps, model load or rollout.
System-wide sample:GPU59%,803MiB,38C; freeRAM~17.14GiB. These system-wide GPU
values are not attributed to a training worker; the40k workers had already exited.

## 13. Six-history late-replay content audit: COMPLETE

The separate offline plan was already pushed in d8ff5ad. Root reviewed the entire
new analysis script in bounded reads, including raw validity,
mode inclusion, sentinel/geometric minima, current/preview/first-hit provenance,
sample freezing, three synthetic witnesses and error/overwrite handling. No
frozen trainer source/config was changed. The only model-related import was
the deterministic feature provider; no checkpoint was deserialized or actor run.

```powershell
.\.venv\Scripts\python.exe -u scripts/audit_final_replay_probe_content.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --out results/masac_v4_sync_40k_late_content_audit_20261105 --device cuda --sample-size 1024 --window 8000 --chunk-size 128 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/late_replay_content_console.log
```

Exit0. Six replay identities and actual SHA256s matched before computation;
all14declared source hashes and both config hashes match. Fixed1024indices in
each final replay's last8000transitions; index manifest SHA256
eca0cfd9c12108af9627b89910fcf6a657c168a9c1bb54622eb628d15d3b3fc7.
48chunks/6144jointobs complete, every full/static trace bitwise equals frozen
provider, maximum error0. All3 independent geometry-only witnesses pass and
are excluded from real-data counts. Per-chunk progress and hashes were streamed
to the screen and the console log; no error.json or nonfinite output occurred.

All18432agent observations have two valid peers; moving0/73728validslots.
Full==static-fill1 on349849/350208tokens=99.89748949%. Only359peer residuals,
allcontrol1522, maximum normalized difference0.4561988711. Other nine columns,
including TTC/brake/reserve/flag, have zero mode differences. Most peer1 values
come from valid-far geometry and a strictly smaller40m padding sentinel, not
absent peers. Static boxes are also present but staticclearance differs from1
in only8tokens; boundary/ego channels retain variation. This describes only
static-trained late replay, not historical full policy or fresh final rollouts.

Root printed and ran a compact metadata consistency check: index/script/provider
SHA256s, six histories/same-variant input SHA,6144samplecount,350208channel counts,
48zero-errortracechunks, zero nonfinite values, and peer residual count equal to
the traced valid-geometric-min-below-scale count. It reports
AUDIT_ARTIFACT_CONSISTENCY PASS residual_peer_tokens359 total350208.
Detailed extraction and check logs are late_content_summary_console.log and
late_content_consistency_console.log. The full artifact set is in the named
late-content audit directory; no .pt/.npz is copied or staged.

Paper/frontier appendices now report these results and their population bounds.
MATH_PATHC_OBJECTIVE_DIAGNOSTIC_20261105.md separately checks actual once-ever
ball memory, capture entry/exit signs, bounded discounted capture component,
legal control examples and delay-versus-finish accounting. Positive near-goal
holding reward alone does not prove a hover preference or the observed cause.
Its feasible one-step finish example returns54.06949, better than a one-step
delay. These are mathematical constructions, not extra trained-policy rollouts.

Next under preparation is a separately frozen offline actor-response diagnostic
on the same hashed observations. It may load only the already verified own final
checkpoint after a new protocol/code review/push; phaseA's no-model contract is
complete and unchanged. No new training, environment step or sealed test is queued.

The first cached diff check stopped with exit2 on Matplotlib-generated SVG
trailing spaces. The SVG exporter now strips line-end whitespace, and the
existing SVG was cleaned after asserting every parsed path's numeric command
tokens remain identical. This is formatting only; plot values and geometry
remain unchanged. No check was bypassed and no commit was made before repair.

PhaseA/code/math/figure stage subsequently passed cached diff check and was
committed/pushed to both branches as9166477c12c4bd0bd06dbfcff504e7eed0fdab1f;
both remote refs verified.33files changed, no weights/replay staged.

## 14. Next protocol: fixed-population offline actor responses

New plan/code were prepared and root reviewed before loading any checkpoint:
`design/MASAC_V4_NEAR_GOAL_ACTOR_OFFLINE_PROTOCOL_20261105.md` and
`scripts/audit_sync_near_goal_actor.py`. Independent mathematical review checked
the scope: tanh(mu) differs from the squashed distribution's expectation; common
64x3epsilon vectors support marginal, not independent-agent joint MC; one-step
kinematics are neither collision feasibility nor episode success. Distance>=4.4
cannot capture in one .2s step ending below2m/s. The tighter state-dependent
necessary bound is also reported, with all observations/bins retained.

Only own final40000actors from verified local checkpoints, same6144obs/hashes,
fixed CPU-generator seed20261105,64draws, no learner/Q/optimizer/update/environment
creation or step. Float32actor/noise precedes float64geometry from quantized
observations. Strict4/2 predicates remain unchanged. Each seed and full/binned
denominator remains visible. Starting-in-set retention is distinct from new
acquisition; zero-of64 is not a true zero probability or significance claim.

Source reads verified V4observe max(world) goal displacement scaling, exact
actor feature/body/heads, checkpoint packaging and the original vector limiter.
Root reviewed the complete new code in two bounded reads; corrected the live
progress reset between histories and applied the protocol's1e-9 tolerance only
to a numerical necessary-bound assertion, never to strict capture.

```powershell
.\.venv\Scripts\python.exe -m py_compile scripts/audit_sync_near_goal_actor.py
git diff --check
```

Both checks exit0. Four fixed-vector limiter cross-checks are scheduled before
actor loading, followed by actual sample/actor/RNG consistency checks in the run.
Output-specific manifest/status/completion/error support recovery; completion
is the last write and includes hashes. Existing nonempty output cannot be reused.
The new protocol and code must be pushed before the printed execution CLI.

The four protocol/code/state files passed cached checks and were pushed to both
branches as3615a66bada42e5e10662ec37f92fc9bcfcfb677, remote refs verified.

## 15. Fixed-state actor audit: COMPLETE, mixed marginal responses

```powershell
.\.venv\Scripts\python.exe -u scripts/audit_sync_near_goal_actor.py --control results/masac_v4_sync_40k_control_cuda_20261105 --treatment results/masac_v4_sync_40k_b_cuda_20261105 --phase-a results/masac_v4_sync_40k_late_content_audit_20261105/audit.json --protocol docs/design/MASAC_V4_NEAR_GOAL_ACTOR_OFFLINE_PROTOCOL_20261105.md --out results/masac_v4_sync_40k_near_goal_actor_audit_20261105 --device cuda --chunk-size 128 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/near_goal_actor_console.log
```

Exit0, six histories/6144jointobs/18432agentobs; four fixed-vector limiter
cross-checks pass,48chunks complete, all actor hashes before/after match and
global CPU/CUDA Torch RNG states unchanged. No learner/environment was created,
no update/Q/TD/rollout/sealed map. The trusted own checkpoints were hashed before
deserialization; only actor states were consumed. Each output has manifest,
atomic live status and a final completion file with artifact sizes/SHA256s.

The first tool display clipped119tokens from the middle of the75-line console;
the full tee log is retained. The affected control1533@384/512/640 chunk lines
were explicitly printed again with Select-String, without rerunning inference.

Primarynear4–10m/speed<2:8000agentobs,7359 outside a necessary one-step distance
bound. Deterministic168/8000 memberships versus marginalMC10440/512000, paired
mean difference-.0609375pp. Per-history differences have3positive/3negative
signs, not a uniform rescue. Mean inwarddetaccel1.90513m/s², dchange-.0712755m;
MC-minus-det dchange+.00255531m. Boundary4–4.4m/speed<2:802obs,161 excluded by
the tighter bound, deterministic168/802, marginalMC10440/51328. These are
correlated, one-step strict-set memberships, not feasible joint/episode outcomes.

Root independently read completion and checked each listed artifact size/hash,
script/protocol SHA, six actors' immutable hashes, RNG marker,48chunklog count,
3072states/history, exhaustive parent bins and their boundary sub-bins, and
within-state paired differences against raw membership fractions. Result:
ACTOR_AUDIT_INTEGRITY PASS. Record:
results/masac_v4_windows_audit_20261105/near_goal_actor_output_verification.json.
Detailed compact differences/console are in the adjacent summary and verification
tee logs. Full report: NEAR_GOAL_ACTOR_OFFLINE_RESULTS_20261105.md.

For the next already-saved-trajectory diagnostic, root inspected one original
final evaluation row and its NPZ schema, without running a policy. Arrays are
positions/velocities(T+1,3,3)float64, actions(T,3,3)float32, goals(3,3), buildings
(M,2,3). JSON has final/collision/event data and reshape coefficients but **no
cumulative reward/return**. Thus any later source-derived component table must
mark reference-return verification UNAVAILABLE; complete dwell/event checking
remains possible. This missing reference is not to be filled from training
episode logs, which are different trajectories. The next protocol/code is
prepared separately, not launched by the completed actor diagnosis.

Actor results/paper/state were explicitly staged, cached checks passed and
19files pushed to both branches as83ea09073bf5461d579835fc1bffc868373c7bf0;
remote refs verified. No Python process remained. The next trajectory-analysis
agent then reported usage unavailable before root review. Its two new files
exist locally and were inspected only at the header/CLI tail, not validated.
Root preserved them as visibly unreviewed drafts and blocked the CLI; no
diagnostic execution occurred. Recovery details and necessary initialization/
time-counting checks are in FINAL_TRAJECTORY_DRAFT_RECOVERY_20261105.md.
Do not read25sampled post-step frames as a proof of continuous5s dwell:25h=5s
occupancy-equivalent count versus24h=4.8s endpoint span. Account/payment settings
were untouched; the same15-minute heartbeat remains active for normal resumption.

## 16. Reviewed final-trajectory audit registration

Resumed from4b92498 without restarting a completed study. Root read original
reward/limiter/writer/helper and scene metadata; independent read-only review
checked reward/reset/time handling and found a provenance issue: repeated
snapshots must compare against the first hash, never replace it. Corrected.
All14 frozen training-source hashes and original config hashes are verified
before measurement. No training source/config was changed.

```powershell
.venv\Scripts\python.exe -m py_compile scripts\audit_sync_final_trajectory_dwell.py
git diff --check
```

Initial checks exit0. Final protocol/code must be explicitly staged and pushed
to both branches before this intended first execution, after checking the
output directory and processes:

```powershell
.venv\Scripts\python.exe -u scripts\audit_sync_final_trajectory_dwell.py --out results/masac_v4_sync_40k_final_dwell_audit_20261105 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/final_dwell_console.log
```

Fixed72 saved trajectories, CPU/NumPy only, no weights, actor, environment,
training or sealed maps. Logged evaluation-return reference remains UNAVAILABLE.

Protocol/code pushed ebaa9c988eaa4d26e7eb35d5300ab909daac3b78 to both branches,
remote refs verified; no Python process, output directory absent before first run.
The printed audit CLI completed exit0 with all72MAP lines,72available traces.

```powershell
.venv\Scripts\python.exe -u scripts\summarize_sync_final_dwell.py --audit results/masac_v4_sync_40k_final_dwell_audit_20261105 --out results/masac_v4_sync_40k_final_dwell_summary_20261105 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/final_dwell_summary_console.log
```

Exit0; integrity PASS183input/six output hashes, full JSONL/summary identity,
all72saved action physics checks max velocity/position error0. Of65never-ball
agentmaps,39have>=25frame near-stall runs and48end near-stall. Occupancy figure
visually checked; sampled fractions are length/censoring-dependent, not arm
causality. B1522=0/12despite lowest near-stall fraction20.09%,58entries/53exits.
Reward components are formula-derived, reference validation UNAVAILABLE.
Report/figure/all negative cases retained. Primary endpoints/gates unchanged.

Primary-source frontier recheck added Vasan2024 goal-reaching/time-on-goal,
Farrahi/Mahmood2025preprint/PMLR2026 continuingSAC and Quartz2024 conditional
stabilization. These are close prior art/limits, not novelty or causal clearance.
Next separate candidate is training near occupancy replacement by a discounted
potential difference, static/B-off; two-arm2k sanity must be registered/pushed
before launch. No next40k or historical continuation is queued.
