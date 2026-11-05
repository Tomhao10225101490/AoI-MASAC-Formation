# Local live training and experiment progress

Date: 2026-11-05. Open **http://127.0.0.1:8765/** on this Windows machine.
The user requested visible training detail and full useful local compute use.

The read-only server is `scripts/serve_research_progress.py`; its page is
`tools/research_progress/index.html`. It refreshes every five seconds, with
pause/manual refresh and study/seed selectors. It does not launch or modify
training. Completed pilot/history outputs are visible; the new independent
40k history shards appear when their registered coordinator creates them.

The overview also shows the full registered queue: running slots, waiting,
completed histories, failures, coordinator PID and status-record age. Counts
remain unknown if the manifest/status identities do not agree. A registered
PENDING history is labeled as waiting for a slot, not as missing work; failure
or interruption pauses pending submissions. Stale coordinator records are
labeled separately. Seed filtering applies to overview cards and raw logs;
the queue summary explicitly remains for all six histories.

Only the currently selected page renders charts/log detail on refresh; other
pages render when selected. Resource and error summaries still update. This
avoids repeated hidden-page rendering without claiming a measured training
speedup. Reload an already-open page after HTML updates; the server and training
processes do not need a restart.

Views include per-seed steps/updates/throughput/estimated time, SAC actor and
critic losses, temperature, entropy, Q/target/TD diagnostics, episode returns
and near correction, every retained evaluation's success/ANY/ALL ball counts,
failure categories, checkpoint paths, raw recent logs and errors. CPU/RAM/disk
and GPU utilization/VRAM/temperature/power are sampled about every ten seconds.
Benchmark results are engineering evidence, not research success outcomes.
Missing data stays missing; old records show their age. ETA is an estimate.

Own-output status/completion/error/manifest are authoritative. The page reads
only fixed known result directories, not the shared registry or sealed maps.
It does not load models/replays or rehash large binaries. Log/JSONL reading is
bounded/incremental. Per-map failure categories may overlap. Mid-training
curves and peaks do not replace final40k registered endpoints.

The HTTP API is `GET /api/progress`. Bind is restricted to loopback; there is
no external CDN. GET unknown paths returns404; POST returns405. Real pilot
data and browser overview/SAC/evaluation views were checked before launch.

## Recovery

New separate **3D几何诊断** tab reads only the named
`results/masac_v4_joint_recourse_geometry_20261105` manifest/start/completion/error
and incremental progressJSONL. It displays real CPU/CUDA repeats, selected
device, all recorded case/first-action brackets and source-check errors. It
does not read the large geometryanalysis, models or oldreplay. No output means
notstarted/unknown, not synthetic zero progress. Geometry has no episode success/
ANY/ALL/checkpoint. Its immutable startstatus is superseded only by its own
completion/error. Backend changes require an owned-server restart and HTMLreload;
never restart training to update this page.

Latest2026-11-05: restarted only this service after six real temporary-file
testsPASS; actualPID36692/session65925 at verification. OldPID27156/38280 and
session71337 are obsolete identities. Recheck command lines/API`dashboard_pid`
before stopping or starting anything. Reader uses WindowsREAD|WRITE|DELETE
sharing, but local tests prove this does NOT ensure os.replace can replace an
open destination. JSON handles close before decode; writer-side bounded retries
remain necessary to investigate before future training. See
[the incident and final study report](NEAR_OBJECTIVE_40K_FINAL_RESULTS_20261105.md).
The six own histories finished; the failed coordinator is intentionally still
visible and cannot be cleared by a dashboard restart.

First check processes for `serve_research_progress.py` and open the URL. Do
not start another server if this one is live. If absent, from repository root
print the following command before executing it, as required by the user:

```powershell
.\.venv\Scripts\python.exe -u scripts\serve_research_progress.py --port 8765 2>&1 | Tee-Object -FilePath results/masac_v4_windows_audit_20261105/progress_dashboard_console.log -Append
```

This server is outside the frozen20 scientific source files and may receive
display fixes without changing training. Preserve study bytes and outcome
rules. Browser connection trouble alone is not evidence of training failure;
inspect process command lines and each output's files before intervention.
