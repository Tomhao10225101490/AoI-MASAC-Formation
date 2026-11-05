#!/usr/bin/env python3
"""Read-only, loopback-only dashboard for the explicitly named local studies.

No torch/numpy dependency, recursive scan, registry, checkpoint/replay load, or
large-file hashing. HTTP cannot choose a filesystem path. All reads are fixed
below or one-level, name-checked metrics/evaluation/checkpoint metadata.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "tools/research_progress/index.html"
RUNS = (
    ("near40_original", "Near 40k · original", "near40",
     "masac_v4_near_objective_40k_original_cuda_20261105", (1611, 1622, 1633), 40000),
    ("near40_potential", "Near 40k · replacement", "near40",
     "masac_v4_near_objective_40k_potential_cuda_20261105", (1611, 1622, 1633), 40000),
    ("near2_original", "Near 2k · original", "pilot",
     "masac_v4_near_objective_2k_original_cuda_20261105", (1644,), 2000),
    ("near2_potential", "Near 2k · replacement", "pilot",
     "masac_v4_near_objective_2k_potential_cuda_20261105", (1644,), 2000),
    ("b40_control", "Previous B 40k · control", "previous",
     "masac_v4_sync_40k_control_cuda_20261105", (1511, 1522, 1533), 40000),
    ("b40_treatment", "Previous B 40k · B", "previous",
     "masac_v4_sync_40k_b_cuda_20261105", (1511, 1522, 1533), 40000),
)
BENCHMARK = ROOT / "results/masac_v4_near_cuda_concurrency_20261105"
SHARDED = ROOT / "results/masac_v4_near_objective_40k_sharded_cuda_20261105"
DEVICE_EVIDENCE = ROOT / "results/masac_v4_windows_audit_20261105/device_benchmark.json"
GEOMETRY = ROOT / "results/masac_v4_joint_recourse_geometry_20261105"
MAX_DOCUMENT = 2 * 1024 * 1024
MAX_LOG_BYTES = 128 * 1024
MAX_METRICS_ROWS = 6000
STALE_AFTER_S = 90


def stamp(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat() if epoch is not None else None


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def decode_json(text):
    return json.loads(text, parse_constant=reject_constant)


def clean(value):
    """Keep malformed/non-finite external values from producing invalid HTTP JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [clean(v) for v in value]
    return value


def inside(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


@lru_cache(maxsize=1)
def _windows_file_api():
    import ctypes
    from ctypes import wintypes
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    create = library.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    close = library.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    return create, close, ctypes.c_void_p(-1).value


@contextmanager
def open_shared_binary(path):
    """Read only with Windows read/write/delete sharing consent.

    FILE_SHARE_DELETE is sharing consent, not delete access for this reader.
    This does NOT guarantee os.replace can replace an open destination on
    Windows. Callers close promptly before decoding; future writers need their
    own bounded retry contract. No frozen study writer is changed here.
    """
    if os.name != "nt":
        with Path(path).open("rb") as stream:
            yield stream
        return
    import ctypes
    import msvcrt
    create, close, invalid = _windows_file_api()
    # GENERIC_READ; FILE_SHARE_READ|WRITE|DELETE; OPEN_EXISTING; normal file.
    handle = create(os.fspath(path), 0x80000000, 0x7, None, 3, 0x80, None)
    if handle == invalid:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    except BaseException:
        close(handle)
        raise
    # The CRT descriptor owns the native handle after open_osfhandle succeeds.
    try:
        msvcrt.setmode(descriptor, os.O_BINARY)
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        yield stream


class Reader:
    def __init__(self):
        self.documents = {}
        self.streams = {}

    @staticmethod
    def metadata(path):
        result = dict(path=str(path), exists=False, updated_at=None, age_s=None, error=None)
        try:
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != ROOT):
                raise ValueError("Symlink reads are disabled")
            stat = path.stat()
            result.update(exists=True, bytes=stat.st_size, updated_at=stamp(stat.st_mtime),
                          age_s=max(0, time.time() - stat.st_mtime),
                          signature=(stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    def document(self, path):
        meta = self.metadata(path)
        cached = self.documents.get(path)
        result = dict(meta=meta, data=None, from_last_good=False)
        if not meta["exists"] or meta["error"]:
            return result
        if cached and cached["signature"] == meta["signature"]:
            result.update(data=cached["data"], from_last_good=cached["error"] is not None)
            meta["error"] = cached["error"]
            return result
        try:
            if meta["bytes"] > MAX_DOCUMENT:
                raise ValueError(f"JSON exceeds {MAX_DOCUMENT} byte read limit")
            with open_shared_binary(path) as handle:
                payload = handle.read(MAX_DOCUMENT + 1)
            if len(payload) > MAX_DOCUMENT:
                raise ValueError(f"JSON exceeds {MAX_DOCUMENT} byte read limit")
            data = decode_json(payload.decode("utf-8-sig"))
            self.documents[path] = dict(signature=meta["signature"], data=data, error=None)
            result["data"] = data
        except (OSError, UnicodeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            data = cached["data"] if cached else None
            self.documents[path] = dict(signature=meta["signature"], data=data, error=error)
            result.update(data=data, from_last_good=data is not None)
            meta["error"] = error
        return result

    def metrics(self, path):
        meta = self.metadata(path)
        empty = dict(meta=meta, rows=[], complete_lines=0, retained_rows=0,
                     dropped_rows=0, incomplete_line=False, errors=[])
        if not meta["exists"] or meta["error"]:
            return empty
        state = self.streams.get(path)
        if state is None or meta["bytes"] < state["offset"]:
            state = dict(offset=0, pending=b"", rows=deque(maxlen=MAX_METRICS_ROWS),
                         complete_lines=0, errors=deque(maxlen=12), signature=None)
            self.streams[path] = state
        if state["signature"] != meta["signature"]:
            try:
                # Metrics are small; parse only newly appended complete lines.
                with open_shared_binary(path) as handle:
                    handle.seek(state["offset"])
                    data = handle.read(8 * 1024 * 1024)
                state["offset"] += len(data)
                lines = (state["pending"] + data).split(b"\n")
                state["pending"] = lines.pop()
                for raw in lines:
                    state["complete_lines"] += 1
                    if not raw.strip():
                        continue
                    try:
                        item = decode_json(raw.decode("utf-8-sig"))
                        if not isinstance(item, dict):
                            raise ValueError("Metrics record is not an object")
                        state["rows"].append(item)
                    except (UnicodeError, ValueError) as exc:
                        state["errors"].append(
                            f"Line {state['complete_lines']}: {type(exc).__name__}: {exc}")
                # A very large first read is continued on the next request.
                state["signature"] = meta["signature"] if state["offset"] >= meta["bytes"] else None
            except OSError as exc:
                meta["error"] = f"{type(exc).__name__}: {exc}"
        return dict(meta=meta, rows=list(state["rows"]), complete_lines=state["complete_lines"],
                    retained_rows=len(state["rows"]),
                    dropped_rows=max(0, state["complete_lines"] - len(state["rows"])),
                    incomplete_line=bool(state["pending"]),
                    unread_bytes=max(0, meta["bytes"] - state["offset"]),
                    errors=list(state["errors"]))

    def tail(self, path, count=60, parse=False):
        meta = self.metadata(path)
        result = dict(meta=meta, lines=[], rows=[], errors=[], clipped=False, incomplete_line=False)
        if not meta["exists"] or meta["error"]:
            return result
        try:
            offset = max(0, meta["bytes"] - MAX_LOG_BYTES)
            with open_shared_binary(path) as handle:
                handle.seek(offset)
                raw = handle.read(MAX_LOG_BYTES)
            result["clipped"] = offset > 0
            if offset:
                first = raw.find(b"\n")
                raw = raw[first + 1:] if first >= 0 else b""
            result["incomplete_line"] = bool(raw and not raw.endswith(b"\n"))
            text = raw.decode("utf-8-sig", errors="replace")
            result["lines"] = text.splitlines()[-count:]
            if parse:
                complete = result["lines"][:-1] if result["incomplete_line"] else result["lines"]
                for index, line in enumerate(complete):
                    try:
                        result["rows"].append(decode_json(line))
                    except ValueError as exc:
                        result["errors"].append(f"Tail line {index + 1}: {exc}")
        except OSError as exc:
            meta["error"] = f"{type(exc).__name__}: {exc}"
        return result


def data_object(document):
    data = document.get("data")
    return data if isinstance(data, dict) else {}


def errors_from(*items):
    result = []
    for item in items:
        if item.get("meta", {}).get("error"):
            result.append(f"{item['meta']['path']}: {item['meta']['error']}")
        result.extend(item.get("errors", []))
        if item.get("from_last_good"):
            result.append(f"{item['meta']['path']}: showing last good document after a read error")
        if item.get("incomplete_line"):
            result.append(f"{item['meta']['path']}: incomplete final line retained for next read")
    return result


class Progress:
    def __init__(self):
        self.reader = Reader()
        self.lock = threading.Lock()
        self.resources = Resources()

    def seed(self, folder, seed, budget, run_status, run_complete, root):
        reader = self.reader
        if folder.is_symlink() or not inside(folder, root):
            return dict(seed=seed, phase="error", step=None, budget=budget, progress=[],
                        episodes=[], evaluations=[], errors=["Seed directory escapes its fixed study root"])
        metrics = reader.metrics(folder / "metrics.jsonl")
        # Return scalar diagnostic series, not repeated map/reshape geometry.
        progress = [{k: v for k, v in r.items() if v is None or isinstance(v, (str, int, float, bool))}
                    for r in metrics["rows"] if r.get("kind") == "progress"]
        episode_fields = (
            "kind", "seed", "step", "episode", "map_seed", "map_family", "return_", "base_return",
            "shaping_return", "near_removed_return", "near_potential_difference_return",
            "near_reward_steps_in_episode", "shaping_return_semantics", "success", "terminated",
            "truncated", "timeout", "ever_ball_any", "ever_ball_all",
            "max_simultaneous_capture_occupancy", "uav_collision", "static_collision",
            "dynamic_collision", "boundary_collision",
        )
        episodes = [{k: r[k] for k in episode_fields if k in r}
                    for r in metrics["rows"] if r.get("kind") == "episode"]
        completion = reader.document(folder / "completion.json")
        error = reader.document(folder / "error.json")
        latest = progress[-1] if progress else {}
        complete = data_object(completion)
        phase = "complete" if complete else "not_started"
        if progress and not complete:
            phase = run_status.get("phase", "observed")
            if run_status.get("seed") != seed and not run_complete:
                phase = "awaiting_or_incomplete"
        if data_object(error):
            phase = "error"
        step = complete.get("step") if complete else latest.get("step")
        updated = latest.get("updated_at")
        age = max(0, time.time() - updated) if finite(updated) else metrics["meta"].get("age_s")
        stale = bool(phase not in ("complete", "not_started") and age is not None and age > STALE_AFTER_S)
        summaries, eval_errors, checkpoints = [], [], []
        if folder.is_dir() and inside(folder, root):
            # A single directory level; never traverse replay/weights or source_snapshot.
            for child in folder.iterdir():
                if child.is_symlink():
                    continue
                match = re.fullmatch(r"evaluation_([0-9]+)_summary\.json", child.name)
                if match:
                    document = reader.document(child)
                    summary = data_object(document)
                    if summary:
                        summaries.append(dict(summary, file=str(child),
                                              updated_at=document["meta"]["updated_at"]))
                    eval_errors.extend(errors_from(document))
                if re.fullmatch(r"checkpoint_[0-9]{6,}", child.name) and child.is_dir():
                    checkpoints.append(child)
        summaries.sort(key=lambda x: x.get("step", -1))
        checkpoint, checkpoint_pending = None, []
        for newest in sorted(checkpoints, key=lambda x: int(x.name.split("_")[-1]), reverse=True):
            manifest = reader.document(newest / "manifest.json")
            eval_errors.extend(errors_from(manifest))
            if data_object(manifest).get("complete") is True:
                checkpoint = dict(path=str(newest), manifest=manifest,
                                  checkpoint_metadata=reader.metadata(newest / "checkpoint.pt"),
                                  replay_metadata=reader.metadata(newest / "replay.npz"))
                break
            checkpoint_pending.append(str(newest))
        # Pilot wrote the stream at root; the formal study writes per-seed.
        near_path = folder / "near_reward_transitions.jsonl"
        if not near_path.exists() and (root / "near_reward_transitions.jsonl").exists():
            near_path = root / "near_reward_transitions.jsonl"
        near = reader.tail(near_path, count=24, parse=True)
        checks = reader.tail(folder / "evaluation_state_checks.jsonl", count=24, parse=True)
        problems = errors_from(metrics, completion, error, near, checks) + eval_errors
        if checkpoint:
            problems.extend(errors_from(checkpoint["manifest"]))
        return dict(seed=seed, directory=str(folder), phase=phase, step=step, budget=budget,
                    updates=latest.get("updates"), steps_per_s=latest.get("steps_per_s"),
                    eta_s=0 if complete else latest.get("eta_s"),
                    age_s=age, stale=stale, latest=latest, completion=completion, error=error,
                    progress=progress, episodes=episodes, evaluations=summaries,
                    metrics_meta={k: v for k, v in metrics.items() if k != "rows"},
                    near_tail=near, evaluation_checks=checks, checkpoint=checkpoint,
                    checkpoint_pending=checkpoint_pending, errors=problems)

    def run(self, definition, root_override=None):
        key, label, group, basename, fallback_seeds, fallback_budget = definition
        root = root_override or ROOT / "results" / basename
        exists = root.is_dir() and not root.is_symlink()
        reader = self.reader
        manifest = reader.document(root / "run_manifest.json")
        status = reader.document(root / "status.json")
        completion = reader.document(root / "completion.json")
        error = reader.document(root / "error.json")
        manifest_data, status_data = data_object(manifest), data_object(status)
        config = manifest_data.get("config", manifest_data.get("protocol_config", {}))
        if not isinstance(config, dict):
            config = {}
        candidates = manifest_data.get("seed_list", manifest_data.get("seeds", config.get("seeds")))
        seeds = (candidates if isinstance(candidates, list) and candidates and
                 all(isinstance(s, int) and not isinstance(s, bool) and s in fallback_seeds for s in candidates)
                 else list(fallback_seeds))
        budget = manifest_data.get("steps_per_seed", manifest_data.get("steps", config.get("steps", fallback_budget)))
        if not finite(budget) or budget <= 0:
            budget = fallback_budget
        phase = status_data.get("phase", manifest_data.get("phase", "observed" if exists else "not_created"))
        if data_object(completion).get("complete") is True:
            phase = "complete"
        if data_object(error):
            phase = "error"
        complete = phase == "complete"
        seed_rows = [self.seed(root / f"maneuver_static_{seed}", seed, budget,
                               status_data, complete, root) for seed in seeds] if exists else []
        logs = reader.tail(root / "train.log")
        issues = errors_from(manifest, status, completion, error, logs)
        for row in seed_rows:
            issues.extend(row["errors"])
        last_age = status["meta"].get("age_s")
        return dict(id=key, label=label, group=group, path=str(root), exists=exists,
                    phase=phase, budget=budget, seed_list=seeds, seeds=seed_rows,
                    seed_source="manifest" if candidates else "fixed expected seeds; manifest unavailable",
                    manifest=manifest, status=status, completion=completion, error=error,
                    stale=bool(exists and not complete and phase != "error" and
                               last_age is not None and last_age > STALE_AFTER_S),
                    logs=logs, errors=list(dict.fromkeys(issues)))

    def sharded(self):
        reader = self.reader
        documents = {name: reader.document(SHARDED / name) for name in
                     ("study_manifest.json", "status.json", "completion.json", "error.json")}
        manifest = data_object(documents["study_manifest.json"])
        errors, runs, seen = [], [], set()
        for document in documents.values():
            errors.extend(errors_from(document))
        if manifest:
            if (manifest.get("schema_version") != 1 or
                    manifest.get("layout") != "near-objective-independent-histories-v1" or
                    manifest.get("seeds") != [1611, 1622, 1633]):
                errors.append("Unsupported sharded manifest schema/layout/seeds; refusing child reads")
            else:
                jobs = manifest.get("jobs", [])
                if not isinstance(jobs, list):
                    errors.append("Coordinator jobs must be a list; refusing child reads")
                    jobs = []
                for job in jobs:
                    if not isinstance(job, dict):
                        errors.append("Invalid coordinator job object")
                        continue
                    mode, seed = job.get("mode"), job.get("training_seed")
                    expected = f"runs/{mode}_{seed}"
                    relative = job.get("relative_root")
                    if (mode not in ("original", "potential") or not isinstance(seed, int) or
                            seed not in (1611, 1622, 1633) or isinstance(seed, bool) or relative != expected or
                            (mode, seed) in seen):
                        errors.append("Invalid or duplicate coordinator child; refusing path: " + str(relative))
                        continue
                    child = SHARDED / relative
                    if child.is_symlink() or not inside(child, SHARDED):
                        errors.append("Coordinator child escapes the fixed root: " + str(relative))
                        continue
                    seen.add((mode, seed))
                    label = f"Near 40k · {'replacement' if mode == 'potential' else mode} · {seed}"
                    runs.append(self.run((f"shard_{mode}_{seed}", label, "sharded",
                                          "", (seed,), 40000), root_override=child))
        return dict(path=str(SHARDED), documents=documents, errors=errors), runs

    def snapshot(self):
        with self.lock:
            runs = [self.run(item) for item in RUNS]
            coordinator, shards = self.sharded()
            runs = shards + runs
            benchmark = {
                name: self.reader.document(BENCHMARK / name) for name in
                ("case_2.json", "case_3.json", "case_4.json", "benchmark.json", "completion.json", "error.json")
            }
            geometry = dict(path=str(GEOMETRY), documents={
                name: self.reader.document(GEOMETRY / name) for name in
                ("manifest.json", "status.json", "completion.json", "error.json")
            }, progress=self.reader.metrics(GEOMETRY / "progress.jsonl"))
            return clean(dict(generated_at=stamp(time.time()), generated_epoch=time.time(),
                              dashboard_pid=os.getpid(),
                              reader_file_sharing="READ|WRITE|DELETE" if os.name == "nt" else "binary read",
                              refresh_s=5, stale_after_s=STALE_AFTER_S, runs=runs,
                              coordinator=coordinator,
                              geometry=geometry,
                              benchmark=dict(path=str(BENCHMARK), documents=benchmark),
                              device_evidence=self.reader.document(DEVICE_EVIDENCE),
                              resources=self.resources.snapshot(),
                              read_only=True, notes=[
                                  "Own output status/completion/error files are authoritative; no global registry.",
                                  "Evaluation: deterministic original Path-C, development maps only.",
                                  "Missing or unreadable values remain unknown; no invented zero series.",
                                  "Checkpoint/replay are metadata only; hashes are recorded manifest values, not recomputed.",
                              ]))


class Resources:
    def __init__(self):
        self.lock = threading.Lock()
        self.value = dict(sampled_at=None, cpu=None, ram=None, processes=[], gpu=None, errors=[])
        self.psutil = None
        self.seen_pids = set()
        try:
            import psutil
            self.psutil = psutil
            psutil.cpu_percent(interval=None)
            print("RESOURCE psutil available (optional import; no installation)", flush=True)
        except ImportError:
            print("RESOURCE psutil unavailable; CPU/RAM/process fields remain unknown", flush=True)
        threading.Thread(target=self.loop, name="dashboard-resources", daemon=True).start()

    def loop(self):
        while True:
            value = dict(sampled_at=stamp(time.time()), cpu=None, ram=None, processes=[], gpu=None, errors=[])
            try:
                disk = shutil.disk_usage(ROOT)
                value["disk"] = dict(total_bytes=disk.total, free_bytes=disk.free)
                if self.psutil:
                    ps = self.psutil
                    memory = ps.virtual_memory()
                    value["cpu"] = dict(percent=ps.cpu_percent(interval=None), logical=ps.cpu_count())
                    value["ram"] = dict(total_bytes=memory.total, available_bytes=memory.available,
                                        percent=memory.percent)
                    for process in ps.process_iter(["pid", "name", "cmdline", "memory_info"]):
                        try:
                            command = " ".join(process.info.get("cmdline") or [])
                            if any(name in command for name in
                                    ("train_masac_v4_near_objective", "benchmark_near", "benchmark_cuda",
                                     "near_cuda_concurrency", "run_near_objective",
                                    "train_masac_v4_sync_capture")):
                                rss = process.info.get("memory_info")
                                cpu = process.cpu_percent(interval=None)
                                value["processes"].append(dict(
                                    pid=process.pid, name=process.info.get("name"),
                                    cpu_percent=cpu if process.pid in self.seen_pids else None,
                                    rss_bytes=rss.rss if rss else None))
                                self.seen_pids.add(process.pid)
                        except (ps.NoSuchProcess, ps.AccessDenied):
                            continue
            except Exception as exc:
                value["errors"].append(f"Resource sample: {type(exc).__name__}: {exc}")
            executable = shutil.which("nvidia-smi")
            if executable:
                command = [executable, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                           "--format=csv,noheader,nounits"]
                # Every external invocation is visible BEFORE subprocess execution.
                print("COMMAND " + subprocess.list2cmdline(command), flush=True)
                try:
                    output = subprocess.run(command, capture_output=True, text=True, timeout=4,
                                            check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    if output.returncode:
                        raise RuntimeError(f"exit={output.returncode}: {output.stderr.strip()}")
                    import csv
                    value["gpu"] = [dict(zip(
                        ("index", "name", "utilization_percent", "used_mib", "total_mib", "temperature_c", "power_w"),
                        [cell.strip() for cell in row]))
                        for row in csv.reader(output.stdout.splitlines()) if row]
                except Exception as exc:
                    value["errors"].append(f"nvidia-smi: {type(exc).__name__}: {exc}")
            else:
                value["errors"].append("nvidia-smi not on PATH; GPU values unavailable")
            with self.lock:
                self.value = value
            time.sleep(10)

    def snapshot(self):
        with self.lock:
            return dict(self.value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use a local unprivileged port, 1024..65535")
    progress = Progress()

    class Handler(BaseHTTPRequestHandler):
        last_progress_access_log = 0.0

        def log_message(self, format_, *values):
            message = format_ % values
            if '"/api/progress HTTP/' in message and " 200 " in message:
                now = time.monotonic()
                if now - Handler.last_progress_access_log < 60:
                    return
                Handler.last_progress_access_log = now
            print(f"HTTP {self.address_string()} {message}", flush=True)

        def do_GET(self):
            allowed_hosts = {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}
            if self.headers.get("Host", "") not in allowed_hosts:
                self.send_error(403, "Loopback Host required")
                return
            address = urlsplit(self.path)
            if address.query:
                self.send_error(400, "Query parameters are not supported")
                return
            try:
                if address.path == "/":
                    with open_shared_binary(HTML) as stream:
                        payload = stream.read()
                    mime = "text/html; charset=utf-8"
                elif address.path == "/api/progress":
                    payload = json.dumps(progress.snapshot(), ensure_ascii=False,
                                         allow_nan=False).encode("utf-8")
                    mime = "application/json; charset=utf-8"
                elif address.path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                    return
                else:
                    self.send_error(404, "Only / and /api/progress exist")
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy",
                                 "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                 "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                                 "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                print(f"DASHBOARD ERROR {type(exc).__name__}: {exc}", flush=True)
                self.send_error(500, "Dashboard read failed; inspect server stdout")

        def do_POST(self):
            self.send_error(405, "Read-only server")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"Read-only research progress: http://127.0.0.1:{args.port}/", flush=True)
    print("Fixed studies only; no replay/weight loading, registry reads, sealed maps, or recursive scans.", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Dashboard stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
