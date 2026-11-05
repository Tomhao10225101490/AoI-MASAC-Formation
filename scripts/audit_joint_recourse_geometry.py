#!/usr/bin/env python3
"""Registered synthetic geometry diagnostic; no policy or environment stepping."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_planning.environment import swept_distance
from uav_planning.structured import DIRECTIONS, limited
from uav_planning.v4_environment import V4Mission
from uav_planning.v4_prediction import ProbeConfig, predict_probes

CONFIG = "configs/masac_v4/joint_recourse_geometry_20261105.json"
PROTOCOL = "docs/design/MASAC_V4_JOINT_RECOURSE_GEOMETRY_PROTOCOL_20261105.md"
SOURCES = ("scripts/audit_joint_recourse_geometry.py", CONFIG, PROTOCOL,
           "tests/test_joint_recourse_geometry.py", "uav_planning/environment.py",
           "uav_planning/structured.py", "uav_planning/v4_environment.py",
           "uav_planning/v4_maps.py", "uav_planning/v4_prediction.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_new(path, value):
    """No replace/overwrite; completion is written separately, last."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def command(args):
    print("COMMAND " + subprocess.list2cmdline(args), flush=True)
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def registered_sources():
    head = command(["git", "rev-parse", "HEAD"])
    command(["git", "ls-files", "--error-unmatch", "--", *SOURCES])
    command(["git", "diff", "--exit-code", "HEAD", "--", *SOURCES])
    return head, {name: (ROOT / name).read_bytes() for name in SOURCES}


class Recorder:
    def __init__(self, out):
        self.out = out
        self.started = time.perf_counter()
        self.sequence = 0

    def emit(self, kind, **values):
        self.sequence += 1
        row = dict(kind=kind, sequence=self.sequence, time_utc=now(),
                   elapsed_s=time.perf_counter() - self.started, **values)
        line = json.dumps(row, allow_nan=False)
        print(line, flush=True)
        with (self.out / "progress.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()


def make_case(cfg, specification):
    origin = np.asarray(cfg["starts"][0], dtype=float)
    kind = specification["kind"]
    signs = ([] if kind == "empty" else [-1] if kind == "single_negative" else
             [1] if kind == "single_positive" else [-1, 1])
    centers = np.repeat(origin[None], len(signs), axis=0)
    velocity = np.zeros_like(centers)
    if len(signs):
        centers[:, specification["approach_axis"]] += 2 * cfg["dt"] * cfg["approach_speed"]
        separation = cfg["wide_half_separation"] if kind == "wide" else cfg["conflict_half_separation"]
        centers[:, specification["separation_axis"]] += np.asarray(signs) * separation
        speed = 0 if kind == "static" else cfg["approach_speed"] * (1 if kind == "receding" else -1)
        velocity[:, specification["approach_axis"]] = speed
    return dict(**specification, dynamic_positions=centers.tolist(),
                dynamic_velocities=velocity.tolist(),
                dynamic_radii=[cfg["sphere_radius"]] * len(signs))


def observations(cfg, case):
    """Call the existing pure observation function on a synthetic data record."""
    record = SimpleNamespace(
        n=3, p=np.asarray(cfg["starts"], dtype=float), v=np.zeros((3, 3)),
        prev_acc=np.zeros((3, 3)), goal=np.asarray(cfg["goals"], dtype=float),
        world=np.asarray(cfg["world"], dtype=float), vmax=cfg["vmax"], amax=cfg["amax"],
        radius=cfg["body_radius"], config=SimpleNamespace(sensor_range=cfg["sensor_range"]),
        boxes=np.empty((0, 2, 3)), lo=np.empty((0, 3)), hi=np.empty((0, 3)),
        dynamic=np.asarray(case["dynamic_positions"], dtype=float).reshape(-1, 3),
        dv=np.asarray(case["dynamic_velocities"], dtype=float).reshape(-1, 3),
        dr=np.asarray(case["dynamic_radii"], dtype=float),
        _visible_order=V4Mission._visible_order)
    return V4Mission.observe(record)


def torch_limited(velocity, action, cfg):
    action = action.clamp(-1, 1)
    action = action / torch.linalg.vector_norm(action, dim=-1, keepdim=True).clamp_min(1)
    speed = velocity + cfg["dt"] * cfg["amax"] * action
    norm = torch.linalg.vector_norm(speed, dim=-1, keepdim=True)
    return speed * torch.minimum(torch.ones_like(norm), cfg["vmax"] / (norm + 1e-12))


def torch_swept(start, end):
    delta = end - start
    fraction = (-(start * delta).sum(-1) / ((delta * delta).sum(-1) + 1e-12)).clamp(0, 1)
    return torch.linalg.vector_norm(start + fraction[..., None] * delta, dim=-1)


def constraint_names(case):
    return ([f"moving_{j}" for j in range(len(case["dynamic_radii"]))] +
            ["peer_1", "peer_2"] + [f"boundary_{side}_{axis}" for side in ("low", "high") for axis in range(3)])


def margin_batch(cfg, case, first, backup):
    """All constraints over two source-semantics segments, for N backup actions."""
    tensor = lambda value: torch.as_tensor(value, dtype=backup.dtype, device=backup.device)
    p0 = tensor(cfg["starts"][0])
    world = tensor(cfg["world"])
    v1 = torch_limited(torch.zeros_like(p0), tensor(first), cfg)
    p1 = p0 + cfg["dt"] * v1
    v2 = torch_limited(v1, backup, cfg)
    p2 = p1 + cfg["dt"] * v2
    centers = np.concatenate((np.asarray(case["dynamic_positions"]).reshape(-1, 3), np.asarray(cfg["starts"])[1:]))
    velocities = np.concatenate((np.asarray(case["dynamic_velocities"]).reshape(-1, 3), np.zeros((2, 3))))
    radii = tensor([radius + cfg["body_radius"] + cfg["safety_margin"] for radius in case["dynamic_radii"]] +
                   [2 * cfg["body_radius"] + cfg["safety_margin"]] * 2)
    centers0, speeds = tensor(centers), tensor(velocities)
    centers1 = centers0 + cfg["dt"] * speeds
    centers2 = centers1 + cfg["dt"] * speeds
    first_segment = torch_swept(p0 - centers0, p1 - centers1) - radii
    second_segment = torch_swept(p1[None, None] - centers1[None], p2[:, None] - centers2[None]) - radii
    spherical = torch.minimum(first_segment[None], second_segment)
    inflation = cfg["body_radius"] + cfg["safety_margin"]
    bound0 = torch.cat((p0 - inflation, world - inflation - p0))
    bound1 = torch.cat((p1 - inflation, world - inflation - p1))
    bound2 = torch.cat((p2 - inflation, world - inflation - p2), dim=-1)
    boundaries = torch.minimum(torch.minimum(bound0, bound1)[None], bound2)
    return torch.cat((spherical, boundaries), dim=-1) / cfg["length_scale"]


def numpy_constraints(cfg, case, first, backup):
    """Independent oracle using original NumPy limiter and swept helper."""
    p = np.asarray(cfg["starts"][0], dtype=float)
    velocity = np.zeros(3)
    world = np.asarray(cfg["world"])
    centers = np.concatenate((np.asarray(case["dynamic_positions"]).reshape(-1, 3), np.asarray(cfg["starts"])[1:]))
    speeds = np.concatenate((np.asarray(case["dynamic_velocities"]).reshape(-1, 3), np.zeros((2, 3))))
    radii = np.asarray([r + cfg["body_radius"] + cfg["safety_margin"] for r in case["dynamic_radii"]] +
                       [2 * cfg["body_radius"] + cfg["safety_margin"]] * 2)
    inflation = cfg["body_radius"] + cfg["safety_margin"]
    sphere_min = np.full(len(centers), np.inf)
    boundary_min = np.r_[p - inflation, world - inflation - p]
    for action in (first, backup):
        old, old_centers = p.copy(), centers.copy()
        velocity = limited(velocity, np.clip(np.asarray(action), -1, 1), cfg["dt"], cfg["amax"], cfg["vmax"])
        p = p + cfg["dt"] * velocity
        centers = centers + cfg["dt"] * speeds
        sphere_min = np.minimum(sphere_min, swept_distance(old - old_centers, p - centers) - radii)
        boundary_min = np.minimum(boundary_min, np.r_[p - inflation, world - inflation - p])
    return np.r_[sphere_min, boundary_min] / cfg["length_scale"]


def grid_chunks(resolution, chunk_size, device):
    for start in range(0, resolution ** 3, chunk_size):
        index = torch.arange(start, min(start + chunk_size, resolution ** 3), device=device)
        coordinates = torch.stack((index // (resolution ** 2), (index // resolution) % resolution, index % resolution), -1)
        yield coordinates.to(torch.float64) * (2 / (resolution - 1)) - 1


def optimize_grid(cfg, case, first, device):
    count = 0
    joint_value = -math.inf
    joint_action = None
    maxima = torch.full((len(constraint_names(case)),), -math.inf, device=device, dtype=torch.float64)
    actions = torch.zeros((len(maxima), 3), device=device, dtype=torch.float64)
    for backups in grid_chunks(cfg["grid_resolution"], cfg["chunk_size"], device):
        margins = margin_batch(cfg, case, first, backups)
        if not bool(torch.isfinite(margins).all()):
            raise FloatingPointError("Nonfinite grid margins")
        scores = margins.amin(-1)
        value, index = scores.max(0)
        scalar = value.item()
        if scalar > joint_value:
            joint_value, joint_action = scalar, backups[index].cpu().tolist()
        values, indices = margins.max(0)
        improved = values > maxima
        actions = torch.where(improved[:, None], backups[indices], actions)
        maxima = torch.maximum(maxima, values)
        count += len(backups)
    return dict(joint=joint_value, joint_raw_action=joint_action,
                pair=maxima.min().item(), per_constraint_max=maxima.cpu().tolist(),
                per_constraint_raw_actions=actions.cpu().tolist(), actions_evaluated=count)


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def benchmark(cfg, case, recorder):
    records = []
    for option in cfg["benchmark_options"]:
        device, threads = option["device"], option["threads"]
        if device == "cuda" and not torch.cuda.is_available():
            recorder.emit("benchmark_unavailable", device=device, reason="torch.cuda.is_available=false")
            continue
        torch.set_num_threads(threads)
        recorder.emit("benchmark_start", device=device, threads=threads,
                      grid_actions=cfg["grid_resolution"] ** 3,
                      warmups=cfg["benchmark_warmups"], repeats=cfg["benchmark_repeats"])
        for _ in range(cfg["benchmark_warmups"]):
            optimize_grid(cfg, case, [0, 0, 0], device)
        repeats, outputs, peak = [], [], 0
        for repeat in range(cfg["benchmark_repeats"]):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            synchronize(device)
            start = time.perf_counter()
            output = optimize_grid(cfg, case, [0, 0, 0], device)
            synchronize(device)
            elapsed = time.perf_counter() - start
            if device == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated())
            repeats.append(elapsed)
            outputs.append(output["joint"])
            recorder.emit("benchmark_repeat", device=device, threads=threads, repeat=repeat,
                          elapsed_s_kernel=elapsed, actions=output["actions_evaluated"],
                          actions_per_s=output["actions_evaluated"] / elapsed, joint=output["joint"],
                          peak_allocated_bytes=peak)
        records.append(dict(**option, repeats_s=repeats, median_s=statistics.median(repeats),
                            joint_values=outputs, peak_allocated_bytes=peak))
    agreement = max(v for row in records for v in row["joint_values"]) - min(v for row in records for v in row["joint_values"])
    if agreement > cfg["numeric_tolerance"]:
        raise AssertionError(f"CPU/device benchmark disagreement {agreement}")
    selected = min(records, key=lambda row: row["median_s"])
    recorder.emit("benchmark_selected", selected=selected, max_joint_difference=agreement,
                  scope="This geometry kernel only; no old trainer default changes")
    torch.set_num_threads(selected["threads"])
    return dict(options=records, selected=selected, max_joint_difference=agreement)


def content(cfg, case, device):
    obs = observations(cfg, case)
    pcfg = ProbeConfig(world=tuple(cfg["world"]), body_radius=cfg["body_radius"],
                       safety_margin=cfg["safety_margin"], dt=cfg["dt"], horizon=cfg["steps"] * cfg["dt"])
    cpu = torch.from_numpy(obs)
    full_cpu = predict_probes(cpu, pcfg, "full")
    selected = predict_probes(cpu.to(device), pcfg, "full").cpu()
    error = (selected - full_cpu).abs().max().item()
    if error > cfg["probe_tolerance"]:
        raise AssertionError(f"Source probe CPU/device disagreement {error}")
    static = predict_probes(cpu, pcfg, "static")
    fill = static.clone()
    fill[..., 4:6] = 1
    moving = obs[0, 111:143].reshape(4, 8)
    return dict(observations=obs.tolist(), observation_sha256=digest(obs.tobytes()),
                moving_valid_count=int((moving[:, 7] > .5).sum()),
                moving_speed_mps=np.linalg.norm(np.asarray(case["dynamic_velocities"]).reshape(-1, 3), axis=-1).tolist(),
                full=full_cpu.tolist(), static=static.tolist(), static_fill_one=fill.tolist(),
                full_vs_fill_different_values=int((full_cpu != fill).sum()),
                ego_moving_nonsaturated_values=int(((full_cpu[0, :, 4] > -1) & (full_cpu[0, :, 4] < 1)).sum()),
                probe_cpu_selected_max_error=error,
                scope="Constant-action source probe content only; NOT the committed-a/backup-b control family")


def scene_checks(cfg, case):
    points = np.asarray(case["dynamic_positions"]).reshape(-1, 3)
    velocities = np.asarray(case["dynamic_velocities"]).reshape(-1, 3)
    minima = []
    for step in range(3):
        current = points + step * cfg["dt"] * velocities
        if len(current):
            assert np.all(current - cfg["sphere_radius"] > 0)
            assert np.all(current + cfg["sphere_radius"] < np.asarray(cfg["world"]))
            distances = np.linalg.norm(current - np.asarray(cfg["starts"][0]), axis=-1)
            minima.append(float(distances.min() - cfg["sphere_radius"] - cfg["body_radius"]))
        if len(current) == 2:
            assert np.linalg.norm(current[0] - current[1]) > 2 * cfg["sphere_radius"]
    assert not minima or minima[0] > 0
    return dict(raw_obstacles_disjoint=True, centers_within_world=True, zero_ego_tick_clearances=minima)


def independent_witness_error(cfg, case, first, grid, library_margins):
    actions = [grid["joint_raw_action"], *grid["per_constraint_raw_actions"]]
    selected = torch.tensor(actions, dtype=torch.float64)
    direct = margin_batch(cfg, case, first, selected).numpy()
    oracle = np.stack([numpy_constraints(cfg, case, first, action) for action in actions])
    library_oracle = np.stack([numpy_constraints(cfg, case, first, action) for action in DIRECTIONS])
    error = max(float(np.abs(direct - oracle).max()), float(np.abs(library_margins - library_oracle).max()),
                abs(float(oracle[0].min()) - grid["joint"]),
                float(np.abs(np.asarray(grid["per_constraint_max"]) - np.diag(oracle[1:])).max()))
    if error > cfg["numeric_tolerance"]:
        raise AssertionError(f"Independent source witness mismatch {error}")
    return error


def execute(cfg, out, recorder):
    cases = [make_case(cfg, specification) for specification in cfg["cases"]]
    validation = {case["id"]: scene_checks(cfg, case) for case in cases}
    selected_benchmark = benchmark(cfg, next(case for case in cases if case["id"] == "dual_conflict_x"), recorder)
    device = selected_benchmark["selected"]["device"]
    allowance = cfg["amax"] * cfg["dt"] ** 2 * math.sqrt(3) / (cfg["grid_resolution"] - 1) / cfg["length_scale"]
    tolerance = cfg["numeric_tolerance"]
    rows, contents = [], {}
    for case in cases:
        contents[case["id"]] = content(cfg, case, device)
        recorder.emit("case_start", case=case["id"], content_difference=contents[case["id"]]["full_vs_fill_different_values"],
                      moving_valid_count=contents[case["id"]]["moving_valid_count"])
        for index, first in enumerate(DIRECTIONS):
            synchronize(device)
            start = time.perf_counter()
            backup = torch.as_tensor(DIRECTIONS, dtype=torch.float64, device=device)
            library = margin_batch(cfg, case, first, backup).cpu().numpy()
            grid = optimize_grid(cfg, case, first, device)
            synchronize(device)
            elapsed = time.perf_counter() - start
            oracle_error = independent_witness_error(cfg, case, first, grid, library)
            jb, pb = float(library.min(-1).max()), float(library.max(0).min())
            assert grid["joint"] + tolerance >= jb
            assert grid["pair"] + tolerance >= pb
            assert grid["joint"] <= grid["pair"] + tolerance
            row = dict(case=case["id"], first_action_index=index, first_action=first.tolist(),
                       constraint_names=constraint_names(case), library_margins=library.tolist(),
                       library_joint=jb, library_pair=pb, grid=grid,
                       joint_bracket=[grid["joint"] - tolerance, grid["joint"] + allowance + tolerance],
                       pair_bracket=[grid["pair"] - tolerance, grid["pair"] + allowance + tolerance],
                       source_oracle_max_error=oracle_error, elapsed_s=elapsed)
            rows.append(row)
            recorder.emit("first_action_complete", case=case["id"], first_action_index=index,
                          first_action=first.tolist(), finished=len(rows), total=len(cases) * len(DIRECTIONS),
                          library_joint=jb, library_pair=pb, joint_bracket=row["joint_bracket"],
                          pair_bracket=row["pair_bracket"], grid_actions=grid["actions_evaluated"],
                          actions_per_s=grid["actions_evaluated"] / elapsed, source_oracle_error=oracle_error)
    checks = {}
    zero = int(np.flatnonzero(np.linalg.norm(DIRECTIONS, axis=-1) == 0)[0])
    lookup = {(row["case"], row["first_action_index"]): row for row in rows}
    analytical = {}
    for case in cases:
        cid = case["id"]
        row = lookup[cid, zero]
        if case["kind"] == "conflict":
            retreat = np.zeros(3)
            retreat[case["approach_axis"]] = -1
            retreat_index = int(np.flatnonzero(np.all(DIRECTIONS == retreat, axis=-1))[0])
            retreat_row = lookup[cid, retreat_index]
            upper = (math.sqrt(cfg["conflict_half_separation"] ** 2 + (cfg["amax"] * cfg["dt"] ** 2) ** 2) -
                     cfg["sphere_radius"] - cfg["body_radius"]) / cfg["length_scale"]
            analytical[cid] = dict(zero_first_joint_upper=upper, retreat_first_action_index=retreat_index)
            checks[cid + "_strict_gap"] = row["pair_bracket"][0] > 0 and row["joint_bracket"][1] < 0
            checks[cid + "_analytic_upper_consistent"] = row["grid"]["joint"] <= upper + tolerance
            checks[cid + "_early_retreat"] = retreat_row["joint_bracket"][0] > 0
            item = contents[cid]
            checks[cid + "_observable_content"] = (item["moving_valid_count"] == 2 and
                all(speed > 0 for speed in item["moving_speed_mps"]) and
                item["ego_moving_nonsaturated_values"] > 0 and item["full_vs_fill_different_values"] > 0)
        else:
            checks[cid + "_zero_first_feasible"] = row["joint_bracket"][0] > 0
    return dict(experiment_id=cfg["experiment_id"], diagnostic_relevance_pass=all(checks.values()),
                checks=checks, cases=cases, scene_validation=validation, benchmark=selected_benchmark,
                grid_cover_allowance=allowance, numeric_allowance=tolerance,
                analytic_conflict_bounds=analytical, observations_and_probes=contents, rows=rows,
                scope="Deterministic two-tick ego recovery with stationary peer backups; no learned advantage, uncertainty certificate, task success or novelty")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    assert cfg["steps"] == 2 and cfg["dtype"] == "float64"
    assert cfg["grid_resolution"] >= 3 and cfg["grid_resolution"] % 2 == 1
    out = args.out.resolve()
    assert out.parent == ROOT / "results" and out.name == cfg["experiment_id"]
    head, sources = registered_sources()
    out.mkdir(exist_ok=False)
    recorder = Recorder(out)
    try:
        hashes = {}
        for name, payload in sources.items():
            path = out / "source_snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
            hashes[name] = digest(payload)
        write_new(out / "manifest.json", dict(experiment_id=cfg["experiment_id"], git_head=head, pid=os.getpid(),
                  source_sha256=hashes, config=cfg, started_at_utc=now(),
                  python=sys.version, torch=torch.__version__, numpy=np.__version__,
                  cuda_available=torch.cuda.is_available(),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  command=sys.argv, no_policy_or_historical_artifact_access=True))
        write_new(out / "status.json", dict(phase="started", complete=False,
                  terminal_authority="completion.json or error.json; status is an immutable start record"))
        recorder.emit("start", experiment_id=cfg["experiment_id"], config=cfg,
                      seed="NOT_APPLICABLE", success="NOT_APPLICABLE", any_ever_ball="NOT_APPLICABLE",
                      all_ever_ball="NOT_APPLICABLE", checkpoint="NOT_APPLICABLE", output=str(out))
        with torch.inference_mode():
            result = execute(cfg, out, recorder)
        for name, expected in hashes.items():
            assert digest((ROOT / name).read_bytes()) == expected, f"Source changed: {name}"
        write_new(out / "analysis.json", result)
        recorder.emit("diagnostic_complete", diagnostic_relevance_pass=result["diagnostic_relevance_pass"],
                      checks=result["checks"], rows=len(result["rows"]), no_automatic_training=True)
        artifacts = {path.relative_to(out).as_posix(): digest(path.read_bytes())
                     for path in out.rglob("*") if path.is_file()}
        write_new(out / "completion.json", dict(complete=True, phase="diagnostic_complete", finished_at_utc=now(),
                  diagnostic_relevance_pass=result["diagnostic_relevance_pass"], source_unchanged=True,
                  artifact_sha256=artifacts, no_automatic_training=True))
    except BaseException:
        failure = dict(complete=False, phase="diagnostic_error", time_utc=now(), traceback=traceback.format_exc())
        print(failure["traceback"], flush=True)
        write_new(out / "error.json", failure)
        raise


if __name__ == "__main__":
    main()
