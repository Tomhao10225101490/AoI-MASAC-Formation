#!/usr/bin/env python3
"""Canonical final-40k analysis of the separately registered near-objective study.

Read-only inputs. No model loading, environment creation, policy evaluation,
training, checkpoint selection, sealed maps, or automatic follow-up launches.
All six frozen histories, actual source/config bytes, final artifacts, finite
sampled learning diagnostics and recorded reward/replay consistency are needed
before the engineering advancement rule can be calculated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1611, 1622, 1633)
STEP = 40000
UPDATES = 39000
MAP_SEEDS = tuple(range(830001, 830013))
MODES = {"control": "original", "treatment": "potential"}
GAMMA, PHI_MAX = .99, .4
TOL = 1e-12
PROTOCOL = "docs/design/MASAC_V4_NEAR_OBJECTIVE_40K_PROTOCOL_20261105.md"
SHARDED_LAYOUT = "near-objective-independent-histories-v1"
COLLISIONS = ("uav_collision", "static_collision", "dynamic_collision", "boundary_collision")
FINITE_FIELDS = (
    "actor_loss", "critic_loss", "alpha", "actor_grad", "critic_grad",
    "actor_grad_post", "critic_grad_post", "q_mean", "target_mean", "td_abs",
    "entropy", "terminal_sample_fraction", "probe_clearance_mean",
    "probe_ttc_mean", "probe_positive_preview_frac",
)
REQUIRED_SOURCES = frozenset((
    "scripts/train_masac_v4_maneuver.py", "uav_planning/v4_maneuver_masac.py",
    "uav_planning/v4_networks.py", "uav_planning/v4_prediction.py",
    "uav_planning/v4_reward_reshape.py", "uav_planning/v4_training.py",
    "uav_planning/v4_environment.py", "uav_planning/v4_maps.py",
    "uav_planning/shared_masac.py", "uav_planning/structured.py", "uav_planning/environment.py",
    "uav_planning/v4_reference.py", "uav_planning/run_lock.py", "uav_planning/v4_sync_capture.py",
    "uav_planning/v4_near_objective.py", "uav_planning/v4_near_study_runtime.py",
    "scripts/train_masac_v4_near_objective_study.py", "scripts/launch_near_objective_study.py", PROTOCOL,
    "scripts/analyze_near_objective_study.py",
))
TRANSITION_NUMBERS = (
    "legacy_base_reward", "replay_reward", "correction", "removed_near",
    "potential_difference", "previous_phi", "following_phi_observed",
    "following_phi_effective_for_potential", "gamma",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b, tolerance=TOL):
    return finite(a) and finite(b) and abs(a - b) <= tolerance


def require(value, message):
    if not value:
        raise ValueError(message)


def inside(base, declared):
    """Reject escapes and symlinks outside the declared source/artifact root."""
    require(isinstance(declared, str) and declared, "declared path is empty or non-string")
    path = (Path(base) / declared).resolve()
    require(path.is_relative_to(Path(base).resolve()), f"declared path escapes allowed root: {declared}")
    return path


def advancement_decision(control, treatment, *, complete, finite_final, integrity_ok, artifact_hashes_verified):
    """Only the preregistered final endpoint can authorize a planning label."""
    if not (complete and finite_final and integrity_ok and artifact_hashes_verified):
        return None
    require(len(control) == len(treatment) == len(SEEDS), "three paired seeds required")
    require(all(type(x) is int and 0 <= x <= len(MAP_SEEDS) for x in [*control, *treatment]),
            "success counts must be integers in [0,12]")
    differences = [b - c for c, b in zip(control, treatment)]
    average = Fraction(sum(differences), len(SEEDS) * len(MAP_SEEDS))
    checks = {
        "mean_paired_fraction_at_least_1_over_12": average >= Fraction(1, 12),
        "at_least_two_strictly_positive_seed_pairs": sum(x > 0 for x in differences) >= 2,
        "every_seed_fraction_difference_at_least_minus_1_over_12": min(differences) >= -1,
        "all_six_complete_finite_frozen_and_actual_artifact_hashes_verified": True,
    }
    advance = all(checks.values())
    return {
        "label": "ADVANCEMENT_RULE_MET" if advance else "NEGATIVE_OR_MIXED_FIXED_BUDGET",
        "advance": advance,
        "rule_checks": checks,
        "integer_differences_by_seed": dict(zip(map(str, SEEDS), differences)),
        "paired_fraction_differences_by_seed": dict(zip(map(str, SEEDS), [x / 12 for x in differences])),
        "mean_paired_fraction_difference": float(average),
        "mean_paired_fraction_difference_exact": f"{average.numerator}/{average.denominator}",
        "negative_seed_pairs": [seed for seed, difference in zip(SEEDS, differences) if difference < 0],
        "control_absolute_gate_passed_seeds": sum(x >= 8 for x in control),
        "treatment_absolute_gate_passed_seeds": sum(x >= 8 for x in treatment),
        "control_absolute_gate_met": all(x >= 8 for x in control),
        "treatment_absolute_gate_met": all(x >= 8 for x in treatment),
        "absolute_gate": "At least 8/12 final joint successes in EACH of the three training seeds.",
        "scope": "Engineering planning confirmation only; no significance, superiority, convergence, safety or novelty claim.",
        "automatic_training_extension_or_sealed_test_authorized": False,
    }


def transition_contract(row, *, mode, seed, step, previous=None):
    """Validate one logged runtime conversion; returns values for aggregation."""
    require(isinstance(row, dict), "transition is not an object")
    require(type(row.get("schema_version")) is int and row["schema_version"] == 1 and
            row.get("kind") == "near_reward_transition" and type(row.get("training_seed")) is int and
            row["training_seed"] == seed and type(row.get("training_step")) is int and
            row["training_step"] == step and row.get("mode") == mode,
            "reward transition kind/seed/exact step/mode mismatch")
    require(all(finite(row.get(key)) for key in TRANSITION_NUMBERS), "missing/nonfinite reward component")
    require(type(row.get("episode")) is int and row["episode"] >= 0 and type(row.get("map_seed")) is int,
            "invalid episode/map identity")
    require(all(type(row.get(key)) is bool for key in
                ("terminated", "truncated", "replay_done", "timeout_without_termination",
                 "absorbing_potential_applied", "bootstrap_allowed_by_flags")),
            "terminal/truncation/replay mask flags must be booleans")
    terminated, truncated = row["terminated"], row["truncated"]
    require(row["replay_done"] is terminated and
            row["timeout_without_termination"] == (truncated and not terminated) and
            row["absorbing_potential_applied"] == (mode == "potential" and terminated) and
            row["bootstrap_allowed_by_flags"] == (not terminated),
            "replay terminal mask or timeout semantics mismatch")
    before, after = row["previous_phi"], row["following_phi_observed"]
    require(-TOL <= before <= PHI_MAX + TOL and -TOL <= after <= PHI_MAX + TOL and row["gamma"] == GAMMA,
            "potential/gamma outside frozen contract")
    effective = 0. if terminated else after
    require(close(row["following_phi_effective_for_potential"], effective), "terminal potential convention mismatch")
    potential = GAMMA * effective - before
    expected = 0. if mode == "original" else potential - after
    require(close(row["correction"], expected) and
            close(row["replay_reward"], row["legacy_base_reward"] + expected), "reward conversion mismatch")
    if mode == "original":
        require(row["removed_near"] == row["potential_difference"] == row["correction"] == 0. and
                row["replay_reward"] == row["legacy_base_reward"], "original numerical reward changed")
    else:
        require(close(row["removed_near"], after) and close(row["potential_difference"], potential),
                "replacement decomposition mismatch")
        lower = -.8 if terminated else -.404
        require(lower - TOL <= row["correction"] <= TOL, "replacement correction violates analytic bound")
    require(row["map_seed"] == 2000000 + seed * 10000 + row["episode"], "training map/reset seed formula mismatch")
    if previous is None:
        require(row["episode"] == 0, "cold history must start at episode zero")
    else:
        ended = previous["terminated"] or previous["truncated"]
        require(row["episode"] == previous["episode"] + int(ended), "episode/reset counter mismatch")
        if not ended:
            require(before == previous["following_phi_observed"] and row["map_seed"] == previous["map_seed"],
                    "potential state discontinuity within an episode")
    return {
        "terminal": int(terminated), "timeout": int(truncated and not terminated),
        "nonzero_phi": int(before != 0. or after != 0.),
        "nonzero_correction": int(row["correction"] != 0.),
    }


class Audit:
    def __init__(self, out=None, *, heavy=True):
        self.out = Path(out) if out is not None else None
        self.heavy = heavy
        self.errors, self.missing, self.warnings = [], [], []
        self.inputs = {}
        self.input_mutations = []
        self.started = now()

    def say(self, message):
        print(message, flush=True)
        if self.out is not None:
            with (self.out / "analysis.log").open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")

    def issue(self, context, message):
        self.errors.append({"context": str(context), "message": str(message)})

    def pending(self, path, reason="required input not yet present"):
        self.missing.append({"path": str(path), "reason": reason})

    def remember(self, path, record):
        """Keep the first byte snapshot; repeat reads cannot erase a change."""
        name = str(Path(path).resolve())
        first = self.inputs.get(name)
        if first is None:
            self.inputs[name] = record
        elif (first["sha256"], first["bytes"]) != (record["sha256"], record["bytes"]):
            self.input_mutations.append({"path": name, "first": dict(first), "observed_again": dict(record)})
            self.issue(path, "input changed between reads; first SHA256/byte count retained")
            self.pending(path, "input changed between reads; fresh stable snapshot required")
        return self.inputs[name]

    def track(self, path, *, heavy=False):
        path = Path(path).resolve()
        if not path.is_file():
            self.pending(path)
            return False
        if not heavy or self.heavy:
            digest, count = hashlib.sha256(), 0
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
                    count += len(block)
            self.remember(path, {"sha256": digest.hexdigest(), "bytes": count, "heavy": heavy})
        return True

    def read(self, path, *, optional=False):
        path = Path(path)
        if not path.is_file():
            if not optional:
                self.pending(path)
            return None
        try:
            payload = path.read_bytes()
            self.remember(path, {
                "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "heavy": False,
            })
            return json.loads(payload.decode("utf-8"))
        except (OSError, ValueError, UnicodeError) as error:
            self.issue(path, f"invalid JSON: {error}")
            return None

    def jsonl(self, path, *, completed):
        path = Path(path)
        if not path.is_file():
            self.pending(path)
            return [], False
        payload = path.read_bytes()
        self.remember(path, {
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "heavy": False,
        })
        records, tail_ignored = [], False
        lines = payload.splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                self.issue(path, f"blank JSONL line {index + 1}")
                continue
            try:
                record = json.loads(line)
                require(isinstance(record, dict), "line is not an object")
                records.append(record)
            except (ValueError, UnicodeError) as error:
                if not completed and index == len(lines) - 1 and not payload.endswith(b"\n"):
                    tail_ignored = True
                    self.pending(path, "live incomplete last JSONL line ignored")
                else:
                    self.issue(path, f"line {index + 1}: {error}")
        return records, tail_ignored

    def check(self, value, context, message):
        if not value:
            self.issue(context, message)
        return bool(value)

    def absent_error(self, directory):
        for name in ("error.json", "error.txt"):
            path = Path(directory) / name
            if path.exists():
                self.track(path)
                self.issue(path, "training error marker exists; completion cannot override it")

    def verify_unchanged_inputs(self):
        changed = []
        for name, record in self.inputs.items():
            path = Path(name)
            if not path.is_file() or sha(path) != record["sha256"]:
                changed.append(name)
        if changed:
            self.pending("INPUT_SNAPSHOT", "inputs changed during analysis; take a fresh snapshot after writes stop")
        return changed


def verify_sources(audit, manifest, context, *, snapshots_required=True):
    sources = manifest.get("sources")
    if not audit.check(isinstance(sources, dict) and bool(sources), context, "nonempty frozen source dictionary required"):
        return False
    audit.check(REQUIRED_SOURCES.issubset(sources), context,
                "required training/physics/model/adapter/runtime/entry/protocol/analyzer sources not all frozen")
    valid = True
    for name, expected in sources.items():
        try:
            path = inside(ROOT, name)
            require(isinstance(expected, str) and len(expected) == 64 and
                    all(c in "0123456789abcdef" for c in expected), "invalid declared SHA256")
            if not audit.track(path):
                valid = False
                continue
            observed = audit.inputs[str(path)]["sha256"]
            audit.check(observed == expected, path, "current source bytes differ from preregistered run manifest")
            if snapshots_required:
                snapshot = inside(Path(context) / "source_snapshot", name)
                if audit.track(snapshot):
                    audit.check(audit.inputs[str(snapshot)]["sha256"] == expected, snapshot,
                                "saved source snapshot differs from frozen manifest")
                else:
                    valid = False
            valid = valid and observed == expected
        except (OSError, ValueError, TypeError) as error:
            valid = False
            audit.issue(context, f"source {name!r}: {error}")
    audit.say(f"SOURCE context={context} declared={len(sources)} actual_bytes_match={valid}")
    return valid


def verify_config(audit, arm, manifest):
    mode = MODES[arm]
    path = inside(ROOT, f"configs/masac_v4/near_objective_40k_{mode}_cuda_20261105.json")
    cfg = audit.read(path)
    if not isinstance(cfg, dict):
        return None
    expected_hash = manifest.get("config_hashes", {}).get("protocol")
    audit.check(set(manifest.get("config_hashes", {})) == {"protocol", "authorization"}, path,
                "unrecognized or missing declared config hash fields")
    audit.check(audit.inputs[str(path)]["sha256"] == expected_hash, path,
                "current registered config SHA256 differs from run manifest")
    snapshot = inside(Path(manifest["_analysis_root"]) / "source_snapshot", f"configs/masac_v4/{path.name}")
    if audit.track(snapshot):
        audit.check(audit.inputs[str(snapshot)]["sha256"] == expected_hash, snapshot,
                    "saved config snapshot differs from registered config hash")
    audit.check(manifest.get("config_hashes", {}).get("authorization") is None, path,
                "unexpected additional authorization protocol/config")
    paired_hashes = manifest.get("paired_config_hashes")
    expected_paths = {f"configs/masac_v4/near_objective_40k_{mode}_cuda_20261105.json" for mode in MODES.values()}
    if audit.check(isinstance(paired_hashes, dict) and set(paired_hashes) == expected_paths, path,
                   "paired config hash manifest must declare exactly both registered config paths"):
        for name, expected in paired_hashes.items():
            paired = inside(ROOT, name)
            if audit.track(paired):
                audit.check(audit.inputs[str(paired)]["sha256"] == expected, paired,
                            "actual paired config bytes differ from manifest")
    expected_near = dict(mode=mode, coefficient=.4, radius_m=15., gamma=.99)
    audit.check(cfg.get("near_objective") == manifest.get("near_objective") == expected_near,
                path, "near-objective mode/coefficient/radius/gamma not frozen")
    audit.check(cfg.get("steps") == STEP and cfg.get("seeds") == list(SEEDS), path,
                "config must contain only registered 40k and three cold seeds")
    for key, expected in dict(warmup=1000, batch_size=128, replay_capacity=100000,
                              eval_interval=2000, checkpoint_interval=10000, log_interval=200,
                              device="cuda", set_mode="pool", n_agents=3).items():
        audit.check(cfg.get(key) == expected, path, f"frozen config {key} differs from {expected!r}")
    audit.check(cfg.get("sync_capture", {}).get("enabled") is False, path, "B must remain off in BOTH arms")
    audit.check(cfg.get("development_maps") == manifest.get("development_maps_used"), path,
                "actual development population differs from the frozen config")
    audit.check(cfg.get("training_map_seed_offset") == 2000000 and cfg.get("training_seed_multiplier") == 10000,
                path, "training episode map seed/reset formula differs from frozen contract")
    audit.check(cfg.get("decision_rule") == dict(minimum_sum_success_gain=3, minimum_positive_seed_pairs=2,
                                                 minimum_pair_success_gain=-1, absolute_success_count_per_seed=8),
                path, "registered decision rule differs from analyzer")
    audit.check(cfg.get("execution") == {"layout": SHARDED_LAYOUT, "max_workers": 4}, path,
                "registered execution layout/max-workers differs from the measured four-worker protocol")
    audit.check(cfg.get("reshape") == manifest.get("reshape"), path, "reshape config differs from manifest")
    reshape = cfg.get("reshape", {})
    audit.check(reshape.get("train_ball_m") == 4. and reshape.get("train_speed_mps") == 2. and
                reshape.get("capture_latch") is False and reshape.get("near_goal_bonus") == .4 and
                reshape.get("near_goal_radius_m") == 15., path, "original 4m/2mps Path-C near/capture contract changed")
    return cfg


def root_contract(audit, arm, root, *, expected_seeds=SEEDS):
    audit.absent_error(root)
    manifest = audit.read(root / "run_manifest.json")
    status = audit.read(root / "status.json")
    completion = audit.read(root / "completion.json")
    record = {"root": str(root), "manifest": manifest, "status": status, "completion": completion,
              "complete": False, "source_bytes_verified": False, "config": None}
    if not isinstance(manifest, dict):
        return record
    expected = dict(steps_per_seed=STEP, warmup=1000, replay_capacity=100000,
                    probe_mode="static", set_mode="pool", device="cuda", smoke=False)
    for key, value in expected.items():
        audit.check(manifest.get(key) == value, root, f"manifest {key} must equal {value!r}")
    audit.check(manifest.get("seed_list") == list(expected_seeds), root, "actual worker seed list/order mismatch")
    audit.check(manifest.get("capture_train_and_gate") == {"ball_m": 4., "speed_mps": 2., "latch": False},
                root, "capture train/gate differs from 4m/2mps unlatched contract")
    audit.check(manifest.get("sync_capture", {}).get("enabled") is False,
                root, "B shaping must be off in BOTH arms")
    audit.check(manifest.get("evaluation_near_objective") == "original_PathC", root,
                "evaluation must use unchanged original Path-C")
    protocol = manifest.get("preregistered_near_protocol", {})
    audit.check(protocol.get("path") == PROTOCOL and
                protocol.get("sha256") == manifest.get("sources", {}).get(PROTOCOL),
                root, "formal protocol path/hash not frozen in manifest")
    record["source_bytes_verified"] = verify_sources(audit, manifest, root)
    # Pass the root internally without altering the retained original manifest.
    record["config"] = verify_config(audit, arm, dict(manifest, _analysis_root=str(root)))
    audit.check(manifest.get("warm_start_path") is None and manifest.get("warm_start_mode") == "cold",
                root, "study must be cold start; continuation is not registered")
    if isinstance(status, dict) and str(status.get("phase", "")).casefold() in ("failed", "error"):
        audit.issue(root, "training status is terminal failed/error")
    if isinstance(completion, dict):
        runs = completion.get("runs")
        valid_runs = (isinstance(runs, list) and len(runs) == len(expected_seeds) and
                      all(isinstance(item, dict) and item.get("step") == STEP for item in runs) and
                      [item.get("seed") for item in runs] == list(expected_seeds))
        record["complete"] = completion.get("complete") is True and completion.get("smoke") is False and valid_runs
        audit.check(record["complete"], root, "worker completion must contain exactly its declared selected final histories")
        if record["complete"]:
            audit.check(isinstance(status, dict) and status.get("complete") is True, root,
                        "root completion/status disagree")
            for key in ("reshape", "probe_mode"):
                audit.check(completion.get(key) == manifest.get(key), root, f"completion {key} differs from manifest")
    return record


def metrics_summary(audit, directory, *, seed, completed):
    path = directory / "metrics.jsonl"
    rows, tail_ignored = audit.jsonl(path, completed=completed)
    progress = [row for row in rows if row.get("kind") == "progress"]
    audit.check(all(row.get("seed") == seed and row.get("probe_mode") == "static" for row in rows), path,
                "metrics history seed/probe identity mismatch")
    trained = [row for row in progress if finite(row.get("updates")) and row["updates"] > 0]
    valid_progress = bool(progress) and all(type(r.get("step")) is int and 0 < r["step"] <= STEP and
                                         type(r.get("updates")) is int and r["updates"] == max(0, r["step"] - 1000)
                                         for r in progress)
    audit.check(valid_progress or not completed, path, "progress steps/real update counts are invalid")
    steps = [r.get("step") for r in progress]
    audit.check(all(type(step) is int for step in steps) and
                all(a < b for a, b in zip(steps, steps[1:])), path, "progress steps not strictly ordered")
    finite_rows = bool(trained) and all(all(finite(row.get(key)) for key in FINITE_FIELDS) for row in trained)
    for row in trained:
        for key, value in row.items():
            if type(value) in (int, float) and not math.isfinite(value):
                audit.issue(path, f"nonfinite progress value {key} at step {row.get('step')}")
    final = trained[-1] if trained else {}
    final_ok = finite_rows and final.get("step") == STEP and final.get("updates") == UPDATES
    if completed:
        audit.check(final_ok, path, "finite final40k/39000update learning diagnostics not established")
        audit.check([row.get("step") for row in trained] == list(range(1200, STEP + 1, 200)), path,
                    "sampled postwarmup diagnostics must cover the frozen 195-row logging schedule")
    elif trained and not finite_rows:
        audit.issue(path, "observed sampled learning diagnostics are missing/nonfinite")
    episodes = [row for row in rows if row.get("kind") == "episode"]
    episode_fields = ("return_", "base_return", "shaping_return", "near_removed_return", "near_potential_difference_return")
    for row in episodes:
        audit.check(type(row.get("step")) is int and 0 < row["step"] <= STEP and
                    all(finite(row.get(key)) for key in episode_fields), path,
                    f"episode at {row.get('step')} has invalid/missing numerical components")
    return {
        "sampled_learning_rows": len(trained), "finite_sampled_learning_rows": finite_rows if trained else None,
        "finite_final_diagnostics_established": final_ok,
        "latest_progress_step": progress[-1].get("step") if progress else None,
        "latest_update_count": final.get("updates"), "jsonl_live_tail_ignored": tail_ignored,
        "checked_diagnostic_fields": list(FINITE_FIELDS), "completed_episode_count": len(episodes),
        "completed_episode_components": {key: sum(row[key] for row in episodes if finite(row.get(key))) for key in episode_fields},
        "diagnostic_scope": "Only sampled top-level progress diagnostics, not all 39000 individual update values.",
        "reward_scope": "Completed episodes only; raw/shaped training returns are not common cross-arm performance scores.",
    }, episodes


def reward_summary(audit, directory, *, mode, seed, completed, episodes):
    path = directory / "near_reward_transitions.jsonl"
    rows, tail_ignored = audit.jsonl(path, completed=completed)
    totals = Counter()
    invalid = 0
    previous = None
    for index, row in enumerate(rows, start=1):
        try:
            require(index <= STEP, "more transitions than the registered fixed budget")
            totals.update(transition_contract(row, mode=mode, seed=seed, step=index, previous=previous))
        except (ValueError, TypeError, KeyError) as error:
            invalid += 1
            if invalid <= 12:
                audit.issue(path, f"transition row {index}: {error}")
        previous = row
    if invalid > 12:
        audit.issue(path, f"{invalid} invalid transition rows total (first 12 listed)")
    count_ok = len(rows) == STEP
    if completed:
        audit.check(count_ok, path, f"expected exact steps 1..40000, observed {len(rows)} rows")
    elif not count_ok:
        audit.pending(path, f"{len(rows)}/{STEP} reward transitions observed")
    start = 0
    component_checks = 0
    for episode in episodes:
        end = episode.get("step")
        if type(end) is not int or not 0 < end <= len(rows) or end <= start:
            audit.issue(path, "completed episode boundary does not align to transition prefix")
            continue
        segment = rows[start:end]
        if not all(all(finite(row.get(key)) for key in TRANSITION_NUMBERS) for row in segment):
            continue
        audit.check(all(row.get("episode") == episode.get("episode") and
                        row.get("map_seed") == episode.get("map_seed") for row in segment) and
                    all(not (row.get("terminated") or row.get("truncated")) for row in segment[:-1]),
                    path, f"metrics/transition episode identity or reset boundary mismatch at step {end}")
        audit.check(episode.get("near_reward_steps_in_episode") == len(segment), path,
                    f"episode reward-row count mismatch at step {end}")
        audit.check(segment[-1].get("terminated") is True or segment[-1].get("truncated") is True,
                    path, f"episode ending step {end} has no terminal/truncation transition")
        for key, source in (("near_removed_return", "removed_near"),
                            ("near_potential_difference_return", "potential_difference"),
                            ("shaping_return", "correction"), ("base_return", "legacy_base_reward"),
                            ("return_", "replay_reward")):
            expected = sum(row[source] for row in segment)
            audit.check(close(episode.get(key), expected, 1e-8), path,
                        f"episode component mismatch at step {end}: {key}")
            component_checks += 1
        start = end
    corrections = [row["correction"] for row in rows if finite(row.get("correction"))]
    afters = [row["following_phi_observed"] for row in rows if finite(row.get("following_phi_observed"))]
    return {
        "rows": len(rows), "exact_final_40000_rows": count_ok, "invalid_rows": invalid,
        "formula_and_continuity_verified": count_ok and invalid == 0,
        "jsonl_live_tail_ignored": tail_ignored, "terminal_transitions": totals["terminal"],
        "timeout_transitions": totals["timeout"], "nonzero_phi_transitions": totals["nonzero_phi"],
        "nonzero_correction_transitions": totals["nonzero_correction"],
        "nonzero_correction_fraction": totals["nonzero_correction"] / len(rows) if rows else None,
        "min_correction": min(corrections) if corrections else None,
        "max_correction": max(corrections) if corrections else None,
        "max_observed_following_phi": max(afters) if afters else None,
        "episode_component_checks": component_checks,
        "reset_scope": "Within-episode phi and map continuity plus exact episode increment after termination/truncation; reset phi is observed, never forced to zero. This is not an independent geometric recomputation.",
        "coverage_scope": "Intervention/timeout coverage is descriptive and does not change the preregistered efficacy rule.",
    }, rows


def evaluation_state_checks(audit, directory, *, mode, seed, completed):
    path = directory / "evaluation_state_checks.jsonl"
    rows, ignored = audit.jsonl(path, completed=completed)
    valid = all(row.get("kind") == "near_eval_state_check" and
                type(row.get("schema_version")) is int and row["schema_version"] == 1 and
                row.get("training_seed") == seed and row.get("mode") == mode and row.get("passed") is True
                for row in rows)
    steps = [row.get("training_step") for row in rows]
    exact = steps == list(range(2000, STEP + 1, 2000))
    audit.check(valid, path, "invalid or failed evaluation training-state guard record")
    if completed:
        audit.check(exact, path, "all twenty scheduled evaluation state guards must be recorded exactly")
    elif not exact:
        audit.pending(path, f"{len(rows)}/20 scheduled evaluation state guards observed")
    return {"records": len(rows), "valid_records": valid, "all_twenty_guard_calls_verified": valid and exact,
            "live_tail_ignored": ignored,
            "scope": "Recorded runtime equality guard around actual development evaluation; not exact CUDA resume validation."}


def final_evaluation(audit, directory, *, completed, expected_maps):
    path = directory / "evaluation_40000.json"
    rows = audit.read(path)
    if not isinstance(rows, list):
        if rows is not None:
            audit.issue(path, "final evaluation must be a list")
        return None, None
    signature, valid = {}, True
    for row in rows:
        if not isinstance(row, dict):
            audit.issue(path, "final map row is not an object")
            valid = False
            continue
        seed = row.get("map_seed")
        ball = row.get("ever_in_ball_by_uav")
        row_valid = (
            type(seed) is int and seed in MAP_SEEDS and seed not in signature and
            row.get("step") == STEP and isinstance(row.get("family"), str) and bool(row["family"]) and
            isinstance(row.get("scene_sha256"), str) and len(row["scene_sha256"]) == 64 and
            all(character in "0123456789abcdef" for character in row["scene_sha256"]) and
            all(type(row.get(key)) is bool for key in
                ("success", "ever_ball_any", "ever_ball_all", "terminated", "truncated", "timeout", *COLLISIONS)) and
            isinstance(ball, list) and len(ball) == 3 and all(type(x) is bool for x in ball) and
            row.get("evaluation_shaping_enabled") is False
        )
        if not audit.check(row_valid, path, f"invalid frozen final map metadata/flags for {seed}"):
            valid = False
            continue
        signature[seed] = (row["family"], row["scene_sha256"])
        audit.check(row["ever_ball_any"] == any(ball) and row["ever_ball_all"] == all(ball), path,
                    f"any/all ever_ball mismatch for map {seed}")
        audit.check(row["timeout"] == (row["truncated"] and not row["terminated"]), path,
                    f"timeout flags mismatch for map {seed}")
        if row["success"]:
            audit.check(row["terminated"] and row["ever_ball_all"] and not any(row[key] for key in COLLISIONS),
                        path, f"success/collision/terminal flags inconsistent for map {seed}")
    valid = audit.check(len(rows) == 12 and set(signature) == set(MAP_SEEDS), path,
                        "must contain exactly the registered twelve development maps") and valid
    expected_pairs = {(item.get("seed"), item.get("family")) for item in expected_maps if isinstance(item, dict)}
    audit.check(len(expected_pairs) == 12 and {(seed, value[0]) for seed, value in signature.items()} == expected_pairs,
                path, "final map seed/family population differs from frozen manifest")
    if not valid:
        return None, signature
    failures = Counter()
    collision_counts = {key: sum(row[key] for row in rows) for key in COLLISIONS}
    maps = []
    for row in rows:
        if row["success"]:
            reason = "success"
        elif any(row[key] for key in COLLISIONS):
            reason = "collision_any"  # individual collision flags are additionally retained nonexclusively.
        elif row["timeout"]:
            reason = ("timeout_after_ball_all" if row["ever_ball_all"] else
                      "timeout_after_ball_any_only" if row["ever_ball_any"] else "timeout_before_ball")
        else:
            reason = "non_success_other"
        failures[reason] += 1
        maps.append({key: row[key] for key in
                     ("map_seed", "family", "scene_sha256", "success", "ever_ball_any", "ever_ball_all", *COLLISIONS)})
        maps[-1]["exclusive_outcome"] = reason
    return {
        "step": STEP, "maps": len(rows), "successes": sum(row["success"] for row in rows),
        "ever_ball_any": sum(row["ever_ball_any"] for row in rows),
        "ever_ball_all": sum(row["ever_ball_all"] for row in rows),
        "exclusive_outcomes": dict(failures), "collision_flags_nonexclusive": collision_counts,
        "per_map": maps, "population": "Repeated frozen development maps; not sealed test data.",
    }, signature


def final_artifacts(audit, directory, *, seed, manifest, reward_rows, completed):
    checkpoint = directory / "checkpoint_040000"
    saved = audit.read(checkpoint / "manifest.json")
    result = {"checkpoint_directory": str(checkpoint), "snapshot_manifest_valid": False,
              "checkpoint_sha256_verified": False, "replay_sha256_verified": False,
              "replay_reward_and_done_verified": False, "binary_hashes_skipped": not audit.heavy}
    if not isinstance(saved, dict):
        return result
    result["snapshot_manifest_valid"] = audit.check(
        saved.get("complete") is True and saved.get("seed") == seed and saved.get("step") == STEP and
        saved.get("replay_size") == STEP and saved.get("replay_index") == STEP and
        saved.get("sources") == manifest.get("sources") and saved.get("warm_start") == {"mode": "cold"} and
        saved.get("set_mode") == "pool" and saved.get("probe_mode") == "static",
        checkpoint, "final snapshot seed/step/completion/replay size/frozen sources mismatch")
    for filename, key in (("checkpoint.pt", "checkpoint_sha256"), ("replay.npz", "replay_sha256")):
        path = inside(checkpoint, filename)
        expected = saved.get(key)
        audit.check(isinstance(expected, str) and len(expected) == 64, path, "missing declared artifact SHA256")
        if audit.track(path, heavy=True) and audit.heavy:
            observed = audit.inputs[str(path)]["sha256"]
            result[key + "_actual"] = observed
            result[key + "_verified"] = audit.check(observed == expected, path, "final artifact SHA256 mismatch")
            audit.say(f"HASH seed={seed} file={path} sha256={observed} matches={observed == expected}")
    replay_path = checkpoint / "replay.npz"
    if replay_path.exists() and len(reward_rows) == STEP:
        try:
            # Only these two numeric arrays are accessed; observations and model weights are never loaded.
            with np.load(replay_path, allow_pickle=False) as saved_replay:
                rewards = saved_replay["rewards"].reshape(-1)
                done = saved_replay["done"].reshape(-1)
                require(rewards.shape == done.shape == (STEP,), "final reward/done arrays have wrong size")
                require(rewards.dtype == np.dtype("float32") and done.dtype == np.dtype("float32"),
                        "unexpected replay numeric storage dtype")
                require(np.isfinite(rewards).all() and np.isfinite(done).all(), "nonfinite replay reward/done")
                expected_rewards = np.asarray([row["replay_reward"] for row in reward_rows], dtype=np.float32)
                expected_done = np.asarray([row["replay_done"] for row in reward_rows], dtype=np.float32)
                require(np.array_equal(rewards, expected_rewards), "actual replay rewards differ from recorded converted float32 rewards")
                require(np.array_equal(done, expected_done), "actual replay terminal mask differs from recorded true terminations")
                result["replay_reward_and_done_verified"] = True
                result["replay_reward_comparison"] = "Exact float32 storage equality for all 40000 rows in insertion order."
        except (OSError, ValueError, TypeError, KeyError) as error:
            audit.issue(replay_path, f"replay reward/done validation: {error}")
    elif completed:
        audit.issue(replay_path, "cannot validate final replay without exact registered transition population")
    result["all_actual_artifact_hashes_verified"] = (
        result["checkpoint_sha256_verified"] and result["replay_sha256_verified"])
    return result


def read_history(audit, arm, root_record, seed):
    root = Path(root_record["root"])
    directory = root / f"maneuver_static_{seed}"
    audit.absent_error(directory)
    completion = audit.read(directory / "completion.json")
    complete = (isinstance(completion, dict) and completion.get("seed") == seed and completion.get("step") == STEP)
    record = {"seed": seed, "mode": MODES[arm], "directory": str(directory), "completed_40000": complete,
              "completion": completion, "final_evaluation": None, "metrics": None,
              "reward_conversion": None, "artifacts": None}
    metrics, episodes = metrics_summary(audit, directory, seed=seed, completed=complete)
    record["metrics"] = metrics
    rewards, reward_rows = reward_summary(audit, directory, mode=MODES[arm], seed=seed,
                                         completed=complete, episodes=episodes)
    record["reward_conversion"] = rewards
    record["evaluation_state_guard"] = evaluation_state_checks(
        audit, directory, mode=MODES[arm], seed=seed, completed=complete)
    manifest = root_record["manifest"] or {}
    evaluation, signature = final_evaluation(audit, directory, completed=complete,
                                              expected_maps=manifest.get("development_maps_used", []))
    record["final_evaluation"] = evaluation
    record["artifacts"] = final_artifacts(audit, directory, seed=seed, manifest=manifest,
                                          reward_rows=reward_rows, completed=complete)
    if complete:
        root_runs = (root_record["completion"] or {}).get("runs", [])
        audit.check(completion in root_runs, directory, "seed completion differs from root completion history")
        audit.check(evaluation is not None and close(completion.get("final_development_success"),
                                                     evaluation["successes"] / 12 if evaluation else None),
                    directory, "seed completion success differs from final40k evaluation")
        for key in ("reshape", "probe_mode"):
            audit.check(completion.get(key) == manifest.get(key), directory, f"seed completion {key} differs from run manifest")
        audit.check(completion.get("warm_start") == {"mode": "cold"}, directory,
                    "seed completion does not establish registered cold initialization")
    audit.say(f"HISTORY arm={arm} mode={MODES[arm]} seed={seed} complete={complete} "
              f"final_success={evaluation['successes'] if evaluation else None}/12 "
              f"any_ball={evaluation['ever_ball_any'] if evaluation else None}/12 "
              f"all_ball={evaluation['ever_ball_all'] if evaluation else None}/12 "
              f"finite_final={metrics['finite_final_diagnostics_established']} "
              f"reward_rows={rewards['rows']} nonzero_correction={rewards['nonzero_correction_transitions']} "
              f"checkpoint={record['artifacts']['checkpoint_directory']}")
    return record, signature


def paired_contract(audit, roots):
    control, treatment = roots["control"], roots["treatment"]
    c, t = control["manifest"], treatment["manifest"]
    if not isinstance(c, dict) or not isinstance(t, dict):
        return False
    sources_match = bool(c.get("sources")) and c.get("sources") == t.get("sources")
    audit.check(sources_match, "PAIRED_MANIFEST", "frozen source dictionaries differ between arms")
    for key in ("development_maps_used", "reshape", "sync_capture", "probe_config", "parameter_counts"):
        audit.check(c.get(key) is not None and c.get(key) == t.get(key), "PAIRED_MANIFEST",
                    f"shared field {key} missing or differs")
    audit.check(c.get("paired_config_hashes") == t.get("paired_config_hashes"), "PAIRED_MANIFEST",
                "registered paired-config hash dictionaries differ")
    configs = [roots[arm]["config"] for arm in MODES]
    if all(isinstance(cfg, dict) for cfg in configs):
        normalized = []
        for cfg in configs:
            copy = json.loads(json.dumps(cfg))
            for name in ("experiment_id", "suite_experiment_id", "change_set"):
                copy.pop(name, None)
            if isinstance(copy.get("near_objective"), dict):
                copy["near_objective"].pop("mode", None)
            normalized.append(copy)
        audit.check(normalized[0] == normalized[1], "PAIRED_CONFIG",
                    "configs differ outside registered mode and nominal experiment/change labels")
    return sources_match


def analyze_study(control, treatment, *, audit=None):
    audit = audit or Audit()
    paths = {"control": Path(control).resolve(), "treatment": Path(treatment).resolve()}
    require(paths["control"] != paths["treatment"], "control and treatment roots must differ")
    for path in paths.values():
        require(path.is_relative_to(ROOT), "study root must stay inside this repository")
    roots = {arm: root_contract(audit, arm, path) for arm, path in paths.items()}
    source_pair = paired_contract(audit, roots)
    arms, signatures = {}, []
    for arm in MODES:
        arms[arm] = []
        for seed in SEEDS:
            try:
                history, signature = read_history(audit, arm, roots[arm], seed)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                audit.issue(f"{arm}/{seed}", f"history validation raised {type(error).__name__}: {error}")
                history = {"seed": seed, "mode": MODES[arm], "completed_40000": False,
                           "validation_error": str(error), "final_evaluation": None,
                           "metrics": None, "reward_conversion": None, "artifacts": None}
                signature = None
            arms[arm].append(history)
            signatures.append(signature)
    return finish_analysis(audit, roots, arms, signatures, source_pair,
                           layout="legacy_two_arm_three_seed_roots")


def finish_analysis(audit, roots, arms, signatures, source_pair, *, layout, coordinator=None):
    final_maps_paired = bool(signatures and all(signature and signature == signatures[0] for signature in signatures))
    complete = all(item["complete"] for item in roots.values()) and all(
        row["completed_40000"] for histories in arms.values() for row in histories)
    if coordinator is not None:
        complete = complete and coordinator["complete"]
    if complete:
        audit.check(final_maps_paired, "FINAL_MAP_PAIRING", "all six final map seed/family/scene signatures must match")
    all_rows = [row for histories in arms.values() for row in histories]
    finite_final = all((row["metrics"] or {}).get("finite_final_diagnostics_established") is True for row in all_rows)
    reward_verified = all((row["reward_conversion"] or {}).get("formula_and_continuity_verified") is True and
                          (row["artifacts"] or {}).get("replay_reward_and_done_verified") is True for row in all_rows)
    artifacts_verified = all((row["artifacts"] or {}).get("all_actual_artifact_hashes_verified") is True for row in all_rows)
    changed = audit.verify_unchanged_inputs()
    integrity_ok = not audit.errors and not audit.missing and not changed and source_pair and final_maps_paired and reward_verified
    ready = complete and finite_final and integrity_ok and artifacts_verified
    decision = advancement_decision(
        [row["final_evaluation"]["successes"] for row in arms["control"]] if ready else [],
        [row["final_evaluation"]["successes"] for row in arms["treatment"]] if ready else [],
        complete=complete, finite_final=finite_final, integrity_ok=integrity_ok,
        artifact_hashes_verified=artifacts_verified,
    )
    status = "COMPLETE" if ready else "INVALID_OR_INCOMPLETE" if audit.errors else "PENDING_FINAL_EVIDENCE"
    paired = []
    for c, t in zip(arms["control"], arms["treatment"]):
        ce, te = c["final_evaluation"], t["final_evaluation"]
        paired.append({"seed": c["seed"], "control_successes": ce["successes"] if ce else None,
                       "treatment_successes": te["successes"] if te else None,
                       "difference_maps": te["successes"] - ce["successes"] if ce and te else None,
                       "scope": "Final40000 descriptive pair; not an advancement decision until all evidence is verified."})
    return {
        "analysis_version": 1, "generated_at_utc": now(), "study_status": status,
        "layout": layout, "coordinator": coordinator,
        "decision_ready": ready, "decision": decision, "fixed_final_step": STEP,
        "all_six_training_completions": complete, "all_six_finite_final_diagnostics": finite_final,
        "all_six_actual_final_artifact_hashes_verified": artifacts_verified,
        "all_six_reward_conversion_and_replay_verified": reward_verified,
        "source_dictionaries_match_between_arms": source_pair, "final_map_pairing_verified": final_maps_paired,
        "heavy_artifact_hashes_requested": audit.heavy,
        "integrity_issues": audit.errors, "missing_evidence": audit.missing, "warnings": audit.warnings,
        "inputs_changed_during_analysis": changed, "roots": roots, "arms": arms, "all_paired_seeds": paired,
        "input_mutations_between_reads": audit.input_mutations,
        "scope": "Three independent cold training seeds and 12 repeated development maps. Correlated observations are not independent binomial trials; no pooled confidence interval, significance, superiority, convergence, safety, dynamic-prediction or novelty claim. No sealed maps, model loading, training or new policy evaluation.",
        "endpoint_policy": "Only evaluation_40000.json is read; no intermediate/best checkpoint selection and no automatic extension.",
        "integrity_scope": "Actual frozen source/config bytes, final artifact SHA256, sampled top-level learning diagnostics, all recorded reward conversions, final replay reward/done arrays and final development map signatures. SHA equality does not establish exact CUDA resume equivalence.",
    }


def job_identities(audit, context, rows, *, require_returncode):
    identities = []
    if not isinstance(rows, list):
        audit.issue(context, "coordinator jobs must be a list")
        return None
    for row in rows:
        if not isinstance(row, dict):
            audit.issue(context, "coordinator job is not an object")
            continue
        identity = row.get("mode"), row.get("training_seed"), row.get("relative_root")
        if not (isinstance(identity[0], str) and type(identity[1]) is int and isinstance(identity[2], str)):
            audit.issue(context, "coordinator job identity must contain string mode/root and integer seed")
            continue
        identities.append(identity)
        if require_returncode or row.get("returncode") is not None:
            audit.check(type(row.get("returncode")) is int and row["returncode"] == 0, context,
                        f"coordinator job returncode is not zero: {identity}")
    expected = {(mode, seed, f"runs/{mode}_{seed}") for mode in MODES.values() for seed in SEEDS}
    audit.check(len(identities) == 6 and all(type(identity[1]) is int for identity in identities) and
                len(set(identities)) == 6 and set(identities) == expected, context,
                "coordinator must declare exactly the six unique registered mode/seed/relative-root jobs")
    return identities


def analyze_sharded_study(study, *, audit=None):
    """Inspect independent single-seed worker roots under one frozen coordinator."""
    audit = audit or Audit()
    study = Path(study).resolve()
    require(study.is_relative_to(ROOT) and study != ROOT, "study root must stay inside this repository")
    audit.absent_error(study)
    manifest = audit.read(study / "study_manifest.json")
    status = audit.read(study / "status.json")
    completion = audit.read(study / "completion.json")
    coordinator = {"root": str(study), "manifest": manifest, "status": status,
                   "completion": completion, "complete": False}
    manifest = manifest if isinstance(manifest, dict) else {}
    audit.check(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1 and
                manifest.get("layout") == SHARDED_LAYOUT and manifest.get("seeds") == list(SEEDS) and
                manifest.get("step") == STEP and manifest.get("max_workers") == 4,
                study, "coordinator schema/layout/seeds/step/max-workers differ from frozen registration")
    identities = job_identities(audit, study / "study_manifest.json", manifest.get("jobs"), require_returncode=False)
    if isinstance(manifest.get("jobs"), list):
        for job in manifest["jobs"]:
            if not isinstance(job, dict):
                continue
            mode, seed = job.get("mode"), job.get("training_seed")
            expected_config = f"configs/masac_v4/near_objective_40k_{mode}_cuda_20261105.json"
            expected_relative = f"runs/{mode}_{seed}"
            audit.check(mode in MODES.values() and type(seed) is int and seed in SEEDS and
                        job.get("relative_root") == expected_relative and job.get("config") == expected_config,
                        study, "job config/path/mode/seed must equal the registered canonical tuple")
            try:
                declared = job.get("relative_root")
                require(isinstance(declared, str) and not Path(declared).is_absolute(), "worker root must be relative")
                inside(study, declared)
                inside(ROOT, job.get("config"))
            except (ValueError, TypeError) as error:
                audit.issue(study, f"invalid declared worker/config path: {error}")
    source_match = verify_sources(audit, manifest, study, snapshots_required=False)
    expected_configs = {f"configs/masac_v4/near_objective_40k_{mode}_cuda_20261105.json" for mode in MODES.values()}
    paired = manifest.get("paired_config_hashes")
    if audit.check(isinstance(paired, dict) and set(paired) == expected_configs, study,
                   "study must freeze both registered config paths/hashes"):
        for relative, expected_hash in paired.items():
            path = inside(ROOT, relative)
            if audit.track(path):
                audit.check(audit.inputs[str(path)]["sha256"] == expected_hash, path,
                            "current config bytes differ from coordinator manifest")
    if isinstance(completion, dict):
        completed_ids = job_identities(audit, study / "completion.json", completion.get("jobs"), require_returncode=True)
        coordinator["complete"] = (completion.get("complete") is True and
                                   completion.get("all_jobs_complete") is True and
                                   completed_ids is not None and len(completed_ids) == 6)
        audit.check(coordinator["complete"], study, "coordinator completion must establish all six jobs completed")
        if coordinator["complete"]:
            audit.check(isinstance(status, dict) and status.get("complete") is True and
                        str(status.get("phase", "")).casefold() == "complete", study,
                        "coordinator status/completion disagree")
    if isinstance(status, dict):
        job_identities(audit, study / "status.json", status.get("jobs"),
                       require_returncode=status.get("complete") is True)
        if str(status.get("phase", "")).casefold() in ("failed", "error"):
            audit.issue(study, "coordinator status is failed/error")
    roots, arms, signatures = {}, {arm: [] for arm in MODES}, []
    for arm, mode in MODES.items():
        for seed in SEEDS:
            key, relative = f"{mode}_{seed}", f"runs/{mode}_{seed}"
            try:
                root = inside(study, relative)
                record = root_contract(audit, arm, root, expected_seeds=(seed,))
                roots[key] = record
                worker_manifest = record["manifest"] or {}
                equal_sources = bool(manifest.get("sources")) and worker_manifest.get("sources") == manifest.get("sources")
                source_match = source_match and equal_sources
                audit.check(equal_sources, root, "worker source dictionary differs from frozen coordinator sources")
                audit.check(worker_manifest.get("paired_config_hashes") == paired, root,
                            "worker paired-config hashes differ from frozen coordinator")
                history, signature = read_history(audit, arm, record, seed)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                audit.issue(f"{mode}/{seed}", f"worker validation raised {type(error).__name__}: {error}")
                roots.setdefault(key, {"root": str(study / relative), "manifest": None, "config": None, "complete": False})
                history = {"seed": seed, "mode": mode, "completed_40000": False,
                           "validation_error": str(error), "final_evaluation": None,
                           "metrics": None, "reward_conversion": None, "artifacts": None}
                signature = None
            arms[arm].append(history)
            signatures.append(signature)
    anchor = roots[f"original_{SEEDS[0]}"]
    for key, record in roots.items():
        source_match = paired_contract(audit, {"control": anchor, "treatment": record}) and source_match
    coordinator["declared_job_identities"] = identities
    return finish_analysis(audit, roots, arms, signatures, source_match,
                           layout=SHARDED_LAYOUT, coordinator=coordinator)


def markdown_report(report):
    lines = ["# Near-objective fixed-budget development study", "", f'**{report["study_status"]}**', "",
             report["scope"], "", report["endpoint_policy"], "",
             "| Training seed | Original /12 | Potential /12 | Potential − original maps |",
             "|---|---:|---:|---:|"]
    for pair in report["all_paired_seeds"]:
        lines.append(f'| {pair["seed"]} | {pair["control_successes"]} | {pair["treatment_successes"]} | {pair["difference_maps"]} |')
    decision = report["decision"]
    if decision is None:
        lines += ["", "No advancement decision is calculated. All six final completions, finite diagnostics, frozen source/config bytes, actual final artifact hashes, reward/replay checks and matched final map signatures are required."]
    else:
        lines += ["", f'**{decision["label"]}**. Mean paired success-fraction difference: '
                  f'{decision["mean_paired_fraction_difference_exact"]} ({decision["mean_paired_fraction_difference"]:.8f}).', "",
                  f'Absolute 8/12 every-seed gate: original **{decision["control_absolute_gate_met"]}** '
                  f'({decision["control_absolute_gate_passed_seeds"]}/3 seeds); potential '
                  f'**{decision["treatment_absolute_gate_met"]}** ({decision["treatment_absolute_gate_passed_seeds"]}/3 seeds).', "",
                  "The advancement label supports planning confirmation only and authorizes no automatic extension or sealed test."]
    lines += ["", "## All histories and intervention coverage", "",
              "| Arm | Seed | Final success | Any ball | All ball | Reward rows | Nonzero corrections | Timeouts | Finite final |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm, histories in report["arms"].items():
        for history in histories:
            evaluation, rewards, metrics = [history.get(key) or {} for key in
                                            ("final_evaluation", "reward_conversion", "metrics")]
            lines.append(f'| {arm} | {history["seed"]} | {evaluation.get("successes")} | '
                         f'{evaluation.get("ever_ball_any")} | {evaluation.get("ever_ball_all")} | '
                         f'{rewards.get("rows")} | {rewards.get("nonzero_correction_transitions")} | '
                         f'{rewards.get("timeout_transitions")} | {metrics.get("finite_final_diagnostics_established")} |')
    lines += ["", "Intervention/timeout coverage is descriptive and does not change the efficacy rule. Finite values are sampled top-level learning logs; they are not a retained value for every update. Completed-episode shaped returns are not common cross-arm performance scores.", "",
              "Per-map outcomes, all negative pairs, any/all ever-ball and nonexclusive collision flags are retained in analysis.json. Exact float32 replay reward/mask checks access only those two arrays, and never deserialize model weights."]
    for key, title in (("integrity_issues", "Integrity issues"), ("missing_evidence", "Missing evidence")):
        if report[key]:
            lines += ["", f"## {title}", ""]
            lines.extend(f'- `{item.get("context", item.get("path"))}`: {item.get("message", item.get("reason"))}' for item in report[key])
    return "\n".join(lines) + "\n"


def json_safe(value):
    """Preserve malformed-input evidence without emitting nonstandard JSON NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_value": repr(value)}
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--study", type=Path, help="registered four-worker coordinator root with six truthful single-seed worker roots")
    selection.add_argument("--control", type=Path, help="legacy two-arm layout only; requires --treatment")
    parser.add_argument("--treatment", type=Path, help="legacy two-arm layout only; requires --control")
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory; existing paths are refused")
    parser.add_argument("--skip-artifact-hashes", action="store_true",
                        help="diagnostic snapshot only; this necessarily disables decision_ready")
    args = parser.parse_args()
    if args.study is not None and args.treatment is not None:
        parser.error("--study cannot be combined with --control or --treatment")
    if args.control is not None and args.treatment is None:
        parser.error("--control requires --treatment")
    out = args.out.resolve()
    input_roots = ([args.study.resolve()] if args.study is not None else
                   [args.control.resolve(), args.treatment.resolve()])
    for root in input_roots:
        require(not out.is_relative_to(root) and not root.is_relative_to(out),
                "analysis output must not overlap either input run")
    require(out.is_relative_to(ROOT) and out != ROOT, "analysis output must stay inside this repository")
    out.mkdir(parents=True, exist_ok=False)
    audit = Audit(out, heavy=not args.skip_artifact_hashes)
    manifest = {
        "analysis_version": 1, "started_at_utc": audit.started, "command": sys.argv,
        "study_root": str(args.study.resolve()) if args.study is not None else None,
        "control_root": str(args.control.resolve()) if args.control is not None else None,
        "treatment_root": str(args.treatment.resolve()) if args.treatment is not None else None,
        "fixed_training_seeds": list(SEEDS), "fixed_final_step": STEP, "development_map_seeds": list(MAP_SEEDS),
        "analyzer_sha256": sha(Path(__file__)), "python_version": sys.version, "numpy_version": np.__version__,
        "heavy_artifact_hashes_requested": audit.heavy,
        "scope": "Read-only final endpoint analysis; no training, model loading, environment or sealed-map access.",
    }
    save(out / "manifest.json", manifest)
    save(out / "status.json", {"complete": False, "phase": "analysis_running", "started_at_utc": audit.started})
    try:
        audit.say(f"START final_step={STEP} seeds={list(SEEDS)} "
                  f"input_roots={[str(root) for root in input_roots]} "
                  f"verify_actual_artifact_sha256={audit.heavy} output={out}")
        report = (analyze_sharded_study(args.study, audit=audit) if args.study is not None else
                  analyze_study(args.control, args.treatment, audit=audit))
        save(out / "analysis.json", json_safe(report))
        (out / "analysis.md").write_text(markdown_report(report), encoding="utf-8")
        with (out / "paired_seeds.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("seed", "control_successes", "treatment_successes", "difference_maps", "scope"))
            writer.writeheader()
            writer.writerows(report["all_paired_seeds"])
        manifest.update(input_sha256=audit.inputs, inputs_rechecked_unchanged=not report["inputs_changed_during_analysis"])
        save(out / "manifest.json", manifest)
        label = report["decision"]["label"] if report["decision"] else None
        audit.say(f"COMPLETE analysis_complete=true study_status={report['study_status']} "
                  f"decision_ready={report['decision_ready']} decision={label} "
                  f"integrity_issues={len(audit.errors)} missing_evidence={len(audit.missing)} output={out}")
        finished = {"complete": True, "phase": "analysis_complete", "study_status": report["study_status"],
                    "decision_ready": report["decision_ready"], "decision_label": label,
                    "analysis_complete_is_not_study_complete": True, "finished_at_utc": now()}
        save(out / "status.json", finished)
        artifacts = {name: sha(out / name) for name in
                     ("manifest.json", "analysis.json", "analysis.md", "paired_seeds.csv", "analysis.log", "status.json")}
        # Authoritative final write; partial analysis files alone never establish completion.
        save(out / "completion.json", {**finished, "artifact_sha256": artifacts})
    except Exception:
        failure = {"complete": False, "phase": "analysis_failed", "traceback": traceback.format_exc(),
                   "finished_at_utc": now()}
        save(out / "error.json", failure)
        save(out / "status.json", failure)
        audit.say(failure["traceback"])
        raise


if __name__ == "__main__":
    main()
