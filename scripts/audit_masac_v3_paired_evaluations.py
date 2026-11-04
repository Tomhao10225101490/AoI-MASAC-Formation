#!/usr/bin/env python3
"""Audit fixed checkpoints and paired development-map outcomes; never overwrite evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("static_collision", "boundary_collision", "dynamic_collision", "uav_collision")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def summarize(rows):
    return {
        "episodes": len(rows),
        "successes": sum(bool(x["success"]) for x in rows),
        "collision_flags": {k: sum(bool(x[k]) for x in rows) for k in FLAGS},
        "episodes_with_multiple_collision_flags": sum(sum(bool(x[k]) for k in FLAGS) > 1 for x in rows),
        "episodes_no_collision_or_success": sum(not x["success"] and not any(x[k] for k in FLAGS) for x in rows),
        "mean_team_remaining_distance_m": float(np.mean([x["remaining_distance"] for x in rows])),
        "mean_team_path_length_m": float(np.mean([x["path_length"] for x in rows])),
        "mean_episode_seconds": float(np.mean([x["time"] for x in rows])),
    }


def rows_at(path, seeds, step):
    rows = read(path)
    if len(rows) != len(seeds) or sorted(x["map_seed"] for x in rows) != sorted(seeds):
        raise ValueError(f"Evaluation map mismatch: {path}")
    if any(x["step"] != step for x in rows):
        raise ValueError(f"Evaluation step mismatch: {path}")
    return rows


def snapshot_at(directory, arm, seed, step):
    snap = directory / "snapshots" / str(step)
    manifest = read(snap / "complete.json")
    if (not manifest["complete"] or manifest["arm"] != arm or manifest["seed"] != seed
            or manifest["step"] != step):
        raise ValueError(f"Incomplete or mismatched snapshot: {snap}")
    paths = {"protocol_sha256": snap / "protocol.json",
             "checkpoint_sha256": snap / "state" / f"checkpoint_{step}.pt",
             "replay_sha256": snap / "state" / "replay_final.npz"}
    for key, path in paths.items():
        if digest(path) != manifest[key]:
            raise ValueError(f"Snapshot digest mismatch: {path}")
    result = {"step": step, "manifest_sha256": digest(snap / "complete.json"),
              "checkpoint_sha256": manifest["checkpoint_sha256"],
              "replay_sha256": manifest["replay_sha256"], "all_manifest_hashes_match": True}
    if step == 300000:
        cp = torch.load(paths["checkpoint_sha256"], map_location="cpu", weights_only=False)
        if (cp["seed"] != seed or cp["step"] != step or cp["replay_size"] != 100000
                or cp["replay_index"] != 0 or not cp["exact_resume"] or cp["history_arm"] != arm):
            raise ValueError(f"Final checkpoint content mismatch: {snap}")
        result.update(replay_size=cp["replay_size"], replay_index=cp["replay_index"],
                      episode=cp["episode"], elapsed_episode_steps=cp["environment"]["t"])
    return result


def trajectory(path):
    with np.load(path) as z:
        positions = z["positions"]
        goals = z["goals"]
        distances = np.linalg.norm(positions - goals[None, :, :], axis=-1)
        speeds = np.zeros_like(distances)
        speeds[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=-1) / .2
        captured = (distances < 4) & (speeds < 2)
        return {
            "sha256": digest(path),
            "sampled_states": int(len(positions)),
            "closest_goal_distance_m_by_uav": distances.min(axis=0).tolist(),
            "entered_goal_4m_by_uav": np.any(distances < 4, axis=0).tolist(),
            "met_capture_condition_at_least_once_by_uav": np.any(captured, axis=0).tolist(),
            "max_simultaneous_capture_count": int(captured.sum(axis=1).max()),
            "distance_and_speed_method": "Stored position samples at 0.2s; speed from consecutive positions. Initial speed is zero.",
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    run = ROOT / "results/masac_v3_paired_300k_20261103"
    complete = read(run / "completion.json")
    cfg = read(run / "protocol.json")
    if (run / "error.json").exists() or not complete["complete"] or complete["runs"] != 6 or complete["smoke"]:
        raise ValueError("The full paired batch must be complete and error-free.")
    if args.out.exists():
        raise FileExistsError(args.out)
    paths = {arm: ROOT / parent for arm, parent in cfg["parents"].items()}
    results, indexed = [], {}
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            d = run / f"{arm}_{seed}"
            parent = paths[arm] / f"standard_{seed}"
            audit_path = d / "audit.json"
            audit = read(audit_path)
            if (audit["arm"] != arm or audit["seed"] != seed or audit["total_steps"] != 300000
                    or audit["start_step"] != 100000 or audit["updates"] != 299000
                    or audit["replay_size"] != 100000 or audit["replay_index"] != 0):
                raise ValueError(f"Unexpected training audit: {audit_path}")
            snapshots = [snapshot_at(d, arm, seed, step) for step in (150000, 200000, 250000, 300000)]
            parent_rows = rows_at(parent / "evaluation_100000.json", cfg["validation_map_seeds"], 100000)
            parent_by_id = {r["map_seed"]: r for r in parent_rows}
            history = [{"step": 100000, **summarize(parent_rows)}]
            for step in range(102000, 300001, 2000):
                rows = rows_at(d / f"evaluation_{step}.json", cfg["validation_map_seeds"], step)
                history.append({"step": step, **summarize(rows)})
            final_rows = rows_at(d / "evaluation_300000.json", cfg["validation_map_seeds"], 300000)
            by_id = {}
            maps = []
            for row in final_rows:
                m = row["map_seed"]
                metadata = d / f"eval_300000_{m}_map.json"
                old_metadata = parent / f"eval_100000_{m}_map.json"
                if read(metadata) != read(old_metadata):
                    raise ValueError(f"Parent/final map metadata differs: {arm}/{seed}/{m}")
                t = trajectory(d / f"eval_300000_{m}.npz")
                item = {
                    "map_seed": m, "layout": row["layout"], "metadata_sha256": digest(metadata),
                    "evaluation_100k": parent_by_id[m], "evaluation_300k": row,
                    "trajectory_300k": t,
                    "delta_remaining_distance_m": row["remaining_distance"] - parent_by_id[m]["remaining_distance"],
                }
                maps.append(item)
                by_id[m] = item
            indexed[(arm, seed)] = by_id
            results.append({
                "history": arm, "training_seed": seed,
                "training_audit_sha256": digest(audit_path), "updates": audit["updates"],
                "initial_100k": summarize(parent_rows), "final_300k": summarize(final_rows),
                "evaluation_100k_sha256": digest(parent / "evaluation_100000.json"),
                "evaluation_300k_sha256": digest(d / "evaluation_300000.json"),
                "full_checkpoints": snapshots, "evaluation_curve": history, "maps": maps,
            })
    pairwise = []
    for seed in cfg["seeds"]:
        for m in cfg["validation_map_seeds"]:
            x, y = indexed[("curriculum", seed)][m], indexed[("original", seed)][m]
            if x["metadata_sha256"] != y["metadata_sha256"]:
                raise ValueError(f"Paired history map metadata differs: {seed}/{m}")
            a, b = x["evaluation_300k"], y["evaluation_300k"]
            pairwise.append({"training_seed": seed, "map_seed": m, "layout": a["layout"],
                             "curriculum_success": bool(a["success"]), "original_success": bool(b["success"]),
                             "curriculum_minus_original_remaining_m": a["remaining_distance"] - b["remaining_distance"]})
    output = {
        "run": str(run), "completion_sha256": digest(run / "completion.json"),
        "protocol_sha256": digest(run / "protocol.json"), "audit_script_sha256": digest(Path(__file__)),
        "analysis_limitations": [
            "Exploratory development maps were repeatedly evaluated and are not an independent confirmation set.",
            "Independent training repeat is the training seed, not each map episode.",
            "Collision flags may overlap; do not sum them as disjoint failure counts.",
            "Distance and trajectory length for failed episodes do not establish navigation quality or success.",
            "300k full replay contains only the latest 100k transitions; no short-stage replay remains in the curriculum history.",
        ],
        "runs": results, "paired_300k_by_map": pairwise,
    }
    with args.out.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Wrote {args.out}")
    for x in results:
        print(x["history"], x["training_seed"], x["final_300k"])


if __name__ == "__main__":
    main()
