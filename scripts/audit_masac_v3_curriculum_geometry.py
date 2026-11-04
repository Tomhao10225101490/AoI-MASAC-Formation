#!/usr/bin/env python3
"""Audit endpoint clearances for reproducible curriculum reset samples."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_planning.masac_v3_curriculum import make_episode
from uav_planning.structured import box_distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    args = parser.parse_args()
    if args.episodes_per_seed < 1:
        parser.error("--episodes-per-seed must be positive")
    config_path = ROOT / "configs/masac_v3/curriculum_protocol.json"
    source_path = ROOT / "uav_planning/masac_v3_curriculum.py"
    cfg = json.loads(config_path.read_text())
    minimum = {f"{kind}_{which}": float("inf") for which in ("start", "goal")
               for kind in ("building", "boundary", "peer", "moving_initial")}
    for seed in cfg["seeds"]:
        for episode in range(args.episodes_per_seed):
            env, obs, stage = make_episode(seed, episode, 0, cfg)
            if stage != "short_edge" or obs.shape != (3, 162):
                raise ValueError("Unexpected curriculum stage or observation shape")
            if not np.allclose(np.linalg.norm(env.goal - env.p, axis=1), 40.0):
                raise ValueError("Unexpected goal distance")
            for label, points in (("start", env.p), ("goal", env.goal)):
                boundary = np.minimum(points - env.radius, env.world - env.radius - points).min()
                building = (box_distance(points, env.lo, env.hi) - env.radius).min()
                pair = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
                np.fill_diagonal(pair, np.inf)
                peer = pair.min() - 2 * env.radius
                moving = (np.linalg.norm(points[:, None] - env.dynamic[None, :], axis=-1)
                          - env.dr[None, :] - env.radius).min()
                for kind, value in (("boundary", boundary), ("building", building),
                                    ("peer", peer), ("moving_initial", moving)):
                    key = f"{kind}_{label}"
                    minimum[key] = min(minimum[key], float(value))
    result = {
        "stage": "short_edge", "seeds": cfg["seeds"],
        "episodes_per_seed": args.episodes_per_seed,
        "maps": len(cfg["seeds"]) * args.episodes_per_seed,
        "minimum_clearance_m": minimum,
        "all_endpoint_clearances_positive": all(value > 0 for value in minimum.values()),
        "limits": "Endpoint geometry and initial dynamic positions only; not joint trajectory feasibility.",
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (config_path, source_path, Path(__file__).resolve())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))
    if not result["all_endpoint_clearances_positive"]:
        raise SystemExit("Endpoint clearance audit failed")


if __name__ == "__main__":
    main()
