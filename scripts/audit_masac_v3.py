#!/usr/bin/env python3
"""Reproducible geometry, mechanism-signal and saved-training diagnostics.

Static witnesses ignore traffic, mutual collisions and the mission deadline.
They are geometry checks, not successful multi-UAV trajectories.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_planning.structured import (  # noqa: E402
    DIRECTIONS, StructuredMission, box_distance, segment_box_hit,
)


def geometry_audit():
    rows, witnesses = [], []
    for layout in ("urban", "warehouse", "courtyard"):
        for n in (3, 4, 6):
            maps = []
            for seed in range(910001, 910011):
                env = StructuredMission(n=n, layout=layout)
                env.reset(seed)
                clearance = float(min(
                    box_distance(env.p, env.lo, env.hi).min(),
                    box_distance(env.goal, env.lo, env.hi).min(),
                ) - env.radius)
                direct = segment_box_hit(
                    env.p, env.goal, env.lo - env.radius, env.hi + env.radius,
                ).any(axis=-1)
                cruise_z = float(env.hi[:, 2].max() + 2 * env.radius)
                up, over = env.p.copy(), env.goal.copy()
                up[:, 2] = cruise_z
                over[:, 2] = cruise_z
                hits = [segment_box_hit(a, b, env.lo - 1, env.hi + 1)
                        for a, b in ((env.p, up), (up, over), (over, env.goal))]
                valid = bool(not np.any(hits) and cruise_z <= env.world[2] - 1)
                record = dict(layout=layout, n=n, map_seed=seed,
                              minimum_endpoint_static_clearance_m=clearance,
                              direct_static_hit_agents=int(direct.sum()),
                              static_witness_valid=valid, cruise_z_m=cruise_z)
                maps.append(record)
                witnesses.append(dict(**record, waypoints=np.stack(
                    (env.p, up, over, env.goal), axis=1).tolist()))
            rows.append(dict(
                layout=layout, n=n, maps=len(maps), agents=n * len(maps),
                direct_static_hit_agents=sum(x['direct_static_hit_agents'] for x in maps),
                minimum_endpoint_static_clearance_m=min(
                    x['minimum_endpoint_static_clearance_m'] for x in maps),
                static_witness_maps=sum(x['static_witness_valid'] for x in maps),
            ))
    return rows, witnesses


def paired_probe():
    records = []
    for amax in (2., 4., 6.):
        env = StructuredMission(mode="maneuver", amax=amax)
        env.reset(910101)
        env.p[0] = [env.lo[0, 0] - 10, 40, 20]
        env.v[0] = [8, 0, 0]
        env.observe()
        ids = np.argsort(box_distance(env.p[0], env.lo, env.hi))[:8]
        margin = env.last_margins[0]
        records.append(dict(
            map_seed=910101, amax=amax, position=env.p[0].tolist(),
            velocity=env.v[0].tolist(),
            current_clearance_m=float(box_distance(env.p[0], env.lo, env.hi).min() - 1),
            cv_preview_margin_m=float(env.local_margins(0, ids, np.zeros((1, 3)))[0]),
            margins_m=margin.tolist(), best_tested_margin_m=float(margin.max()),
            positive_probe_fraction=float(np.mean(margin > 0)),
            directions=DIRECTIONS.tolist(), nearest_buildings=env.boxes[ids].tolist(),
        ))
    return records


def saved_training_audit(parent):
    records = []
    for directory in sorted(parent.glob("standard_*")):
        path = directory / "training.jsonl"
        if not path.exists():
            continue
        rows = []
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A live writer may have a partial final line.
        episodes = [r for r in rows if r.get("kind") == "episode"]
        progress = [r for r in rows if r.get("kind") == "progress"]
        evaluations = []
        for file in directory.glob("evaluation_*.json"):
            data = json.loads(file.read_text())
            if data:
                evaluations.append(dict(
                    step=data[0]['step'], successes=sum(x['success'] for x in data),
                    episodes=len(data), remaining_distance_m=float(np.mean(
                        [x['remaining_distance'] for x in data])),
                ))
        evaluations.sort(key=lambda x: x['step'])
        flags = ('success', 'static_collision', 'boundary_collision',
                 'dynamic_collision', 'uav_collision')
        records.append(dict(
            run=directory.name, episodes=len(episodes),
            last_step=progress[-1]['step'] if progress else 0,
            complete=(directory / 'audit.json').exists(),
            terminal_flag_counts={k: sum(bool(r.get(k)) for r in episodes) for k in flags},
            mean_episode_seconds=float(np.mean([r['time'] for r in episodes])) if episodes else None,
            mean_final_remaining_distance_m=float(np.mean(
                [r['remaining_distance'] for r in episodes])) if episodes else None,
            final_validation=evaluations[-1] if evaluations else None,
            evaluations=evaluations,
        ))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path,
                        default=ROOT / 'results/masac_v3_baseline_20261102')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    geometry, witnesses = geometry_audit()
    result = dict(
        evidence_level="development diagnosis; no learned-method efficacy or novelty claim",
        geometry=geometry, paired_probe=paired_probe(),
        training=saved_training_audit(args.parent),
        limitations=[
            "Static witnesses ignore dynamics, other UAVs, moving obstacles and deadline.",
            "Probe margins use a 0.2s time grid and constant-velocity traffic prediction.",
            "A positive sampled margin is not a continuous safety guarantee.",
            "Repeated six-map validations are development data, not final held-out evidence.",
        ],
        source_hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in ('uav_planning/structured.py', 'scripts/audit_masac_v3.py')},
    )
    (args.out / 'diagnosis.json').write_text(json.dumps(result, indent=2))
    (args.out / 'static_witnesses.json').write_text(json.dumps(witnesses, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('paired_probe', 'source_hashes')}, indent=2))


if __name__ == '__main__':
    main()
