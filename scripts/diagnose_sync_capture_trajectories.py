#!/usr/bin/env python3
"""Post-hoc CPU trajectory diagnosis; never changes the 40k efficacy endpoint.

Reads only the requested development evaluation JSON and saved trajectory NPZs.
No weights, learned-policy evaluation, sealed maps, or training are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

DT_S = 0.2
BALL_M = 4.0
SPEED_MPS = 2.0
COLLISIONS = ('uav_collision', 'static_collision', 'dynamic_collision', 'boundary_collision')
CLASSES = ('never_ball', 'ball_without_capture', 'ever_capture')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def longest_streak(mask):
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def first_time(mask):
    indices = np.flatnonzero(mask)
    return float((int(indices[0]) + 1) * DT_S) if len(indices) else None


def diagnose_map(directory, row, step):
    map_seed = row['map_seed']
    context = f'{directory.name} step={step} map_seed={map_seed}'
    require(row.get('step') == step, f'{context}: JSON step mismatch')
    require(type(row.get('success')) is bool, f'{context}: missing/nonboolean JSON success')
    require(all(type(row.get(key)) is bool for key in COLLISIONS),
            f'{context}: missing/nonboolean collision flags')
    path = directory / f'eval_{step}_{map_seed}.npz'
    with np.load(path, allow_pickle=False) as trace:
        arrays = [trace[key] for key in ('positions', 'velocities', 'goals')]
        require(all(np.issubdtype(x.dtype, np.number) and np.isrealobj(x) for x in arrays),
                f'{context}: trajectories must contain real numeric arrays')
        positions, velocities, goals = [x.astype(np.float64, copy=False) for x in arrays]
    require(positions.ndim == 3 and positions.shape[1:] == (3, 3) and len(positions) >= 2,
            f'{context}: positions must have shape (T+1,3,3), T>=1')
    require(velocities.shape == positions.shape and goals.shape == (3, 3),
            f'{context}: velocity/goal shapes do not match positions')
    require(all(np.isfinite(x).all() for x in (positions, velocities, goals)),
            f'{context}: nonfinite trajectory value')
    full_distance = np.linalg.norm(goals[None] - positions, axis=-1)
    # Frame zero is the reset state, not a completed environment transition.
    distance = full_distance[1:]
    speed = np.linalg.norm(velocities[1:], axis=-1)
    in_ball = distance < BALL_M
    captured = in_ball & (speed < SPEED_MPS)
    ever_ball = in_ball.any(axis=0)
    ever_capture = captured.any(axis=0)
    counts = captured.sum(axis=1)
    joint_capture = counts == 3
    maximum_occupancy = int(counts.max())
    closest_including_initial = full_distance.min(axis=0)
    saved_ball = row.get('ever_in_ball_by_uav')
    require(isinstance(saved_ball, list) and len(saved_ball) == 3 and all(type(x) is bool for x in saved_ball),
            f'{context}: missing/nonboolean ever_in_ball_by_uav')
    require(saved_ball == ever_ball.tolist(),
            f'{context}: ever_in_ball mismatch JSON={saved_ball}, trajectory={ever_ball.tolist()}')
    saved_closest = np.asarray(row.get('closest_goal_m_by_uav'), dtype=np.float64)
    require(saved_closest.shape == (3,) and np.isfinite(saved_closest).all(),
            f'{context}: invalid closest_goal_m_by_uav')
    # evaluate() initializes closest at reset, then updates it after every step.
    require(np.allclose(saved_closest, closest_including_initial, rtol=1e-7, atol=1e-6),
            f'{context}: closest mismatch JSON={saved_closest.tolist()}, '
            f'trajectory_including_initial={closest_including_initial.tolist()}')
    collision = any(row[key] for key in COLLISIONS)
    if not collision:
        require(bool(joint_capture.any()) == row['success'],
                f'{context}: collision-free success mismatch JSON={row["success"]}, '
                f'trajectory={bool(joint_capture.any())}')
    if 'max_simultaneous_capture_occupancy' in row:
        require(type(row['max_simultaneous_capture_occupancy']) is int and
                row['max_simultaneous_capture_occupancy'] == maximum_occupancy,
                f'{context}: maximum capture occupancy mismatch JSON='
                f'{row["max_simultaneous_capture_occupancy"]}, trajectory={maximum_occupancy}')
    agents = []
    for index in range(3):
        label = ('ever_capture' if ever_capture[index] else
                 'ball_without_capture' if ever_ball[index] else 'never_ball')
        agents.append({
            'agent_index': index,
            'classification': label,
            'ever_in_ball': bool(ever_ball[index]),
            'ever_captured': bool(ever_capture[index]),
            'min_distance_m': float(closest_including_initial[index]),
            'post_step_min_distance_m': float(distance[:, index].min()),
            'final_distance_m': float(distance[-1, index]),
            'final_speed_mps': float(speed[-1, index]),
            'capture_frames': int(captured[:, index].sum()),
            'max_capture_streak_frames': longest_streak(captured[:, index]),
            'first_capture_time_s': first_time(captured[:, index]),
            'first_ball_time_s': first_time(in_ball[:, index]),
        })
    first_captures = [a['first_capture_time_s'] for a in agents]
    arrival_gap = (float(max(first_captures) - min(first_captures))
                   if all(value is not None for value in first_captures) else None)
    classifications = Counter(a['classification'] for a in agents)
    return {
        'map_seed': map_seed, 'family': row['family'], 'scene_sha256': row['scene_sha256'],
        'trajectory': str(path.resolve()), 'post_step_frames': len(distance),
        'duration_s': float(len(distance) * DT_S),
        'json_success': row['success'], 'collision': collision,
        'collision_flags_nonexclusive': {key: row[key] for key in COLLISIONS},
        'recomputed_joint_capture': bool(joint_capture.any()),
        'recomputed_collision_free_success': bool(joint_capture.any() and not collision),
        'first_joint_capture_time_s': first_time(joint_capture),
        'any_agent_ever_in_ball': bool(ever_ball.any()),
        'all_agents_ever_in_ball': bool(ever_ball.all()),
        'all_agents_ever_captured': bool(ever_capture.all()),
        'max_simultaneous_capture_occupancy': maximum_occupancy,
        'worst_closest_goal_distance_m': float(closest_including_initial.max()),
        'worst_final_goal_distance_m': float(distance[-1].max()),
        'worst_final_speed_mps': float(speed[-1].max()),
        'first_capture_arrival_gap_s': arrival_gap,
        'agent_classification_counts': {key: classifications[key] for key in CLASSES},
        'agents': agents,
        'cross_checks': {
            'ever_in_ball_matches': True, 'closest_including_initial_matches': True,
            'collision_free_success_matches': True if not collision else None,
            'max_occupancy_matches': True if 'max_simultaneous_capture_occupancy' in row else None,
        },
    }


def diagnose_arm(root, seed, step):
    directory = Path(root).resolve() / f'maneuver_static_{seed}'
    path = directory / f'evaluation_{step}.json'
    rows = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(rows, list) and len(rows) == 12, f'{path}: expected 12 development map rows')
    require(all(isinstance(row, dict) and type(row.get('map_seed')) is int and
                isinstance(row.get('family'), str) and isinstance(row.get('scene_sha256'), str)
                for row in rows), f'{path}: missing map seed/family/scene hash')
    require(len({row['map_seed'] for row in rows}) == 12, f'{path}: duplicate map seed')
    maps = [diagnose_map(directory, row, step) for row in rows]
    counts = Counter(a['classification'] for row in maps for a in row['agents'])
    return {
        'root': str(Path(root).resolve()), 'evaluation': str(path), 'training_seed': seed,
        'step': step, 'map_count': len(maps), 'agent_map_observations': 3 * len(maps),
        'agent_classification_counts': {key: counts[key] for key in CLASSES},
        'success_maps': sum(row['json_success'] for row in maps),
        'collision_maps': sum(row['collision'] for row in maps),
        'collision_flags_nonexclusive': {key: sum(row['collision_flags_nonexclusive'][key] for row in maps)
                                        for key in COLLISIONS},
        'any_agent_ever_in_ball_maps': sum(row['any_agent_ever_in_ball'] for row in maps),
        'all_agents_ever_in_ball_maps': sum(row['all_agents_ever_in_ball'] for row in maps),
        'all_agents_ever_captured_maps': sum(row['all_agents_ever_captured'] for row in maps),
        'max_simultaneous_capture_occupancy_histogram': dict(sorted(Counter(
            str(row['max_simultaneous_capture_occupancy']) for row in maps).items())),
        'maps': maps,
    }


def diagnose_study(control, treatment, seed, step):
    require(type(seed) is int and seed >= 0 and type(step) is int and step > 0,
            'seed must be a nonnegative integer; step must be a positive integer')
    arms = {'control': diagnose_arm(control, seed, step),
            'treatment': diagnose_arm(treatment, seed, step)}
    signatures = [{row['map_seed']: (row['family'], row['scene_sha256']) for row in arm['maps']}
                  for arm in arms.values()]
    require(signatures[0] == signatures[1], 'Control/B map seed, family or scene hashes differ')
    return {
        'diagnosis': 'POSTHOC_INTERMEDIATE_TRAJECTORY_DESCRIPTION',
        'training_seed': seed, 'step': step, 'dt_s': DT_S,
        'capture_predicate': 'post-step distance<4m AND speed<2m/s; unlatched',
        'closest_distance_scope': 'min_distance_m and JSON closest include reset frame, matching evaluate(); capture/ever-ball/streak/timing use only post-step frames.',
        'arrival_gap_scope': 'Range of first-capture times only if every agent ever captures; null otherwise. Eventual capture and first-arrival gap do not establish simultaneous capture.',
        'counting_scope': 'Each arm has 36 correlated agent-map observations from one training seed and 12 development maps; these are descriptive counts, not 36 independent trials.',
        'efficacy_decision': None,
        'protocol_status': 'Post-hoc intermediate diagnosis only. The registered final40k primary endpoint and six-history advancement rule remain unchanged; no training change, automatic extension or sealed-test evaluation is authorized.',
        'paired_map_metadata_verified': True,
        'arms': arms,
    }


def markdown_report(summary):
    lines = [f'# Post-hoc trajectory diagnosis: seed {summary["training_seed"]}, step {summary["step"]}',
             '', summary['protocol_status'], '', summary['counting_scope'], '',
             'Capture occupancy counts position/speed only. Success remains collision-free joint capture; all four nonexclusive collision flags are retained per map and per arm in summary.json.', '',
             summary['closest_distance_scope'], '', summary['arrival_gap_scope'], '',
             '| Arm | Never ball | Ball without capture | Ever capture | Success maps | Any-ball maps | All-ball maps | All-ever-capture maps |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, arm in summary['arms'].items():
        c = arm['agent_classification_counts']
        lines.append(f'| {name} | {c["never_ball"]} | {c["ball_without_capture"]} | {c["ever_capture"]} | '
                     f'{arm["success_maps"]}/12 | {arm["any_agent_ever_in_ball_maps"]}/12 | '
                     f'{arm["all_agents_ever_in_ball_maps"]}/12 | {arm["all_agents_ever_captured_maps"]}/12 |')
    for name, arm in summary['arms'].items():
        lines += ['', f'## {name}: map descriptions', '',
                  '| Map seed | Family | Never / ball-only / captured agents | Max occupancy | Worst closest m | Worst final m | Worst final speed m/s | First-capture gap s |',
                  '|---|---|---|---:|---:|---:|---:|---|']
        for row in arm['maps']:
            c = row['agent_classification_counts']
            gap = row['first_capture_arrival_gap_s']
            lines.append(f'| {row["map_seed"]} | {row["family"]} | '
                         f'{c["never_ball"]} / {c["ball_without_capture"]} / {c["ever_capture"]} | '
                         f'{row["max_simultaneous_capture_occupancy"]} | '
                         f'{row["worst_closest_goal_distance_m"]:.4f} | {row["worst_final_goal_distance_m"]:.4f} | '
                         f'{row["worst_final_speed_mps"]:.4f} | {"None" if gap is None else f"{gap:.4f}"} |')
    lines += ['', 'Per-agent minimum/final distance, final speed, capture-frame count, longest capture streak and first-capture time are retained in summary.json. All requested JSON/trajectory cross-checks passed; inconsistencies raise an error instead of being hidden.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--treatment', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summary = diagnose_study(args.control, args.treatment, args.seed, args.step)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (args.out / 'summary.md').write_text(markdown_report(summary), encoding='utf-8')
    for name, arm in summary['arms'].items():
        print(f'DIAGNOSIS arm={name} seed={args.seed} step={args.step} '
              f'agent_classes={arm["agent_classification_counts"]} '
              f'any_ball_maps={arm["any_agent_ever_in_ball_maps"]}/12 '
              f'all_ball_maps={arm["all_agents_ever_in_ball_maps"]}/12 '
              f'all_ever_capture_maps={arm["all_agents_ever_captured_maps"]}/12 '
              f'occupancy_hist={arm["max_simultaneous_capture_occupancy_histogram"]}', flush=True)
    print(f'ARTIFACTS {args.out.resolve() / "summary.json"} {args.out.resolve() / "summary.md"}', flush=True)


if __name__ == '__main__':
    main()
