#!/usr/bin/env python3
"""Audit a completed curriculum run against its fixed-budget V3 baseline."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

FLAGS = ('static_collision', 'boundary_collision', 'dynamic_collision', 'uav_collision')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def require_complete(root):
    if (root/'error.json').exists():
        raise ValueError(f'Run has error marker: {root / "error.json"}')
    path = root/'completion.json'
    if not path.is_file() or not read_json(path).get('complete'):
        raise ValueError(f'Run has no completed batch marker: {root}')


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wait_for_run(root, pid, interval, max_wait):
    start = time.monotonic()
    while True:
        if (root/'error.json').exists():
            raise RuntimeError(f'Training reported an error: {root / "error.json"}')
        complete = (root/'completion.json').exists()
        alive = process_alive(pid)
        if complete and not alive:
            require_complete(root)
            return
        if not alive and not complete:
            raise RuntimeError(f'Training PID {pid} exited without a completion marker')
        if time.monotonic()-start >= max_wait:
            raise TimeoutError(f'Training PID {pid} did not finish within {max_wait} seconds')
        time.sleep(interval)


def evaluation(root, seed, steps, map_seeds):
    path = root/f'standard_{seed}'/f'evaluation_{steps}.json'
    rows = read_json(path)
    ids = [row['map_seed'] for row in rows]
    if len(ids) != len(map_seeds) or set(ids) != set(map_seeds) or len(ids) != len(set(ids)):
        raise ValueError(f'Unexpected or duplicate evaluation map seeds: {path}')
    return {row['map_seed']: row for row in rows}


def eval_summary(rows):
    rows = list(rows.values())
    return dict(episodes=len(rows), successes=sum(bool(r['success']) for r in rows),
        collision_flag_counts={key:sum(bool(r[key]) for r in rows) for key in FLAGS},
        no_collision_or_success=sum(not r['success'] and not any(r[k] for k in FLAGS) for r in rows),
        mean_team_remaining_distance_m=float(np.mean([r['remaining_distance'] for r in rows])),
        mean_team_path_m=float(np.mean([r['path_length'] for r in rows])),
        mean_episode_seconds=float(np.mean([r['time'] for r in rows])))


def stage_summary(path):
    stages = {}
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            if row.get('kind') != 'episode':
                continue
            stage = row['actual_stage']
            item = stages.setdefault(stage, dict(completed_episodes=0, successes=0,
                terminated=0, truncated=0, collision_flag_counts={k:0 for k in FLAGS},
                sum_remaining_distance_m=0., sum_episode_seconds=0., first_start_step=row['episode_start_step'],
                first_end_step=row['step'], last_end_step=row['step']))
            item['completed_episodes'] += 1
            item['successes'] += int(row['success'])
            item['terminated'] += int(row['terminated'])
            item['truncated'] += int(row['truncated'])
            for k in FLAGS:
                item['collision_flag_counts'][k] += int(row[k])
            item['sum_remaining_distance_m'] += row['remaining_distance']
            item['sum_episode_seconds'] += row['time']
            item['last_end_step'] = row['step']
    for item in stages.values():
        n = item['completed_episodes']
        item['mean_remaining_distance_m'] = item.pop('sum_remaining_distance_m')/n
        item['mean_episode_seconds'] = item.pop('sum_episode_seconds')/n
    return stages


def coverage(path, vmax, expected_size):
    with np.load(path) as replay:
        observations = replay['obs']
        if observations.ndim != 3 or observations.shape[1:] != (3, 162) or observations.shape[0] != expected_size:
            raise ValueError(f'Unexpected replay dimensions: {path}: {observations.shape}')
        distances = np.linalg.norm(observations[..., :3]*240., axis=-1)
        speeds = np.linalg.norm(observations[..., 3:6]*vmax, axis=-1)
        result = dict(transitions=int(len(distances)), agent_states=int(distances.size),
            minimum_distance_m=float(distances.min()),
            distance_percentiles_m=dict(zip(('p01','p10','p50','p90'), np.percentile(distances,[1,10,50,90]).tolist())),
            agent_states_below_distance_m={str(d):int(np.count_nonzero(distances < d)) for d in (4,10,30,50,100)},
            agent_states_with_goal_capture=int(np.count_nonzero((distances < 4) & (speeds < 2))))
    result['replay_sha256'] = digest(path)
    return result


def analyze(run, baseline):
    require_complete(run); require_complete(baseline)
    cfg, original = read_json(run/'protocol.json'), read_json(baseline/'protocol.json')
    if cfg.get('smoke') or cfg['arms'] != ['standard'] or cfg['seeds'] != [411,422,433] or cfg['steps'] != 100000:
        raise ValueError('Expected the full three-seed, 100000-step Standard curriculum run')
    if original['steps'] != cfg['steps'] or original['validation_map_seeds'] != cfg['validation_map_seeds']:
        raise ValueError('Baseline budget or development maps do not match the curriculum run')
    rows = []
    for seed in cfg['seeds']:
        directory = run/f'standard_{seed}'
        audit = read_json(directory/'audit.json')
        if audit['total_steps'] != cfg['steps'] or audit['replay_checkpoint_alignment_step'] != cfg['steps']:
            raise ValueError(f'Incomplete final audit for seed {seed}')
        if not (directory/f'checkpoint_{cfg["steps"]}.pt').is_file():
            raise ValueError(f'Missing final checkpoint for seed {seed}')
        new = evaluation(run, seed, cfg['steps'], cfg['validation_map_seeds'])
        old = evaluation(baseline, seed, cfg['steps'], cfg['validation_map_seeds'])
        comparisons = []
        for map_seed in cfg['validation_map_seeds']:
            a, b = old[map_seed], new[map_seed]
            if a['layout'] != b['layout']:
                raise ValueError(f'Map layout mismatch for seed {seed}, map {map_seed}')
            map_name = f'eval_{cfg["steps"]}_{map_seed}_map.json'
            if read_json(baseline/f'standard_{seed}'/map_name) != read_json(directory/map_name):
                raise ValueError(f'Map geometry or initial state mismatch for seed {seed}, map {map_seed}')
            comparisons.append(dict(map_seed=map_seed, layout=a['layout'],
                baseline_success=bool(a['success']), curriculum_success=bool(b['success']),
                baseline_remaining_distance_m=a['remaining_distance'], curriculum_remaining_distance_m=b['remaining_distance']))
        rows.append(dict(training_seed=seed, baseline_final=eval_summary(old), curriculum_final=eval_summary(new),
            paired_development_maps=comparisons, completed_training_episodes_by_stage=stage_summary(directory/'training.jsonl'),
            final_replay_coverage=coverage(directory/'replay_final.npz', cfg['vmax'], audit['replay_size'])))
    return dict(run=str(run.resolve()), baseline=str(baseline.resolve()), final_steps=cfg['steps'],
        independent_unit='training seed', development_maps=cfg['validation_map_seeds'],
        warning='Repeated development maps are not independent training seeds. Collision flags and terminated/truncated flags may overlap; do not add them. Replay states are correlated pre-action training states, not independent evaluations. This is exploratory, not confirmation.',
        analysis_sha256=digest(Path(__file__)), seeds=rows)


def write_new(path, data):
    with path.open('x') as dest:
        json.dump(data, dest, indent=2, allow_nan=False)
        dest.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait', action='store_true', help='Wait for this training process to exit before analysis')
    parser.add_argument('--pid', type=int, help='Training PID; defaults to run protocol pid')
    parser.add_argument('--poll-seconds', type=float, default=10.)
    parser.add_argument('--max-wait-seconds', type=float, default=86400.)
    args = parser.parse_args()
    error = args.out.with_name(args.out.name+'.error.json')
    try:
        if args.out.exists() or error.exists():
            raise FileExistsError('Analysis or error output already exists; choose a new output path')
        if args.wait:
            pid = args.pid if args.pid else read_json(args.run/'protocol.json')['pid']
            if pid <= 0 or args.poll_seconds <= 0 or args.max_wait_seconds <= 0:
                raise ValueError('Invalid PID or wait interval')
            wait_for_run(args.run, pid, args.poll_seconds, args.max_wait_seconds)
        result = analyze(args.run, args.baseline)
        write_new(args.out, result)
        print(f'Wrote {args.out}', flush=True)
    except Exception as exc:
        if not error.exists() and not args.out.exists():
            write_new(error, dict(error=repr(exc), run=str(args.run), updated_at=time.time()))
        raise


if __name__ == '__main__':
    main()
