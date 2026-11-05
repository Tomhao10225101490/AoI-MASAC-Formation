#!/usr/bin/env python3
"""Fixed 72-trajectory CPU-only descriptive audit; no environment/model imports."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

from diagnose_sync_capture_trajectories import diagnose_map, longest_streak, require

ROOT = Path(__file__).resolve().parents[1]
STEP, DT, GAMMA = 40000, 0.2, 0.99
SEEDS, MAPS = (1511, 1522, 1533), tuple(range(830001, 830013))
ARMS = ('control', 'b')
PROTOCOL = 'docs/design/MASAC_V4_FINAL_TRAJECTORY_DWELL_PROTOCOL_20261105.md'
SOURCES = ('scripts/train_masac_v4_maneuver.py', 'uav_planning/v4_reward_reshape.py',
           'scripts/diagnose_sync_capture_trajectories.py', 'scripts/audit_sync_final_trajectory_dwell.py',
           'docs/design/MASAC_V4_SYNC_CAPTURE_40K_PROTOCOL_20261105.md', PROTOCOL)
DISTANCE_BANDS = ((0., 4., 'd0_4'), (4., 5., 'd4_5'), (5., 10., 'd5_10'),
                  (10., np.inf, 'd10_inf'))
COLLISIONS = ('uav_collision', 'static_collision', 'dynamic_collision', 'boundary_collision')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    if not path.is_file():
        return 'UNAVAILABLE'
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [(int(start), int(end)) for start, end in
            zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]


def occupancy(mask):
    count, longest = int(mask.sum()), longest_streak(mask)
    return dict(frames=count, occupancy_s=count * DT, fraction=count / len(mask),
                longest_frames=longest, longest_occupancy_s=longest * DT,
                longest_endpoint_span_s=(longest - 1) * DT if longest else None)


def statistics(values):
    return dict(min=float(values.min()), mean=float(values.mean()),
                median=float(np.median(values)), max=float(values.max()))


def event_times(mask):
    return [float((int(t) + 1) * DT) for t in np.flatnonzero(mask)]


def reward_components(full_distance, velocity, ball, captured, joint, reshape):
    previous = np.vstack((np.zeros((1, 3), dtype=bool), captured[:-1]))
    ever = np.maximum.accumulate(ball, axis=0)
    previous_ever = np.vstack((np.zeros((1, 3), dtype=bool), ever[:-1]))
    real_acc = np.diff(velocity, axis=0) / DT
    components = {
        'progress': reshape['progress_scale'] * -np.diff(full_distance, axis=0),
        'living': np.full(captured.shape, -reshape['living_cost'], dtype=np.float64),
        'real_acceleration_penalty': -reshape['acc_penalty'] * np.sum(real_acc ** 2, axis=-1),
        'near': reshape['near_goal_bonus'] * np.clip(
            1 - full_distance[1:] / reshape['near_goal_radius_m'], 0, 1),
        'first_ever_ball': reshape['ball_enter_bonus'] * (ball & ~previous_ever),
        'capture_delta': reshape['capture_bonus'] * (
            captured.astype(float) - previous.astype(float)),
        'team_terminal': np.repeat((reshape['team_success_bonus'] * joint)[:, None], 3, axis=1),
        'collision': np.zeros(captured.shape, dtype=np.float64),
    }
    require(np.array_equal((ball & ~previous_ever).sum(axis=0), ball.any(axis=0).astype(int)),
            'first-ever ball memory mismatch')
    weights = GAMMA ** np.arange(len(captured))
    totals = {}
    for name, values in {**components, 'total': sum(components.values())}.items():
        totals[name] = {
            'undiscounted_by_agent': values.sum(axis=0).tolist(),
            'undiscounted_team_mean': float(values.mean(axis=1).sum()),
            'raw_discounted_by_agent': (values * weights[:, None]).sum(axis=0).tolist(),
            'raw_discounted_team_mean': float(np.dot(weights, values.mean(axis=1))),
        }
    return dict(component_totals=totals, formula_components_available=True,
                reward_sum_reference='UNAVAILABLE', reference_validation='UNAVAILABLE',
                reason='Audited evaluation writer saves no base-return/reward field; no training return substituted.',
                reference_tolerance='1e-4 + 1e-5*abs(reference)', gamma=GAMMA,
                discounted_scope='Raw finite-trajectory base reward only; no bootstrap value or entropy.')


def audit_map(directory, row, old, reshape):
    path = directory / f'eval_{STEP}_{row["map_seed"]}.npz'
    require(row['step'] == STEP and type(row['success']) is bool, 'Invalid evaluation endpoint')
    require(all(type(row.get(key)) is bool for key in COLLISIONS), 'Invalid collision flags')
    require(not any(row[key] for key in COLLISIONS),
            'Fixed population unexpectedly has a collision; event timing is unavailable')
    require(row['reshape'] == reshape, 'Recorded reshape differs from frozen configuration')
    require(not any('return' in key.lower() or 'reward' in key.lower() for key in row),
            'Unexpected return-like evaluation field requires a separate writer audit')
    require(old['map_seed'] == row['map_seed'] and old['family'] == row['family'] and
            old['scene_sha256'] == row['scene_sha256'], 'Prior map identity mismatch')
    result = dict(map_seed=row['map_seed'], family=row['family'], scene_sha256=row['scene_sha256'],
                  json_success=row['success'], collision=False,
                  collision_flags_nonexclusive={key: row[key] for key in COLLISIONS},
                  saved_ever_ball_any=row['ever_ball_any'], saved_ever_ball_all=row['ever_ball_all'])
    metadata_path = directory / f'eval_{STEP}_{row["map_seed"]}_map.json'
    metadata = read_json(metadata_path) if metadata_path.is_file() else None
    if metadata is not None:
        require(metadata['seed'] == row['map_seed'] and metadata['scene_sha256'] == row['scene_sha256']
                and metadata['config']['family'] == row['family'], 'Scene metadata identity mismatch')
        require(metadata['config']['dt'] == DT and metadata['config']['horizon'] == reshape['horizon'],
                'Scene timestep/horizon mismatch')
    if not path.is_file():
        return dict(result, dwell_status='UNAVAILABLE', reason='Missing trajectory NPZ',
                    reward_sum_reference='UNAVAILABLE', reward_formula_status='UNAVAILABLE')
    with np.load(path, allow_pickle=False) as trace:
        missing = [key for key in ('positions', 'velocities', 'goals') if key not in trace.files]
        if missing:
            return dict(result, dwell_status='UNAVAILABLE', reason=f'Missing NPZ arrays: {missing}',
                        reward_sum_reference='UNAVAILABLE', reward_formula_status='UNAVAILABLE')
        position, velocity, goal = [trace[key] for key in ('positions', 'velocities', 'goals')]
        actions = trace['actions'] if 'actions' in trace.files else None
    # Reuse the previously audited pure-array JSON/trajectory checks, without an environment.
    result = diagnose_map(directory, row, STEP)
    require(not result['collision'], 'Fixed population unexpectedly has a collision; timing is unavailable')
    for key in ('all_agents_ever_in_ball', 'all_agents_ever_captured', 'recomputed_joint_capture',
                'first_joint_capture_time_s', 'max_simultaneous_capture_occupancy'):
        require(result[key] == old[key], f'Existing trajectory summary mismatch: {key}')
    position, velocity, goal = [x.astype(np.float64, copy=False) for x in (position, velocity, goal)]
    physics = dict(status='UNAVAILABLE', reason='Missing actions or scene metadata')
    if metadata is not None:
        require(np.allclose(position[0], metadata['starts'], rtol=0, atol=1e-9) and
                np.allclose(goal, metadata['goals'], rtol=0, atol=1e-9), 'Scene start/goal mismatch')
    if actions is not None and metadata is not None:
        require(actions.shape == (len(position) - 1, 3, 3) and np.isrealobj(actions) and
                np.issubdtype(actions.dtype, np.number) and np.isfinite(actions).all(),
                'Invalid saved actions')
        # Pure-array transcription of source limited(); no environment is instantiated.
        action = np.clip(actions.astype(np.float64), -1, 1)
        action /= np.maximum(1, np.linalg.norm(action, axis=-1, keepdims=True))
        candidate = velocity[:-1] + DT * metadata['config']['amax'] * action
        expected_v = candidate * np.minimum(
            1, metadata['config']['vmax'] / (np.linalg.norm(candidate, axis=-1, keepdims=True) + 1e-12))
        velocity_error = float(np.max(np.abs(expected_v - velocity[1:])))
        position_error = float(np.max(np.abs(position[:-1] + DT * velocity[1:] - position[1:])))
        require(velocity_error <= 1e-9 and position_error <= 1e-9, 'Saved physics transition mismatch')
        physics = dict(status='PASS', max_velocity_abs_error=velocity_error,
                       max_position_abs_error=position_error, tolerance=1e-9,
                       amax=metadata['config']['amax'], vmax=metadata['config']['vmax'])
    full_distance = np.linalg.norm(goal[None] - position, axis=-1)
    distance, speed = full_distance[1:], np.linalg.norm(velocity[1:], axis=-1)
    real_acc = np.diff(velocity, axis=0) / DT
    ball, captured = distance < 4, (distance < 4) & (speed < 2)
    joint = captured.all(axis=1)
    require(len(distance) <= reshape['horizon'], 'Trajectory exceeds frozen horizon')
    require(not joint[:-1].any(), 'Saved trajectory continues after joint terminal capture')
    require(all(type(row.get(key)) is bool for key in ('terminated', 'truncated', 'timeout')),
            'Nonboolean termination flags')
    require(row['terminated'] == bool(joint[-1]) and
            row['timeout'] == (row['truncated'] and not row['terminated']),
            'Evaluation termination/timeout does not match collision-free trajectory')
    require(row['truncated'] == (len(distance) == reshape['horizon']), 'Truncation/horizon mismatch')
    require(bool(joint[-1]) or len(distance) == reshape['horizon'], 'Nonterminal trajectory ends early')
    require(row['ever_ball_any'] == bool(ball.any()) and row['ever_ball_all'] == bool(ball.any(axis=0).all()),
            'Evaluation any/all-ball mismatch')
    require(np.allclose(row['final_goal_m_by_uav'], distance[-1], rtol=1e-7, atol=1e-6),
            'Evaluation final distances mismatch')
    previous = np.vstack((np.zeros((1, 3), dtype=bool), captured[:-1]))
    for i, agent in enumerate(result['agents']):
        prior_agent = old['agents'][i]
        require(prior_agent['agent_index'] == i, 'Prior agent ordering mismatch')
        for key in ('ever_in_ball', 'ever_captured', 'first_ball_time_s', 'first_capture_time_s',
                    'capture_frames', 'max_capture_streak_frames'):
            require(agent[key] == prior_agent[key], f'Existing agent summary mismatch: {key}')
        bands = {f'{name}_{label}': occupancy((distance[:, i] >= lo) & (distance[:, i] < hi) & mask)
                 for lo, hi, name in DISTANCE_BANDS
                 for label, mask in (('v0_2', speed[:, i] < 2), ('v2_inf', speed[:, i] >= 2))}
        require(sum(b['frames'] for b in bands.values()) == len(distance), 'Band partition mismatch')
        stall = (distance[:, i] >= 4) & (distance[:, i] < 10) & (speed[:, i] < 0.5)
        long_runs = [dict(first_poststep_time_s=(start + 1) * DT,
                          last_poststep_time_s=end * DT, frames=end - start,
                          occupancy_s=(end - start) * DT, endpoint_span_s=(end - start - 1) * DT)
                     for start, end in runs(stall) if end - start >= 25]
        agent.update(bands=bands, near_stall=occupancy(stall), near_stall_runs_25plus=long_runs,
                     capture_entry_times_s=event_times(captured[:, i] & ~previous[:, i]),
                     capture_exit_times_s=event_times(~captured[:, i] & previous[:, i]),
                     distance_m=statistics(distance[:, i]), speed_mps=statistics(speed[:, i]),
                     realized_acceleration_norm_mps2=statistics(np.linalg.norm(real_acc[:, i], axis=-1)),
                     mean_realized_acceleration_xyz_mps2=real_acc[:, i].mean(axis=0).tolist())
    result.update(dwell_status='AVAILABLE', reward_formula_status='AVAILABLE',
                  saved_physics_cross_check=physics,
                  joint_capture_entry_times_s=event_times(joint & ~np.r_[False, joint[:-1]]),
                  joint_capture_exit_times_s=event_times(~joint & np.r_[False, joint[:-1]]),
                  reward=reward_components(full_distance, velocity, ball, captured, joint, reshape),
                  prior_summary_cross_checks='PASS')
    return result


def audit(out):
    inputs, histories, signatures = {}, [], {}

    def snapshot(path):
        key, value = str(path.resolve()), digest(path)
        require(inputs.setdefault(key, value) == value, f'Input changed after initial freeze: {path}')

    # Freeze every input before measurements. Only trajectories/scene metadata may be absent.
    for relative in SOURCES:
        require((ROOT / relative).is_file(), f'Missing audited source/protocol: {relative}')
        snapshot(ROOT / relative)
    for arm in ARMS:
        root = ROOT / f'results/masac_v4_sync_40k_{arm}_cuda_20261105'
        config_path = ROOT / f'configs/masac_v4/sync_capture_40k_{arm}_cuda_20261105.json'
        manifest_path, completion_path = root / 'run_manifest.json', root / 'completion.json'
        for path in (config_path, manifest_path, completion_path):
            snapshot(path)
        manifest, completion, config = map(read_json, (manifest_path, completion_path, config_path))
        require(manifest['config_hashes']['protocol'] == digest(config_path), 'Original config hash changed')
        require(manifest['seed_list'] == list(SEEDS) and manifest['steps_per_seed'] == STEP and
                completion['complete'] is True and len(completion['runs']) == 3 and
                {r['seed'] for r in completion['runs']} == set(SEEDS), 'Incomplete frozen study')
        require(manifest['reshape'] == config['reshape'] == completion['reshape'] and
                manifest['sync_capture'] == config['sync_capture'] and
                manifest['development_maps_used'] == config['development_maps'], 'Run manifest/config mismatch')
        require(len(manifest['sources']) == 14, 'Unexpected frozen source set')
        for relative, expected_sha in manifest['sources'].items():
            path = (ROOT / relative).resolve()
            require(path.is_relative_to(ROOT), 'Frozen source outside repository')
            require(digest(path) == expected_sha, f'Original frozen source changed: {relative}')
            snapshot(path)
        for seed in SEEDS:
            directory = root / f'maneuver_static_{seed}'
            final = directory / 'completion.json'
            snapshot(final)
            record = read_json(final)
            require(record['seed'] == seed and record['step'] == STEP and
                    record['reshape'] == config['reshape'] and record in completion['runs'],
                    'Seed completion differs from frozen study')
            for path in (directory / f'evaluation_{STEP}.json', ROOT /
                         f'results/masac_v4_sync_40k_progress_20261105/trajectory_40000_seed{seed}/summary.json'):
                snapshot(path)
                require(path.is_file(), f'Missing required evaluation/prior summary: {path}')
            for map_seed in MAPS:
                for suffix in ('.npz', '_map.json'):
                    snapshot(directory / f'eval_{STEP}_{map_seed}{suffix}')
    provenance = dict(inputs_sha256=inputs, fixed_seeds=SEEDS, fixed_maps=MAPS, step=STEP,
                      original_training_source_config_hashes_verified=True,
                      numerical_environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
                      input_hashes_rechecked=False)
    write_json(out / 'manifest.json', provenance)

    with (out / 'audit.log').open('x', encoding='utf-8', buffering=1) as log:
        def say(line):
            print(line, flush=True)
            print(line, file=log, flush=True)

        try:
            say(f'START CPU fixed_maps=72 seeds={SEEDS} step={STEP} gamma={GAMMA} '
                'training=false environment=false weights=false reward_reference=UNAVAILABLE')
            write_json(out / 'status.json', dict(phase='RUNNING', processed_maps=0, total_maps=72))
            with (out / 'maps.jsonl').open('x', encoding='utf-8') as stream:
                for arm in ARMS:
                    config_path = ROOT / f'configs/masac_v4/sync_capture_40k_{arm}_cuda_20261105.json'
                    snapshot(config_path)
                    config = read_json(config_path)
                    reshape = config['reshape']
                    require(reshape['train_ball_m'] == 4 and reshape['train_speed_mps'] == 2 and
                            reshape['capture_latch'] is False, 'Frozen capture configuration changed')
                    require(reshape['near_goal_radius_m'] > 0, 'Unexpected inactive near-goal term')
                    require(config['seeds'] == list(SEEDS) and config['steps'] == STEP,
                            'Unexpected training population')
                    expected = {entry['seed']: entry['family'] for entry in config['development_maps']}
                    require(set(expected) == set(MAPS), 'Unexpected development maps')
                    for seed in SEEDS:
                        directory = ROOT / f'results/masac_v4_sync_40k_{arm}_cuda_20261105/maneuver_static_{seed}'
                        evaluation = directory / f'evaluation_{STEP}.json'
                        prior = ROOT / f'results/masac_v4_sync_40k_progress_20261105/trajectory_40000_seed{seed}/summary.json'
                        for path in (evaluation, prior):
                            snapshot(path)
                        rows = read_json(evaluation)
                        require(len(rows) == 12 and {r['map_seed'] for r in rows} == set(MAPS),
                                'Expected every fixed development map exactly once')
                        prior_record = read_json(prior)
                        require(prior_record['training_seed'] == seed and prior_record['step'] == STEP and
                                prior_record['paired_map_metadata_verified'] is True, 'Invalid prior population')
                        old_arm = prior_record['arms']['control' if arm == 'control' else 'treatment']
                        require(old_arm['training_seed'] == seed and old_arm['step'] == STEP and
                                len(old_arm['maps']) == 12 and
                                {m['map_seed'] for m in old_arm['maps']} == set(MAPS), 'Invalid prior arm')
                        old_maps = {r['map_seed']: r for r in old_arm['maps']}
                        maps = []
                        for row in sorted(rows, key=lambda r: r['map_seed']):
                            map_seed = row['map_seed']
                            require(row['family'] == expected[map_seed], 'Map family differs from protocol')
                            signature = (row['family'], row['scene_sha256'])
                            require(signatures.setdefault(map_seed, signature) == signature,
                                    'Map scene hash differs across histories')
                            for suffix in ('.npz', '_map.json'):
                                snapshot(directory / f'eval_{STEP}_{map_seed}{suffix}')
                            record = audit_map(directory, row, old_maps[map_seed], reshape)
                            record.update(arm=arm, training_seed=seed)
                            maps.append(record)
                            stream.write(json.dumps(record, allow_nan=False) + '\n')
                            stream.flush()
                            processed = sum(len(h['maps']) for h in histories) + len(maps)
                            say(f'MAP {processed}/72 arm={arm} seed={seed} map={map_seed} '
                                f'dwell={record["dwell_status"]} A={record.get("all_agents_ever_in_ball")} '
                                f'C={record.get("all_agents_ever_captured")} J={record.get("recomputed_joint_capture")} '
                                'reward_reference=UNAVAILABLE')
                            write_json(out / 'status.json', dict(phase='RUNNING', processed_maps=processed, total_maps=72))
                        agents = [a for m in maps if m['dwell_status'] == 'AVAILABLE' for a in m['agents']]
                        bands = Counter({name: sum(a['bands'][name]['frames'] for a in agents)
                                         for name in agents[0]['bands']}) if agents else Counter()
                        histories.append(dict(arm=arm, training_seed=seed, maps=maps,
                                              success_maps=sum(m['json_success'] for m in maps),
                                              all_ball_maps=sum(m.get('all_agents_ever_in_ball',
                                                                     m.get('saved_ever_ball_all')) for m in maps),
                                              all_capture_maps=(sum(m['all_agents_ever_captured'] for m in maps)
                                                  if all(m['dwell_status'] == 'AVAILABLE' for m in maps) else None),
                                              dwell_available_maps=sum(m['dwell_status'] == 'AVAILABLE' for m in maps),
                                              agent_frames=sum(bands.values()), band_agent_frames=dict(bands),
                                              near_stall_agent_frames=sum(a['near_stall']['frames'] for a in agents),
                                              agent_maps_with_25plus_run=sum(bool(a['near_stall_runs_25plus']) for a in agents)))
            for path, before in inputs.items():
                require(digest(Path(path)) == before, f'Input changed during audit: {path}')
            available = sum(m['dwell_status'] == 'AVAILABLE' for h in histories for m in h['maps'])
            summary = dict(histories=histories, total_maps=72, dwell_available_maps=available,
                           reward_formula_available_maps=available, reward_reference_available_maps=0,
                           reward_reference_validation='UNAVAILABLE', dt_s=DT, gamma=GAMMA,
                           scope='Post-hoc descriptive fixed development trajectories; no efficacy-rule change.',
                           denominators='Correlated agent-frames, agent-maps and repeated development maps; no independent CI.',
                           near_stall='4<=d<10 and speed<0.5; 25 sampled occupancy frames=5s, endpoint span=4.8s.')
            write_json(out / 'summary.json', summary)
            lines = ['# Fixed final-trajectory dwell audit', '', summary['scope'], '', summary['denominators'], '',
                     'Reward components are formula-derived. Logged evaluation-return reference: **UNAVAILABLE**.', '',
                     '| Arm | Seed | Agent-frames | Near-stall frames | Agent-maps with >=25-frame run |',
                     '|---|---:|---:|---:|---:|']
            lines += [f'| {h["arm"]} | {h["training_seed"]} | {h["agent_frames"]} | '
                      f'{h["near_stall_agent_frames"]} | {h["agent_maps_with_25plus_run"]} |' for h in histories]
            (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
            provenance['input_hashes_rechecked'] = True
            write_json(out / 'manifest.json', provenance)
            status = dict(phase='COMPLETE', complete=True, processed_maps=72, dwell_available_maps=available,
                          reward_reference_validation='UNAVAILABLE', missing_input_measurements=72 - available)
            write_json(out / 'status.json', status)
            say(f'COMPLETE maps=72 dwell_available={available} reward_reference=UNAVAILABLE out={out}')
            artifact_hashes = {name: digest(out / name) for name in
                               ('maps.jsonl', 'summary.json', 'summary.md', 'manifest.json', 'status.json', 'audit.log')}
            write_json(out / 'completion.json', dict(status, artifact_sha256=artifact_hashes))
            # Last artifact write; complete means processed, never return-reference validation.
        except Exception:
            error = traceback.format_exc()
            (out / 'error.txt').write_text(error, encoding='utf-8')
            write_json(out / 'error.json', dict(complete=False, traceback=error))
            write_json(out / 'status.json', dict(phase='FAILED', complete=False, error=error))
            say(error)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    # This check is outside the error handler: an existing output is never modified.
    out.mkdir(parents=True, exist_ok=False)
    try:
        audit(out)
    except Exception:
        if not (out / 'error.txt').exists():
            error = traceback.format_exc()
            (out / 'error.txt').write_text(error, encoding='utf-8')
            write_json(out / 'error.json', dict(complete=False, traceback=error))
            write_json(out / 'status.json', dict(phase='FAILED', complete=False, error=error))
            (out / 'audit.log').write_text(error, encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
