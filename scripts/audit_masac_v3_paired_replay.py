#!/usr/bin/env python3
"""Audit aligned paired V3 snapshots and describe correlated replay states."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/masac_v3_paired_300k_20261103'
CAPACITY = 100000
EXPECTED = {'obs': (CAPACITY, 3, 162), 'next_obs': (CAPACITY, 3, 162),
            'actions': (CAPACITY, 3, 3), 'rewards': (CAPACITY,), 'done': (CAPACITY,)}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def inspect_npz_headers(path):
    shapes = {}
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {name + '.npy' for name in EXPECTED}:
            raise ValueError(f'Unexpected replay arrays in {path}')
        for name, expected in EXPECTED.items():
            with archive.open(name + '.npy') as source:
                version = np.lib.format.read_magic(source)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(source)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(source)
                else:
                    raise ValueError(f'Unsupported npy header {version} in {path}')
                if shape != expected or dtype != np.float32:
                    raise ValueError(f'Wrong replay shape or dtype: {path} {name}: {shape} {dtype}')
                shapes[name] = list(shape)
    return shapes


def counts(obs):
    dist = np.linalg.norm(obs[..., :3] * 240., axis=-1)
    speed = np.linalg.norm(obs[..., 3:6] * 10., axis=-1)
    return dict(transitions=int(len(obs)), agent_states=int(dist.size),
                minimum_distance_m=float(dist.min()),
                distance_percentiles_m=dict(zip(('p01','p10','p50','p90'),
                                                 np.percentile(dist, [1,10,50,90]).tolist())),
                agent_states_below_distance_m={str(d): int(np.count_nonzero(dist < d))
                                               for d in (4, 10, 30, 50, 100)},
                agent_states_with_goal_capture=int(np.count_nonzero((dist < 4) & (speed < 2))))


def analyze(out):
    completion = read_json(RUN / 'completion.json')
    if not completion.get('complete') or completion.get('runs') != 6 or completion.get('smoke') or (RUN/'error.json').exists():
        raise ValueError('Full six-run batch is not complete and error-free')
    proto = read_json(RUN / 'protocol.json')
    if (proto['arms'] != ['curriculum', 'original'] or proto['seeds'] != [411,422,433] or
            proto['start_step'] != 100000 or proto['steps'] != 300000 or
            proto['replay_capacity'] != CAPACITY):
        raise ValueError('Unexpected paired protocol')
    parents = read_json(RUN / 'parent_provenance.json')
    prior = read_json(ROOT/'results/masac_v3_goal_curriculum_20261102/goal_coverage_by_stage.json')
    split = {row['seed']:row['short_end_step'] for row in prior['rows']}
    rows = []
    for arm in proto['arms']:
        for seed in proto['seeds']:
            parent = Path(parents[arm]['path']) / f'standard_{seed}'
            parent_cp = parent/'checkpoint_100000.pt'
            parent_replay = parent/'replay_final.npz'
            evidence = parents[arm]['seeds'][str(seed)]
            if digest(parent_cp) != evidence['checkpoint_sha256'] or digest(parent_replay) != evidence['replay_sha256']:
                raise ValueError(f'Parent differs from recorded provenance: {arm}/{seed}')
            parent_state = torch.load(parent_cp, map_location='cpu', weights_only=False)
            if (parent_state['step'] != 100000 or parent_state['seed'] != seed or
                    parent_state['replay_size'] != CAPACITY or parent_state['replay_index'] != 0):
                raise ValueError(f'Parent checkpoint alignment error: {arm}/{seed}')
            inspect_npz_headers(parent_replay)
            with np.load(parent_replay) as replay:
                obs = replay['obs']
                parent_stats = counts(obs)
                if arm == 'curriculum':
                    parent_stats['short_edge'] = counts(obs[:split[seed]])
                    parent_stats['full_task'] = counts(obs[split[seed]:])
                else:
                    parent_stats['full_task'] = counts(obs)
            checks = []
            samples = {}
            for step in (150000,200000,250000,300000):
                snap = RUN/f'{arm}_{seed}'/'snapshots'/str(step)
                manifest = read_json(snap/'complete.json')
                ckpt = snap/'state'/f'checkpoint_{step}.pt'
                replay_path = snap/'state'/'replay_final.npz'
                config = snap/'protocol.json'
                if (not manifest['complete'] or manifest['step'] != step or
                        manifest['arm'] != arm or manifest['seed'] != seed or
                        digest(ckpt) != manifest['checkpoint_sha256'] or
                        digest(replay_path) != manifest['replay_sha256'] or
                        digest(config) != manifest['protocol_sha256']):
                    raise ValueError(f'Snapshot manifest mismatch: {snap}')
                state = torch.load(ckpt, map_location='cpu', weights_only=False)
                if (state['step'] != step or state['seed'] != seed or
                        state['history_arm'] != arm or not state['exact_resume'] or
                        state['replay_size'] != CAPACITY or state['replay_index'] != step % CAPACITY):
                    raise ValueError(f'Snapshot replay alignment error: {snap}')
                if arm == 'curriculum' and state.get('curriculum_stage') != 'full':
                    raise ValueError(f'Unexpected curriculum stage: {snap}')
                shapes = inspect_npz_headers(replay_path)
                checks.append(dict(step=step, replay_index=state['replay_index'],
                                   checkpoint_sha256=manifest['checkpoint_sha256'],
                                   replay_sha256=manifest['replay_sha256'],
                                   protocol_sha256=manifest['protocol_sha256'], shapes=shapes))
                if step in (200000,300000):
                    with np.load(replay_path) as replay:
                        obs = replay['obs']
                        samples[str(step)] = counts(obs)
            rows.append(dict(arm=arm, seed=seed, parent_replay_sha256=evidence['replay_sha256'],
                             parent_100000=parent_stats, snapshot_integrity=checks,
                             full_task_200000_and_300000=samples))
            print(f'checked {arm}/{seed}', flush=True)
    result = dict(batch=str(RUN), source_sha256=digest(Path(__file__)),
                  meaning='Each replay transition stores three pre-action agent states. States from the same trajectories are correlated; counts are not independent episodes or team successes.',
                  fifo='Parents have size 100000 and write index 0. Curriculum short stage ends around 30k, so those earliest slots are overwritten within about 30k additional full-task steps. At 200k and 300k replay contains the latest 100k full-task transitions; not the same distribution as the parent curriculum buffer.',
                  limitations='Capture uses distance <4 m and speed <2 m/s on stored pre-action observations. It can exclude terminal post-action capture states and does not require all agents to arrive simultaneously.',
                  rows=rows)
    with out.open('x') as dest:
        json.dump(result, dest, indent=2, allow_nan=False)
        dest.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite audit')
    analyze(args.out)
