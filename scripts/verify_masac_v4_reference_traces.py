#!/usr/bin/env python3
"""Independently replay saved V4 reference actions through the mission."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from uav_planning.v4_environment import V4Mission
from uav_planning.v4_maps import SceneConfig


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(root, out):
    audit_path = root/'map_audit.json'
    audit = json.loads(audit_path.read_text())
    for name, expected in audit['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Source changed since the map audit: {name}')
    checked = []
    for row in audit['records']:
        if 'generation_or_audit_error' in row:
            checked.append(dict(level=row['level'], family=row['family'], seed=row['seed'],
                                skipped_reason=row['generation_or_audit_error']))
            continue
        scene_path = root/f'{row["level"]}_{row["family"]}_{row["seed"]}_scene.json'
        scene = json.loads(scene_path.read_text())
        config = SceneConfig(**scene['config']).validate()
        for index, expected in enumerate(row['attempts']):
            trace_path = root/expected['trajectory_file']
            if sha(trace_path) != expected['trajectory_sha256']:
                raise ValueError(f'Trajectory SHA mismatch: {trace_path}')
            with np.load(trace_path) as z:
                positions, velocities, dynamic, actions = (z[k] for k in ('positions','velocities','dynamic','actions'))
            if (len(positions) != len(actions)+1 or len(velocities) != len(positions)
                    or len(dynamic) != len(positions) or len(actions) > config.horizon):
                raise ValueError(f'Trajectory lengths are invalid: {trace_path}')
            env = V4Mission(config)
            env.reset(row['seed'])
            if env.scene.scene_hash != row['scene_sha256']:
                raise ValueError(f'Scene hash differs: {trace_path}')
            error = max(float(np.max(np.abs(env.p-positions[0]))),
                        float(np.max(np.abs(env.v-velocities[0]))))
            if len(env.dynamic):
                error = max(error,float(np.max(np.abs(env.dynamic-dynamic[0]))))
            max_acc = max_speed = 0.
            for t, action in enumerate(actions):
                if action.shape != (config.n_agents,3) or not np.isfinite(action).all() or np.linalg.norm(action,axis=1).max() > 1+1e-9:
                    raise ValueError(f'Invalid constrained action: {trace_path} step {t}')
                old_v = env.v.copy()
                _, _, done, trunc, info = env.step(action)
                error = max(error,float(np.max(np.abs(env.p-positions[t+1]))),
                            float(np.max(np.abs(env.v-velocities[t+1]))))
                if len(env.dynamic):
                    error = max(error,float(np.max(np.abs(env.dynamic-dynamic[t+1]))))
                max_speed = max(max_speed,float(np.linalg.norm(env.v,axis=1).max()))
                max_acc = max(max_acc,float(np.linalg.norm((env.v-old_v)/env.dt,axis=1).max()))
                if (done or trunc) and t != len(actions)-1:
                    raise ValueError(f'Trajectory continues after termination: {trace_path}')
            if not len(actions):
                raise ValueError(f'Empty reference trajectory: {trace_path}')
            flags = ('success','static_collision','dynamic_collision','uav_collision','boundary_collision')
            if (error > 1e-9 or env.t != expected['steps'] or
                    any(bool(info[k]) != bool(expected['final_info'][k]) for k in flags) or
                    bool(info['success']) != bool(expected['success']) or
                    max_speed > config.vmax+1e-8 or max_acc > config.amax+1e-8):
                raise ValueError(f'Replay verification failed: {trace_path}, position/velocity/dynamic error {error}')
            checked.append(dict(level=row['level'],family=row['family'],seed=row['seed'],attempt=index+1,
                                success=bool(info['success']), steps=env.t,
                                max_state_error=error, max_speed_mps=max_speed, max_acceleration_mps2=max_acc,
                                trajectory_file=trace_path.name, trajectory_sha256=expected['trajectory_sha256']))
    summary = dict(attempts_replayed=len([x for x in checked if 'attempt' in x]),
                   successful_replayed=sum(bool(x.get('success')) for x in checked),
                   generation_failures=sum('skipped_reason' in x for x in checked),
                   max_state_error=max((x['max_state_error'] for x in checked if 'attempt' in x),default=0.))
    with out.open('x') as dest:
        json.dump(dict(map_audit_sha256=sha(audit_path), verifier_sha256=sha(Path(__file__)),
                       summary=summary, records=checked,
                       limits='Replay verifies saved executions under this simulator, not real-world safety or policy learning.',
                       completed_at=time.time()),dest,indent=2,allow_nan=False)
        dest.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite verification')
    verify(args.run,args.out)
