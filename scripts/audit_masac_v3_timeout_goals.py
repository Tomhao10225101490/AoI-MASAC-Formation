#!/usr/bin/env python3
"""Audit goal proximity and sampled velocity in final V3 timeout trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'results/masac_v3_paired_300k_20261103'
FLAGS = ('static_collision','boundary_collision','dynamic_collision','uav_collision')
DT = .2


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def summarize(cases):
    agents = [u for case in cases for u in case['uavs']]
    return {
        'timeout_episodes':len(cases), 'uav_trajectories':len(agents),
        'mean_episode_final_team_goal_distance_m':float(np.mean([np.mean([u['terminal_goal_distance_m'] for u in c['uavs']]) for c in cases])) if cases else None,
        'mean_uav_terminal_goal_distance_m':float(np.mean([u['terminal_goal_distance_m'] for u in agents])) if agents else None,
        'mean_uav_terminal_speed_mps':float(np.mean([u['terminal_speed_mps'] for u in agents])) if agents else None,
        'mean_uav_closest_recorded_goal_distance_m':float(np.mean([u['closest_recorded_goal_distance_m'] for u in agents])) if agents else None,
        'uav_terminal_speed_below_2mps':sum(u['terminal_speed_mps'] < 2 for u in agents),
        'uav_terminal_distance_below_4m':sum(u['terminal_goal_distance_m'] < 4 for u in agents),
        'uav_terminal_capture_count':sum(u['captured_at_terminal'] for u in agents),
        'uav_ever_capture_count':sum(u['captured_at_any_sample'] for u in agents),
        'episodes_any_uav_ever_captured':sum(any(u['captured_at_any_sample'] for u in c['uavs']) for c in cases),
        'episodes_all_uavs_simultaneously_captured':sum(c['max_simultaneous_capture_count']==3 for c in cases),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    completion = read(RUN/'completion.json')
    if not completion['complete'] or completion['runs'] != 6 or completion['smoke'] or (RUN/'error.json').exists():
        raise ValueError('Expected completed six-run paired batch without error')
    cfg = read(RUN/'protocol.json')
    collision_path = RUN/'final_collision_attribution_audit.json'
    collision = read(collision_path)
    cases_by_id = {(x['history'],x['training_seed'],x['map_seed']):x for x in collision['cases']}
    environment_hashes = read(RUN/'source_hashes.json')
    for name in ('structured.py','environment.py'):
        if digest(RUN/'source_snapshot'/name) != environment_hashes[name]:
            raise ValueError(f'Frozen environment snapshot mismatch: {name}')
    cases = []
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            directory = RUN/f'{arm}_{seed}'
            eval_path = directory/'evaluation_300000.json'
            rows = read(eval_path)
            if len(rows)!=6 or sorted(r['map_seed'] for r in rows)!=sorted(cfg['validation_map_seeds']):
                raise ValueError(f'Unexpected evaluation maps: {eval_path}')
            for row in rows:
                if row['success'] or any(row[k] for k in FLAGS):
                    continue
                if not np.isclose(row['time'],70.):
                    raise ValueError('Failed without collision at a non-horizon time')
                map_seed=row['map_seed']
                path=directory/f'eval_300000_{map_seed}.npz'
                prior=cases_by_id[(arm,seed,map_seed)]
                if prior['terminal_condition']!='timeout' or digest(path)!=prior['trajectory_sha256']:
                    raise ValueError(f'Collision attribution mismatch: {path}')
                with np.load(path) as z:
                    positions,goals=z['positions'],z['goals']
                if positions.shape!=(351,3,3) or goals.shape!=(3,3):
                    raise ValueError(f'Unexpected trajectory shape: {path}')
                distances=np.linalg.norm(positions-goals[None,:,:],axis=-1)
                speeds=np.vstack((np.zeros((1,3)),np.linalg.norm(np.diff(positions,axis=0),axis=-1)/DT))
                captures=(distances<4)&(speeds<2)
                if (not np.allclose(distances[-1],prior['terminal_goal_distance_m_by_uav']) or
                        not np.allclose(speeds[-1],prior['terminal_speed_mps_by_uav'])):
                    raise ValueError(f'Collision attribution distance/speed mismatch: {path}')
                uavs=[]
                for i in range(3):
                    idx=np.flatnonzero(captures[:,i])
                    nearest=int(np.argmin(distances[:,i]))
                    uavs.append(dict(uav_index=i,initial_position_m=positions[0,i].tolist(),
                        goal_m=goals[i].tolist(),terminal_position_m=positions[-1,i].tolist(),
                        terminal_goal_distance_m=float(distances[-1,i]),terminal_speed_mps=float(speeds[-1,i]),
                        closest_recorded_goal_distance_m=float(distances[nearest,i]),
                        closest_recorded_time_s=float(nearest*DT),
                        captured_at_any_sample=bool(len(idx)), captured_at_terminal=bool(captures[-1,i]),
                        capture_sample_count=int(len(idx)),
                        first_captured_time_s=float(idx[0]*DT) if len(idx) else None,
                        last_captured_time_s=float(idx[-1]*DT) if len(idx) else None))
                cases.append(dict(history=arm,training_seed=seed,map_seed=map_seed,layout=row['layout'],
                    time_s=float(row['time']),success=False,collision_flags={k:bool(row[k]) for k in FLAGS},
                    evaluation_json_sha256=digest(eval_path),trajectory_npz_sha256=digest(path),
                    max_simultaneous_capture_count=int(captures.sum(axis=1).max()),uavs=uavs))
    result=dict(run=str(RUN), completion_sha256=digest(RUN/'completion.json'),
        paired_protocol_sha256=digest(RUN/'protocol.json'),
        collision_attribution_sha256=digest(collision_path),
        audit_script_sha256=digest(Path(__file__)),
        frozen_environment_source_sha256={name:environment_hashes[name] for name in ('structured.py','environment.py')},
        methodology=['Only no-collision no-success final evaluation episodes ending at 350 steps (70s) are timeouts.',
            'Goal distance uses recorded UAV and goal positions; sampled speed is norm(p[t]-p[t-1])/0.2, with zero initial speed.',
            'Capture is distance <4m and speed <2m/s at a sampled state; sample counts from a trajectory are correlated.',
            'These 18 episodes use six repeated development maps and three independent training seeds per history.'],
        by_history={arm:summarize([c for c in cases if c['history']==arm]) for arm in cfg['arms']},
        by_history_and_seed={f'{arm}_{seed}':summarize([c for c in cases if c['history']==arm and c['training_seed']==seed])
                             for arm in cfg['arms'] for seed in cfg['seeds']},
        cases=cases)
    with args.out.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    print(f'Wrote {args.out}')
    for k,v in result['by_history'].items():
        print(k,json.dumps(v,ensure_ascii=False))


if __name__=='__main__':
    main()
