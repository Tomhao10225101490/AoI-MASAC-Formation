#!/usr/bin/env python3
"""Replay 300k evaluation trajectories under V3's recorded collision geometry."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_planning.environment import swept_distance
from uav_planning.structured import segment_box_hit

FLAGS = ('static_collision', 'boundary_collision', 'dynamic_collision', 'uav_collision')
AXES = ('x', 'y', 'z')
DT = .2


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def slab_entry(a, b, lo, hi):
    """Segment entry time and contacted faces of a (radius-inflated) box."""
    delta = b-a
    near = np.full(3, -np.inf)
    far = np.full(3, np.inf)
    for k in range(3):
        if abs(delta[k]) < 1e-12:
            if a[k] < lo[k] or a[k] > hi[k]:
                return None
        else:
            t0, t1 = (lo[k]-a[k])/delta[k], (hi[k]-a[k])/delta[k]
            near[k], far[k] = min(t0,t1), max(t0,t1)
    entry, leave = max(0., float(np.max(near))), min(1., float(np.min(far)))
    if entry > leave + 1e-12:
        return None
    faces = [f'{AXES[k]}_{"min" if delta[k] > 0 else "max"}'
             for k in range(3) if abs(near[k]-entry) < 1e-8]
    return entry, faces


def sphere_entry(r0, r1, radius):
    """First contact on the segment, when V3's strict swept check is true."""
    delta = r1-r0
    a, b, c = float(delta@delta), float(2*r0@delta), float(r0@r0-radius*radius)
    if c <= 0:
        return 0.
    disc = b*b-4*a*c
    if a == 0 or disc < 0:
        raise ValueError('Swept collision has no sphere entry')
    root = (-b-np.sqrt(max(0.,disc)))/(2*a)
    if not -1e-7 <= root <= 1+1e-7:
        raise ValueError('Sphere entry outside swept segment')
    return float(np.clip(root, 0., 1.))


def masks(old, new, dynamic_old, dynamic_new, boxes, dr, radius, world):
    static = segment_box_hit(old, new, boxes[:,0]-radius, boxes[:,1]+radius)
    boundary_low = new < radius
    boundary_high = new > world-radius
    peer = swept_distance(old[:,None]-old[None], new[:,None]-new[None]) < 2*radius
    np.fill_diagonal(peer, False)
    moving = swept_distance(old[:,None]-dynamic_old[None], new[:,None]-dynamic_new[None]) < dr[None]+radius
    return static, boundary_low, boundary_high, peer, moving


def terminal_events(old, new, dyn0, dyn1, boxes, dr, radius, world, goals, step):
    static, low, high, peer, moving = masks(old, new, dyn0, dyn1, boxes, dr, radius, world)
    velocities = (new-old)/DT
    events = []

    def add(kind, uav, tau, **extra):
        pos = old[uav] + tau*(new[uav]-old[uav])
        events.append(dict(kind=kind, uav_index=int(uav), tau=float(tau),
            contact_time_s=float((step-1+tau)*DT), interpolated_contact_position_m=pos.tolist(),
            tick_velocity_mps=velocities[uav].tolist(), tick_speed_mps=float(np.linalg.norm(velocities[uav])),
            goal_distance_at_contact_m=float(np.linalg.norm(goals[uav]-pos)), **extra))

    for i, j in np.argwhere(static):
        hit = slab_entry(old[i], new[i], boxes[j,0]-radius, boxes[j,1]+radius)
        if hit is None:
            raise ValueError('V3 static flag disagrees with slab entry')
        tau, faces = hit
        add('static_collision', i, tau, building_index=int(j), inflated_box_faces=faces,
            building_bounds_m=boxes[j].tolist())
    for side, mask in (('min',low),('max',high)):
        for i,k in np.argwhere(mask):
            plane = radius if side == 'min' else world[k]-radius
            tau = (plane-old[i,k])/(new[i,k]-old[i,k])
            add('boundary_collision', i, tau, boundary_face=f'{AXES[k]}_{side}')
    for i,j in np.argwhere(np.triu(peer,1)):
        tau = sphere_entry(old[i]-old[j], new[i]-new[j], 2*radius)
        add('uav_collision', i, tau, other_uav_index=int(j),
            other_interpolated_contact_position_m=(old[j]+tau*(new[j]-old[j])).tolist(),
            other_tick_speed_mps=float(np.linalg.norm(velocities[j])))
    for i,j in np.argwhere(moving):
        tau = sphere_entry(old[i]-dyn0[j], new[i]-dyn1[j], dr[j]+radius)
        add('dynamic_collision', i, tau, dynamic_obstacle_index=int(j),
            obstacle_interpolated_position_m=(dyn0[j]+tau*(dyn1[j]-dyn0[j])).tolist())
    events.sort(key=lambda item: item['tau'])
    return events, static, low, high, peer, moving


def audit_case(arm, seed, row, directory):
    map_seed = row['map_seed']
    path = directory / f'eval_300000_{map_seed}.npz'
    map_path = directory / f'eval_300000_{map_seed}_map.json'
    meta = load(map_path)
    with np.load(path) as z:
        positions, dynamic, goals = z['positions'], z['dynamic'], z['goals']
        boxes, dr = z['buildings'], z['dynamic_radii']
    world, radius = np.array(meta['world']), float(meta['radius'])
    if (positions.shape[1:] != (3,3) or dynamic.shape[1:] != (4,3)
            or not np.allclose(positions[0],meta['starts']) or not np.allclose(goals,meta['goals'])
            or not np.allclose(boxes,meta['buildings']) or not np.allclose(dynamic[0],meta['dynamic'])
            or not np.isclose((len(positions)-1)*DT,row['time'])):
        raise ValueError(f'Trajectory/metadata mismatch: {path}')
    if len(positions) != len(dynamic) or len(positions) < 2:
        raise ValueError(f'Missing recorded transition: {path}')
    # The environment terminates immediately on a collision. Validate all
    # previous ticks as well as the final tick against the saved flags.
    for k in range(1,len(positions)-1):
        ms = masks(positions[k-1],positions[k],dynamic[k-1],dynamic[k],boxes,dr,radius,world)
        if any(np.any(a) for a in ms):
            raise ValueError(f'Earlier collision than saved terminal event: {path}, step {k}')
    step = len(positions)-1
    events, static, low, high, peer, moving = terminal_events(
        positions[-2],positions[-1],dynamic[-2],dynamic[-1],boxes,dr,radius,world,goals,step)
    observed = {'static_collision':bool(np.any(static)), 'boundary_collision':bool(np.any(low)|np.any(high)),
                'dynamic_collision':bool(np.any(moving)), 'uav_collision':bool(np.any(peer))}
    if any(observed[k] != bool(row[k]) for k in FLAGS):
        raise ValueError(f'Terminal collision flags disagree with saved result: {path}')
    if row['success'] and events:
        raise ValueError('Collision and success cannot both be true')
    if not row['success'] and not events and not np.isclose(row['time'],70.):
        raise ValueError(f'Unidentified terminal condition: {path}')
    distances = np.linalg.norm(positions-goals[None],axis=-1)
    velocity = np.diff(positions,axis=0)/DT
    speed = np.vstack((np.zeros((1,3)),np.linalg.norm(velocity,axis=-1)))
    capture = (distances < 4)&(speed < 2)
    first = [] if not events else [e for e in events if abs(e['tau']-events[0]['tau']) < 1e-8]
    return dict(history=arm, training_seed=seed, map_seed=map_seed, layout=row['layout'],
        trajectory_sha256=digest(path), map_metadata_sha256=digest(map_path),
        initial_positions_m=positions[0].tolist(), goals_m=goals.tolist(),
        episode_time_s=float(row['time']), episode_steps=step, success=bool(row['success']),
        terminal_condition='success' if row['success'] else ('collision' if events else 'timeout'),
        collision_flags=observed, closest_goal_distance_m_by_uav=distances.min(axis=0).tolist(),
        terminal_goal_distance_m_by_uav=distances[-1].tolist(),
        terminal_speed_mps_by_uav=speed[-1].tolist(),
        captured_at_any_sample_by_uav=np.any(capture,axis=0).tolist(),
        all_final_tick_collision_events=events, earliest_interpolated_event_ties=first)


def aggregate(cases):
    face, building_face, first_kind, first_uav = Counter(), Counter(), Counter(), Counter()
    speeds, goal_distances = {}, {}
    for case in cases:
        for e in case['all_final_tick_collision_events']:
            if 'boundary_face' in e:
                face[e['boundary_face']]+=1
            for f in e.get('inflated_box_faces',[]):
                building_face[f]+=1
        for e in case['earliest_interpolated_event_ties']:
            first_kind[e['kind']]+=1
            first_uav[str(e['uav_index'])]+=1
            if 'other_uav_index' in e:
                first_uav[str(e['other_uav_index'])]+=1
            speeds.setdefault(e['kind'],[]).append(e['tick_speed_mps'])
            goal_distances.setdefault(e['kind'],[]).append(e['goal_distance_at_contact_m'])
    return dict(episodes=len(cases), timeout_episodes=sum(c['terminal_condition']=='timeout' for c in cases),
        success_episodes=sum(c['success'] for c in cases),
        collision_flag_counts={k:sum(c['collision_flags'][k] for c in cases) for k in FLAGS},
        boundary_face_events=dict(face), inflated_building_face_events=dict(building_face),
        first_contact_event_counts=dict(first_kind), first_contact_uav_counts=dict(first_uav),
        first_contact_speed_mps={k:{'n':len(v),'mean':float(np.mean(v)),'min':float(min(v)),'max':float(max(v))} for k,v in speeds.items()},
        first_contact_goal_distance_m={k:{'n':len(v),'mean':float(np.mean(v)),'min':float(min(v)),'max':float(max(v))} for k,v in goal_distances.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    root = ROOT/'results/masac_v3_paired_300k_20261103'
    if (root/'error.json').exists() or not load(root/'completion.json')['complete']:
        raise ValueError('Batch is not complete and error-free')
    cfg, hashes = load(root/'protocol.json'), load(root/'source_hashes.json')
    for name, live in [('structured.py',ROOT/'uav_planning/structured.py'),('environment.py',ROOT/'uav_planning/environment.py')]:
        if digest(live) != hashes[name] or digest(root/'source_snapshot'/name) != hashes[name]:
            raise ValueError(f'Environment source mismatch: {name}')
    cases = []
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            directory = root/f'{arm}_{seed}'
            rows = load(directory/'evaluation_300000.json')
            if sorted(r['map_seed'] for r in rows) != sorted(cfg['validation_map_seeds']):
                raise ValueError(f'Map seeds mismatch: {directory}')
            cases.extend(audit_case(arm,seed,row,directory) for row in rows)
    out = dict(run=str(root), run_completion_sha256=digest(root/'completion.json'),
        audit_script_sha256=digest(Path(__file__)),
        methodology=['Replay every 0.2s position segment using frozen V3 swept collision predicates.',
            'V3 static collision uses axis-aligned building boxes inflated 1m on each axis; faces are of inflated boxes and conservative at corners.',
            'Boundary collision is reported when a final tick endpoint passes the 1m inset; reported position interpolates the plane crossing.',
            'Peer/dynamic first contacts use straight-line interpolation and V3 radii. Multiple flags may coexist in one final tick.',
            'Earliest-event order is an interpolation within the final tick; the environment reports the whole tick and does not itself select a first cause.',
            'UAV indices are zero-based; collision-event and UAV counts may exceed episode counts when contacts overlap.',
            'Speed is endpoint displacement divided by 0.2 seconds, matching V3 constant velocity within each step.',
            'Goal minima use recorded tick positions only; maps are repeated development evaluations, not independent training seeds.'],
        by_history_and_seed={f'{arm}_{seed}':aggregate([c for c in cases if c['history']==arm and c['training_seed']==seed])
            for arm in cfg['arms'] for seed in cfg['seeds']},
        by_history={arm:aggregate([c for c in cases if c['history']==arm]) for arm in cfg['arms']}, cases=cases)
    with args.out.open('x') as f:
        json.dump(out,f,indent=2,allow_nan=False)
        f.write('\n')
    print(f'Wrote {args.out}')
    for key, val in out['by_history_and_seed'].items():
        print(key,json.dumps(val,ensure_ascii=False))


if __name__ == '__main__':
    main()
