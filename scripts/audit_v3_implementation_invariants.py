#!/usr/bin/env python3
"""Read-only numerical checks of V3 semantics and saved timeout trajectories."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_planning.environment import swept_distance  # noqa: E402
from uav_planning.shared_masac import SharedMASAC  # noqa: E402
from uav_planning.structured import StructuredMission, box_distance, limited, segment_box_hit  # noqa: E402

FLAGS = ('static_collision', 'boundary_collision', 'uav_collision', 'dynamic_collision')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def semantics():
    torch.set_num_threads(1)
    torch.manual_seed(20261103)
    lo, hi = np.array([[0., 0., 0.]]), np.array([[1., 1., 1.]])
    point = np.array([[-.9, -.9, .5]])
    dist = float(box_distance(point, lo, hi)[0, 0])
    hit = bool(segment_box_hit(point, point, lo - 1., hi + 1.)[0, 0])
    if not (dist > 1. and hit):
        raise AssertionError('Corner counterexample no longer holds')

    z = torch.tensor([-4., -1., 0., 1., 4.], dtype=torch.float64)
    stable = 2*(math.log(2)-z-F.softplus(-2*z))
    direct = torch.log1p(-torch.tanh(z)**2)
    tanh_error = float(torch.max(torch.abs(stable-direct)))
    if tanh_error > 1e-10:
        raise AssertionError('Tanh Jacobian identity mismatch')
    a, b, v0 = np.array([[.8,.8,0.]]), np.array([[.9,.9,0.]]), np.zeros((1,3))
    proj_error = float(np.max(np.abs(limited(v0,a,.2,4.,10.)-limited(v0,b,.2,4.,10.))))
    if proj_error > 1e-12:
        raise AssertionError('Expected many-to-one radial map')

    env = StructuredMission()
    obs = env.reset(800001)
    env.t = 349
    no_clock = bool(np.array_equal(obs, env.observe()))
    if not no_clock:
        raise AssertionError('Observation unexpectedly depends on t')
    clip_horizon = StructuredMission(horizon=1)
    observation = clip_horizon.reset(800001)
    nxt, reward, terminated, truncated, _ = clip_horizon.step(np.zeros((3,3)))
    if terminated or not truncated:
        raise AssertionError('Expected pure horizon truncation')
    model = SharedMASAC(capacity=4)
    model.push(observation, np.zeros((3,3),np.float32), reward, nxt, terminated)
    if model.done[0] != 0.:
        raise AssertionError('Pure horizon truncation unexpectedly masks bootstrap')

    obs_batch = torch.as_tensor(np.stack([obs, obs]))
    model.q.requires_grad_(False)
    actions, logp = model.actor(obs_batch)
    qs = [q(obs_batch, actions) for q in model.q]
    objective = -torch.minimum(*qs).mean()
    grads = torch.autograd.grad(objective, tuple(model.actor.parameters()), allow_unused=True)
    actor_q_grad = float(torch.sqrt(sum(torch.sum(g*g) for g in grads if g is not None)))
    if not math.isfinite(actor_q_grad) or actor_q_grad <= 0:
        raise AssertionError('Actor lacks Q-to-action gradient')
    q1param, q2param = next(model.q[0].parameters()), next(model.q[1].parameters())
    targetparam = next(model.target[0].parameters())
    independent = (q1param.data_ptr()!=q2param.data_ptr() and
                   q1param.data_ptr()!=targetparam.data_ptr())
    if not independent or any(p.requires_grad for p in model.target.parameters()):
        raise AssertionError('Critic/target alias or target requires grad')

    goals = StructuredMission()
    goals.reset(800001)
    goals.goal[0] = goals.p[0] + [0.,9.,0.]
    goals.p[0,1] = goals.goal[0,1]-4.1
    goals.v[0] = [0.,1.,0.]
    captures, rewards, distances, velocities = [], [], [], []
    for step in range(4):
        control = np.zeros((3,3))
        if step: control[0,1] = -1.
        _, r, done, trunc, info = goals.step(control)
        if done or trunc or any(info[k] for k in FLAGS):
            raise AssertionError('Unexpected terminal event in capture/exit check')
        captures.append(bool(goals.captured[0]))
        rewards.append(float(r))
        distances.append(float(np.linalg.norm(goals.goal[0]-goals.p[0])))
        velocities.append(float(np.linalg.norm(goals.v[0])))
    if captures != [True,True,True,False]:
        raise AssertionError(f'Unexpected instantaneous capture sequence: {captures}')
    crossing = float(swept_distance(np.array([[-2.,0.,0.]]), np.array([[2.,0.,0.]]))[0])
    if crossing > 1e-8:
        raise AssertionError('Swept relative segment missed crossing')
    return dict(corner=dict(box_distance=dist, radius=1., inflated_aabb_collision=hit),
        tanh_jacobian_max_abs_error=tanh_error, radial_many_to_one_max_abs_delta=proj_error,
        identical_observation_when_only_time_changed=no_clock,
        pure_horizon=dict(terminated=bool(terminated),truncated=bool(truncated),stored_done=float(model.done[0])),
        actor_q_gradient_norm=actor_q_grad, critic_and_target_storage_distinct=independent,
        capture_enter_exit=dict(capture_sequence=captures,goal_distance_m=distances,speed_mps=velocities,
            total_team_rewards=rewards,discounted_capture_component_only=(5./3.)*(1.-.99**3),
            note='Constructed valid discrete motion, not a learned rollout; the positive discounted capture component omits all other rewards.'),
        swept_crossing_min_distance=crossing)


def low_speed_timeouts(root):
    result = {}
    for arm in ('curriculum','original'):
        total=near_any=near_80=small_displacement=0
        sources=[]
        for seed in (411,422,433):
            directory = root/f'{arm}_{seed}'
            eval_path=directory/'evaluation_300000.json'
            for row in json.loads(eval_path.read_text()):
                if row['success'] or any(row[k] for k in FLAGS):
                    continue
                if not np.isclose(row['time'],70.):
                    raise ValueError('Noncollision failure is not a 70-second timeout')
                file=directory/f'eval_300000_{row["map_seed"]}.npz'
                with np.load(file,allow_pickle=False) as data:
                    p,g=data['positions'],data['goals']
                    velocity=np.linalg.norm(np.diff(p,axis=0),axis=-1)/.2
                    distance=np.linalg.norm(p[1:]-g,axis=-1)
                    if velocity.shape != (350,3):
                        raise ValueError('Unexpected timeout trajectory shape')
                    fraction=((distance[-50:]<30.)&(velocity[-50:]<2.)).mean(axis=0)
                    net_displacement=np.linalg.norm(p[-1]-p[-51],axis=-1)
                total+=3
                near_any+=int((fraction>0).sum())
                near_80+=int((fraction>=.8).sum())
                small_displacement+=int((net_displacement<5.).sum())
                sources.append(dict(seed=seed,map_seed=row['map_seed'],trajectory_sha256=sha(file)))
        result[arm]=dict(timeout_uav_trajectories=total,with_any_near_low_speed_last_10s=near_any,
            with_near_low_speed_at_least_40_of_last_50_steps=near_80,
            with_net_displacement_lt5m_last_10s=small_displacement,sources=sources)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    root=args.run.resolve()
    if not json.loads((root/'completion.json').read_text()).get('complete') or (root/'error.json').exists():
        raise ValueError('A completed error-free V3 batch is required')
    manifest=json.loads((root/'source_hashes.json').read_text())
    hashes={}
    for name in ('structured.py','shared_masac.py','environment.py'):
        live=ROOT/'uav_planning'/name
        snap=root/'source_snapshot'/name
        if sha(live)!=manifest[name] or sha(snap)!=manifest[name]:
            raise ValueError(f'Frozen source mismatch: {name}')
        hashes[name]=manifest[name]
    result=dict(source_sha256=hashes,script_sha256=sha(Path(__file__)),
        completion_sha256=sha(root/'completion.json'),checks=semantics(),
        timeout_low_speed=low_speed_timeouts(root),
        timeout_criterion='Exploratory posthoc: during last 50 steps (10 s), per-UAV distance<30 m and speed<2 m/s for at least 40 steps; correlated steps and fixed development maps, not independent tests or proof of causation.',
        caveat='These are numerical code invariants and descriptive samples; they do not establish general policy effectiveness or safety.')
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(f'Wrote {args.out}')


if __name__=='__main__': main()
