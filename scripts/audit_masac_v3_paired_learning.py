#!/usr/bin/env python3
"""Read-only audit of sampled gradient clipping and saved evaluation rewards.

The evaluation decomposition reconstructs rewards from recorded positions;
it does not reconstruct the SAC entropy term or the training objective.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_planning.structured import StructuredMission  # noqa: E402

FLAGS = ('static_collision', 'boundary_collision', 'uav_collision', 'dynamic_collision')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def reward_components(positions, goals, info, dt=.2):
    """Per-step team reward components; all evaluation resets have zero velocity."""
    p = np.asarray(positions, dtype=np.float64)
    g = np.asarray(goals, dtype=np.float64)
    v = np.diff(p, axis=0) / dt
    prev_v = np.concatenate([np.zeros_like(v[:1]), v[:-1]], axis=0)
    acc = (v - prev_v) / dt
    d = np.linalg.norm(p - g[None], axis=-1)
    arrived = (d[1:] < 4.) & (np.linalg.norm(v, axis=-1) < 2.)
    prev_arrived = np.concatenate([np.zeros_like(arrived[:1]), arrived[:-1]], axis=0)
    terms = {
        'goal_progress': ((d[:-1] - d[1:]) / 2).mean(axis=1),
        'time_cost': np.full(len(v), -.02),
        'acceleration_cost': -.002 * np.sum(acc * acc, axis=-1).mean(axis=1),
        'capture_change': 5 * (arrived.astype(float) - prev_arrived.astype(float)).mean(axis=1),
        'collision': np.zeros(len(v)),
        'team_success': np.zeros(len(v)),
    }
    terms['collision'][-1] = -20. if any(info[k] for k in FLAGS) else 0.
    terms['team_success'][-1] = 20. if info['success'] else 0.
    return terms


def synthetic_reward_check():
    """Compare reconstruction against the actual environment per-step rewards."""
    checks = []
    for action in (np.zeros((3, 3)), np.tile([-1., 0., 0.], (3, 1))):
        env = StructuredMission(layout='urban', horizon=40)
        env.reset(910501)
        positions, rewards = [env.p.copy()], []
        for _ in range(env.horizon):
            _, r, done, trunc, info = env.step(action)
            rewards.append(r)
            positions.append(env.p.copy())
            if done or trunc:
                break
        terms = reward_components(positions, env.goal, info)
        reconstructed = sum(terms.values())
        difference = float(np.max(np.abs(reconstructed - rewards)))
        if difference > 1e-8:
            raise AssertionError(f'Per-step reward reconstruction mismatch: {difference}')
        checks.append(dict(steps=len(rewards), collision=bool(any(info[k] for k in FLAGS)),
                           max_abs_difference=difference))
    return checks


def summary(values):
    arr = np.asarray(values, dtype=np.float64)
    if not len(arr) or not np.isfinite(arr).all():
        raise ValueError('Empty or nonfinite progress samples')
    return dict(n=int(len(arr)), minimum=float(arr.min()), median=float(np.median(arr)),
                p90=float(np.percentile(arr, 90)), p99=float(np.percentile(arr, 99)),
                maximum=float(arr.max()), count_above_5=int(np.count_nonzero(arr > 5.)))


def run(root):
    marker = json.loads((root / 'completion.json').read_text())
    if not marker.get('complete') or marker.get('runs') != 6 or (root / 'error.json').exists():
        raise ValueError('Expected an error-free completed six-run batch')
    manifest = json.loads((root / 'source_hashes.json').read_text())
    source_hashes = {}
    for name in ('structured.py', 'shared_masac.py'):
        file = root / 'source_snapshot' / name
        source_hashes[name] = sha(file)
        if source_hashes[name] != manifest[name] or sha(ROOT / 'uav_planning' / name) != manifest[name]:
            raise ValueError(f'Source snapshot or live source changed: {name}')
    qsource = (root / 'source_snapshot' / 'shared_masac.py').read_text()
    if 'clip_grad_norm_(self.q.parameters(),5.)' not in qsource or 'clip_grad_norm_(self.actor.parameters(),5.)' not in qsource:
        raise ValueError('Expected gradient clipping implementation not found')
    rows = []
    for arm in ('curriculum', 'original'):
        for seed in (411, 422, 433):
            folder = root / f'{arm}_{seed}'
            log = folder / 'training.jsonl'
            progress, episodes = [], []
            with log.open() as src:
                for line in src:
                    row = json.loads(line)
                    if row.get('kind') == 'progress': progress.append(row)
                    if row.get('kind') == 'episode': episodes.append(row)
            if [x['step'] for x in progress] != list(range(100100, 300001, 100)):
                raise ValueError(f'Unexpected progress sampling grid: {log}')
            windows = {}
            for label, lo, hi in (('100k_to_200k',100000,200000), ('200k_to_300k',200000,300000)):
                subset = [x for x in progress if lo < x['step'] <= hi]
                windows[label] = dict(critic_preclip_norm=summary([x['critic_grad'] for x in subset]),
                                      actor_preclip_norm=summary([x['actor_grad'] for x in subset]),
                                      critic_loss=summary([x['critic_loss'] for x in subset]),
                                      alpha=summary([x['alpha'] for x in subset]))
            collisions = [x for x in episodes if any(x[k] for k in FLAGS)]
            evaluation = json.loads((folder / 'evaluation_300000.json').read_text())
            eval_rows = []
            for info in evaluation:
                path = folder / f'eval_300000_{info["map_seed"]}.npz'
                with np.load(path, allow_pickle=False) as trajectory:
                    terms = reward_components(trajectory['positions'], trajectory['goals'], info)
                weights = .99 ** np.arange(len(next(iter(terms.values()))))
                raw = {k:float(v.sum()) for k, v in terms.items()}
                discounted = {k:float(np.dot(v, weights)) for k, v in terms.items()}
                eval_rows.append(dict(map_seed=info['map_seed'], layout=info['layout'],
                    collision_flags={k:bool(info[k]) for k in FLAGS}, success=bool(info['success']),
                    steps=len(weights), reward_components_undiscounted=raw,
                    reward_components_discounted=discounted,
                    discounted_environment_return=sum(discounted.values()), trajectory_sha256=sha(path)))
            rows.append(dict(arm=arm, seed=seed, log_sha256=sha(log), windows=windows,
                completed_episodes=len(episodes), collision_episodes=len(collisions),
                collision_episodes_positive_undiscounted_return=sum(x['return_'] > 0 for x in collisions),
                evaluation=eval_rows))
    return dict(evidence='Progress rows are sampled every 100 updates, not all updates. Gradient norm is pre-clipping and clip threshold is 5. Evaluation reward components are environment returns, not SAC entropy-augmented objective. Episode returns are undiscounted; collision flags may overlap.',
                gamma=.99, source_sha256=source_hashes, audit_sha256=sha(Path(__file__)),
                synthetic_reward_reconstruction_checks=synthetic_reward_check(), rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    result = run(args.run)
    with args.out.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
