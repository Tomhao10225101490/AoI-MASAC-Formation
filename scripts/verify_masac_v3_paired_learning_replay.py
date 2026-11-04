#!/usr/bin/env python3
"""Replay two saved policies and check trajectory-derived environment rewards."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_masac_v3_paired_learning import FLAGS, reward_components, sha  # noqa: E402
from uav_planning.shared_masac import Actor  # noqa: E402
from uav_planning.structured import StructuredMission  # noqa: E402


def replay_one(root, arm, seed, map_seed):
    folder = root / f'{arm}_{seed}'
    evaluation = folder / 'evaluation_300000.json'
    row = next(x for x in json.loads(evaluation.read_text()) if x['map_seed'] == map_seed)
    checkpoint = folder / 'checkpoint_300000.pt'
    trajectory = folder / f'eval_300000_{map_seed}.npz'
    actor = Actor()
    actor.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['actor'])
    env = StructuredMission(mode='standard', layout=row['layout'])
    obs = env.reset(map_seed)
    positions, rewards = [env.p.copy()], []
    for _ in range(env.horizon):
        with torch.no_grad():
            action = actor(torch.as_tensor(obs).unsqueeze(0), True)[0][0].numpy()
        obs, reward, done, trunc, info = env.step(action)
        positions.append(env.p.copy())
        rewards.append(reward)
        if done or trunc:
            break
    with np.load(trajectory, allow_pickle=False) as z:
        if z['positions'].shape != np.asarray(positions).shape:
            raise ValueError('Saved and replayed trajectory lengths differ')
        pos_error = float(np.max(np.abs(z['positions'] - np.asarray(positions))))
        if not np.array_equal(z['goals'], env.goal):
            raise ValueError('Saved and replayed goals differ')
    reconstructed = sum(reward_components(positions, env.goal, info).values())
    reward_error = float(np.max(np.abs(reconstructed - rewards)))
    if pos_error > 1e-8 or reward_error > 1e-8:
        raise ValueError('Trajectory or reconstructed environment reward differs')
    return dict(arm=arm, seed=seed, map_seed=map_seed, layout=row['layout'],
                steps=len(rewards), success=bool(info['success']),
                collision_flags={k:bool(info[k]) for k in FLAGS},
                saved_trajectory_max_abs_error=pos_error,
                reconstructed_step_reward_max_abs_error=reward_error,
                checkpoint_sha256=sha(checkpoint), saved_trajectory_sha256=sha(trajectory),
                evaluation_json_sha256=sha(evaluation))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    run = args.run.resolve()
    if not json.loads((run / 'completion.json').read_text()).get('complete') or (run / 'error.json').exists():
        raise ValueError('An error-free completed run is required')
    manifest = json.loads((run / 'source_hashes.json').read_text())
    hashes = {}
    for name in ('structured.py', 'shared_masac.py'):
        expected = manifest[name]
        snapshot = run / 'source_snapshot' / name
        live = ROOT / 'uav_planning' / name
        if sha(snapshot) != expected or sha(live) != expected:
            raise ValueError(f'Live or snapshot source mismatch: {name}')
        hashes[name] = expected
    torch.set_num_threads(1)
    checks = [replay_one(run, 'curriculum', 411, 800001),
              replay_one(run, 'original', 411, 800002)]
    result = dict(limitation='Two fixed development episodes only; validates deterministic trajectory and per-step environment reward reconstruction, not SAC entropy or generalization.',
                  run=str(run), source_sha256=hashes,
                  reward_audit_script_sha256=sha(ROOT / 'scripts/audit_masac_v3_paired_learning.py'),
                  verification_script_sha256=sha(Path(__file__)), checks=checks)
    with args.out.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
