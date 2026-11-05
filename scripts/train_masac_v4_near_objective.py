#!/usr/bin/env python3
"""Explicit two-arm near-objective pilot entry point; frozen trainer unchanged.

Only the training reward callback is replaced. Original environment/evaluation,
SAC, replay, capture and snapshots are reused; source hashes include this entry
point and the pure reward adapter. No continuation or policy warm start.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_masac_v4_maneuver as trainer
from uav_planning.v4_near_objective import near_potential, replace_near_reward

PROTOCOL = 'docs/design/MASAC_V4_NEAR_OBJECTIVE_2K_PROTOCOL_20261105.md'


def install_training_reward(cfg, out):
    near = cfg['near_objective']
    mode = near['mode']
    if mode not in ('original', 'potential') or near != dict(
            mode=mode, coefficient=.4, radius_m=15., gamma=.99):
        raise ValueError('Unknown or modified fixed near-objective intervention')
    if cfg['sync_capture']['enabled'] or cfg['steps'] != 2000 or cfg['seeds'] != [1644]:
        raise ValueError('This registered pilot requires B-off, 2k, cold seed1644')
    if cfg['reshape']['near_goal_bonus'] != .4 or cfg['reshape']['near_goal_radius_m'] != 15:
        raise ValueError('Potential scale must match the removed original near term')
    original_audit, original_append, original_write = trainer.audited_episode, trainer.append, trainer.write
    original_evaluate = trainer.evaluate
    state = dict(env=None, previous=None, step_total=0, episode_removed=0., episode_pbrs=0.)

    def phi(env):
        return near_potential(env.p, env.goal, coefficient=.4, radius_m=15.)

    def audited_episode(*args, **kwargs):
        env, obs = original_audit(*args, **kwargs)
        state.update(env=env, previous=phi(env), episode_removed=0., episode_pbrs=0.)
        return env, obs

    def reward_callback(base, previous, following, *, terminated, truncated, gamma, config):
        if config.enabled or gamma != .99 or state['env'] is None:
            raise ValueError('Unexpected reward callback context')
        current = phi(state['env'])
        result, correction, detail = replace_near_reward(
            base, state['previous'], current, terminated=terminated,
            truncated=truncated, gamma=gamma, mode=mode)
        lower = -.8 if terminated else -.404
        if correction < lower - 1e-12 or correction > 1e-12:
            raise ValueError('Near correction outside analytic bounded range')
        state['previous'] = current
        state['step_total'] += 1
        state['episode_removed'] += detail['removed_near']
        state['episode_pbrs'] += detail['potential_difference']
        original_append(out / 'near_reward_transitions.jsonl', dict(
            kind='near_reward', training_seed=cfg['seeds'][0], map_seed=state['env'].seed,
            training_step=state['step_total'], replay_reward=float(result), **detail))
        if state['step_total'] % cfg['log_interval'] == 0:
            trainer.message(out, f'NEAR-REWARD mode={mode} training_step_total={state["step_total"]} '
                            f'original_base={base:.8f} removed_near={detail["removed_near"]:.8f} '
                            f'potential_difference={detail["potential_difference"]:.8f} '
                            f'correction={correction:.8f} replay_reward={result:.8f} '
                            f'terminated={terminated} truncated={truncated}')
        return result, correction

    def evaluate(*args, **kwargs):
        env, before, count = state['env'], state['previous'], state['step_total']
        positions, velocities = env.p.copy(), env.v.copy()
        result = original_evaluate(*args, **kwargs)
        if not (state['env'] is env and state['previous'] == before and state['step_total'] == count
                and np.array_equal(positions, env.p) and np.array_equal(velocities, env.v)):
            raise ValueError('Evaluation mutated training reward state')
        trainer.message(out, f'NEAR-EVAL-STATE PASS training_step={count} mode={mode}')
        return result

    def append(path, obj):
        if obj.get('kind') == 'episode':
            obj = dict(obj, near_objective=near,
                       near_removed_return=state['episode_removed'],
                       near_potential_difference_return=state['episode_pbrs'],
                       shaping_return_semantics='near replacement correction, not B shaping')
        original_append(path, obj)

    def write(path, obj):
        if Path(path).name == 'run_manifest.json':
            obj = dict(obj, near_objective=near, training_entry_point=__file__,
                       evaluation_near_objective='original Path-C; no reward callback during evaluation',
                       shaping_return_semantics='near replacement correction; base_return remains original Path-C',
                       preregistered_near_protocol=dict(path=PROTOCOL, sha256=trainer.digest(ROOT / PROTOCOL)))
        original_write(path, obj)

    trainer.audited_episode = audited_episode
    trainer.evaluate = evaluate
    trainer.shape_reward = reward_callback
    trainer.append = append
    trainer.write = write
    trainer.SOURCES += ('scripts/train_masac_v4_near_objective.py',
                        'uav_planning/v4_near_objective.py', PROTOCOL)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--torch-threads', type=int, default=1)
    args = p.parse_args()
    cfg = json.loads(args.protocol.read_text(encoding='utf-8'))
    legacy = json.loads((ROOT / 'results/masac_v4_sync_40k_control_cuda_20261105/run_manifest.json').read_text())
    if len(legacy['sources']) != 14 or any(trainer.digest(ROOT / p) != sha
                                         for p, sha in legacy['sources'].items()):
        raise ValueError('Original14 frozen sources changed; do not run this pilot')
    print(f'NEAR-OBJECTIVE PILOT protocol={args.protocol} config={cfg} '
          'training_only=true evaluation=original_PathC resume=false', flush=True)
    install_training_reward(cfg, args.out.resolve())
    args.authorization_protocol = None
    args.probe_mode = 'static'
    args.smoke = False
    args.seed = None
    args.warm_start = None
    args.lock_path = None
    args.primary = False
    args.caffeinate_pid = None
    trainer.run(args)


if __name__ == '__main__':
    main()
