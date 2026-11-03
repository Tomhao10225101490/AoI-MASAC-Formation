#!/usr/bin/env python3
"""Path-C SharedMASAC reward/capture reshape on empty (then optional single) maps.

Fresh learner. Does not edit frozen train_masac_v4_l0.py / v4_environment.py.
Does NOT auto-launch L0 full stack; empty-map gate must PASS first.

Authorization: user widget t4s5 choosing C (2026-11-03).
Forbidden: Attention / Predictive Maneuverability / L1 / silent +50k continue restack.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uav_planning.structured import box_distance
from uav_planning.v4_reward_reshape import (
    DEFAULT_RESHAPE,
    make_reshape_env,
    reshape_from_protocol,
)
from uav_planning.v4_sanity_maps import SANITY_KINDS, development_map_entries, straight_line_clear
from uav_planning.v4_training import InstrumentedLegacyMASAC

SOURCES = (
    'scripts/train_masac_v4_c_reshape.py',
    'uav_planning/v4_reward_reshape.py',
    'uav_planning/v4_sanity_maps.py',
    'uav_planning/v4_training.py',
    'uav_planning/v4_environment.py',
    'uav_planning/v4_maps.py',
    'uav_planning/shared_masac.py',
    'uav_planning/structured.py',
    'uav_planning/environment.py',
)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, allow_nan=False) + '\n')
        f.flush()


def message(root, text):
    print(text, flush=True)
    with (root / 'train.log').open('a') as f:
        f.write(text + '\n')
        f.flush()


def source_hashes():
    return {p: digest(ROOT / p) for p in SOURCES}


def audited_episode(kind, cfg, reshape, seed, episode, directory):
    env, obs = make_reshape_env(kind, seed, sensor_range=cfg['sensor_range'], reshape=reshape)
    record = {
        'episode': episode,
        'map_seed': int(seed),
        'kind': kind,
        'level': 'L0',
        'scene_sha256': env.scene.scene_hash,
        'n_boxes': int(len(env.boxes)),
        'straight_line_clear': bool(straight_line_clear(env)),
        'reshape': reshape.to_dict(),
        'sanity_feasible_endpoints': True,
    }
    append(directory / 'training_map_audit.jsonl', record)
    if kind == 'empty' and not straight_line_clear(env):
        raise ValueError(f'Empty map unexpectedly blocked: seed={seed}')
    if kind == 'single_obstacle' and len(env.boxes) != 1:
        raise ValueError(f'single_obstacle expected 1 box, got {len(env.boxes)}')
    return env, obs


def evaluate(model, maps, directory, step, cfg, reshape):
    records = []
    for entry in maps:
        seed = int(entry['seed'])
        kind = entry['kind']
        env, obs = make_reshape_env(kind, seed, sensor_range=cfg['sensor_range'], reshape=reshape)
        initial = env.p.copy()
        positions = [env.p.copy()]
        velocities = [env.v.copy()]
        actions = []
        near_stall = np.zeros(env.n, dtype=int)
        closest = np.linalg.norm(env.goal - env.p, axis=-1)
        ever = np.zeros(env.n, dtype=bool)
        ever_ball = np.zeros(env.n, dtype=bool)
        while True:
            action = model.act(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            positions.append(env.p.copy())
            velocities.append(env.v.copy())
            actions.append(action.copy())
            distance = np.linalg.norm(env.goal - env.p, axis=-1)
            speed = np.linalg.norm(env.v, axis=-1)
            closest = np.minimum(closest, distance)
            ever |= env.captured
            ever_ball |= np.asarray(info['ever_in_ball_by_uav'])
            near_stall += ((distance < 10) & (speed < 2) & ~env.captured)
            if done or truncated:
                break
        path = np.asarray(info['path_length_by_uav'])
        direct = np.linalg.norm(env.goal - initial, axis=-1)
        row = {
            **{k: v for k, v in info.items() if k != 'reshape'},
            'reshape': info.get('reshape'),
            'map_seed': seed,
            'kind': kind,
            'level': 'L0',
            'step': step,
            'terminated': bool(done),
            'truncated': bool(truncated),
            'timeout': bool(truncated and not done),
            'closest_goal_m_by_uav': closest.tolist(),
            'final_goal_m_by_uav': np.linalg.norm(env.goal - env.p, axis=-1).tolist(),
            'ever_arrived_by_uav': ever.tolist(),
            'ever_in_ball_by_uav': ever_ball.tolist(),
            'simultaneously_arrived': bool(info['success']),
            'near_goal_low_speed_uncaptured_steps_by_uav': near_stall.tolist(),
            'success_efficiency_direct_over_path_by_uav': (
                (direct / np.maximum(path, 1e-12)).tolist() if info['success'] else None
            ),
            'eval_capture_rule': {
                'ball_m': reshape.train_ball_m,
                'speed_mps': reshape.train_speed_mps,
                'latch': reshape.capture_latch,
                'note': 'Path-C experiment eval uses soft train thresholds; frozen L0 is 4m/2mps.',
            },
        }
        prefix = directory / f'eval_{step}_{seed}'
        with open(str(prefix) + '.npz', 'xb') as f:
            np.savez(
                f,
                positions=np.asarray(positions),
                velocities=np.asarray(velocities),
                actions=np.asarray(actions),
                goals=env.goal,
                buildings=env.boxes,
            )
        write(Path(str(prefix) + '_map.json'), env.metadata())
        records.append(row)
    write(directory / f'evaluation_{step}.json', records)
    write(directory / 'evaluation.json', {'step': step, 'episodes': records})
    return records


def save_snapshot(directory, model, env, explore, state):
    step = state['step']
    folder = directory / f'checkpoint_{step:06d}'
    folder.mkdir(exist_ok=False)
    n = model.size
    replay = folder / 'replay.npz'
    with replay.open('xb') as f:
        np.savez(
            f,
            obs=model.obs[:n],
            next=model.next[:n],
            actions=model.actions[:n],
            rewards=model.rewards[:n],
            done=model.done[:n],
        )
        f.flush()
        os.fsync(f.fileno())
    # Environment may hold non-picklable reshape dataclass; store dict form.
    env_state = copy.deepcopy({k: v for k, v in env.__dict__.items() if k != 'reshape'})
    env_state['reshape'] = env.reshape.to_dict()
    checkpoint = {
        'model': model.checkpoint(),
        'environment': env_state,
        'exploration_rng': copy.deepcopy(explore.bit_generator.state),
        'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'state': copy.deepcopy(state),
    }
    cp = folder / 'checkpoint.pt'
    with cp.open('xb') as f:
        torch.save(checkpoint, f)
        f.flush()
        os.fsync(f.fileno())
    write(
        folder / 'manifest.json',
        {
            'version': 1,
            'step': step,
            'seed': state['seed'],
            'replay_size': n,
            'replay_index': model.index,
            'checkpoint_sha256': digest(cp),
            'replay_sha256': digest(replay),
            'sources': source_hashes(),
            'arm': 'c_reshape',
            'complete': True,
        },
    )
    return folder


def validate_protocol(cfg):
    required = [
        'experiment_id', 'seeds', 'kinds', 'steps', 'warmup', 'batch_size',
        'replay_capacity', 'eval_interval', 'checkpoint_interval', 'log_interval',
        'device', 'sensor_range', 'development_maps', 'gate', 'n_agents', 'reshape',
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f'protocol missing {key}')
    if cfg['n_agents'] != 3 or cfg['device'] != 'cpu':
        raise ValueError('c-reshape protocol expects n_agents=3 device=cpu')
    if any(k not in SANITY_KINDS for k in cfg['kinds']):
        raise ValueError(f'kinds must be subset of {SANITY_KINDS}')
    development_map_entries(cfg['development_maps'])
    reshape_from_protocol(cfg['reshape'])


def run(args):
    cfg = json.loads(args.protocol.read_text())
    auth = json.loads(args.authorization_protocol.read_text()) if args.authorization_protocol else {}
    validate_protocol(cfg)
    reshape = reshape_from_protocol(cfg['reshape'])
    confhash = {
        'protocol': digest(args.protocol),
        'authorization': digest(args.authorization_protocol) if args.authorization_protocol else None,
    }
    seeds = cfg['seeds'][:1] if args.smoke else list(cfg['seeds'])
    steps = 256 if args.smoke else int(cfg['steps'])
    warmup = 128 if args.smoke else int(cfg['warmup'])
    batch = int(cfg['batch_size'])
    capacity = 3000 if args.smoke else int(cfg['replay_capacity'])
    eval_every = 256 if args.smoke else int(cfg['eval_interval'])
    save_every = 256 if args.smoke else int(cfg['checkpoint_interval'])
    eval_maps = development_map_entries(cfg['development_maps'])
    if args.smoke:
        eval_maps = eval_maps[:2]
    kinds = list(cfg['kinds'])
    torch.set_num_threads(max(1, int(args.torch_threads)))

    if args.lock_path is None:
        lockpath = ROOT / 'results' / 'locks' / f'masac_v4_c_reshape_{args.out.name}.lock'
    else:
        lockpath = args.lock_path if args.lock_path.is_absolute() else ROOT / args.lock_path
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    lock = lockpath.open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f'Another runner holds c-reshape lock {lockpath}') from exc
    lock.write(f'pid={os.getpid()} out={args.out}\n')
    lock.flush()

    args.out.mkdir(parents=True, exist_ok=False)
    root = args.out.resolve()
    experiment_id = auth.get('experiment_id', cfg['experiment_id'])

    manifest = {
        'experiment_id': experiment_id,
        'mode': 'shared_masac_c_reshape_empty',
        'path': 'C',
        'smoke': args.smoke,
        'level': 'L0',
        'created_at': time.time(),
        'pid': os.getpid(),
        'config_hashes': confhash,
        'sources': source_hashes(),
        'seed_list': seeds,
        'steps_per_seed': steps,
        'warmup': warmup,
        'replay_capacity': capacity,
        'kinds': kinds,
        'development_maps_used': eval_maps,
        'reshape': reshape.to_dict(),
        'frozen_l0_capture': {'ball_m': 4.0, 'speed_mps': 2.0, 'latch': False},
        'authorization_protocol': str(args.authorization_protocol.resolve()) if args.authorization_protocol else None,
        'authorization': auth.get('authorization'),
        'failure_action': auth.get('failure_action', 'save_and_diagnose'),
        'success_action': auth.get(
            'success_action',
            'declare empty-map learnability floor; only then authorize L0 return',
        ),
        'do_not_auto_launch_l0': True,
        'command': sys.argv,
        'note': (
            'Path C: denser approach + soft capture + latch on empty maps. '
            'No Attention / maneuverability / L1. No silent continue restack.'
        ),
    }
    write(root / 'run_manifest.json', manifest)
    if args.authorization_protocol:
        shutil.copy2(args.authorization_protocol, root / 'authorization_protocol.json')
    for name in SOURCES:
        target = root / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (root / 'source_snapshot' / 'configs' / 'masac_v4').mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.protocol, root / 'source_snapshot' / 'configs' / 'masac_v4' / args.protocol.name)

    if not args.smoke:
        registry_path = ROOT / 'results/masac_v4_parallel_runs.json'
        registry = {'primary_active_run_file': 'results/masac_v4_active_run.json', 'arms': []}
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text())
            except Exception:
                pass
        arms = [a for a in registry.get('arms', []) if a.get('out') != str(root)]
        arms.append({
            'out': str(root),
            'pid': os.getpid(),
            'started_at': time.time(),
            'kind': 'v4_c_reshape_empty',
            'phase': 'running',
            'experiment_id': experiment_id,
            'seeds': seeds,
            'steps': steps,
            'reshape': reshape.to_dict(),
            'lock_path': str(lockpath),
            'log': str(root / 'train.log'),
            'authorization': 'user widget t4s5 path C 2026-11-03',
            'do_not_auto_launch_l0': True,
        })
        registry['arms'] = arms
        registry['updated_at'] = time.time()
        registry['primary'] = {
            'out': str(root),
            'pid': os.getpid(),
            'experiment_id': experiment_id,
            'phase': 'running',
            'kind': 'v4_c_reshape_empty',
            'note': 'Path C empty-map reshape; wait for gate before L0',
        }
        write(registry_path, registry)
        write(
            ROOT / 'results/masac_v4_active_run.json',
            {
                'out': str(root),
                'pid': os.getpid(),
                'started_at': time.time(),
                'kind': 'v4_c_reshape_empty',
                'phase': 'running',
                'experiment_id': experiment_id,
                'seed': seeds[0] if seeds else None,
                'target_total_steps': steps,
                'reshape': reshape.to_dict(),
                'authorization': 'user widget t4s5 path C 2026-11-03',
                'log': str(root / 'train.log'),
                'do_not_auto_launch_l0': True,
                'updated_at_asia_shanghai': time.strftime('%Y-%m-%d %H:%M Asia/Shanghai'),
            },
        )

    seed = None
    step = 0
    phase = 'initializing'
    completed = []
    begin = time.perf_counter()
    try:
        message(
            root,
            f'C-RESHAPE | PID={os.getpid()} | seeds={seeds} | steps={steps} | '
            f'kinds={kinds} | ball={reshape.train_ball_m} speed={reshape.train_speed_mps} '
            f'latch={reshape.capture_latch} | out={root}',
        )
        for seed in seeds:
            directory = root / f'reshape_{seed}'
            directory.mkdir()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            explore = np.random.default_rng(seed)
            model = InstrumentedLegacyMASAC(n=3, capacity=capacity, seed=seed, device=cfg['device'])
            episode = 0
            kind0 = kinds[0]
            env, obs = audited_episode(
                kind0, cfg, reshape,
                cfg.get('training_map_seed_offset', 880000) + seed * cfg.get('training_seed_multiplier', 1000),
                episode, directory,
            )
            state = dict(
                seed=seed, step=0, episode=0, updates=0, episode_start=0, episode_return=0.,
                recent=[], losses={}, last_eval=None, last_eval_step=None,
                closest=np.linalg.norm(env.goal - env.p, axis=-1).tolist(), near_stall=[0] * 3,
            )
            message(root, f'START seed={seed} kind={kind0} reshape={reshape.to_dict()}')
            start_step = 0
            tick = time.perf_counter()
            for step in range(1, steps + 1):
                phase = 'training'
                state['step'] = step
                action = (
                    explore.uniform(-1, 1, (3, 3)).astype(np.float32)
                    if step <= warmup else model.act(obs)
                )
                next_obs, reward, done, truncated, info = env.step(action)
                model.push(obs, action, reward, next_obs, done)
                obs = next_obs
                state['episode_return'] += reward
                distance = np.linalg.norm(env.goal - env.p, axis=-1)
                speed = np.linalg.norm(env.v, axis=-1)
                state['closest'] = np.minimum(state['closest'], distance).tolist()
                state['near_stall'] = (
                    np.asarray(state['near_stall']) + ((distance < 10) & (speed < 2) & ~env.captured)
                ).tolist()
                if step > warmup:
                    state['losses'] = model.update(batch)
                    state['updates'] += 1
                    if not all(np.isfinite(v) for v in state['losses'].values()):
                        raise ValueError('nonfinite learning diagnostic')
                if done or truncated:
                    row = {
                        'kind': 'episode',
                        'experiment_id': experiment_id,
                        'level': 'L0',
                        'seed': seed,
                        'step': step,
                        'episode': state['episode'],
                        'map_seed': env.seed,
                        'map_kind': kinds[state['episode'] % len(kinds)],
                        'return_': state['episode_return'],
                        'terminated': bool(done),
                        'truncated': bool(truncated),
                        'timeout': bool(truncated and not done),
                        'closest_goal_m_by_uav': state['closest'],
                        'near_goal_low_speed_uncaptured_steps_by_uav': state['near_stall'],
                        **{k: v for k, v in info.items() if k != 'reshape'},
                        'reshape': reshape.to_dict(),
                    }
                    append(directory / 'metrics.jsonl', row)
                    state['recent'] = (state['recent'] + [row])[-30:]
                    state['episode'] += 1
                    state['episode_start'] = step
                    state['episode_return'] = 0.0
                    next_kind = kinds[state['episode'] % len(kinds)]
                    map_seed = (
                        cfg.get('training_map_seed_offset', 880000)
                        + seed * cfg.get('training_seed_multiplier', 1000)
                        + state['episode']
                    )
                    env, obs = audited_episode(next_kind, cfg, reshape, map_seed, state['episode'], directory)
                    state['closest'] = np.linalg.norm(env.goal - env.p, axis=-1).tolist()
                    state['near_stall'] = [0] * 3
                if step % cfg['log_interval'] == 0 or step == steps:
                    recent = state['recent']
                    elapsed = time.perf_counter() - tick
                    rate = (step - start_step) / elapsed if elapsed else 0.0
                    status = {
                        'phase': phase,
                        'experiment_id': experiment_id,
                        'arm': 'c_reshape',
                        'path': 'C',
                        'seed': seed,
                        'step': step,
                        'total_steps': steps,
                        'updates': state['updates'],
                        'episodes': state['episode'],
                        'steps_per_s': rate,
                        'eta_s': (steps - step) / rate if rate else None,
                        'last_eval': state['last_eval'],
                        'last_eval_step': state['last_eval_step'],
                        'pid': os.getpid(),
                        'updated_at': time.time(),
                        'reshape': reshape.to_dict(),
                        'recent_success': sum(x['success'] for x in recent) / len(recent) if recent else None,
                        **state['losses'],
                    }
                    write(root / 'status.json', status)
                    append(directory / 'metrics.jsonl', {'kind': 'progress', **status})
                    message(
                        root,
                        f'[c-reshape seed={seed}] {step}/{steps} upd={state["updates"]} '
                        f'{rate:.1f}step/s val={state["last_eval"]}',
                    )
                if step % eval_every == 0 or step == steps:
                    phase = 'evaluating'
                    records = evaluate(model, eval_maps, directory, step, cfg, reshape)
                    state['last_eval'] = sum(x['success'] for x in records) / len(records)
                    state['last_eval_step'] = step
                    n_ball = sum(1 for x in records if any(x.get('ever_in_ball_by_uav') or []))
                    message(
                        root,
                        f'EVAL seed={seed} step={step}: success={sum(x["success"] for x in records)}/{len(records)} '
                        f'ever_ball_maps={n_ball}/{len(records)}',
                    )
                if step % save_every == 0 or step == steps:
                    phase = 'saving'
                    save_snapshot(directory, model, env, explore, state)
            gate = float(cfg['gate']['per_seed_joint_success_fraction'])
            completed.append({
                'seed': seed,
                'step': steps,
                'final_development_success': state['last_eval'],
                'gate_met': bool(state['last_eval'] is not None and state['last_eval'] >= gate),
                'reshape': reshape.to_dict(),
            })
            write(directory / 'completion.json', completed[-1])
        complete = {
            'complete': True,
            'scope': 'smoke' if args.smoke else 'c_reshape empty floor',
            'path': 'C',
            'runs': completed,
            'gate_threshold': cfg['gate']['per_seed_joint_success_fraction'],
            'all_seeds_gate_met': bool(completed) and all(x['gate_met'] for x in completed),
            'smoke': args.smoke,
            'finished_at': time.time(),
            'elapsed_s': time.perf_counter() - begin,
            'authorization': auth.get('authorization'),
            'reshape': reshape.to_dict(),
            'do_not_auto_launch_l0': True,
            'next_if_pass': 'User/monitor may authorize return to L0 12-map gate with frozen capture; do not auto-launch.',
            'next_if_fail': 'save_and_diagnose; further reshape or audited longer empty protocol — no Attention/L1.',
        }
        write(root / 'completion.json', complete)
        write(root / 'status.json', {'phase': 'complete', 'complete': True, **complete})
        if not args.smoke:
            registry_path = ROOT / 'results/masac_v4_parallel_runs.json'
            if registry_path.exists():
                registry = json.loads(registry_path.read_text())
                for arm in registry.get('arms', []):
                    if arm.get('out') == str(root):
                        arm['phase'] = 'complete'
                        arm['completed_at'] = time.time()
                        arm['all_seeds_gate_met'] = complete['all_seeds_gate_met']
                if registry.get('primary', {}).get('out') == str(root):
                    registry['primary']['phase'] = 'complete'
                    registry['primary']['all_seeds_gate_met'] = complete['all_seeds_gate_met']
                registry['updated_at'] = time.time()
                write(registry_path, registry)
            active = {
                'out': str(root),
                'pid': os.getpid(),
                'kind': 'v4_c_reshape_empty',
                'phase': 'complete',
                'experiment_id': experiment_id,
                'all_seeds_gate_met': complete['all_seeds_gate_met'],
                'do_not_auto_launch_l0': True,
                'completed_at': time.time(),
                'updated_at_asia_shanghai': time.strftime('%Y-%m-%d %H:%M Asia/Shanghai'),
            }
            write(ROOT / 'results/masac_v4_active_run.json', active)
        message(root, f'COMPLETE c-reshape gate_met={complete["all_seeds_gate_met"]}')
    except BaseException as e:
        write(
            root / 'error.json',
            {
                'error': repr(e),
                'traceback': traceback.format_exc(),
                'seed': seed,
                'step': step,
                'phase': phase,
                'updated_at': time.time(),
            },
        )
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument(
        '--protocol',
        type=Path,
        default=ROOT / 'configs/masac_v4/c_reshape_empty_20261103.json',
    )
    parser.add_argument('--authorization-protocol', type=Path, default=None)
    parser.add_argument('--lock-path', type=Path, default=None)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
