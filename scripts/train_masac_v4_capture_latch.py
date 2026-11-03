#!/usr/bin/env python3
"""A/B capture-latch isolation on Standard L0 maps (SharedMASAC, Standard obs).

Baseline arm: capture_latch=False (frozen simultaneous dist<4 AND speed<2).
Latch arm: capture_latch=True (sticky once captured; no -5 on leave).

Does not modify frozen train_masac_v4_l0.py / v4_environment.py hashes.
Per-arm lock; does not overwrite masac_v4_active_run.json.

Authorization: user intensity t1u 2026-11-03 + local wiring t2u/t3u.
Forbidden: Attention / 19-probe maneuverability.
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
from uav_planning.v4_capture_latch import V4MissionCaptureVariant, make_standard_env
from uav_planning.v4_maps import FAMILIES, default_config
from uav_planning.v4_reference import joint_reference
from uav_planning.v4_training import InstrumentedLegacyMASAC

SOURCES = (
    'scripts/train_masac_v4_capture_latch.py',
    'uav_planning/v4_capture_latch.py',
    'uav_planning/v4_training.py',
    'uav_planning/v4_environment.py',
    'uav_planning/v4_maps.py',
    'uav_planning/shared_masac.py',
    'uav_planning/structured.py',
    'uav_planning/environment.py',
    'uav_planning/v4_reference.py',
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


def make_env(family, cfg, capture_latch):
    return make_standard_env(family, sensor_range=cfg['sensor_range'], capture_latch=capture_latch)


def audited_episode(family, cfg, seed, episode, directory, capture_latch):
    env = make_env(family, cfg, capture_latch)
    obs = env.reset(seed)
    check = joint_reference(env.config, seed)
    same_hash = check['scene']['scene_sha256'] == env.scene.scene_hash
    record = {
        'episode': episode,
        'map_seed': seed,
        'family': family,
        'level': 'L0',
        'capture_latch': bool(capture_latch),
        'scene_sha256': env.scene.scene_hash,
        'reference_scene_hash_matches': same_hash,
        'joint_reference_success': check['joint_reference_success'],
        'attempts': check['attempts'],
    }
    append(directory / 'training_map_audit.jsonl', record)
    if not same_hash or not check['joint_reference_success']:
        raise ValueError(f'Training map reference failed or scene hash mismatched: {seed}')
    return env, obs


def evaluate(model, maps, directory, step, cfg, capture_latch):
    records = []
    for entry in maps:
        seed = int(entry['seed'])
        family = entry['family']
        env = make_env(family, cfg, capture_latch)
        obs = env.reset(seed)
        initial = env.p.copy()
        near_stall = np.zeros(env.n, dtype=int)
        closest = np.linalg.norm(env.goal - env.p, axis=-1)
        ever = np.zeros(env.n, dtype=bool)
        positions = [env.p.copy()]
        velocities = [env.v.copy()]
        actions = []
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
            near_stall += ((distance < 10) & (speed < 2) & ~env.captured)
            if done or truncated:
                break
        path = np.asarray(info['path_length_by_uav'])
        direct = np.linalg.norm(env.goal - initial, axis=-1)
        row = {
            **info,
            'map_seed': seed,
            'family': family,
            'level': 'L0',
            'step': step,
            'capture_latch': bool(capture_latch),
            'terminated': bool(done),
            'truncated': bool(truncated),
            'timeout': bool(truncated and not done),
            'closest_goal_m_by_uav': closest.tolist(),
            'final_goal_m_by_uav': np.linalg.norm(env.goal - env.p, axis=-1).tolist(),
            'ever_arrived_by_uav': ever.tolist(),
            'simultaneously_arrived': bool(info['success']),
            'near_goal_low_speed_uncaptured_steps_by_uav': near_stall.tolist(),
            'success_efficiency_direct_over_path_by_uav': (
                (direct / np.maximum(path, 1e-12)).tolist() if info['success'] else None
            ),
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
    write(directory / 'evaluation.json', {'step': step, 'episodes': records, 'capture_latch': capture_latch})
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
    checkpoint = {
        'model': model.checkpoint(),
        'environment': copy.deepcopy(env.__dict__),
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
            'arm': 'capture_latch',
            'capture_latch': bool(getattr(env, 'capture_latch', False)),
            'complete': True,
        },
    )
    return folder


def validate_protocol(cfg):
    required = [
        'experiment_id', 'seeds', 'families', 'steps', 'warmup', 'batch_size',
        'replay_capacity', 'eval_interval', 'checkpoint_interval', 'log_interval',
        'device', 'sensor_range', 'development_maps', 'gate', 'n_agents', 'capture_latch',
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f'protocol missing {key}')
    if cfg['n_agents'] != 3 or cfg['device'] != 'cpu':
        raise ValueError('latch protocol expects n_agents=3 device=cpu')
    if cfg['families'] != list(FAMILIES):
        raise ValueError('families must match frozen FAMILIES order for Standard L0')
    if not isinstance(cfg['capture_latch'], bool):
        raise ValueError('capture_latch must be bool')


def run(args):
    cfg = json.loads(args.protocol.read_text())
    auth = json.loads(args.authorization_protocol.read_text()) if args.authorization_protocol else {}
    validate_protocol(cfg)
    capture_latch = bool(cfg['capture_latch'])
    if args.force_latch is not None:
        capture_latch = bool(args.force_latch)
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
    eval_maps = cfg['development_maps'][:2] if args.smoke else list(cfg['development_maps'])
    families = list(cfg['families'])
    torch.set_num_threads(max(1, int(args.torch_threads)))

    tag = 'latch' if capture_latch else 'baseline'
    if args.lock_path is None:
        lockpath = ROOT / 'results' / 'locks' / f'masac_v4_capture_{tag}_{args.out.name}.lock'
    else:
        lockpath = args.lock_path if args.lock_path.is_absolute() else ROOT / args.lock_path
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    lock = lockpath.open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f'Another runner holds capture-latch lock {lockpath}') from exc
    lock.write(f'pid={os.getpid()} out={args.out} capture_latch={capture_latch}\n')
    lock.flush()

    args.out.mkdir(parents=True, exist_ok=False)
    root = args.out.resolve()
    experiment_id = auth.get('experiment_id', cfg['experiment_id'])

    manifest = {
        'experiment_id': experiment_id,
        'mode': f'capture_latch_ab_{tag}',
        'capture_latch': capture_latch,
        'smoke': args.smoke,
        'level': 'L0',
        'observation': 'Standard (Pooling); no attention; no maneuverability',
        'created_at': time.time(),
        'pid': os.getpid(),
        'config_hashes': confhash,
        'sources': source_hashes(),
        'seed_list': seeds,
        'steps_per_seed': steps,
        'warmup': warmup,
        'replay_capacity': capacity,
        'development_maps_used': eval_maps,
        'authorization_protocol': str(args.authorization_protocol.resolve()) if args.authorization_protocol else None,
        'authorization': auth.get('authorization'),
        'failure_action': auth.get('failure_action', 'save_and_diagnose'),
        'command': sys.argv,
        'note': (
            'Isolates sticky capture vs frozen simultaneous ball+speed gate. '
            'Frozen v4_environment.py left byte-identical.'
        ),
    }
    write(root / 'run_manifest.json', manifest)
    if args.authorization_protocol:
        shutil.copy2(args.authorization_protocol, root / 'authorization_protocol.json')
    for name in SOURCES:
        target = root / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
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
            'kind': f'v4_capture_{tag}',
            'phase': 'running',
            'experiment_id': experiment_id,
            'capture_latch': capture_latch,
            'seeds': seeds,
            'steps': steps,
            'lock_path': str(lockpath),
            'log': str(root / 'train.log'),
            'authorization': 'user intensity t1u 2026-11-03',
        })
        registry['arms'] = arms
        registry['updated_at'] = time.time()
        write(registry_path, registry)

    seed = None
    step = 0
    phase = 'initializing'
    completed = []
    begin = time.perf_counter()
    try:
        message(
            root,
            f'CAPTURE-{tag.upper()} | PID={os.getpid()} | latch={capture_latch} | '
            f'seeds={seeds} | steps={steps} | out={root}',
        )
        for seed in seeds:
            directory = root / f'{tag}_{seed}'
            directory.mkdir()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            explore = np.random.default_rng(seed)
            model = InstrumentedLegacyMASAC(n=3, capacity=capacity, seed=seed, device=cfg['device'])
            episode = 0
            env, obs = audited_episode(
                families[0], cfg,
                cfg.get('training_map_seed_offset', 2000000) + seed * cfg.get('training_seed_multiplier', 10000),
                episode, directory, capture_latch,
            )
            state = dict(
                seed=seed, step=0, episode=0, updates=0, episode_start=0, episode_return=0.,
                recent=[], losses={}, last_eval=None, last_eval_step=None,
                closest=np.linalg.norm(env.goal - env.p, axis=-1).tolist(), near_stall=[0] * 3,
            )
            message(root, f'START seed={seed} family={env.config.family} latch={capture_latch}')
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
                        'family': env.config.family,
                        'capture_latch': capture_latch,
                        'return_': state['episode_return'],
                        'terminated': bool(done),
                        'truncated': bool(truncated),
                        'timeout': bool(truncated and not done),
                        'closest_goal_m_by_uav': state['closest'],
                        'near_goal_low_speed_uncaptured_steps_by_uav': state['near_stall'],
                        **info,
                    }
                    append(directory / 'metrics.jsonl', row)
                    state['recent'] = (state['recent'] + [row])[-30:]
                    state['episode'] += 1
                    state['episode_start'] = step
                    state['episode_return'] = 0.0
                    env, obs = audited_episode(
                        families[state['episode'] % 6], cfg,
                        cfg.get('training_map_seed_offset', 2000000)
                        + seed * cfg.get('training_seed_multiplier', 10000)
                        + state['episode'],
                        state['episode'], directory, capture_latch,
                    )
                    state['closest'] = np.linalg.norm(env.goal - env.p, axis=-1).tolist()
                    state['near_stall'] = [0] * 3
                if step % cfg['log_interval'] == 0 or step == steps:
                    recent = state['recent']
                    elapsed = time.perf_counter() - tick
                    rate = (step - start_step) / elapsed if elapsed else 0.0
                    status = {
                        'phase': phase,
                        'experiment_id': experiment_id,
                        'arm': f'capture_{tag}',
                        'capture_latch': capture_latch,
                        'seed': seed,
                        'step': step,
                        'total_steps': steps,
                        'updates': state['updates'],
                        'episodes': state['episode'],
                        'family': env.config.family,
                        'steps_per_s': rate,
                        'eta_s': (steps - step) / rate if rate else None,
                        'last_eval': state['last_eval'],
                        'last_eval_step': state['last_eval_step'],
                        'pid': os.getpid(),
                        'updated_at': time.time(),
                        'recent_success': sum(x['success'] for x in recent) / len(recent) if recent else None,
                        **state['losses'],
                    }
                    write(root / 'status.json', status)
                    append(directory / 'metrics.jsonl', {'kind': 'progress', **status})
                    message(
                        root,
                        f'[capture-{tag} seed={seed}] {step}/{steps} upd={state["updates"]} '
                        f'{rate:.1f}step/s val={state["last_eval"]}',
                    )
                if step % eval_every == 0 or step == steps:
                    phase = 'evaluating'
                    records = evaluate(model, eval_maps, directory, step, cfg, capture_latch)
                    state['last_eval'] = sum(x['success'] for x in records) / len(records)
                    state['last_eval_step'] = step
                    message(
                        root,
                        f'EVAL seed={seed} step={step}: success={sum(x["success"] for x in records)}/{len(records)}',
                    )
                if step % save_every == 0 or step == steps:
                    phase = 'saving'
                    save_snapshot(directory, model, env, explore, state)
            gate = float(cfg['gate']['per_seed_joint_success_fraction'])
            completed.append({
                'seed': seed,
                'step': steps,
                'capture_latch': capture_latch,
                'final_development_success': state['last_eval'],
                'gate_met': bool(state['last_eval'] is not None and state['last_eval'] >= gate),
            })
            write(directory / 'completion.json', completed[-1])
        complete = {
            'complete': True,
            'scope': 'smoke' if args.smoke else f'capture_{tag}',
            'capture_latch': capture_latch,
            'runs': completed,
            'gate_threshold': cfg['gate']['per_seed_joint_success_fraction'],
            'all_seeds_gate_met': bool(completed) and all(x['gate_met'] for x in completed),
            'smoke': args.smoke,
            'finished_at': time.time(),
            'elapsed_s': time.perf_counter() - begin,
            'authorization': auth.get('authorization'),
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
                registry['updated_at'] = time.time()
                write(registry_path, registry)
        message(root, f'COMPLETE capture-{tag} gate_met={complete["all_seeds_gate_met"]}')
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
        default=ROOT / 'configs/masac_v4/l0_capture_latch_on_20261103.json',
    )
    parser.add_argument('--authorization-protocol', type=Path, default=None)
    parser.add_argument('--lock-path', type=Path, default=None)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument(
        '--force-latch',
        type=lambda s: s.lower() in ('1', 'true', 'yes', 'on'),
        default=None,
        help='Override protocol capture_latch bool',
    )
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
