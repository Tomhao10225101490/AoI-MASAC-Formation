#!/usr/bin/env python3
"""Path-C SharedMASAC reward reshape on frozen L0 12-map gate (capture 4m/2mps).

Keeps Path-C reward magnitudes; TRAIN+EVAL use frozen capture dist<4 AND speed<2
with L0-default latch=False. Optional warm-start: load actor/critic/opt from the
empty-floor Path-C checkpoint; discard soft-capture replay / env / step counters.

Does NOT mutate frozen train_masac_v4_l0.py / v4_environment.py.
Does NOT auto-launch Attention / maneuver / L1 / further stages.

Authorization: user widget t4s11 (2026-11-03).
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

from uav_planning.v4_maps import FAMILIES
from uav_planning.v4_reference import joint_reference
from uav_planning.v4_reward_reshape import (
    FROZEN_CAPTURE_SPEED_MPS,
    FROZEN_GOAL_BALL_M,
    make_reshape_l0_env,
    reshape_from_protocol,
)
from uav_planning.v4_training import InstrumentedLegacyMASAC

SOURCES = (
    'scripts/train_masac_v4_c_reshape_l0.py',
    'uav_planning/v4_reward_reshape.py',
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


def warm_start_weights(model, checkpoint_dir: Path):
    """Load network + optimizer state; clear replay. Fresh RNGs stay with caller seed."""
    folder = Path(checkpoint_dir)
    meta = json.loads((folder / 'manifest.json').read_text())
    cpfile = folder / 'checkpoint.pt'
    if digest(cpfile) != meta['checkpoint_sha256']:
        raise ValueError(f'Warm-start checkpoint digest mismatch: {cpfile}')
    blob = torch.load(cpfile, map_location='cpu', weights_only=False)
    m = blob['model']
    model.actor.load_state_dict(m['actor'])
    model.q.load_state_dict(m['critics'])
    model.target.load_state_dict(m['targets'])
    with torch.no_grad():
        model.logalpha.copy_(m['logalpha'].to(model.device))
    for optimizer, key in (
        (model.ao, 'actor_optimizer'),
        (model.qo, 'critic_optimizer'),
        (model.eo, 'alpha_optimizer'),
    ):
        optimizer.load_state_dict(m[key])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(model.device)
    model.size = 0
    model.index = 0
    return {
        'warm_start_from': str(folder.resolve()),
        'parent_step': int(meta.get('step', blob.get('state', {}).get('step', -1))),
        'parent_seed': int(meta.get('seed', blob.get('state', {}).get('seed', -1))),
        'parent_manifest': meta,
        'mode': 'weights_and_optimizers_only',
        'replay_discarded': True,
        'env_rng_step_discarded': True,
    }


def audited_episode(family, cfg, reshape, seed, episode, directory):
    env, obs = make_reshape_l0_env(family, seed, sensor_range=cfg['sensor_range'], reshape=reshape)
    check = joint_reference(env.config, seed)
    same_hash = check['scene']['scene_sha256'] == env.scene.scene_hash
    record = {
        'episode': episode,
        'map_seed': int(seed),
        'family': family,
        'level': 'L0',
        'scene_sha256': env.scene.scene_hash,
        'reference_scene_hash_matches': same_hash,
        'joint_reference_success': check['joint_reference_success'],
        'attempts': check['attempts'],
        'reshape': reshape.to_dict(),
        'capture_rule': {
            'ball_m': reshape.train_ball_m,
            'speed_mps': reshape.train_speed_mps,
            'latch': reshape.capture_latch,
        },
    }
    append(directory / 'training_map_audit.jsonl', record)
    if not same_hash or not check['joint_reference_success']:
        raise ValueError(f'Training map reference failed or scene hash mismatched: {seed}')
    return env, obs


def evaluate(model, maps, directory, step, cfg, reshape):
    records = []
    for entry in maps:
        seed = int(entry['seed'])
        family = entry['family']
        env, obs = make_reshape_l0_env(family, seed, sensor_range=cfg['sensor_range'], reshape=reshape)
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
            'family': family,
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
                'note': (
                    f'Train+gate use frozen L0 capture ball<{FROZEN_GOAL_BALL_M} '
                    f'speed<{FROZEN_CAPTURE_SPEED_MPS} latch={reshape.capture_latch}.'
                ),
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
            'arm': 'c_reshape_l0_frozen42',
            'warm_start': state.get('warm_start'),
            'complete': True,
        },
    )
    return folder


def validate_protocol(cfg):
    required = [
        'experiment_id', 'seeds', 'families', 'steps', 'warmup', 'batch_size',
        'replay_capacity', 'eval_interval', 'checkpoint_interval', 'log_interval',
        'device', 'sensor_range', 'development_maps', 'gate', 'n_agents', 'reshape',
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f'protocol missing {key}')
    if cfg['n_agents'] != 3 or cfg['device'] != 'cpu':
        raise ValueError('c-reshape-l0 protocol expects n_agents=3 device=cpu')
    if list(cfg['families']) != list(FAMILIES):
        raise ValueError('families must match frozen FAMILIES order for Standard L0')
    maps = cfg['development_maps']
    if len(maps) != 12:
        raise ValueError('expected 12 L0 development maps')
    for entry in maps:
        if entry.get('level') != 'L0' or entry.get('family') not in FAMILIES:
            raise ValueError(f'bad development map entry: {entry}')
    reshape = reshape_from_protocol(cfg['reshape'])
    if abs(reshape.train_ball_m - FROZEN_GOAL_BALL_M) > 1e-9:
        raise ValueError(f'train_ball_m must be frozen {FROZEN_GOAL_BALL_M}, got {reshape.train_ball_m}')
    if abs(reshape.train_speed_mps - FROZEN_CAPTURE_SPEED_MPS) > 1e-9:
        raise ValueError(
            f'train_speed_mps must be frozen {FROZEN_CAPTURE_SPEED_MPS}, got {reshape.train_speed_mps}'
        )
    if reshape.capture_latch is not False:
        raise ValueError('L0-return protocol requires capture_latch=False (project L0 default)')
    gate = float(cfg['gate']['per_seed_joint_success_fraction'])
    if abs(gate - 0.6) > 1e-9:
        raise ValueError('L0 gate must be 0.6')


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
    if args.seed is not None:
        seeds = [int(args.seed)]
    steps = 256 if args.smoke else int(cfg['steps'])
    warmup = 128 if args.smoke else int(cfg['warmup'])
    batch = int(cfg['batch_size'])
    capacity = 3000 if args.smoke else int(cfg['replay_capacity'])
    eval_every = 256 if args.smoke else int(cfg['eval_interval'])
    save_every = 256 if args.smoke else int(cfg['checkpoint_interval'])
    eval_maps = list(cfg['development_maps'])
    if args.smoke:
        eval_maps = eval_maps[:2]
    families = list(cfg['families'])
    torch.set_num_threads(max(1, int(args.torch_threads)))

    warm_path = None
    if args.warm_start is not None:
        warm_path = args.warm_start if args.warm_start.is_absolute() else ROOT / args.warm_start
    elif args.no_warm_start:
        warm_path = None
    elif cfg.get('warm_start', {}).get('enabled_by_default') and cfg.get('warm_start', {}).get('checkpoint'):
        warm_path = ROOT / cfg['warm_start']['checkpoint']
    if warm_path is not None:
        warm_path = warm_path.resolve()
        if not (warm_path / 'checkpoint.pt').exists():
            raise FileNotFoundError(f'warm-start checkpoint missing: {warm_path}')

    if args.lock_path is None:
        lockpath = ROOT / 'results' / 'locks' / f'masac_v4_c_reshape_l0_{args.out.name}.lock'
    else:
        lockpath = args.lock_path if args.lock_path.is_absolute() else ROOT / args.lock_path
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    lock = lockpath.open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f'Another runner holds c-reshape-l0 lock {lockpath}') from exc
    lock.write(f'pid={os.getpid()} out={args.out}\n')
    lock.flush()

    args.out.mkdir(parents=True, exist_ok=False)
    root = args.out.resolve()
    experiment_id = auth.get('experiment_id', cfg['experiment_id'])

    manifest = {
        'experiment_id': experiment_id,
        'mode': 'shared_masac_c_reshape_l0_frozen42',
        'path': 'C',
        'smoke': args.smoke,
        'level': 'L0',
        'created_at': time.time(),
        'created_at_asia_shanghai': time.strftime('%Y-%m-%d %H:%M Asia/Shanghai'),
        'pid': os.getpid(),
        'config_hashes': confhash,
        'sources': source_hashes(),
        'seed_list': seeds,
        'steps_per_seed': steps,
        'warmup': warmup,
        'replay_capacity': capacity,
        'families': families,
        'development_maps_used': eval_maps,
        'reshape': reshape.to_dict(),
        'capture_train_and_gate': {
            'ball_m': reshape.train_ball_m,
            'speed_mps': reshape.train_speed_mps,
            'latch': reshape.capture_latch,
            'note': 'Frozen L0 4/2 latch=False for train+gate; Path-C reward magnitudes kept.',
        },
        'warm_start_path': str(warm_path) if warm_path else None,
        'warm_start_mode': 'weights_and_optimizers_only' if warm_path else 'cold',
        'authorization_protocol': str(args.authorization_protocol.resolve()) if args.authorization_protocol else None,
        'authorization': auth.get('authorization'),
        'failure_action': auth.get('failure_action', 'save_and_diagnose'),
        'success_action': auth.get(
            'success_action',
            'declare L0 12-map gate under Path-C reshape + frozen capture; do not auto-launch further',
        ),
        'do_not_auto_launch_further': True,
        'command': sys.argv,
        'note': (
            'Path C L0 return: C rewards + frozen capture 4/2 latch off. '
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
            'kind': 'v4_c_reshape_l0_frozen42',
            'phase': 'running',
            'experiment_id': experiment_id,
            'seeds': seeds,
            'steps': steps,
            'reshape': reshape.to_dict(),
            'warm_start': str(warm_path) if warm_path else None,
            'lock_path': str(lockpath),
            'log': str(root / 'train.log'),
            'authorization': 'user widget t4s11 path C L0 return 2026-11-03',
            'do_not_auto_launch_further': True,
        })
        registry['arms'] = arms
        registry['updated_at'] = time.time()
        registry['updated_at_asia_shanghai'] = time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')
        registry['primary'] = {
            'out': str(root),
            'pid': os.getpid(),
            'experiment_id': experiment_id,
            'phase': 'running',
            'kind': 'v4_c_reshape_l0_frozen42',
            'seeds': seeds,
            'warm_start': str(warm_path) if warm_path else None,
            'do_not_auto_launch_further': True,
        }
        write(registry_path, registry)
        active = {
            'out': str(root),
            'pid': os.getpid(),
            'kind': 'v4_c_reshape_l0_frozen42',
            'phase': 'running',
            'experiment_id': experiment_id,
            'seeds': seeds,
            'target_total_steps': steps,
            'reshape': reshape.to_dict(),
            'capture': {'ball_m': 4.0, 'speed_mps': 2.0, 'latch': False},
            'warm_start': str(warm_path) if warm_path else None,
            'authorization': 'user widget t4s11 path C L0 return 2026-11-03',
            'log': str(root / 'train.log'),
            'do_not_auto_launch_further': True,
            'started_at_asia_shanghai': time.strftime('%Y-%m-%d %H:%M Asia/Shanghai'),
            'updated_at_asia_shanghai': time.strftime('%Y-%m-%d %H:%M Asia/Shanghai'),
            'monitor_note': 'Path C L0 return running; gate ≥0.6 on 12 maps frozen 4/2; no auto further stages',
        }
        write(ROOT / 'results/masac_v4_active_run.json', active)

    completed = []
    begin = time.perf_counter()
    seed = None
    step = 0
    phase = 'init'
    try:
        for run_index, seed in enumerate(seeds):
            directory = root / f'reshape_l0_{seed}'
            directory.mkdir()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            explore = np.random.default_rng(seed)
            model = InstrumentedLegacyMASAC(n=3, capacity=capacity, seed=seed, device=cfg['device'])
            warm_meta = None
            if warm_path is not None:
                warm_meta = warm_start_weights(model, warm_path)
                message(root, f'WARM-START seed={seed} from={warm_path} parent_step={warm_meta["parent_step"]}')
            else:
                message(root, f'COLD-START seed={seed}')
            episode = 0
            family0 = families[0]
            map_seed0 = (
                cfg.get('training_map_seed_offset', 2000000)
                + seed * cfg.get('training_seed_multiplier', 10000)
            )
            env, obs = audited_episode(family0, cfg, reshape, map_seed0, episode, directory)
            state = dict(
                seed=seed,
                step=0,
                episode=0,
                updates=0,
                episode_start=0,
                episode_return=0.0,
                recent=[],
                losses={},
                last_eval=None,
                last_eval_step=None,
                closest=np.linalg.norm(env.goal - env.p, axis=-1).tolist(),
                near_stall=[0] * 3,
                warm_start=warm_meta,
            )
            write(directory / 'warm_start.json', warm_meta or {'mode': 'cold'})
            message(root, f'START seed={seed} family={family0} reshape={reshape.to_dict()}')
            start_step = 0
            tick = time.perf_counter()
            for step in range(1, steps + 1):
                phase = 'training'
                state['step'] = step
                action = (
                    explore.uniform(-1, 1, (3, 3)).astype(np.float32)
                    if step <= warmup
                    else model.act(obs)
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
                        'map_family': families[state['episode'] % len(families)],
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
                    next_family = families[state['episode'] % len(families)]
                    map_seed = (
                        cfg.get('training_map_seed_offset', 2000000)
                        + seed * cfg.get('training_seed_multiplier', 10000)
                        + state['episode']
                    )
                    env, obs = audited_episode(
                        next_family, cfg, reshape, map_seed, state['episode'], directory
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
                        'arm': 'c_reshape_l0_frozen42',
                        'path': 'C',
                        'seed': seed,
                        'seed_index': run_index,
                        'seeds_total': len(seeds),
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
                        'updated_at_asia_shanghai': time.strftime('%Y-%m-%d %H:%M Asia/Shanghai'),
                        'reshape': reshape.to_dict(),
                        'warm_start': bool(warm_meta),
                        'recent_success': sum(x['success'] for x in recent) / len(recent) if recent else None,
                        **state['losses'],
                    }
                    write(root / 'status.json', status)
                    append(directory / 'metrics.jsonl', {'kind': 'progress', **status})
                    message(
                        root,
                        f'[c-reshape-l0 seed={seed}] {step}/{steps} upd={state["updates"]} '
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
                'warm_start': warm_meta,
            })
            write(directory / 'completion.json', completed[-1])
        complete = {
            'complete': True,
            'scope': 'smoke' if args.smoke else 'c_reshape L0 frozen42 12-map gate',
            'path': 'C',
            'runs': completed,
            'gate_threshold': cfg['gate']['per_seed_joint_success_fraction'],
            'all_seeds_gate_met': bool(completed) and all(x['gate_met'] for x in completed),
            'smoke': args.smoke,
            'finished_at': time.time(),
            'finished_at_asia_shanghai': time.strftime('%Y-%m-%d %H:%M Asia/Shanghai'),
            'elapsed_s': time.perf_counter() - begin,
            'authorization': auth.get('authorization'),
            'reshape': reshape.to_dict(),
            'capture_train_and_gate': {
                'ball_m': reshape.train_ball_m,
                'speed_mps': reshape.train_speed_mps,
                'latch': reshape.capture_latch,
            },
            'warm_start_path': str(warm_path) if warm_path else None,
            'do_not_auto_launch_further': True,
            'next_if_pass': 'Declare L0 gate under Path-C reshape+frozen capture; await user before further stages.',
            'next_if_fail': 'save_and_diagnose; no Attention/L1/silent budget restack.',
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
                registry['updated_at_asia_shanghai'] = time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')
                write(registry_path, registry)
            active = {
                'out': str(root),
                'pid': os.getpid(),
                'kind': 'v4_c_reshape_l0_frozen42',
                'phase': 'complete',
                'experiment_id': experiment_id,
                'all_seeds_gate_met': complete['all_seeds_gate_met'],
                'do_not_auto_launch_further': True,
                'completed_at': time.time(),
                'updated_at_asia_shanghai': time.strftime('%Y-%m-%d %H:%M Asia/Shanghai'),
            }
            write(ROOT / 'results/masac_v4_active_run.json', active)
        message(root, f'COMPLETE c-reshape-l0 gate_met={complete["all_seeds_gate_met"]}')
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
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--authorization-protocol', type=Path, default=None)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='optional single-seed override')
    parser.add_argument(
        '--warm-start',
        type=Path,
        default=None,
        help='checkpoint_* folder; weights+opt only (default: protocol warm_start.checkpoint)',
    )
    parser.add_argument('--no-warm-start', action='store_true', help='force cold start')
    parser.add_argument('--lock-path', type=Path, default=None)
    run(parser.parse_args())
