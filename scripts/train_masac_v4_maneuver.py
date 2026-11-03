#!/usr/bin/env python3
"""Predictive Maneuverability-Aware MASAC on frozen L0 12-map gate.

Paper-facing candidate arm after Standard L0 Path-C pass:
  - V4 SetEncoder pool + RiskEncoder (19×11 probes); NO Attention in this suite
  - Path-C reward reshape retained; capture frozen 4m/2mps latch=False
  - probe_mode ∈ {full, static, zero, shuffle}; capacity/reward/budget matched
  - Cold start (Legacy SharedMASAC Path-C checkpoints are architecture-incompatible)
  - Does NOT auto-launch Attention / L1 / further stages

Authorization: user t4s17 + phone steering for comprehensive paper suite (2026-11-03).
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
from uav_planning.v4_maneuver_masac import ManeuverMASAC
from uav_planning.v4_prediction import ProbeConfig, predict_probes
from uav_planning.v4_reference import joint_reference
from uav_planning.v4_reward_reshape import (
    FROZEN_CAPTURE_SPEED_MPS,
    FROZEN_GOAL_BALL_M,
    make_reshape_l0_env,
    reshape_from_protocol,
)

SOURCES = (
    'scripts/train_masac_v4_maneuver.py',
    'uav_planning/v4_maneuver_masac.py',
    'uav_planning/v4_networks.py',
    'uav_planning/v4_prediction.py',
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


def probe_summary(obs, probe_mode, probe_config, generator=None):
    o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    risk = predict_probes(o, probe_config, mode=probe_mode, generator=generator)
    # columns: dir3, clearances4, ttc, brake, reserve, positive_flag
    clear = risk[..., 3:7].amin(-1)
    return {
        'probe_min_clearance_mean': float(clear.mean()),
        'probe_ttc_mean': float(risk[..., 7].mean()),
        'probe_brake_mean': float(risk[..., 8].mean()),
        'probe_reserve_mean': float(risk[..., 9].mean()),
        'probe_positive_preview_frac': float(risk[..., 10].mean()),
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


def evaluate(model, maps, directory, step, cfg, reshape, probe_mode, probe_config):
    records = []
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(step) * 1009 + 7)
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
        probe_acc = []
        while True:
            probe_acc.append(probe_summary(obs, probe_mode, probe_config, generator=gen))
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
        probe_means = {
            k: float(np.mean([p[k] for p in probe_acc])) for k in probe_acc[0]
        } if probe_acc else {}
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
            'mean_path_efficiency_direct_over_path': (
                float(np.mean(direct / np.maximum(path, 1e-12))) if info['success'] else None
            ),
            'probe_mode': probe_mode,
            **probe_means,
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
            'arm': f'maneuver_{model.probe_mode}',
            'set_mode': model.set_mode,
            'probe_mode': model.probe_mode,
            'warm_start': state.get('warm_start'),
            'complete': True,
        },
    )
    return folder


def validate_protocol(cfg, probe_mode):
    required = [
        'experiment_id', 'seeds', 'families', 'steps', 'warmup', 'batch_size',
        'replay_capacity', 'eval_interval', 'checkpoint_interval', 'log_interval',
        'device', 'sensor_range', 'development_maps', 'gate', 'n_agents', 'reshape',
        'set_mode', 'probe_modes',
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f'protocol missing {key}')
    if cfg['n_agents'] != 3 or cfg['device'] != 'cpu':
        raise ValueError('maneuver protocol expects n_agents=3 device=cpu')
    if cfg['set_mode'] != 'pool':
        raise ValueError('maneuver suite freezes set_mode=pool')
    if probe_mode not in cfg['probe_modes']:
        raise ValueError(f'probe_mode {probe_mode} not in protocol probe_modes {cfg["probe_modes"]}')
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
        raise ValueError(f'train_ball_m must be frozen {FROZEN_GOAL_BALL_M}')
    if abs(reshape.train_speed_mps - FROZEN_CAPTURE_SPEED_MPS) > 1e-9:
        raise ValueError(f'train_speed_mps must be frozen {FROZEN_CAPTURE_SPEED_MPS}')
    if reshape.capture_latch is not False:
        raise ValueError('requires capture_latch=False')
    gate = float(cfg['gate']['per_seed_joint_success_fraction'])
    if abs(gate - 0.6) > 1e-9:
        raise ValueError('L0 gate must be 0.6')


def update_registry(root, payload, primary=False):
    registry_path = ROOT / 'results/masac_v4_parallel_runs.json'
    registry = {'primary_active_run_file': 'results/masac_v4_active_run.json', 'arms': []}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
        except Exception:
            pass
    arms = [a for a in registry.get('arms', []) if a.get('out') != str(root)]
    arms.append(payload)
    registry['arms'] = arms
    registry['updated_at'] = time.time()
    registry['updated_at_asia_shanghai'] = time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')
    if primary:
        registry['primary'] = {
            'out': payload['out'],
            'pid': payload['pid'],
            'experiment_id': payload['experiment_id'],
            'phase': payload['phase'],
            'kind': payload['kind'],
            'probe_mode': payload.get('probe_mode'),
            'seeds': payload.get('seeds'),
            'do_not_auto_launch_further': True,
            'suite': payload.get('suite_experiment_id'),
        }
        registry['maneuver_suite'] = {
            'suite_experiment_id': payload.get('suite_experiment_id'),
            'updated_at_asia_shanghai': registry['updated_at_asia_shanghai'],
            'note': 'Predictive Maneuverability paper suite; do_not_auto_launch further stages',
        }
    write(registry_path, registry)


def run(args):
    cfg = json.loads(args.protocol.read_text())
    auth = json.loads(args.authorization_protocol.read_text()) if args.authorization_protocol else {}
    probe_mode = args.probe_mode or cfg.get('default_probe_mode', 'full')
    validate_protocol(cfg, probe_mode)
    reshape = reshape_from_protocol(cfg['reshape'])
    probe_config = ProbeConfig(
        world=tuple(cfg.get('world', [240.0, 240.0, 90.0])),
        body_radius=float(cfg.get('body_radius', 1.0)),
        safety_margin=float(cfg.get('safety_margin', 0.0)),
        dt=float(cfg.get('dt', 0.2)),
        horizon=float(cfg.get('probe_horizon_s', 2.0)),
        clearance_scale=float(cfg.get('clearance_scale', 40.0)),
    )
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
    warm_note = (
        'Cold start: Legacy SharedMASAC Path-C L0 checkpoints are architecture-incompatible '
        'with V4 pool+RiskEncoder; paper suite trains maneuver arms from scratch under matched '
        'C-reshape + frozen 4/2. Standard-C baseline comparison uses completed Path-C L0 results.'
    )
    if args.warm_start is not None:
        raise ValueError('Warm-start not supported for maneuver arms (architecture mismatch); omit --warm-start')

    if args.lock_path is None:
        lockpath = ROOT / 'results' / 'locks' / f'masac_v4_maneuver_{probe_mode}_{args.out.name}.lock'
    else:
        lockpath = args.lock_path if args.lock_path.is_absolute() else ROOT / args.lock_path
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    lock = lockpath.open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f'Another runner holds maneuver lock {lockpath}') from exc
    lock.write(f'pid={os.getpid()} out={args.out} probe_mode={probe_mode}\n')
    lock.flush()

    args.out.mkdir(parents=True, exist_ok=False)
    root = args.out.resolve()
    suite_id = auth.get('suite_experiment_id', cfg.get('suite_experiment_id', cfg['experiment_id']))
    experiment_id = f"{suite_id}_{probe_mode}"

    model0 = ManeuverMASAC(
        n=3, capacity=8, seed=0, device=cfg['device'], set_mode=cfg['set_mode'],
        probe_mode=probe_mode, probe_config=probe_config,
    )
    param_counts = model0.parameter_counts()
    del model0

    manifest = {
        'experiment_id': experiment_id,
        'suite_experiment_id': suite_id,
        'mode': 'predictive_maneuverability_aware_masac',
        'smoke': args.smoke,
        'level': 'L0',
        'set_mode': cfg['set_mode'],
        'probe_mode': probe_mode,
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
        },
        'probe_config': {
            'world': list(probe_config.world),
            'body_radius': probe_config.body_radius,
            'dt': probe_config.dt,
            'horizon_s': probe_config.horizon,
            'clearance_scale': probe_config.clearance_scale,
        },
        'parameter_counts': param_counts,
        'warm_start_path': None,
        'warm_start_mode': 'cold',
        'warm_start_note': warm_note,
        'baseline_comparison': cfg.get('baseline_comparison'),
        'authorization_protocol': str(args.authorization_protocol.resolve()) if args.authorization_protocol else None,
        'authorization': auth.get('authorization'),
        'failure_action': auth.get('failure_action', 'save_and_diagnose'),
        'success_action': auth.get(
            'success_action',
            'declare maneuver arm evidence; do not auto-launch Attention/L1/further stages',
        ),
        'do_not_auto_launch_further': True,
        'command': sys.argv,
        'note': (
            f'Predictive Maneuverability probe_mode={probe_mode}; V4 pool+RiskEncoder; '
            'C reshape + frozen capture 4/2; cold start; no Attention/L1 auto-launch.'
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

    kind = f'v4_maneuver_{probe_mode}'
    if not args.smoke:
        arm_payload = {
            'out': str(root),
            'pid': os.getpid(),
            'started_at': time.time(),
            'kind': kind,
            'phase': 'running',
            'experiment_id': experiment_id,
            'suite_experiment_id': suite_id,
            'probe_mode': probe_mode,
            'set_mode': cfg['set_mode'],
            'seeds': seeds,
            'steps': steps,
            'reshape': reshape.to_dict(),
            'warm_start': None,
            'lock_path': str(lockpath),
            'log': str(root / 'train.log'),
            'authorization': 'user t4s17 + phone comprehensive maneuver suite 2026-11-03',
            'do_not_auto_launch_further': True,
            'caffeinate_pid': args.caffeinate_pid,
        }
        update_registry(root, arm_payload, primary=(probe_mode == 'full' or args.primary))
        active = {
            'out': str(root),
            'pid': os.getpid(),
            'kind': kind,
            'phase': 'running',
            'experiment_id': experiment_id,
            'suite_experiment_id': suite_id,
            'probe_mode': probe_mode,
            'seeds': seeds,
            'target_total_steps': steps,
            'reshape': reshape.to_dict(),
            'capture': {'ball_m': 4.0, 'speed_mps': 2.0, 'latch': False},
            'warm_start': None,
            'warm_start_note': warm_note,
            'authorization': 'user t4s17 + phone comprehensive maneuver suite 2026-11-03',
            'log': str(root / 'train.log'),
            'do_not_auto_launch_further': True,
            'started_at_asia_shanghai': time.strftime('%Y-%m-%d %H:%M Asia/Shanghai'),
            'updated_at_asia_shanghai': time.strftime('%Y-%m-%d %H:%M Asia/Shanghai'),
            'monitor_note': (
                f'Maneuver {probe_mode} running; gate ≥0.6 on 12 maps frozen 4/2; '
                'paper suite; no auto further stages'
            ),
            'baseline_comparison': cfg.get('baseline_comparison'),
        }
        if probe_mode == 'full' or args.primary:
            write(ROOT / 'results/masac_v4_active_run.json', active)

    completed = []
    begin = time.perf_counter()
    seed = None
    step = 0
    phase = 'init'
    try:
        for run_index, seed in enumerate(seeds):
            directory = root / f'maneuver_{probe_mode}_{seed}'
            directory.mkdir()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            explore = np.random.default_rng(seed)
            model = ManeuverMASAC(
                n=3,
                capacity=capacity,
                seed=seed,
                device=cfg['device'],
                set_mode=cfg['set_mode'],
                probe_mode=probe_mode,
                probe_config=probe_config,
            )
            write(directory / 'warm_start.json', {'mode': 'cold', 'note': warm_note})
            write(directory / 'parameter_counts.json', model.parameter_counts())
            message(root, f'COLD-START probe_mode={probe_mode} seed={seed} params={model.parameter_counts()}')
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
                warm_start={'mode': 'cold'},
                probe_mode=probe_mode,
            )
            message(root, f'START probe={probe_mode} seed={seed} family={family0} reshape={reshape.to_dict()}')
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
                        'probe_mode': probe_mode,
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
                        'suite_experiment_id': suite_id,
                        'arm': f'maneuver_{probe_mode}',
                        'probe_mode': probe_mode,
                        'set_mode': cfg['set_mode'],
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
                        'warm_start': False,
                        'recent_success': sum(x['success'] for x in recent) / len(recent) if recent else None,
                        **state['losses'],
                    }
                    write(root / 'status.json', status)
                    append(directory / 'metrics.jsonl', {'kind': 'progress', **status})
                    message(
                        root,
                        f'[maneuver-{probe_mode} seed={seed}] {step}/{steps} upd={state["updates"]} '
                        f'{rate:.1f}step/s val={state["last_eval"]}',
                    )
                if step % eval_every == 0 or step == steps:
                    phase = 'evaluating'
                    records = evaluate(
                        model, eval_maps, directory, step, cfg, reshape, probe_mode, probe_config
                    )
                    state['last_eval'] = sum(x['success'] for x in records) / len(records)
                    state['last_eval_step'] = step
                    n_ball = sum(1 for x in records if any(x.get('ever_in_ball_by_uav') or []))
                    eff = [
                        x['mean_path_efficiency_direct_over_path']
                        for x in records
                        if x.get('mean_path_efficiency_direct_over_path') is not None
                    ]
                    message(
                        root,
                        f'EVAL probe={probe_mode} seed={seed} step={step}: '
                        f'success={sum(x["success"] for x in records)}/{len(records)} '
                        f'ever_ball_maps={n_ball}/{len(records)} '
                        f'mean_eff={float(np.mean(eff)) if eff else None}',
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
                'probe_mode': probe_mode,
                'warm_start': {'mode': 'cold'},
            })
            write(directory / 'completion.json', completed[-1])
        complete = {
            'complete': True,
            'scope': 'smoke' if args.smoke else f'maneuver_{probe_mode} L0 12-map gate',
            'suite_experiment_id': suite_id,
            'experiment_id': experiment_id,
            'probe_mode': probe_mode,
            'set_mode': cfg['set_mode'],
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
            'warm_start_path': None,
            'do_not_auto_launch_further': True,
            'baseline_comparison': cfg.get('baseline_comparison'),
            'next_if_pass': 'Record maneuver arm evidence; await user before Attention/L1/further stages.',
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
            if probe_mode == 'full' or args.primary:
                write(
                    ROOT / 'results/masac_v4_active_run.json',
                    {
                        'out': str(root),
                        'pid': os.getpid(),
                        'kind': kind,
                        'phase': 'complete',
                        'experiment_id': experiment_id,
                        'suite_experiment_id': suite_id,
                        'probe_mode': probe_mode,
                        'all_seeds_gate_met': complete['all_seeds_gate_met'],
                        'do_not_auto_launch_further': True,
                        'completed_at': time.time(),
                        'updated_at_asia_shanghai': time.strftime('%Y-%m-%d %H:%M Asia/Shanghai'),
                    },
                )
        message(root, f'COMPLETE maneuver-{probe_mode} gate_met={complete["all_seeds_gate_met"]}')
    except BaseException as e:
        write(
            root / 'error.json',
            {
                'error': repr(e),
                'traceback': traceback.format_exc(),
                'seed': seed,
                'step': step,
                'phase': phase,
                'probe_mode': probe_mode,
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
    parser.add_argument('--probe-mode', type=str, default=None, choices=list(ManeuverMASAC.PROBE_MODES))
    parser.add_argument('--torch-threads', type=int, default=3)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--warm-start', type=Path, default=None)
    parser.add_argument('--lock-path', type=Path, default=None)
    parser.add_argument('--primary', action='store_true', help='write this arm into masac_v4_active_run.json')
    parser.add_argument('--caffeinate-pid', type=int, default=None)
    run(parser.parse_args())
