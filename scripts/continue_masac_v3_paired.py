#!/usr/bin/env python3
"""Continue two completed V3 Standard MASAC histories with matched budgets.

Intermediate model checkpoints do not have a replay.  Full snapshots include
the replay, environment, optimizer, episode and all RNG state, plus hashes.
"""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from scripts.continue_masac_v3 import capture, restore_completed_run, save_replay
from scripts.train_masac_v3 import evaluate, write
from scripts.train_masac_v3_curriculum import (
    evaluation_diagnostics, new_episode_stats, restore_curriculum, update_stats,
)
from uav_planning.masac_v3_curriculum import make_episode
from uav_planning.shared_masac import SharedMASAC
from uav_planning.structured import StructuredMission

COMMON = {
    'structured.py': 'uav_planning/structured.py',
    'environment.py': 'uav_planning/environment.py',
    'shared_masac.py': 'uav_planning/shared_masac.py',
    'train_masac_v3.py': 'scripts/train_masac_v3.py',
    'continue_masac_v3.py': 'scripts/continue_masac_v3.py',
    'protocol.json': 'configs/masac_v3/protocol.json',
}
CURRICULUM = {
    'masac_v3_curriculum.py': 'uav_planning/masac_v3_curriculum.py',
    'train_masac_v3_curriculum.py': 'scripts/train_masac_v3_curriculum.py',
    'curriculum_protocol.json': 'configs/masac_v3/curriculum_protocol.json',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def restore_parent(arm, parent_dir, seed):
    """Restore the actual parent implementation without a reset or warmup."""
    parent = Path(parent_dir)
    cfg = read_json(parent / 'protocol.json')
    directory = parent / f'standard_{seed}'
    if arm == 'curriculum':
        return restore_curriculum(directory)
    if arm != 'original':
        raise ValueError(f'Unknown training history {arm}')
    model = SharedMASAC(n=cfg.get('n_agents', 3), capacity=cfg['replay_capacity'], seed=seed)
    cp, env, obs, exploration = restore_completed_run(model, directory, cfg)
    return model, env, obs, exploration, cp


def check_parent(arm, parent_dir, seeds, protocol):
    """Verify completion, source snapshots, saved state and replay provenance."""
    parent = Path(parent_dir)
    marker = parent / 'completion.json'
    if not marker.exists() or (parent / 'error.json').exists() or not read_json(marker).get('complete'):
        raise ValueError(f'Parent must be complete and error-free: {parent}')
    cfg = read_json(parent / 'protocol.json')
    if (cfg.get('smoke') or cfg['steps'] != protocol['start_step'] or
            cfg['n_agents'] != 3 or cfg['arms'] != ['standard'] or
            cfg['seeds'] != protocol['seeds'] or cfg['warmup'] != protocol['warmup'] or
            cfg['batch_size'] != protocol['batch_size'] or
            cfg['replay_capacity'] != protocol['replay_capacity'] or
            cfg['validation_map_seeds'] != protocol['validation_map_seeds'] or
            cfg['training_map_seed_offset'] != protocol['training_map_seed_offset'] or
            cfg['layouts'] != protocol['layouts']):
        raise ValueError(f'Parent protocol does not match paired budget: {parent}')
    if any(cfg.get(k) != v for k, v in dict(lr=.0003, gamma=.99, tau=.005, initial_alpha=.1, hidden=128).items()):
        raise ValueError('Parent protocol disagrees with implemented SAC constants')
    if not set(seeds).issubset(cfg['seeds']):
        raise ValueError('Requested seed absent from parent')
    hashes = read_json(parent / 'source_hashes.json')
    # Check every stored snapshot against its manifest, including design docs.
    for name, digest in hashes.items():
        if sha(parent / 'source_snapshot' / name) != digest:
            raise ValueError(f'Parent snapshot hash mismatch: {name}')
    live = {**COMMON, **(CURRICULUM if arm == 'curriculum' else {})}
    for name, path in live.items():
        if name not in hashes or sha(ROOT / path) != hashes[name]:
            raise ValueError(f'Active source differs from parent: {path}')
    if arm == 'curriculum':
        report = read_json(parent / 'final_analysis.json')
        replay_hashes = {x['training_seed']: x['final_replay_coverage']['replay_sha256'] for x in report['seeds']}
    elif arm == 'original':
        report = read_json(ROOT / 'results/masac_v3_diagnosis_20261102/budget_replay_coverage_all_final.json')
        replay_hashes = {int(x['run'].split('_')[-1]): x['replay_sha256'] for x in report['rows']}
    else:
        raise ValueError(f'Unknown training history {arm}')
    provenance = {}
    for seed in seeds:
        directory = parent / f'standard_{seed}'
        audit = read_json(directory / 'audit.json')
        step = protocol['start_step']
        if (audit['updates'] != step - cfg['warmup'] or audit['total_steps'] != step or
                audit['replay_size'] != cfg['replay_capacity'] or
                audit['replay_checkpoint_alignment_step'] != step):
            raise ValueError(f'Parent audit is not aligned for {arm}/{seed}')
        ckpt_path, replay_path = directory / f'checkpoint_{step}.pt', directory / 'replay_final.npz'
        if not (directory / f'evaluation_{step}.json').is_file():
            raise ValueError(f'Missing parent final evaluation for {arm}/{seed}')
        cp = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        required = ('actor', 'critics', 'targets', 'logalpha', 'actor_optimizer', 'critic_optimizer',
                    'alpha_optimizer', 'replay_rng', 'torch_rng', 'python_rng', 'numpy_rng',
                    'exploration_rng', 'environment', 'episode', 'episode_return')
        if any(k not in cp for k in required):
            raise ValueError(f'Incomplete checkpoint for {arm}/{seed}')
        if (cp['cfg'] != cfg or cp['step'] != step or cp['seed'] != seed or cp['mode'] != 'standard' or
                cp['replay_size'] != cfg['replay_capacity'] or cp['replay_index'] != 0):
            raise ValueError(f'Checkpoint protocol or replay position mismatch for {arm}/{seed}')
        env = cp['environment']
        if not 0 <= env['t'] < cfg['horizon'] or not np.isfinite(cp['episode_return']):
            raise ValueError(f'Invalid in-progress episode for {arm}/{seed}')
        if arm == 'curriculum' and (not cp.get('exact_resume') or cp.get('curriculum_stage') != 'full' or
                                    'episode_start_step' not in cp or 'episode_stats' not in cp):
            raise ValueError(f'Incomplete curriculum full-stage checkpoint for seed {seed}')
        replay_digest = sha(replay_path)
        if replay_digest != replay_hashes[seed]:
            raise ValueError(f'Parent replay differs from prior recorded hash for {arm}/{seed}')
        with np.load(replay_path) as replay:
            expected = {'obs': (cfg['replay_capacity'], 3, 162),
                        'next_obs': (cfg['replay_capacity'], 3, 162),
                        'actions': (cfg['replay_capacity'], 3, 3),
                        'rewards': (cfg['replay_capacity'],), 'done': (cfg['replay_capacity'],)}
            for key, shape in expected.items():
                if replay[key].shape != shape:
                    raise ValueError(f'Parent replay shape mismatch: {arm}/{seed}/{key}')
        provenance[str(seed)] = dict(checkpoint_sha256=sha(ckpt_path), replay_sha256=replay_digest,
                                     episode=cp['episode'], episode_return=cp['episode_return'],
                                     replay_size=cp['replay_size'], replay_index=cp['replay_index'])
    return cfg, provenance


def save_full_snapshot(directory, model, env, exploration, *, step, episode, episode_return,
                       episode_start, episode_stats, partial_episode, seed, parent_cfg, arm):
    """Save an aligned state pair; the manifest is written only after hashes."""
    snap = directory / 'snapshots' / str(step)
    snap.mkdir(parents=True, exist_ok=False)
    state = snap / 'state'
    state.mkdir()
    snapshot_cfg = copy.deepcopy(parent_cfg)
    snapshot_cfg['steps'] = step
    write(snap / 'protocol.json', snapshot_cfg)
    cp = capture(model, env, exploration, step=step, episode=episode,
                 episode_return=episode_return, seed=seed, cfg=snapshot_cfg)
    cp.update(curriculum_stage='full' if arm == 'curriculum' else None,
              episode_start_step=episode_start, episode_stats=copy.deepcopy(episode_stats),
              partial_episode=partial_episode, exact_resume=True, history_arm=arm)
    ckpt = state / f'checkpoint_{step}.pt'
    torch.save(cp, ckpt)
    save_replay(model, state)
    manifest = dict(complete=True, arm=arm, seed=seed, step=step,
                    checkpoint_sha256=sha(ckpt), replay_sha256=sha(state / 'replay_final.npz'),
                    protocol_sha256=sha(snap / 'protocol.json'))
    write(snap / 'complete.json', manifest)
    return manifest


def restore_snapshot(snapshot):
    """Restore a verified complete pair created by save_full_snapshot."""
    snapshot = Path(snapshot)
    manifest = read_json(snapshot / 'complete.json')
    if not manifest.get('complete'):
        raise ValueError('Snapshot is not complete')
    state = snapshot / 'state'
    step, arm, seed = manifest['step'], manifest['arm'], manifest['seed']
    if (sha(snapshot / 'protocol.json') != manifest['protocol_sha256'] or
            sha(state / f'checkpoint_{step}.pt') != manifest['checkpoint_sha256'] or
            sha(state / 'replay_final.npz') != manifest['replay_sha256']):
        raise ValueError('Snapshot hash mismatch')
    if arm == 'curriculum':
        model, env, obs, exploration, cp = restore_curriculum(state)
    elif arm == 'original':
        cfg = read_json(snapshot / 'protocol.json')
        model = SharedMASAC(n=cfg['n_agents'], capacity=cfg['replay_capacity'], seed=seed)
        cp, env, obs, exploration = restore_completed_run(model, state, cfg)
    else:
        raise ValueError('Unknown snapshot arm')
    if cp.get('history_arm') != arm or cp['step'] != step or cp['seed'] != seed or not cp.get('exact_resume'):
        raise ValueError('Snapshot metadata mismatch')
    return model, env, obs, exploration, cp


def validate_snapshot_source(snapshot, protocol, protocol_path, *, source_root=ROOT):
    """Reject recovery if the producing run or current execution code changed."""
    snapshot = Path(snapshot).resolve()
    source_root = Path(source_root)
    producing_run = snapshot.parents[2]
    producing_protocol = read_json(producing_run / 'protocol.json')
    if any(producing_protocol.get(k) != v for k, v in protocol.items()):
        raise ValueError('Snapshot producing run has a different paired protocol')
    hashes = read_json(producing_run / 'source_hashes.json')
    for name, digest in hashes.items():
        if sha(producing_run / 'source_snapshot' / name) != digest:
            raise ValueError(f'Producing run snapshot hash mismatch: {name}')
    live = {**COMMON, **CURRICULUM,
            'continue_masac_v3_paired.py': 'scripts/continue_masac_v3_paired.py',
            'paired_300k_protocol.json': str(Path(protocol_path).resolve())}
    for name, path in live.items():
        if name not in hashes or sha(source_root / path) != hashes[name]:
            raise ValueError(f'Current code or paired protocol differs from producing run: {name}')
    return producing_run


def run(args):
    protocol = read_json(args.protocol)
    if (protocol['arms'] != ['curriculum', 'original'] or protocol['seeds'] != [411, 422, 433] or
            protocol['start_step'] != 100000 or protocol['steps'] != 300000 or
            protocol['full_resume_interval'] != 50000 or protocol['checkpoint_interval'] != 10000 or
            protocol['eval_interval'] != 2000):
        raise ValueError('Unexpected paired continuation protocol')
    torch.set_num_threads(1)
    lock = (ROOT / 'results/masac_v3_training.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another V3 training job holds the training lock')
    try:
        args.out.mkdir(parents=True, exist_ok=False)
    except BaseException:
        lock.close()
        raise
    out = args.out.resolve()
    arm, seed, step = None, None, protocol['start_step']
    try:
        parents = {a: (ROOT / protocol['parents'][a]).resolve() for a in protocol['arms']}
        items = [(a, s) for a in protocol['arms'] for s in protocol['seeds']]
        recovery = None
        if args.resume_snapshot:
            if args.run_item:
                raise ValueError('Snapshot recovery and selected parent runs use separate shards')
            recovery = args.resume_snapshot.resolve()
            manifest = read_json(recovery / 'complete.json')
            arm, seed = manifest['arm'], manifest['seed']
            if (arm, seed) not in items or not protocol['start_step'] < manifest['step'] < protocol['steps']:
                raise ValueError('Snapshot is not a recoverable in-progress paired run')
            items = [(arm, seed)]
        elif args.run_item:
            requested = []
            for item in args.run_item:
                try:
                    group, text_seed = item.split(':', 1)
                    requested.append((group, int(text_seed)))
                except ValueError as exc:
                    raise ValueError(f'Invalid selected run {item!r}; expected arm:seed') from exc
            if len(set(requested)) != len(requested) or not set(requested).issubset(items):
                raise ValueError('Duplicate or unknown selected paired run')
            items = [item for item in items if item in requested]
        elif args.smoke:
            items = [(a, 411) for a in protocol['arms']]
        partial_run = bool(recovery or args.run_item)
        # All checks occur while the global writer lock is held, before any new
        # learner step. A failure writes an error marker to this new directory.
        parent_data = {}
        for a in sorted({x[0] for x in items}):
            seeds = [s for group, s in items if group == a]
            parent_data[a] = check_parent(a, parents[a], seeds, protocol)
        initial_step = manifest['step'] if recovery else protocol['start_step']
        target = initial_step + 32 if args.smoke else protocol['steps']
        if recovery:
            parent_cfg = parent_data[arm][0]
            expected_cfg = copy.deepcopy(parent_cfg)
            expected_cfg['steps'] = initial_step
            if read_json(recovery / 'protocol.json') != expected_cfg:
                raise ValueError('Snapshot protocol does not match original parent provenance')
            validate_snapshot_source(recovery, protocol, args.protocol)
        runtime = copy.deepcopy(protocol)
        runtime.update(pid=os.getpid(), command=sys.argv, started_at=time.time(), smoke=args.smoke,
                       target_step=target, recovery_only=partial_run, selected_runs=items,
                       resume_snapshot=str(recovery) if recovery else None)
        write(out / 'protocol.json', runtime)
        snapshot = out / 'source_snapshot'
        snapshot.mkdir()
        sources = {**COMMON, **CURRICULUM,
                   'continue_masac_v3_paired.py': 'scripts/continue_masac_v3_paired.py',
                   'MASAC_V3_PAIRED_300K_20261103.md': 'docs/design/MASAC_V3_PAIRED_300K_20261103.md',
                   'MASAC_V3_PAIRED_300K_RECOVERY_20261103.md': 'docs/design/MASAC_V3_PAIRED_300K_RECOVERY_20261103.md',
                   'paired_300k_protocol.json': str(args.protocol.resolve())}
        for name, src in sources.items():
            shutil.copy2(ROOT / src, snapshot / name)
        if read_json(snapshot / 'paired_300k_protocol.json') != protocol:
            raise ValueError('Paired protocol changed during startup')
        write(out / 'source_hashes.json', {p.name: sha(p) for p in snapshot.iterdir()})
        write(out / 'parent_provenance.json', {a: {'path': str(parents[a]), 'seeds': v[1]}
                                              for a, v in parent_data.items()})
        if not args.smoke:
            write(ROOT / 'results/masac_v3_active_run.json', dict(run=str(out), log=str(args.log.resolve())))
        total_new = sum(target - (initial_step if recovery else protocol['start_step']) for _ in items)
        finished_new = 0
        print(f'PAIRED STANDARD MASAC BUDGET | PID={os.getpid()} | CPU threads=1 | runs={items} '
              f'| total new steps={total_new:,}\nOutput: {out}', flush=True)
        for run_index, (arm, seed) in enumerate(items):
            parent_cfg = parent_data[arm][0]
            directory = out / f'{arm}_{seed}'
            directory.mkdir()
            if recovery:
                model, env, obs, exploration, old = restore_snapshot(recovery)
                start_step = initial_step
            else:
                model, env, obs, exploration, old = restore_parent(arm, parents[arm], seed)
                start_step = protocol['start_step']
            # Check the source files again after decompression: an edit during
            # the load must never be presented as an exact continuation.
            if not recovery:
                source = parents[arm] / f'standard_{seed}'
                saved = parent_data[arm][1][str(seed)]
                if (sha(source / f'checkpoint_{start_step}.pt') != saved['checkpoint_sha256'] or
                        sha(source / 'replay_final.npz') != saved['replay_sha256']):
                    raise ValueError(f'Parent changed while loading: {arm}/{seed}')
            if arm == 'curriculum' and (old.get('curriculum_stage') != 'full' or env.curriculum_stage != 'full'):
                raise ValueError('Curriculum state must remain in the full stage')
            episode, episode_return = old['episode'], old['episode_return']
            if arm == 'curriculum' or recovery:
                episode_start = old.get('episode_start_step')
                stats = copy.deepcopy(old['episode_stats'])
                partial_episode = old.get('partial_episode', False)
            else:
                # Original parent checkpoint has no pre-resume per-UAV stats.
                episode_start = None if env.t else start_step
                stats = new_episode_stats(env)
                partial_episode = bool(env.t)
            updates = start_step - protocol['warmup']
            initial_actor = torch.cat([x.detach().flatten() for x in model.actor.parameters()]).clone()
            losses, recent, begin = {}, [], time.perf_counter()
            if recovery:
                last_eval, eval_step = None, None
            else:
                records = read_json(parents[arm] / f'standard_{seed}' / f'evaluation_{start_step}.json')
                last_eval = sum(x['success'] for x in records) / len(records)
                eval_step = start_step
            write(directory / 'resume.json', dict(arm=arm, seed=seed, parent=str(parents[arm]),
                parent_provenance=parent_data[arm][1][str(seed)], source_snapshot=str(recovery) if recovery else None,
                start_step=start_step, episode=episode, episode_return=episode_return,
                replay_size=model.size, replay_index=model.index, partial_episode=partial_episode,
                restored_rng=['torch', 'numpy_global', 'python', 'replay', 'exploration']))
            write(directory / 'parameters.json', dict(actor=sum(p.numel() for p in model.actor.parameters()),
                twin_critic=sum(p.numel() for p in model.q.parameters())))
            print(f'RESUMED {arm}/{seed} | step={start_step:,} | updates={updates:,} '
                  f'| replay={model.size:,} index={model.index} | episode={episode}', flush=True)

            def status(phase='training'):
                elapsed = time.perf_counter() - begin
                new = step - start_step
                rate = new / elapsed if new and elapsed else 0.
                return dict(phase=phase, mode=arm, seed=seed, step=step, total_steps=target,
                    resume_step=start_step, new_steps=new, updates=updates, episodes=episode,
                    overall_completed_new_steps=finished_new + new, overall_total_new_steps=total_new,
                    run_index=run_index, run_count=len(items), elapsed_s=elapsed,
                    steps_per_s=rate, eta_s=(target-step)/rate if rate else None,
                    validation_success=last_eval, last_evaluation_step=eval_step,
                    recent_training_success=sum(x['success'] for x in recent)/len(recent) if recent else None,
                    updated_at=time.time(), pid=os.getpid(), **losses)

            step = start_step
            write(out / 'status.json', status())
            with (directory / 'training.jsonl').open('x') as log:
                for step in range(start_step + 1, target + 1):
                    action = model.act(obs)
                    old_pos = env.p.copy()
                    next_obs, reward, done, trunc, info = env.step(action)
                    model.push(obs, action, reward, next_obs, done)
                    update_stats(stats, env, old_pos, action)
                    obs = next_obs
                    episode_return += reward
                    losses = model.update(protocol['batch_size'])
                    updates += 1
                    if not all(np.isfinite(x) for x in losses.values()):
                        raise ValueError('Non-finite loss')
                    if done or trunc:
                        row = dict(kind='episode', arm=arm, step=step, episode=episode,
                            episode_start_step=episode_start, partial_episode_stats=partial_episode,
                            return_=episode_return, terminated=done, truncated=trunc,
                            closest_goal_m_by_uav=stats['closest_goal_m'].tolist(),
                            min_boundary_m=stats['min_boundary_m'], path_m_by_uav=stats['path_m'].tolist(),
                            mean_action_norm=stats['action_norm_sum']/max(stats['samples'],1), **info)
                        log.write(json.dumps(row)+'\n')
                        recent = (recent + [row])[-30:]
                        episode += 1
                        episode_return, episode_start, partial_episode = 0., step, False
                        if arm == 'curriculum':
                            env, obs, stage = make_episode(seed, episode, step, parent_cfg)
                            if stage != 'full':
                                raise ValueError('Curriculum unexpectedly left the full stage')
                        else:
                            env = StructuredMission(mode='standard', layout=parent_cfg['layouts'][episode % 3])
                            obs = env.reset(parent_cfg['training_map_seed_offset'] + seed * 1000 + episode)
                        stats = new_episode_stats(env)
                    if step % 100 == 0 or step == target:
                        s = status(); write(out / 'status.json', s)
                        log.write(json.dumps(dict(kind='progress', **s))+'\n'); log.flush()
                        val = '?' if last_eval is None else f'{last_eval:.1%}'
                        print(f'[{arm} seed={seed}] {step:,}/{target:,} (+{step-start_step:,}) '
                              f'| upd={updates:,} ep={episode} | actor={losses["actor_loss"]:+.4f} '
                              f'critic={losses["critic_loss"]:.4f} alpha={losses["alpha"]:.4f} '
                              f'| {s["steps_per_s"]:.1f} steps/s ETA={s["eta_s"]/60 if s["eta_s"] is not None else 0:.1f}min '
                              f'| val@{eval_step}={val}', flush=True)
                    if step % protocol['checkpoint_interval'] == 0 or step == target:
                        write(out / 'status.json', status('saving_model'))
                        cp = capture(model, env, exploration, step=step, episode=episode,
                                     episode_return=episode_return, seed=seed, cfg=runtime)
                        cp.update(history_arm=arm, exact_resume=False, note='Model-only; no aligned replay')
                        torch.save(cp, directory / f'checkpoint_{step}.pt')
                    if step % protocol['full_resume_interval'] == 0 or step == target:
                        write(out / 'status.json', status('saving_full_snapshot'))
                        saved = save_full_snapshot(directory, model, env, exploration, step=step,
                            episode=episode, episode_return=episode_return, episode_start=episode_start,
                            episode_stats=stats, partial_episode=partial_episode, seed=seed,
                            parent_cfg=parent_cfg, arm=arm)
                        print(f'SNAPSHOT {arm}/{seed} step={step:,} replay_sha256={saved["replay_sha256"]}', flush=True)
                    if step % protocol['eval_interval'] == 0 or step == target:
                        write(out / 'status.json', status('evaluating'))
                        maps = protocol['validation_map_seeds'][:2] if args.smoke else protocol['validation_map_seeds']
                        records = evaluate(model, 'standard', maps, directory, step)
                        evaluation_diagnostics(directory, step, records)
                        eval_step = step; last_eval = sum(x['success'] for x in records) / len(records)
                        print(f'EVAL [{arm} seed={seed} step={step}] success={sum(x["success"] for x in records)}/{len(records)} '
                              f'| static={sum(x["static_collision"] for x in records)} '
                              f'boundary={sum(x["boundary_collision"] for x in records)} '
                              f'peer={sum(x["uav_collision"] for x in records)} '
                              f'moving={sum(x["dynamic_collision"] for x in records)} (flags may overlap)', flush=True)
                        write(out / 'latest_evaluation.json', dict(mode=arm, seed=seed, step=step, results=records))
                        write(out / 'status.json', status())
            change = float(torch.linalg.vector_norm(torch.cat([x.detach().flatten() for x in model.actor.parameters()])-initial_actor))
            if updates != target - protocol['warmup'] or not np.isfinite(change) or change <= 0:
                raise ValueError(f'Update audit failed for {arm}/{seed}')
            write(directory / 'audit.json', dict(arm=arm, seed=seed, total_steps=target, start_step=start_step,
                added_steps=target-start_step, updates=updates, replay_size=model.size, replay_index=model.index,
                actor_parameter_change_since_resume=change, final_validation_success=last_eval,
                full_snapshot=str(directory / 'snapshots' / str(target)), wall_s=time.perf_counter()-begin))
            finished_new += target - start_step
            print(f'FINISHED {arm}/{seed} | total={target:,} | updates={updates:,} | val={last_eval}', flush=True)
            del model, env, obs, exploration
            gc.collect()
        completion_name = 'recovery_completion.json' if partial_run else 'completion.json'
        write(out / completion_name, dict(complete=True, runs=len(items), target_steps=target,
            recovery_only=partial_run, selected_runs=items, smoke=args.smoke, finished_at=time.time()))
        final_status = status('complete')
        final_status.update(complete=True, recovery_only=partial_run,
                            overall_completed_new_steps=total_new)
        write(out / 'status.json', final_status)
    except BaseException as exc:
        write(out / 'error.json', dict(error=repr(exc), arm=arm, seed=seed, step=step, updated_at=time.time()))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume-snapshot', type=Path, help='Recover one arm/seed into a new result directory')
    parser.add_argument('--run-item', action='append', default=[], metavar='ARM:SEED',
                        help='Continue one or more specified parent runs as an explicitly incomplete shard')
    args = parser.parse_args()
    if not args.smoke and args.log is None:
        parser.error('--log is required for the active-run pointer')
    run(args)
