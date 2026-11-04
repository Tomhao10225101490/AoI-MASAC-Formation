#!/usr/bin/env python3
"""Extend completed V3 seeds without changing their environment or update rule.

Only final checkpoints with a matching complete replay archive can be resumed.
Intermediate evaluation checkpoints are intentionally not accepted as resumable.
"""
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scripts.train_masac_v3 import evaluate, write  # noqa: E402
from uav_planning.shared_masac import SharedMASAC  # noqa: E402
from uav_planning.structured import StructuredMission  # noqa: E402

ALGORITHM_FILES = ('uav_planning/structured.py', 'uav_planning/environment.py',
                   'uav_planning/shared_masac.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_parent(parent, seeds, total_steps):
    if not (parent / 'completion.json').exists() or (parent / 'error.json').exists():
        raise ValueError('Parent batch must be complete and error-free before continuation')
    if not json.loads((parent / 'completion.json').read_text()).get('complete'):
        raise ValueError('Parent batch must be complete and error-free before continuation')
    cfg = json.loads((parent / 'protocol.json').read_text())
    if cfg['smoke'] or cfg['n_agents'] != 3 or cfg['arms'] != ['standard']:
        raise ValueError('This development continuation requires a three-agent standard batch')
    # The original SharedMASAC implements these constants. Do not merely log
    # arbitrary configuration values as if they were effective hyperparameters.
    effective = dict(lr=3e-4, gamma=.99, tau=.005, initial_alpha=.1, hidden=128)
    if any(cfg.get(k) != v for k, v in effective.items()):
        raise ValueError('Parent protocol disagrees with effective V3 update constants')
    hashes = json.loads((parent / 'source_hashes.json').read_text())
    for name in ALGORITHM_FILES:
        snapshot = parent / 'source_snapshot' / Path(name).name
        if sha(ROOT / name) != hashes[Path(name).name] or sha(snapshot) != hashes[Path(name).name]:
            raise ValueError(f'Algorithm source changed since parent run: {name}')
    for seed in seeds:
        if seed not in cfg['seeds']:
            raise ValueError(f'No parent training seed {seed}')
        directory = parent / f'standard_{seed}'
        audit = json.loads((directory / 'audit.json').read_text())
        if audit['updates'] != cfg['steps'] - cfg['warmup']:
            raise ValueError(f'Unexpected parent update count for {seed}')
        if total_steps <= cfg['steps']:
            raise ValueError('Continuation must add environment steps')
        if not (directory / f'checkpoint_{cfg["steps"]}.pt').is_file():
            raise ValueError(f'Missing final checkpoint for {seed}')
        if not (directory / 'replay_final.npz').is_file():
            raise ValueError(f'Missing final replay for {seed}')
    return cfg


def restore_completed_run(model, directory, cfg):
    """Restore all saved learner, environment and RNG state; no warmup/reset."""
    checkpoint = torch.load(directory / f'checkpoint_{cfg["steps"]}.pt',
                            map_location='cpu', weights_only=False)
    if checkpoint['step'] != cfg['steps'] or checkpoint['cfg'] != cfg:
        raise ValueError('Final checkpoint and parent protocol do not match')
    model.actor.load_state_dict(checkpoint['actor'])
    model.q.load_state_dict(checkpoint['critics'])
    model.target.load_state_dict(checkpoint['targets'])
    with torch.no_grad():
        model.logalpha.copy_(checkpoint['logalpha'])
    model.ao.load_state_dict(checkpoint['actor_optimizer'])
    model.qo.load_state_dict(checkpoint['critic_optimizer'])
    model.eo.load_state_dict(checkpoint['alpha_optimizer'])
    size, index = checkpoint['replay_size'], checkpoint['replay_index']
    if size > model.capacity or not 0 <= index < model.capacity:
        raise ValueError('Replay capacity is incompatible with parent state')
    with np.load(directory / 'replay_final.npz') as replay:
        for attr, key in (('obs', 'obs'), ('next', 'next_obs'), ('actions', 'actions'),
                          ('rewards', 'rewards'), ('done', 'done')):
            values = replay[key]
            array = getattr(model, attr)
            if values.shape != array[:size].shape:
                raise ValueError(f'Invalid replay shape: {key}')
            array[:size] = values
    model.size, model.index = size, index
    model.rng.bit_generator.state = checkpoint['replay_rng']
    env = StructuredMission(n=model.n, mode=checkpoint['mode'],
                            layout=checkpoint['environment']['layout'])
    env.__dict__.update(copy.deepcopy(checkpoint['environment']))
    obs = env.observe()
    exploration = np.random.default_rng()
    exploration.bit_generator.state = checkpoint['exploration_rng']
    random.setstate(checkpoint['python_rng'])
    np.random.set_state(checkpoint['numpy_rng'])
    torch.set_rng_state(checkpoint['torch_rng'])
    return checkpoint, env, obs, exploration


def capture(model, env, exploration, *, step, episode, episode_return, seed, cfg):
    checkpoint = model.checkpoint()
    checkpoint.update(step=step, episode=episode, mode='standard', seed=seed, cfg=cfg,
                      environment=copy.deepcopy(env.__dict__), episode_return=episode_return,
                      python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                      exploration_rng=exploration.bit_generator.state)
    return checkpoint


def save_replay(model, directory):
    np.savez_compressed(directory / 'replay_final.npz', obs=model.obs[:model.size],
                        next_obs=model.next[:model.size], actions=model.actions[:model.size],
                        rewards=model.rewards[:model.size], done=model.done[:model.size])


def run(args):
    parent = args.parent.resolve()
    cfg = check_parent(parent, args.seeds, args.steps)
    # Single-writer guard for continuation jobs; the original job must already
    # have completed. The advisory lock is released automatically on process exit.
    lock = (ROOT / 'results/masac_v3_training.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another V3 continuation job holds the training lock')
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out.resolve()
    start_step = cfg['steps']
    parent_cfg = copy.deepcopy(cfg)
    cfg.update(phase='controlled budget extension; development only', steps=args.steps,
               seeds=args.seeds, pid=os.getpid(), command=sys.argv, parent=str(parent),
               resume_step=start_step, eval_interval=2000, checkpoint_interval=10000,
               started_at=time.time(), diagnostic_plan='docs/design/MASAC_V3_DIAGNOSTIC_PLAN.md')
    write(out / 'protocol.json', cfg)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    for name in (*ALGORITHM_FILES, 'scripts/train_masac_v3.py', 'scripts/continue_masac_v3.py',
                 'docs/design/MASAC_V3_DIAGNOSTIC_PLAN.md', 'configs/masac_v3/protocol.json'):
        shutil.copy2(ROOT / name, snapshot / Path(name).name)
    write(out / 'source_hashes.json', {f.name: sha(f) for f in snapshot.iterdir()})
    write(ROOT / 'results/masac_v3_active_run.json', dict(run=str(out), log=str(out) + '.log'))
    torch.set_num_threads(1)
    additional_per_seed = args.steps - start_step
    total_new = additional_per_seed * len(args.seeds)
    print(f'CONTROLLED BUDGET EXTENSION | PID={os.getpid()} | CPU threads=1\n'
          f'Parent: {parent}\nOutput: {out}\n'
          f'Network / reward / optimizer unchanged | seeds={args.seeds}\n'
          f'Each seed: {start_step:,} -> {args.steps:,}; new steps total={total_new:,}', flush=True)
    step, seed = start_step, args.seeds[0]
    try:
        for run_index, seed in enumerate(args.seeds):
            directory = out / f'standard_{seed}'
            directory.mkdir()
            model = SharedMASAC(capacity=cfg['replay_capacity'], seed=seed)
            old, env, obs, exploration = restore_completed_run(
                model, parent / f'standard_{seed}', parent_cfg)
            initial = torch.cat([x.detach().flatten() for x in model.actor.parameters()]).clone()
            episode, episode_return = old['episode'], old['episode_return']
            updates = start_step - cfg['warmup']
            last_records = json.loads((parent / f'standard_{seed}/evaluation_{start_step}.json').read_text())
            last_eval = sum(x['success'] for x in last_records) / len(last_records)
            eval_step = start_step
            losses, recent, begin = {}, [], time.perf_counter()
            write(directory / 'parameters.json', dict(
                actor=sum(p.numel() for p in model.actor.parameters()),
                twin_critic=sum(p.numel() for p in model.q.parameters())))
            write(directory / 'resume.json', dict(
                source=str(parent / f'standard_{seed}'), step=start_step,
                episode=episode, episode_return=episode_return, replay_size=model.size,
                checkpoint_sha256=sha(parent / f'standard_{seed}/checkpoint_{start_step}.pt'),
                replay_sha256=sha(parent / f'standard_{seed}/replay_final.npz'),
                rng_restored=['torch', 'numpy_global', 'python', 'replay', 'exploration']))
            print(f'RESUMED standard/{seed} | step={start_step:,} updates={updates:,} '
                  f'replay={model.size:,} | next map episode={episode}', flush=True)

            def status(phase='training'):
                elapsed = time.perf_counter() - begin
                new_steps = step - start_step
                rate = new_steps / elapsed if new_steps else 0.
                return dict(
                    phase=phase, mode='standard', seed=seed, step=step, total_steps=args.steps,
                    resume_step=start_step, new_steps=new_steps, updates=updates, episodes=episode,
                    elapsed_s=elapsed, steps_per_s=rate,
                    eta_s=(args.steps - step) / rate if rate else None,
                    overall_completed_new_steps=run_index * additional_per_seed + new_steps,
                    overall_total_new_steps=total_new, run_index=run_index, run_count=len(args.seeds),
                    validation_success=last_eval, last_evaluation_step=eval_step,
                    recent_training_success=sum(x['success'] for x in recent) / len(recent) if recent else None,
                    recent_remaining_distance_m=float(np.mean([x['remaining_distance'] for x in recent])) if recent else None,
                    updated_at=time.time(), pid=os.getpid(), **losses,
                )

            step = start_step
            write(out / 'status.json', status())
            with (directory / 'training.jsonl').open('w') as log:
                for step in range(start_step + 1, args.steps + 1):
                    action = model.act(obs)
                    next_obs, reward, done, trunc, info = env.step(action)
                    model.push(obs, action, reward, next_obs, done)
                    obs = next_obs
                    episode_return += reward
                    losses = model.update(cfg['batch_size'])
                    updates += 1
                    if not all(np.isfinite(v) for v in losses.values()):
                        raise ValueError('Non-finite training loss')
                    if done or trunc:
                        row = dict(kind='episode', step=step, episode=episode,
                                   return_=episode_return, **info)
                        log.write(json.dumps(row) + '\n')
                        recent = (recent + [row])[-30:]
                        episode += 1
                        episode_return = 0.
                        env = StructuredMission(mode='standard', layout=cfg['layouts'][episode % 3])
                        obs = env.reset(cfg['training_map_seed_offset'] + seed * 1000 + episode)
                    if step % 100 == 0 or step == args.steps:
                        s = status()
                        write(out / 'status.json', s)
                        log.write(json.dumps(dict(kind='progress', **s)) + '\n')
                        log.flush()
                        print(f'[standard seed={seed}] {step:6d}/{args.steps} '
                              f'(+{step-start_step:,}) | upd={updates:,} ep={episode} '
                              f'| actor={losses["actor_loss"]:+.4f} critic={losses["critic_loss"]:.4f} '
                              f'alpha={losses["alpha"]:.4f} | {s["steps_per_s"]:.1f} steps/s '
                              f'ETA={s["eta_s"]/60:.1f}min | val@{eval_step}={last_eval:.1%}', flush=True)
                    if step % cfg['checkpoint_interval'] == 0 or step == args.steps:
                        write(out / 'status.json', status('saving_checkpoint'))
                        checkpoint = capture(model, env, exploration, step=step, episode=episode,
                                             episode_return=episode_return, seed=seed, cfg=cfg)
                        torch.save(checkpoint, directory / f'checkpoint_{step}.pt')
                    if step % cfg['eval_interval'] == 0 or step == args.steps:
                        write(out / 'status.json', status('evaluating'))
                        records = evaluate(model, 'standard', cfg['validation_map_seeds'], directory, step)
                        eval_step = step
                        last_eval = sum(x['success'] for x in records) / len(records)
                        print(f'EVAL [standard seed={seed} step={step}] '
                              f'success={sum(x["success"] for x in records)}/{len(records)} '
                              f'| static={sum(x["static_collision"] for x in records)} '
                              f'boundary={sum(x["boundary_collision"] for x in records)} '
                              f'peer={sum(x["uav_collision"] for x in records)} '
                              f'moving={sum(x["dynamic_collision"] for x in records)}', flush=True)
                        write(out / 'latest_evaluation.json', dict(mode='standard', seed=seed, step=step, results=records))
                        write(out / 'status.json', status())
                write(out / 'status.json', status('saving_final_replay'))
                save_replay(model, directory)
                final = torch.cat([x.detach().flatten() for x in model.actor.parameters()])
                change = float(torch.linalg.vector_norm(final - initial))
                if change <= 0 or updates != args.steps - cfg['warmup']:
                    raise ValueError('Continuation update audit failed')
                write(directory / 'audit.json', dict(
                    updates=updates, added_updates=additional_per_seed, total_steps=args.steps,
                    replay_size=model.size, actor_parameter_change_since_resume=change,
                    wall_s=time.perf_counter()-begin, success_rate=last_eval, parent=str(parent),
                    replay_checkpoint_alignment_step=args.steps))
                print(f'FINISHED standard/{seed}: total={args.steps:,}, '
                      f'added={additional_per_seed:,}, actor_delta={change:.6f}', flush=True)
        write(out / 'completion.json', dict(complete=True, runs=len(args.seeds), steps=args.steps,
                                            additional_steps_per_seed=additional_per_seed, finished_at=time.time()))
        write(out / 'status.json', {**status('complete'), 'complete': True})
    except BaseException as error:
        write(out / 'error.json', dict(error=repr(error), seed=seed, step=step, updated_at=time.time()))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT / 'results/masac_v3_baseline_20261102')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[411, 422, 433])
    run(parser.parse_args())
