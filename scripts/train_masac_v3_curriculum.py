#!/usr/bin/env python3
"""Run the frozen, baseline-only V3 curriculum from independent initializations."""
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
import numpy as np
import torch
from scripts.train_masac_v3 import evaluate, write
from scripts.continue_masac_v3 import capture, save_replay, restore_completed_run
from uav_planning.shared_masac import SharedMASAC
from uav_planning.masac_v3_curriculum import make_episode


def stage_for_step(stages, step):
    return [stage['name'] for stage in stages if step >= stage['start_step']][-1]


def new_episode_stats(env):
    return dict(closest_goal_m=np.linalg.norm(env.goal-env.p, axis=1),
                min_boundary_m=float(np.min(np.minimum(env.p-env.radius, env.world-env.radius-env.p))),
                path_m=np.zeros(env.n), action_norm_sum=0., samples=0)


def update_stats(stats, env, old_pos, action):
    stats['closest_goal_m'] = np.minimum(stats['closest_goal_m'], np.linalg.norm(env.goal-env.p, axis=1))
    stats['min_boundary_m'] = min(stats['min_boundary_m'], float(np.min(np.minimum(env.p-env.radius, env.world-env.radius-env.p))))
    stats['path_m'] += np.linalg.norm(env.p-old_pos, axis=1)
    stats['action_norm_sum'] += float(np.linalg.norm(action, axis=1).sum())
    stats['samples'] += env.n


def evaluation_diagnostics(directory, step, records):
    for row in records:
        with np.load(directory / f'eval_{step}_{row["map_seed"]}.npz') as data:
            distances = np.linalg.norm(data['positions']-data['goals'][None], axis=-1)
            row['closest_goal_m_by_uav'] = distances.min(axis=0).tolist()
            row['final_goal_m_by_uav'] = distances[-1].tolist()
    write(directory / f'evaluation_{step}.json', records)


def restore_curriculum(directory):
    """Restore a final checkpoint with its matching replay and curriculum state."""
    directory = Path(directory)
    cfg = json.loads((directory.parent/'protocol.json').read_text())
    meta = torch.load(directory/f'checkpoint_{cfg["steps"]}.pt', map_location='cpu', weights_only=False)
    if not meta.get('exact_resume'):
        raise ValueError('Checkpoint does not have an aligned replay')
    env, _, stage = make_episode(meta['seed'], meta['episode'], meta['episode_start_step'], cfg)
    if stage != meta['curriculum_stage']:
        raise ValueError('Saved curriculum stage does not match the episode schedule')
    model = SharedMASAC(capacity=3000 if cfg['smoke'] else cfg['replay_capacity'], seed=meta['seed'])
    cp, base_env, _, exploration = restore_completed_run(model, directory, cfg)
    env.__dict__.update(copy.deepcopy(base_env.__dict__))
    obs = env.observe()
    return model, env, obs, exploration, cp


def run(args):
    base = json.loads((ROOT / 'configs/masac_v3/protocol.json').read_text())
    curriculum = json.loads(args.protocol.read_text())
    cfg = {**base, **curriculum}
    seeds = [411] if args.smoke else cfg['seeds']
    steps = 256 if args.smoke else cfg['steps']
    warmup = 128 if args.smoke else cfg['warmup']
    eval_interval = 256 if args.smoke else cfg['eval_interval']
    checkpoint_interval = 256 if args.smoke else cfg['checkpoint_interval']
    cfg.update(arms=['standard'], seeds=seeds, steps=steps, warmup=warmup,
               smoke=args.smoke, pid=os.getpid(), command=sys.argv, started_at=time.time())
    if cfg['n_agents'] != 3 or not cfg['stages'] or cfg['stages'][0]['start_step'] != 0:
        raise ValueError('Expected three agents and a curriculum starting at step zero')
    torch.set_num_threads(1)
    lock = (ROOT/'results/masac_v3_training.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another V3 training job holds the training lock')
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out.resolve()
    step, seed = 0, seeds[0]
    try:
        write(out/'protocol.json', cfg)
        snapshot = out/'source_snapshot'
        snapshot.mkdir()
        files = ('uav_planning/structured.py', 'uav_planning/environment.py',
                 'uav_planning/shared_masac.py', 'uav_planning/masac_v3_curriculum.py',
                 'scripts/train_masac_v3.py', 'scripts/continue_masac_v3.py',
                 'scripts/train_masac_v3_curriculum.py', 'configs/masac_v3/protocol.json',
                 'docs/design/MASAC_V3_GOAL_CURRICULUM_20261102.md')
        for name in files:
            shutil.copy2(ROOT/name, snapshot/Path(name).name)
        shutil.copy2(args.protocol, snapshot/'curriculum_protocol.json')
        write(out/'source_hashes.json', {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.iterdir()})
        if not args.smoke:
            write(ROOT/'results/masac_v3_active_run.json', dict(run=str(out), log=str(args.log.resolve())))
        print(f'CURRICULUM STANDARD MASAC | PID={os.getpid()} | CPU threads=1 | seeds={seeds} | steps/seed={steps:,}\nOutput: {out}', flush=True)
        for run_index, seed in enumerate(seeds):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            exploration = np.random.default_rng(seed)
            directory = out/f'standard_{seed}'
            directory.mkdir()
            model = SharedMASAC(capacity=3000 if args.smoke else cfg['replay_capacity'], seed=seed)
            write(directory/'parameters.json', dict(actor=sum(p.numel() for p in model.actor.parameters()),
                                                     twin_critic=sum(p.numel() for p in model.q.parameters())))
            initial = torch.cat([x.detach().flatten() for x in model.actor.parameters()]).clone()
            step = episode = updates = 0
            env, obs, stage = make_episode(seed, episode, step, cfg)
            episode_start = step
            stats = new_episode_stats(env)
            episode_return, last_eval, eval_step, losses = 0., None, None, {}
            begin = time.perf_counter()
            recent = []

            def status(phase='training'):
                elapsed = time.perf_counter()-begin
                rate = step/elapsed if elapsed and step else 0.
                return dict(phase=phase, mode='standard', seed=seed, step=step,
                    total_steps=steps, updates=updates, episodes=episode, elapsed_s=elapsed,
                    steps_per_s=rate, eta_s=(steps-step)/rate if rate else None,
                    validation_success=last_eval, last_evaluation_step=eval_step,
                    actual_stage=stage, planned_stage=stage_for_step(cfg['stages'], step),
                    recent_training_success=sum(x['success'] for x in recent)/len(recent) if recent else None,
                    run_index=run_index, run_count=len(seeds), updated_at=time.time(), pid=os.getpid(), **losses)

            write(out/'status.json', status())
            print(f'START seed={seed} | stage={stage}', flush=True)
            with (directory/'training.jsonl').open('w') as log:
                for step in range(1, steps+1):
                    action = exploration.uniform(-1, 1, (3, 3)).astype(np.float32) if step <= warmup else model.act(obs)
                    old_pos = env.p.copy()
                    next_obs, reward, done, trunc, info = env.step(action)
                    model.push(obs, action, reward, next_obs, done)
                    update_stats(stats, env, old_pos, action)
                    obs = next_obs
                    episode_return += reward
                    if step > warmup:
                        losses = model.update(cfg['batch_size']); updates += 1
                        if not all(np.isfinite(v) for v in losses.values()):
                            raise ValueError('Non-finite training loss')
                    if done or trunc:
                        row = dict(kind='episode', step=step, episode=episode, episode_start_step=episode_start,
                            actual_stage=stage, return_=episode_return, terminated=done, truncated=trunc,
                            closest_goal_m_by_uav=stats['closest_goal_m'].tolist(),
                            min_boundary_m=stats['min_boundary_m'], path_m_by_uav=stats['path_m'].tolist(),
                            mean_action_norm=stats['action_norm_sum']/max(stats['samples'],1), **info)
                        log.write(json.dumps(row)+'\n'); recent = (recent+[row])[-30:]
                        episode += 1; episode_return = 0.; episode_start = step
                        prior_stage = stage
                        env, obs, stage = make_episode(seed, episode, step, cfg)
                        stats = new_episode_stats(env)
                        if stage != prior_stage:
                            print(f'STAGE seed={seed} actual_step={step:,} {prior_stage} -> {stage}', flush=True)
                    if step % 100 == 0 or step == steps:
                        s = status(); write(out/'status.json', s)
                        log.write(json.dumps(dict(kind='progress', **s))+'\n'); log.flush()
                        val = '?' if last_eval is None else f'{last_eval:.1%}'
                        print(f'[standard seed={seed}] {step:,}/{steps:,} | stage={stage} planned={s["planned_stage"]} | upd={updates:,} ep={episode} | actor={losses.get("actor_loss",0):+.4f} critic={losses.get("critic_loss",0):.4f} alpha={losses.get("alpha",.1):.4f} | {s["steps_per_s"]:.1f} steps/s ETA={s["eta_s"]/60 if s["eta_s"] is not None else 0:.1f}min | val@{eval_step}={val}', flush=True)
                    if step % checkpoint_interval == 0 or step == steps:
                        cp = capture(model, env, exploration, step=step, episode=episode,
                                     episode_return=episode_return, seed=seed, cfg=cfg)
                        cp.update(curriculum_stage=stage, episode_start_step=episode_start,
                                  episode_stats=copy.deepcopy(stats), exact_resume=(step == steps))
                        torch.save(cp, directory/f'checkpoint_{step}.pt')
                    if step % eval_interval == 0 or step == steps:
                        write(out/'status.json', status('evaluating'))
                        records = evaluate(model, 'standard', cfg['validation_map_seeds'][:2] if args.smoke else cfg['validation_map_seeds'], directory, step)
                        evaluation_diagnostics(directory, step, records)
                        eval_step = step; last_eval = sum(x['success'] for x in records)/len(records)
                        print(f'EVAL [seed={seed} step={step}] success={sum(x["success"] for x in records)}/{len(records)} | static={sum(x["static_collision"] for x in records)} boundary={sum(x["boundary_collision"] for x in records)} peer={sum(x["uav_collision"] for x in records)} moving={sum(x["dynamic_collision"] for x in records)}', flush=True)
                        write(out/'latest_evaluation.json', dict(mode='standard', seed=seed, step=step, results=records))
                        write(out/'status.json', status())
                write(out/'status.json', status('saving_final_replay'))
                save_replay(model, directory)
                change = float(torch.linalg.vector_norm(torch.cat([x.detach().flatten() for x in model.actor.parameters()])-initial))
                if updates != steps-warmup or change <= 0:
                    raise ValueError('Update audit failed')
                write(directory/'audit.json', dict(total_steps=steps, updates=updates, replay_size=model.size,
                    replay_checkpoint_alignment_step=steps, exact_resume_checkpoint=f'checkpoint_{steps}.pt',
                    intermediate_checkpoints_exact_resume=False, actor_parameter_change=change,
                    success_rate=last_eval, wall_s=time.perf_counter()-begin))
                print(f'FINISHED seed={seed} | updates={updates:,} | final val={last_eval:.1%}', flush=True)
        write(out/'completion.json', dict(complete=True, runs=len(seeds), steps=steps, finished_at=time.time()))
        write(out/'status.json', {**status('complete'), 'complete':True})
    except BaseException as error:
        write(out/'error.json', dict(error=repr(error), seed=seed, step=step, updated_at=time.time()))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--log', type=Path, help='Path used for the active-run pointer')
    parser.add_argument('--smoke', action='store_true')
    opts = parser.parse_args()
    if not opts.smoke and opts.log is None:
        parser.error('--log is required for an auditable active-run pointer')
    run(opts)
