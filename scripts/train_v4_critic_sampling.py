#!/usr/bin/env python3
"""Run the frozen V3 Standard SAC Critic-sampling isolation pilot.

This is not the V4 architecture or maneuverability experiment.  Both arms
fork the same verified 300k Standard checkpoints, with independent Actor and
Critic batches and fully saved Replay and sampler state.
"""
import argparse
import copy
import fcntl
import gc
import json
import math
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.continue_masac_v3 import capture, restore_completed_run, save_replay, sha  # noqa: E402
from scripts.continue_masac_v3_paired import restore_snapshot, validate_snapshot_source  # noqa: E402
from scripts.train_masac_v3 import evaluate, write  # noqa: E402
from uav_planning.critic_sampling_diagnostic import CriticSamplingUpdater  # noqa: E402
from uav_planning.environment import swept_distance  # noqa: E402
from uav_planning.shared_masac import SharedMASAC  # noqa: E402
from uav_planning.structured import StructuredMission, segment_box_hit  # noqa: E402

SOURCE_FILES = {
    'structured.py':'uav_planning/structured.py',
    'environment.py':'uav_planning/environment.py',
    'shared_masac.py':'uav_planning/shared_masac.py',
    'critic_sampling_diagnostic.py':'uav_planning/critic_sampling_diagnostic.py',
    'train_v4_critic_sampling.py':'scripts/train_v4_critic_sampling.py',
    'train_masac_v3.py':'scripts/train_masac_v3.py',
    'continue_masac_v3.py':'scripts/continue_masac_v3.py',
    'continue_masac_v3_paired.py':'scripts/continue_masac_v3_paired.py',
    'train_masac_v3_curriculum.py':'scripts/train_masac_v3_curriculum.py',
    'masac_v3_curriculum.py':'uav_planning/masac_v3_curriculum.py',
    'V4_CRITIC_SAMPLING_PROTOCOL.md':'docs/design/V4_CRITIC_SAMPLING_PROTOCOL.md',
    'critic_sampling_diagnostic.json':'configs/masac_v4/critic_sampling_diagnostic.json',
    'paired_300k_protocol.json':'configs/masac_v3/paired_300k_protocol.json',
    'curriculum_protocol.json':'configs/masac_v3/curriculum_protocol.json',
    'protocol.json':'configs/masac_v3/protocol.json',
}


def read(path):
    return json.loads(Path(path).read_text())


def validate_config(cfg):
    required = dict(parent_history='original', parent_step=300000, additional_steps=20000,
        target_total_steps=320000, parent_updates=299000, n_agents=3, batch_size=128,
        stratified_done_quota=32, replay_capacity=100000, gamma=.99, tau=.005,
        learning_rate=.0003, gradient_clip_norm=5., eval_interval=2000,
        checkpoint_interval=10000, device='cpu', seed_multiplier=104729)
    if any(cfg.get(k)!=v for k,v in required.items()):
        raise ValueError('Configuration differs from the frozen diagnostic')
    if cfg['arms'] != ['uniform','stratified'] or cfg['seeds'] != [411,422,433]:
        raise ValueError('Unexpected diagnostic arms or seeds')
    if cfg['rng_offsets'] != dict(critic_numpy=31001, actor_numpy=31002,
                                   target_torch=31003, actor_torch=31004):
        raise ValueError('Sampling RNG streams differ from frozen protocol')


def preflight(cfg):
    parent=(ROOT/cfg['parent_run']).resolve()
    completion=read(parent/'completion.json')
    if not completion.get('complete') or completion.get('runs')!=6 or completion.get('smoke') or (parent/'error.json').exists():
        raise ValueError('Parent paired batch must be complete and error-free')
    paired_file=ROOT/'configs/masac_v3/paired_300k_protocol.json'
    paired=read(paired_file)
    provenance={}
    for seed in cfg['seeds']:
        snap=parent/f'original_{seed}'/'snapshots/300000'
        validate_snapshot_source(snap,paired,paired_file)
        manifest=read(snap/'complete.json')
        checkpoint=snap/'state/checkpoint_300000.pt'
        replay=snap/'state/replay_final.npz'
        if (not manifest.get('complete') or manifest['arm']!='original' or manifest['seed']!=seed or
                manifest['step']!=cfg['parent_step'] or sha(checkpoint)!=manifest['checkpoint_sha256'] or
                sha(replay)!=manifest['replay_sha256'] or sha(snap/'protocol.json')!=manifest['protocol_sha256']):
            raise ValueError(f'Parent checkpoint/Replay mismatch for seed {seed}')
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if (cp['step']!=cfg['parent_step'] or cp['seed']!=seed or cp['history_arm']!='original' or
                cp['replay_size']!=cfg['replay_capacity'] or not cp.get('exact_resume')):
            raise ValueError(f'Parent checkpoint metadata mismatch for seed {seed}')
        provenance[str(seed)]=dict(snapshot=str(snap),checkpoint_sha256=manifest['checkpoint_sha256'],
                                   replay_sha256=manifest['replay_sha256'],protocol_sha256=manifest['protocol_sha256'])
    return provenance


def snapshot_sources(out):
    snapshot=out/'source_snapshot'; snapshot.mkdir()
    for name,path in SOURCE_FILES.items():
        shutil.copy2(ROOT/path,snapshot/name)
    hashes={p.name:sha(p) for p in snapshot.iterdir()}
    write(out/'source_hashes.json',hashes)
    return hashes


def validate_diagnostic_source(snapshot, cfg):
    producing=Path(snapshot).resolve().parents[2]
    manifest=read(producing/'run_manifest.json')
    if manifest['configuration'] != cfg:
        raise ValueError('Recovery source used a different diagnostic protocol')
    hashes=read(producing/'source_hashes.json')
    for name,path in SOURCE_FILES.items():
        if name not in hashes or sha(producing/'source_snapshot'/name)!=hashes[name]:
            raise ValueError(f'Recovery source or live code changed: {name}')
        # A report can be updated after the run; the saved copy still has to
        # match its manifest. Runtime code and configs must remain identical.
        if not name.endswith('.md') and sha(ROOT/path)!=hashes[name]:
            raise ValueError(f'Recovery source or live code changed: {name}')


def save_snapshot(directory,model,updater,env,exploration,*,step,episode,episode_return,
                  episode_start,episode_stats,partial_episode,parent_cfg,seed,arm):
    snap=directory/'snapshots'/str(step); snap.mkdir(parents=True,exist_ok=False)
    state=snap/'state'; state.mkdir()
    cfg=copy.deepcopy(parent_cfg);cfg['steps']=step
    write(snap/'protocol.json',cfg)
    cp=capture(model,env,exploration,step=step,episode=episode,episode_return=episode_return,seed=seed,cfg=cfg)
    cp.update(diagnostic_sampling=updater.state_dict(),diagnostic_arm=arm,diagnostic_seed=seed,
              episode_start_step=episode_start,episode_stats=copy.deepcopy(episode_stats),
              partial_episode=partial_episode,exact_resume=True)
    checkpoint=state/f'checkpoint_{step}.pt'
    torch.save(cp,checkpoint)
    save_replay(model,state)
    manifest=dict(complete=True,arm=arm,seed=seed,step=step,device=updater.device.type,
        checkpoint_sha256=sha(checkpoint),replay_sha256=sha(state/'replay_final.npz'),
        protocol_sha256=sha(snap/'protocol.json'))
    write(snap/'complete.json',manifest)
    return manifest


def restore_diagnostic_snapshot(snapshot, *, device):
    snap=Path(snapshot).resolve(); manifest=read(snap/'complete.json'); step=manifest['step']
    state=snap/'state'; checkpoint=state/f'checkpoint_{step}.pt'; replay=state/'replay_final.npz'
    if (not manifest.get('complete') or manifest['device']!=device or
            sha(checkpoint)!=manifest['checkpoint_sha256'] or sha(replay)!=manifest['replay_sha256'] or
            sha(snap/'protocol.json')!=manifest['protocol_sha256']):
        raise ValueError('Incomplete, changed or wrong-device diagnostic snapshot')
    cfg=read(snap/'protocol.json')
    model=SharedMASAC(n=cfg['n_agents'],capacity=cfg['replay_capacity'],seed=manifest['seed'])
    cp,env,obs,exploration=restore_completed_run(model,state,cfg)
    if (not cp.get('exact_resume') or cp['diagnostic_arm']!=manifest['arm'] or
            cp['diagnostic_seed']!=manifest['seed'] or cp['step']!=step):
        raise ValueError('Diagnostic snapshot metadata mismatch')
    updater=CriticSamplingUpdater(model,arm=manifest['arm'],seed=manifest['seed'],device=device)
    updater.load_state_dict(cp['diagnostic_sampling'])
    return model,updater,env,obs,exploration,cp


class PolicyAdapter:
    def __init__(self,updater): self.updater=updater
    def act(self,obs,deterministic=False): return self.updater.act(obs,deterministic)


def enrich_evaluation(directory,step,rows):
    """Add per-UAV metrics and terminal-event attribution from saved geometry."""
    for row in rows:
        path=directory/f'eval_{step}_{row["map_seed"]}.npz'
        with np.load(path,allow_pickle=False) as data:
            p,g,dynamic,boxes,dr=(data[k] for k in ('positions','goals','dynamic','buildings','dynamic_radii'))
            velocity=np.vstack([np.zeros((1,3)), np.linalg.norm(np.diff(p,axis=0),axis=-1)/.2])
            distance=np.linalg.norm(p-g[None],axis=-1)
            captures=(distance<4.)&(velocity<2.)
            row['closest_goal_m_by_uav']=distance.min(axis=0).tolist()
            row['final_goal_m_by_uav']=distance[-1].tolist()
            row['final_speed_mps_by_uav']=velocity[-1].tolist()
            row['path_m_by_uav']=np.linalg.norm(np.diff(p,axis=0),axis=-1).sum(axis=0).tolist()
            row['ever_captured_by_uav']=captures.any(axis=0).tolist()
            row['max_simultaneous_captures']=int(captures.sum(axis=1).max())
            row['near_30m_slow_2mps_fraction_by_uav']=((distance<30.)&(velocity<2.)).mean(axis=0).tolist()
            row['timeout']=bool(not row['success'] and not any(row[k] for k in
                ('static_collision','boundary_collision','dynamic_collision','uav_collision')) and np.isclose(row['time'],70.))
            contact={}
            if len(p)>1:
                before,after=p[-2],p[-1]
                static=segment_box_hit(before,after,boxes[:,0]-1.,boxes[:,1]+1.).any(axis=1)
                moving=(swept_distance(before[:,None]-dynamic[-2][None],after[:,None]-dynamic[-1][None]) < dr[None]+1.).any(axis=1)
                pair=swept_distance(before[:,None]-before[None],after[:,None]-after[None]);np.fill_diagonal(pair,np.inf)
                peer=(pair<2.).any(axis=1)
                boundary=((after<1.)|(after>np.array([239.,239.,89.]))).any(axis=1)
                for name,mask in (('static',static),('dynamic',moving),('peer',peer),('boundary',boundary)):
                    contact[name]=velocity[-1][mask].tolist()
            row['terminal_contact_speed_mps_by_type']=contact
            row['terminal_contact_speed_note']='Discrete-step endpoint speed of involved UAV; not continuous-time impact speed.'
    write(directory/f'evaluation_{step}.json',rows)


def new_stats(env):
    return dict(closest_goal_m=np.linalg.norm(env.goal-env.p,axis=1),
        path_m=np.zeros(env.n),min_boundary_m=float(np.min(np.minimum(env.p-env.radius,env.world-env.radius-env.p))))


def update_stats(stats,env,old):
    stats['closest_goal_m']=np.minimum(stats['closest_goal_m'],np.linalg.norm(env.goal-env.p,axis=1))
    stats['path_m']+=np.linalg.norm(env.p-old,axis=1)
    stats['min_boundary_m']=min(stats['min_boundary_m'],float(np.min(np.minimum(env.p-env.radius,env.world-env.radius-env.p))))


def run(args):
    cfg=read(args.protocol); validate_config(cfg)
    if cfg != read(ROOT/'configs/masac_v4/critic_sampling_diagnostic.json'):
        raise ValueError('Runtime protocol does not match the frozen source protocol')
    torch.set_num_threads(1)
    lock=(ROOT/'results/masac_v3_training.lock').open('a+')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close(); raise RuntimeError('Another V3/V4 training job holds the lock')
    try: args.out.mkdir(parents=True,exist_ok=False)
    except BaseException:
        lock.close(); raise
    out=args.out.resolve();arm=None;seed=None;step=cfg['parent_step']
    try:
        provenance=preflight(cfg)
        items=[(arm,seed) for seed in cfg['seeds'] for arm in cfg['arms']]
        recovery=None
        if args.resume_snapshot:
            if args.run_item: raise ValueError('Recovery and selected branch are separate modes')
            recovery=args.resume_snapshot.resolve();meta=read(recovery/'complete.json')
            arm,seed=meta['arm'],meta['seed']
            if (arm,seed) not in items or not cfg['parent_step']<meta['step']<cfg['target_total_steps']:
                raise ValueError('Recovery snapshot is outside this diagnostic')
            validate_diagnostic_source(recovery,cfg)
            items=[(arm,seed)]
        elif args.run_item:
            arm,s= args.run_item.split(':',1);seed=int(s)
            if (arm,seed) not in items: raise ValueError('Unknown arm/seed')
            items=[(arm,seed)]
        elif args.smoke:
            items=[('uniform',411),('stratified',411)]
        partial=bool(recovery or args.run_item)
        target=(meta['step']+32 if args.smoke and recovery else cfg['parent_step']+32) if args.smoke else cfg['target_total_steps']
        start_override=meta['step'] if recovery else cfg['parent_step']
        runtime=copy.deepcopy(cfg);runtime.update(pid=os.getpid(),started_at=time.time(),command=sys.argv,
            smoke=args.smoke,steps=target,total_steps=target,selected_runs=items,recovery_only=partial)
        write(out/'protocol.json',runtime)
        source_hashes=snapshot_sources(out)
        write(out/'run_manifest.json',dict(configuration=cfg,provenance=provenance,source_hashes=source_hashes,
              pid=os.getpid(),smoke=args.smoke,selected_runs=items,recovery_only=partial,started_at=time.time()))
        if not args.smoke:
            write(ROOT/'results/masac_v3_active_run.json',dict(run=str(out),log=str(args.log.resolve())))
        total_new=sum(target-(start_override if recovery else cfg['parent_step']) for _ in items)
        completed_new=0; evaluations={}
        with (out/'train.log').open('x',buffering=1) as log,(out/'metrics.jsonl').open('x',buffering=1) as metrics:
          def emit(message):
              print(message,flush=True);log.write(message+'\n');log.flush()
          emit(f'V4 CRITIC SAMPLING DIAGNOSTIC | pid={os.getpid()} | CPU | runs={items} | new={total_new:,}')
          for run_index,(arm,seed) in enumerate(items):
            directory=out/f'{arm}_{seed}';directory.mkdir()
            if recovery:
                model,updater,env,obs,exploration,old=restore_diagnostic_snapshot(recovery,device=cfg['device'])
                start=meta['step']
            else:
                model,env,obs,exploration,old=restore_snapshot(Path(provenance[str(seed)]['snapshot']))
                updater=CriticSamplingUpdater(model,arm=arm,seed=seed,batch_size=cfg['batch_size'],
                                               done_quota=cfg['stratified_done_quota'],device=cfg['device'])
                start=cfg['parent_step']
            write(directory/'parameters.json',dict(
                actor=sum(p.numel() for p in model.actor.parameters()),
                twin_critic=sum(p.numel() for p in model.q.parameters()),
                target_twin_critic=sum(p.numel() for p in model.target.parameters()),
                replay_capacity=model.capacity,device=cfg['device']))
            episode,episode_return=old['episode'],old['episode_return']
            episode_start=old.get('episode_start_step')
            if old.get('episode_stats') is not None:
                stats=copy.deepcopy(old['episode_stats'])
                partial_episode=bool(old.get('partial_episode',False))
            else:
                partial_episode=bool(env.t)
                episode_start=None if partial_episode else start
                stats=new_stats(env)
            updates=start-1000;begin=time.perf_counter();recent=[];window=[];latest={};losses={}
            start_step=start;step=start
            def status(phase='training'):
                elapsed=time.perf_counter()-begin;new=step-start;rate=new/elapsed if new else 0.
                clip_fraction=sum(x['critic_clipped'] for x in window)/len(window) if window else None
                env_time=sum(x['environment_s'] for x in window)
                update_time=sum(x['update_s'] for x in window)
                return dict(experiment_id='v4_critic_sampling_diagnostic',phase=phase,mode=arm,seed=seed,
                    map_difficulty='V3-full',step=step,total_steps=target,resume_step=start,new_steps=new,
                    updates=updates,episodes=episode,overall_completed_new_steps=completed_new+new,
                    overall_total_new_steps=total_new,run_index=run_index,run_count=len(items),
                    elapsed_s=elapsed,steps_per_s=rate,eta_s=(target-step)/rate if rate else None,
                    validation_success=latest.get('old_success_rate'),last_evaluation_step=latest.get('step'),
                    new_validation_success=latest.get('new_success_rate'),
                    recent_training_success=sum(x['success'] for x in recent)/len(recent) if recent else None,
                    recent_training_episodes=len(recent),
                    recent_terminal_reasons={k:sum(bool(x[k]) for x in recent) for k in
                        ('static_collision','boundary_collision','dynamic_collision','uav_collision','truncated')},
                    critic_clip_frequency_window=clip_fraction,
                    update_window_size=len(window),
                    population_done_fraction_window=(sum(x['population_done_fraction'] for x in window)/len(window) if window else None),
                    sampled_done_fraction_window=(sum(x['sampled_done_fraction'] for x in window)/len(window) if window else None),
                    fallback_uniform_count_window=sum(x['fallback_uniform'] for x in window),
                    sampling_ms_window=(1000*sum(x['sampling_s'] for x in window)/len(window) if window else None),
                    critic_ms_window=(1000*sum(x['critic_s'] for x in window)/len(window) if window else None),
                    actor_alpha_ms_window=(1000*sum(x['actor_alpha_s'] for x in window)/len(window) if window else None),
                    environment_steps_per_s_window=len(window)/env_time if env_time else None,
                    updates_per_s_window=len(window)/update_time if update_time else None,
                    max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)),
                    updated_at=time.time(),pid=os.getpid(),**losses)
            write(out/'status.json',status())
            emit(f'START {arm}/{seed} | step={start:,} | updates={updates:,}')
            with (directory/'training.jsonl').open('x',buffering=1) as local:
              for step in range(start+1,target+1):
                environment_start=time.perf_counter()
                action=updater.act(obs);before=env.p.copy()
                next_obs,reward,done,trunc,info=env.step(action)
                updater.push(obs,action,reward,next_obs,done)
                update_stats(stats,env,before);obs=next_obs;episode_return+=reward
                environment_s=time.perf_counter()-environment_start
                losses=updater.update();losses['environment_s']=environment_s
                updates+=1;window=(window+[losses])[-100:]
                if done or trunc:
                    event=dict(kind='episode',arm=arm,seed=seed,step=step,episode=episode,
                        episode_start_step=episode_start,partial_episode_stats=partial_episode,
                        return_=episode_return,terminated=done,truncated=trunc,
                        closest_goal_m_by_uav=stats['closest_goal_m'].tolist(),path_m_by_uav=stats['path_m'].tolist(),
                        min_boundary_m=stats['min_boundary_m'],**info)
                    text=json.dumps(event,allow_nan=False);local.write(text+'\n');metrics.write(text+'\n')
                    recent=(recent+[event])[-30:];episode+=1;episode_return=0.;episode_start=step;partial_episode=False
                    env=StructuredMission(mode='standard',layout=cfg['layouts'][episode%3])
                    obs=env.reset(cfg['training_map_seed_offset']+seed*1000+episode);stats=new_stats(env)
                if step%100==0 or step==target:
                    state=status();write(out/'status.json',state)
                    text=json.dumps(dict(kind='progress',**state),allow_nan=False);local.write(text+'\n');metrics.write(text+'\n')
                    val='?' if state['validation_success'] is None else f'{state["validation_success"]:.1%}'
                    val_new='?' if state['new_validation_success'] is None else f'{state["new_validation_success"]:.1%}'
                    train='?' if state['recent_training_success'] is None else f'{state["recent_training_success"]:.1%}'
                    reasons=state['recent_terminal_reasons']
                    emit(f'[{arm} seed={seed}] {step:,}/{target:,} | ep={episode} upd={updates:,} '
                         f'| actor={losses["actor_loss"]:+.4f} critic={losses["critic_loss"]:.4f} '
                         f'alpha={losses["alpha"]:.4f} entropy={losses["entropy"]:.3f} '
                         f'| Q={losses["q1_mean"]:.2f}/{losses["q2_mean"]:.2f} '
                         f'TD={losses["absolute_td_q1_mean"]:.3f}/{losses["absolute_td_q2_mean"]:.3f} '
                         f'| qgrad={losses["critic_grad_preclip"]:.1f}->{losses["critic_grad_postclip"]:.1f} '
                         f'clip100={state["critic_clip_frequency_window"]:.1%} '
                         f'| done={losses["population_done_fraction"]:.2%}->{losses["sampled_done_fraction"]:.2%} '
                         f'| train30={train} static/boundary/dynamic/peer/trunc='
                         f'{reasons["static_collision"]}/{reasons["boundary_collision"]}/'
                         f'{reasons["dynamic_collision"]}/{reasons["uav_collision"]}/{reasons["truncated"]} '
                         f'| overall={state["steps_per_s"]:.1f} env={state["environment_steps_per_s_window"]:.1f} '
                         f'update={state["updates_per_s_window"]:.1f} steps/s '
                         f'elapsed={state["elapsed_s"]/60:.1f}min ETA={(state["eta_s"] or 0)/60:.1f}min '
                         f'| val@{state["last_evaluation_step"]} old={val} new={val_new}')
                if step%cfg['checkpoint_interval']==0 or step==target:
                    write(out/'status.json',status('saving_checkpoint'))
                    snapshot=save_snapshot(directory,model,updater,env,exploration,step=step,episode=episode,
                        episode_return=episode_return,episode_start=episode_start,episode_stats=stats,
                        partial_episode=partial_episode,parent_cfg=old['cfg'],seed=seed,arm=arm)
                    emit(f'SNAPSHOT {arm}/{seed} step={step:,} replay_sha256={snapshot["replay_sha256"]}')
                if step%cfg['eval_interval']==0 or step==target:
                    write(out/'status.json',status('evaluating'))
                    maps=(cfg['original_development_map_seeds'][:2] if args.smoke else
                          cfg['original_development_map_seeds']+cfg['new_development_map_seeds'])
                    rows=evaluate(PolicyAdapter(updater),'standard',maps,directory,step)
                    enrich_evaluation(directory,step,rows)
                    split=2 if args.smoke else 6
                    old_rows,new_rows=rows[:split],rows[split:]
                    latest=dict(step=step,old_success_rate=sum(x['success'] for x in old_rows)/len(old_rows),
                        new_success_rate=(sum(x['success'] for x in new_rows)/len(new_rows) if new_rows else None))
                    evaluations.setdefault(f'{arm}_{seed}',{})[str(step)]=rows
                    write(out/'evaluation.json',evaluations)
                    write(out/'latest_evaluation.json',dict(mode=arm,seed=seed,**latest,results=rows))
                    write(out/'status.json',status())
                    emit(f'EVAL {arm}/{seed} step={step:,} old={sum(x["success"] for x in old_rows)}/{len(old_rows)} '
                         f'new={sum(x["success"] for x in new_rows)}/{len(new_rows)}')
            if updates!=target-1000:
                raise ValueError('Unexpected update count')
            write(directory/'audit.json',dict(arm=arm,seed=seed,start_step=start,total_steps=target,
                added_steps=target-start,updates=updates,replay_size=model.size,replay_index=model.index,
                final_evaluation=latest,checkpoint=str(directory/'snapshots'/str(target))))
            completed_new+=target-start
            emit(f'FINISHED {arm}/{seed} | step={target:,} updates={updates:,}')
            del model,updater,env,obs,exploration;gc.collect()
          name='recovery_completion.json' if partial else 'completion.json'
          write(out/name,dict(complete=True,runs=len(items),selected_runs=items,smoke=args.smoke,
                              recovery_only=partial,target_steps=target,finished_at=time.time()))
          final=status('complete');final.update(complete=True,recovery_only=partial,
                                                overall_completed_new_steps=total_new)
          write(out/'status.json',final)
    except BaseException as error:
        write(out/'error.json',dict(error=repr(error),arm=arm,seed=seed,step=step,updated_at=time.time()))
        raise
    finally:
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/masac_v4/critic_sampling_diagnostic.json')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--log',type=Path)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume-snapshot',type=Path)
    p.add_argument('--run-item',help='Selected shard as arm:seed; never marks the six-run batch complete')
    options=p.parse_args()
    if not options.smoke and not options.log: p.error('--log is required for the active-run pointer')
    if options.smoke and options.run_item:
        p.error('Smoke and selected parent branches are separate modes')
    run(options)
