#!/usr/bin/env python3
"""Auditable V4 L0 legacy baseline runner. Never starts on import.

Snapshots contain optimizer, replay, RNG and live environment state. Recovery
uses a new segment directory and verifies source and snapshot hashes; previous
files are never overwritten. Smoke runs never update the active pointer.
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
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from uav_planning.structured import box_distance
from uav_planning.v4_environment import V4Mission
from uav_planning.v4_maps import FAMILIES, default_config
from uav_planning.v4_training import InstrumentedLegacyMASAC
from uav_planning.v4_reference import joint_reference

SOURCES=('scripts/train_masac_v4_l0.py','uav_planning/v4_training.py',
 'uav_planning/v4_environment.py','uav_planning/v4_maps.py',
 'uav_planning/shared_masac.py','uav_planning/structured.py','uav_planning/environment.py',
 'uav_planning/v4_reference.py')


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()


def write(path,obj):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f: json.dump(obj,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def append(path,obj):
    with Path(path).open('a') as f: f.write(json.dumps(obj,allow_nan=False)+'\n'); f.flush()


def message(root,text):
    print(text,flush=True)
    with (root/'train.log').open('a') as f: f.write(text+'\n'); f.flush()


def source_hashes(): return {p:digest(ROOT/p) for p in SOURCES}


def current_clearance(env):
    vals=[np.min(np.minimum(env.p-env.radius,env.world-env.radius-env.p))]
    if len(env.boxes): vals.append(np.min(box_distance(env.p,env.lo,env.hi))-env.radius)
    for i in range(env.n):
        for j in range(i+1,env.n): vals.append(np.linalg.norm(env.p[i]-env.p[j])-2*env.radius)
    if len(env.dynamic): vals.append(np.min(np.linalg.norm(env.p[:,None]-env.dynamic[None,:],axis=-1)-env.dr[None,:]-env.radius))
    return float(min(vals))


def dist_stats(x):
    x=np.asarray(x,dtype=float).ravel()
    if not len(x): return None
    return {'min':float(x.min()),'mean':float(x.mean()),'p50':float(np.percentile(x,50)),
            'p95':float(np.percentile(x,95)),'max':float(x.max())}


def make_env(family,cfg):
    return V4Mission(default_config(family,'L0',sensor_range=cfg['sensor_range']))


def audited_episode(family,cfg,seed,episode,directory):
    env=make_env(family,cfg); obs=env.reset(seed)
    check=joint_reference(env.config,seed)
    same_hash=check['scene']['scene_sha256']==env.scene.scene_hash
    record={'episode':episode,'map_seed':seed,'family':family,'level':'L0',
            'scene_sha256':env.scene.scene_hash,'reference_scene_hash_matches':same_hash,
            'joint_reference_success':check['joint_reference_success'],'attempts':check['attempts']}
    append(directory/'training_map_audit.jsonl',record)
    if not same_hash or not check['joint_reference_success']:
        raise ValueError(f'Training map reference failed or scene hash mismatched: {seed}')
    return env,obs


def evaluate(model,maps,directory,step,cfg):
    records=[]
    for entry in maps:
        seed=int(entry['seed']); family=entry['family']; env=make_env(family,cfg)
        obs=env.reset(seed); initial=env.p.copy()
        positions=[env.p.copy()]; velocities=[env.v.copy()]; dynamics=[env.dynamic.copy()]; actions=[]; acc=[]; clearances=[current_clearance(env)]
        near_stall=np.zeros(env.n,dtype=int); closest=np.linalg.norm(env.goal-env.p,axis=-1); ever=np.zeros(env.n,dtype=bool)
        while True:
            action=model.act(obs,deterministic=True); old_v=env.v.copy()
            obs,reward,done,truncated,info=env.step(action)
            positions.append(env.p.copy()); velocities.append(env.v.copy()); dynamics.append(env.dynamic.copy()); actions.append(action.copy()); acc.append((env.v-old_v)/env.dt); clearances.append(current_clearance(env))
            distance=np.linalg.norm(env.goal-env.p,axis=-1); speed=np.linalg.norm(env.v,axis=-1)
            closest=np.minimum(closest,distance); ever |= env.captured; near_stall += ((distance<10)&(speed<2)&~env.captured)
            if done or truncated: break
        pos=np.asarray(positions); vel=np.asarray(velocities); acts=np.asarray(actions); path=np.asarray(info['path_length_by_uav'])
        collision=any(info[k] for k in ('uav_collision','static_collision','dynamic_collision','boundary_collision'))
        direct=np.linalg.norm(env.goal-initial,axis=-1)
        row={**info,'map_seed':seed,'family':family,'level':'L0','step':step,
             'terminated':bool(done),'truncated':bool(truncated),'timeout':bool(truncated and not done),
             'closest_goal_m_by_uav':closest.tolist(),'final_goal_m_by_uav':np.linalg.norm(env.goal-env.p,axis=-1).tolist(),
             'ever_arrived_by_uav':ever.tolist(),'simultaneously_arrived':bool(info['success']),
             'near_goal_low_speed_uncaptured_steps_by_uav':near_stall.tolist(),
             'success_efficiency_direct_over_path_by_uav':(direct/np.maximum(path,1e-12)).tolist() if info['success'] else None,
             'action_norm_distribution':dist_stats(np.linalg.norm(acts,axis=-1)),
             'speed_distribution':dist_stats(np.linalg.norm(vel,axis=-1)),
             'acceleration_distribution':dist_stats(np.linalg.norm(np.asarray(acc),axis=-1)),
             'collision_speed_by_uav':np.linalg.norm(env.v,axis=-1).tolist() if collision else None}
        prefix=directory/f'eval_{step}_{seed}'
        with open(str(prefix)+'.npz','xb') as f:
            np.savez(f,positions=pos,velocities=vel,actions=acts,dynamic=np.asarray(dynamics),goals=env.goal,
                     buildings=env.boxes,dynamic_radii=env.dr,clearances=np.asarray(clearances))
        write(Path(str(prefix)+'_map.json'),env.metadata()); records.append(row)
    write(directory/f'evaluation_{step}.json',records)
    write(directory/'evaluation.json',{'step':step,'episodes':records})
    return records


def save_snapshot(directory,model,env,explore,state):
    step=state['step']; folder=directory/f'checkpoint_{step:06d}'
    folder.mkdir(exist_ok=False)
    n=model.size
    replay=folder/'replay.npz'
    with replay.open('xb') as f:
        np.savez(f,obs=model.obs[:n],next=model.next[:n],actions=model.actions[:n],rewards=model.rewards[:n],done=model.done[:n])
        f.flush(); os.fsync(f.fileno())
    checkpoint={'model':model.checkpoint(),'environment':copy.deepcopy(env.__dict__),
       'exploration_rng':copy.deepcopy(explore.bit_generator.state),'python_rng':random.getstate(),
       'numpy_rng':np.random.get_state(),'state':copy.deepcopy(state)}
    cp=folder/'checkpoint.pt'
    with cp.open('xb') as f: torch.save(checkpoint,f); f.flush(); os.fsync(f.fileno())
    write(folder/'manifest.json',{'version':1,'step':step,'seed':state['seed'],'replay_size':n,
          'replay_index':model.index,'checkpoint_sha256':digest(cp),'replay_sha256':digest(replay),
          'sources':source_hashes(),'complete':True})
    return folder


def restore_snapshot(folder,capacity,device):
    folder=Path(folder); meta=json.loads((folder/'manifest.json').read_text())
    if not meta.get('complete') or meta['sources']!=source_hashes(): raise ValueError('Snapshot incomplete or source hash differs')
    cpfile=folder/'checkpoint.pt'; replay=folder/'replay.npz'
    if digest(cpfile)!=meta['checkpoint_sha256'] or digest(replay)!=meta['replay_sha256']: raise ValueError('Snapshot digest mismatch')
    cp=torch.load(cpfile,map_location='cpu',weights_only=False); s=cp['state']; seed=s['seed']
    if cp['model']['replay_size']!=meta['replay_size'] or cp['model']['replay_index']!=meta['replay_index'] or s['step']!=meta['step']: raise ValueError('Snapshot state mismatch')
    model=InstrumentedLegacyMASAC(capacity=capacity,seed=seed,device=device)
    with np.load(replay,allow_pickle=False) as r:
        n=meta['replay_size']
        if n>capacity or any(len(r[k])!=n for k in ('obs','next','actions','rewards','done')): raise ValueError('Replay size mismatch')
        for name,key in (('obs','obs'),('next','next'),('actions','actions'),('rewards','rewards'),('done','done')): getattr(model,name)[:n]=r[key]
    model.load_state(cp['model'])
    env=V4Mission(cp['environment']['config']); env.__dict__.update(cp['environment'])
    explore=np.random.default_rng(); explore.bit_generator.state=cp['exploration_rng']
    random.setstate(cp['python_rng']); np.random.set_state(cp['numpy_rng'])
    return model,env,env.observe(),explore,s


def validate_protocol(cfg,sets):
    if cfg['seeds']!=sets['training_seeds']['l0'] or cfg['development_maps']!=sets['development']['l0']:
        raise ValueError('seed namespace disagrees with protocol')
    expected=dict(seeds=[511,522,533],level='L0',steps=50000,warmup=1000,batch_size=128,replay_capacity=100000,
                  eval_interval=2000,checkpoint_interval=10000,log_interval=200,training_map_seed_offset=2000000,
                  training_seed_multiplier=10000,n_agents=3,device='cpu',sensor_range=400,
                  world=[240,240,90],dt=.2,horizon=350,vmax=10,amax=4,body_radius=1,safety_margin=0,
                  learning_rate=.0003,gamma=.99,tau=.005,initial_alpha=.1,gradient_clip=5,arms=['standard'])
    if any(cfg.get(k)!=v for k,v in expected.items()) or cfg['families']!=list(FAMILIES):
        raise ValueError('configuration differs from frozen L0 protocol')
    if [e['seed'] for e in cfg['development_maps']]!=list(range(830001,830013)):
        raise ValueError('unexpected development namespace')
    if any(e['family']!=FAMILIES[i%6] or e['level']!='L0' for i,e in enumerate(cfg['development_maps'])):
        raise ValueError('unexpected development family schedule')
    if sets['final_test']['l0_seed_range']!=[930001,930060]: raise ValueError('unexpected final test namespace')


def run(args):
    cfg=json.loads(args.protocol.read_text()); seed_path=ROOT/cfg['seed_sets']; sets=json.loads(seed_path.read_text()); validate_protocol(cfg,sets)
    confhash={'protocol':digest(args.protocol),'seed_sets':digest(seed_path)}
    seeds=cfg['seeds'][:1] if args.smoke else cfg['seeds']
    steps=256 if args.smoke else cfg['steps']; warmup=128 if args.smoke else cfg['warmup']
    batch=cfg['batch_size']; capacity=3000 if args.smoke else cfg['replay_capacity']
    eval_every=256 if args.smoke else cfg['eval_interval']; save_every=256 if args.smoke else cfg['checkpoint_interval']
    eval_maps=cfg['development_maps'][:2] if args.smoke else cfg['development_maps']
    torch.set_num_threads(1)
    lockpath=ROOT/'results/masac_v4_l0_training.lock'; lockpath.parent.mkdir(parents=True,exist_ok=True)
    lock=lockpath.open('a+')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another V4 L0 runner holds the global training lock')
    old_manifest=None; resume=None
    if args.resume:
        resume=args.resume.resolve()
        old_root=resume.parent.parent
        old_manifest=json.loads((old_root/'run_manifest.json').read_text())
        if old_manifest['config_hashes']!=confhash or old_manifest['smoke']!=args.smoke:
            raise ValueError('Resume protocol, seed namespace, or smoke mode does not match')
        if old_manifest['sources']!=source_hashes(): raise ValueError('Resume source hash mismatch')
    args.out.mkdir(parents=True,exist_ok=False); root=args.out.resolve()
    manifest={'experiment_id':cfg['experiment_id'],'mode':'legacy standard','smoke':args.smoke,
       'level':'L0','created_at':time.time(),'pid':os.getpid(),'config_hashes':confhash,'sources':source_hashes(),
       'seed_list':seeds,'steps_per_seed':steps,'warmup':warmup,'replay_capacity':capacity,
       'development_maps_used':eval_maps,'final_test':'reserved and not generated',
       'resume_from':str(resume) if resume else None,'command':sys.argv}
    write(root/'run_manifest.json',manifest)
    for name in SOURCES:
        target=root/'source_snapshot'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
        if digest(target)!=manifest['sources'][name]: raise ValueError('Source changed while taking snapshot')
    for source,name in ((args.protocol,'l0_protocol.json'),(seed_path,'seed_sets.json')):
        target=root/'source_snapshot'/'configs/masac_v4'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        if digest(source)!=digest(target): raise ValueError('Configuration changed while taking snapshot')
    if not args.smoke: write(ROOT/'results/masac_v4_active_run.json',{'run':str(root),'pid':os.getpid(),'experiment_id':cfg['experiment_id'],'log':str(root/'train.log')})
    seed=None; step=0; phase='initializing'; completed=[]; begin=time.perf_counter()
    try:
        if resume:
            snapshot_meta=json.loads((resume/'manifest.json').read_text()); resume_seed=snapshot_meta['seed']
            if resume_seed not in seeds: raise ValueError('resume seed outside frozen seed set')
            seeds=seeds[seeds.index(resume_seed):]
        message(root,f'L0 STANDARD | PID={os.getpid()} | seeds={seeds} | steps={steps} | smoke={args.smoke} | out={root}')
        for run_index,seed in enumerate(seeds):
            directory=root/f'standard_{seed}'; directory.mkdir()
            if resume and run_index==0:
                model,env,obs,explore,state=restore_snapshot(resume,capacity,cfg['device'])
                if state['seed']!=seed or state['step']>steps: raise ValueError('invalid resume state')
            else:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                explore=np.random.default_rng(seed); model=InstrumentedLegacyMASAC(n=3,capacity=capacity,seed=seed,device=cfg['device'])
                episode=0; env,obs=audited_episode(cfg['families'][0],cfg,cfg['training_map_seed_offset']+seed*cfg['training_seed_multiplier'],episode,directory)
                state=dict(seed=seed,step=0,episode=0,updates=0,episode_start=0,episode_return=0.,recent=[],losses={},last_eval=None,last_eval_step=None,
                           closest=np.linalg.norm(env.goal-env.p,axis=-1).tolist(),near_stall=[0]*3)
            message(root,f'START seed={seed} from_step={state["step"]} | family={env.config.family}')
            start_step=state['step']; tick=time.perf_counter()
            for step in range(start_step+1,steps+1):
                phase='training'; state['step']=step
                action=explore.uniform(-1,1,(3,3)).astype(np.float32) if step<=warmup else model.act(obs)
                next_obs,reward,done,truncated,info=env.step(action)
                model.push(obs,action,reward,next_obs,done); obs=next_obs; state['episode_return']+=reward
                distance=np.linalg.norm(env.goal-env.p,axis=-1); speed=np.linalg.norm(env.v,axis=-1)
                state['closest']=np.minimum(state['closest'],distance).tolist()
                state['near_stall']=(np.asarray(state['near_stall'])+((distance<10)&(speed<2)&~env.captured)).tolist()
                if step>warmup:
                    state['losses']=model.update(batch); state['updates']+=1
                    if not all(np.isfinite(v) for v in state['losses'].values()): raise ValueError('nonfinite learning diagnostic')
                if done or truncated:
                    row={'kind':'episode','experiment_id':cfg['experiment_id'],'level':'L0','seed':seed,'step':step,
                         'episode':state['episode'],'episode_start_step':state['episode_start'],'map_seed':env.seed,'family':env.config.family,
                         'return_':state['episode_return'],'terminated':bool(done),'truncated':bool(truncated),
                         'timeout':bool(truncated and not done),'closest_goal_m_by_uav':state['closest'],
                         'near_goal_low_speed_uncaptured_steps_by_uav':state['near_stall'],**info}
                    append(directory/'metrics.jsonl',row); state['recent']=(state['recent']+[row])[-30:]
                    state['episode']+=1; state['episode_start']=step; state['episode_return']=0.
                    env,obs=audited_episode(cfg['families'][state['episode']%6],cfg,
                        cfg['training_map_seed_offset']+seed*cfg['training_seed_multiplier']+state['episode'],state['episode'],directory)
                    state['closest']=np.linalg.norm(env.goal-env.p,axis=-1).tolist(); state['near_stall']=[0]*3
                if step % cfg['log_interval']==0 or step==steps:
                    recent=state['recent']; elapsed=time.perf_counter()-tick; rate=(step-start_step)/elapsed if elapsed else 0.
                    status={'phase':phase,'experiment_id':cfg['experiment_id'],'level':'L0','seed':seed,'step':step,
                      'total_steps':steps,'updates':state['updates'],'episodes':state['episode'],'family':env.config.family,
                      'steps_per_s':rate,'eta_s':(steps-step)/rate if rate else None,'last_eval':state['last_eval'],
                      'last_eval_step':state['last_eval_step'],'pid':os.getpid(),'updated_at':time.time(),
                      'recent_count':len(recent),'recent_success':sum(x['success'] for x in recent)/len(recent) if recent else None,
                      'recent_terminations':{k:sum(bool(x.get(k)) for x in recent) for k in ('success','static_collision','dynamic_collision','uav_collision','boundary_collision','timeout')},**state['losses']}
                    write(root/'status.json',status); append(directory/'metrics.jsonl',{'kind':'progress',**status})
                    message(root,f'[L0 seed={seed}] {step}/{steps} upd={state["updates"]} ep={state["episode"]} {rate:.1f}step/s ETA={status["eta_s"]}s val={state["last_eval"]} q={state["losses"].get("q_mean")} td={state["losses"].get("td_abs")} entropy={state["losses"].get("entropy")}')
                if step%eval_every==0 or step==steps:
                    phase='evaluating'; write(root/'status.json',{'phase':phase,'seed':seed,'step':step,'pid':os.getpid(),'experiment_id':cfg['experiment_id'],'level':'L0','updated_at':time.time()})
                    records=evaluate(model,eval_maps,directory,step,cfg)
                    state['last_eval']=sum(x['success'] for x in records)/len(records); state['last_eval_step']=step
                    message(root,f'EVAL seed={seed} step={step}: success={sum(x["success"] for x in records)}/{len(records)}')
                if step%save_every==0 or step==steps:
                    phase='saving'; save_snapshot(directory,model,env,explore,state)
            completed.append({'seed':seed,'step':steps,'final_development_success':state['last_eval'],
                              'gate_met':bool(state['last_eval']>=cfg['gate']['per_seed_joint_success_fraction'])})
            write(directory/'completion.json',completed[-1])
        complete={'complete':True,'scope':'smoke' if args.smoke else ('resumed segment' if resume else 'full batch'),'runs':completed,'seeds_expected':cfg['seeds'],
                  'all_three_gate_met':len(completed)==3 and all(x['gate_met'] for x in completed),
                  'smoke':args.smoke,'finished_at':time.time(),'elapsed_s':time.perf_counter()-begin}
        write(root/'completion.json',complete); write(root/'status.json',{'phase':'complete','complete':True,**complete})
        message(root,f'COMPLETE scope={complete["scope"]} smoke={args.smoke} runs={len(completed)}')
    except BaseException as e:
        write(root/'error.json',{'error':repr(e),'traceback':traceback.format_exc(),'seed':seed,'step':step,'phase':phase,'updated_at':time.time()})
        raise
    finally: lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/masac_v4/l0_protocol.json')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',type=Path,help='complete checkpoint folder; output must be a new directory')
    run(parser.parse_args())
