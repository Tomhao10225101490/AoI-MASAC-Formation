#!/usr/bin/env python3
"""Visible-terminal training for the frozen structured MASAC design."""
import argparse,json,os,random,shutil,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from uav_planning.structured import StructuredMission
from uav_planning.shared_masac import SharedMASAC


def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def evaluate(model,mode,seeds,directory,step):
    records=[]
    for i,seed in enumerate(seeds):
        env=StructuredMission(mode=mode,layout=('urban','warehouse','courtyard')[i%3]);obs=env.reset(seed)
        meta=env.metadata();positions=[env.p.copy()];moving=[env.dynamic.copy()];margins=[env.last_margins.copy()];lat=[]
        for _ in range(env.horizon):
            start=time.perf_counter();a=model.act(obs,True);lat.append(time.perf_counter()-start)
            obs,r,done,trunc,info=env.step(a);positions.append(env.p.copy());moving.append(env.dynamic.copy());margins.append(env.last_margins.copy())
            if done or trunc:break
        info.update(map_seed=seed,layout=env.layout,step=step,inference_ms=float(np.mean(lat)*1000));records.append(info)
        np.savez_compressed(directory/f'eval_{step}_{seed}.npz',positions=positions,dynamic=moving,margins=margins,goals=env.goal,buildings=env.boxes,dynamic_radii=env.dr)
        write(directory/f'eval_{step}_{seed}_map.json',meta)
    write(directory/f'evaluation_{step}.json',records)
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=20000);p.add_argument('--seeds',type=int,nargs='+',default=[411,422,433]);p.add_argument('--arms',nargs='+',default=['standard'],choices=['standard','distance','prediction','maneuver']);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);out=args.out.resolve();torch.set_num_threads(1)
    cfg=json.loads((ROOT/'configs/masac_v3/protocol.json').read_text());warmup=128 if args.smoke else cfg['warmup'];interval=500 if args.smoke else 2000
    cfg.update(steps=args.steps,seeds=args.seeds,arms=args.arms,warmup=warmup,smoke=args.smoke,command=sys.argv,pid=os.getpid(),platform=sys.platform)
    write(out/'protocol.json',cfg);snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ('uav_planning/structured.py','uav_planning/environment.py','uav_planning/shared_masac.py','scripts/train_masac_v3.py','docs/design/MASAC_V3_DESIGN.md','configs/masac_v3/protocol.json'):
        shutil.copy2(ROOT/name,snapshot/Path(name).name)
    write(out/'source_hashes.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.iterdir()})
    print(f'RUN {out}\nPID {os.getpid()} | device=CPU | seeds={args.seeds} | arms={args.arms} | steps/run={args.steps}',flush=True)
    try:
      for mode in args.arms:
       for seed in args.seeds:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);rng=np.random.default_rng(seed)
        d=out/f'{mode}_{seed}';d.mkdir();model=SharedMASAC(capacity=3000 if args.smoke else cfg['replay_capacity'],seed=seed)
        parameters=dict(actor=sum(p.numel() for p in model.actor.parameters()),twin_critic=sum(p.numel() for p in model.q.parameters()));write(d/'parameters.json',parameters)
        initial=torch.cat([x.detach().flatten() for x in model.actor.parameters()]).clone()
        episode=0;env=StructuredMission(mode=mode,layout=cfg['layouts'][0]);obs=env.reset(100000+seed*1000);ret=0.;updates=0;begin=time.perf_counter();recent=[];loss={};last_eval=None
        print(f'START arm={mode} seed={seed} actor={parameters["actor"]:,} critic={parameters["twin_critic"]:,}',flush=True)
        with (d/'training.jsonl').open('w') as log:
          for step in range(1,args.steps+1):
            a=rng.uniform(-1,1,(3,3)).astype(np.float32) if step<=warmup else model.act(obs)
            no,r,done,trunc,info=env.step(a);model.push(obs,a,r,no,done);obs=no;ret+=r
            if step>warmup:
                loss=model.update(cfg['batch_size']);updates+=1
                if not all(np.isfinite(v) for v in loss.values()):raise ValueError('Non-finite training loss')
            if done or trunc:
                row=dict(kind='episode',step=step,episode=episode,return_=ret,**info);log.write(json.dumps(row)+'\n');recent.append(row);recent=recent[-30:];episode+=1;ret=0.
                env=StructuredMission(mode=mode,layout=cfg['layouts'][episode%3]);obs=env.reset(100000+seed*1000+episode)
            if step%100==0 or step==args.steps:
                elapsed=time.perf_counter()-begin;rate=step/elapsed;eta=(args.steps-step)/rate
                status=dict(mode=mode,seed=seed,step=step,total_steps=args.steps,updates=updates,episodes=episode,elapsed_s=elapsed,steps_per_s=rate,eta_s=eta,validation_success=last_eval,**loss)
                write(out/'status.json',status);log.write(json.dumps(dict(kind='progress',**status))+'\n');log.flush()
                print(f'[{mode} seed={seed}] {step:6d}/{args.steps} ({100*step/args.steps:5.1f}%) | upd={updates:5d} ep={episode:4d} | actor={loss.get("actor_loss",0):+.4f} critic={loss.get("critic_loss",0):.4f} alpha={loss.get("alpha",.1):.4f} | {rate:.1f} steps/s | elapsed={elapsed:.0f}s ETA={eta:.0f}s | val={last_eval}',flush=True)
            if step%interval==0 or step==args.steps:
                checkpoint=model.checkpoint();checkpoint.update(step=step,episode=episode,mode=mode,seed=seed,cfg=cfg,environment=env.__dict__,episode_return=ret,python_rng=random.getstate(),numpy_rng=np.random.get_state(),exploration_rng=rng.bit_generator.state)
                torch.save(checkpoint,d/f'checkpoint_{step}.pt')
                records=evaluate(model,mode,cfg['validation_map_seeds'][:2] if args.smoke else cfg['validation_map_seeds'],d,step);last_eval=sum(x['success'] for x in records)/len(records)
                print(f'EVAL [{mode} seed={seed} step={step}] success={sum(x["success"] for x in records)}/{len(records)} | UAV={sum(x["uav_collision"] for x in records)} static={sum(x["static_collision"] for x in records)} dynamic={sum(x["dynamic_collision"] for x in records)} boundary={sum(x["boundary_collision"] for x in records)}',flush=True)
                write(out/'latest_evaluation.json',dict(mode=mode,seed=seed,step=step,results=records))
          final=torch.cat([x.detach().flatten() for x in model.actor.parameters()]);change=float(torch.linalg.vector_norm(final-initial));assert change>0 and updates>0
          np.savez_compressed(d/'replay_final.npz',obs=model.obs[:model.size],next_obs=model.next[:model.size],actions=model.actions[:model.size],rewards=model.rewards[:model.size],done=model.done[:model.size])
          write(d/'audit.json',dict(updates=updates,replay_size=model.size,actor_parameter_change=change,wall_s=time.perf_counter()-begin,success_rate=last_eval))
          print(f'FINISHED {mode}/{seed}: actor delta={change:.6f}, checkpoint/replay saved',flush=True)
      write(out/'completion.json',dict(complete=True,runs=len(args.arms)*len(args.seeds),steps=args.steps))
    except BaseException as error:
      write(out/'error.json',dict(error=repr(error)));raise
if __name__=='__main__':main()
