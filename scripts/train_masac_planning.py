#!/usr/bin/env python3
import sys,json,time,hashlib,shutil,random,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from algos.masac import MASAC
from uav_planning.environment import Mission

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=2000);p.add_argument('--arms',nargs='+',choices=['standard','distance','prediction','maneuver'],default=['standard','distance','prediction','maneuver']);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    cfg=dict(hidden=64,batch_size=64,buffer_size=10000,grad_clip=5.)
    protocol=dict(steps=args.steps,seeds=[101,202,303],arms=args.arms,cfg=cfg,phase='engineering pilot; not convergence or novelty confirmation',evaluation_seeds=list(range(9000,9020)),observation='perfect local relative state; padded six-agent capacity',mechanism='observation-only single-channel surrogate; identical reward and network capacity',device='cpu',training_agents=3)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));snap=out/'source_snapshot';snap.mkdir()
    for file in ['uav_planning/environment.py','algos/masac.py','algos/buffer.py','algos/networks.py','algos/common.py','scripts/train_masac_planning.py']:
        shutil.copy2(ROOT/file,snap/Path(file).name)
    (out/'source_hashes.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in snap.iterdir()},indent=2))
    for mode in protocol['arms']:
      for seed in protocol['seeds']:
        d=out/f'{mode}_{seed}';d.mkdir();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);rng=np.random.default_rng(seed)
        env=Mission(mode=mode);obs,state=env.reset(seed*1000);algo=MASAC([70]*3,210,3,3,cfg,torch.device('cpu'),seed)
        initial=torch.cat([x.detach().flatten() for actor in algo.actors for x in actor.parameters()]).clone();updates=0;episode=0;ret=0.;begin=time.time()
        with (d/'training.jsonl').open('w') as log:
          for step in range(args.steps):
            action=rng.uniform(-1,1,(3,3)) if step<256 else algo.act(obs,True)
            nxt,ns,reward,term,trunc,info=env.step(action);algo.push(obs,state,action,reward,nxt,ns,term);obs,state=nxt,ns;ret+=float(reward.mean())
            if step>=256:
                loss=algo.update();updates+=1
                assert all(np.isfinite(v) for v in loss.values())
                if step%100==0:log.write(json.dumps(dict(step=step,**loss))+'\n');log.flush()
            if (step+1)%2000==0:
                torch.save(dict(model=algo.state_dict(),step=step+1,seed=seed,mode=mode),d/'progress_checkpoint.pt')
                (out/'status.json').write_text(json.dumps(dict(mode=mode,seed=seed,step=step+1,updates=updates)))
                print(mode,seed,'step',step+1,flush=True)
            if term or trunc:
                log.write(json.dumps(dict(step=step,episode=episode,episode_return=ret,**info))+'\n');episode+=1;ret=0.;obs,state=env.reset(seed*1000+episode)
        final=torch.cat([x.detach().flatten() for actor in algo.actors for x in actor.parameters()]);change=float(torch.linalg.vector_norm(final-initial));assert change>0 and updates>0
        torch.save(dict(model=algo.state_dict(),cfg=cfg,seed=seed,mode=mode,steps=args.steps,updates=updates,torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state),d/'checkpoint.pt')
        # Replay checkpoint permits auditing the real transitions used for updates.
        torch.save(algo.buffer,d/'replay.pt')
        evaluation=[]
        for es in protocol['evaluation_seeds']:
            obs,state=env.reset(es);trajectory=[env.p.copy()];lat=[]
            for t in range(env.horizon):
                b=time.perf_counter();a=algo.act(obs,False);lat.append(time.perf_counter()-b)
                obs,state,_,term,trunc,info=env.step(a);trajectory.append(env.p.copy())
                if term or trunc:break
            info.update(seed=es,inference_ms=float(np.mean(lat)*1000));evaluation.append(info)
            np.savez_compressed(d/f'trajectory_{es}.npz',positions=np.array(trajectory),goals=env.goal,obstacles=env.obstacles,radii=env.radii)
        (d/'evaluation.json').write_text(json.dumps(evaluation,indent=2));(d/'audit.json').write_text(json.dumps(dict(updates=updates,replay_size=algo.buffer.size,actor_parameter_l2_change=change,wall_s=time.time()-begin),indent=2))
        print(mode,seed,'updates',updates,'success',sum(x['success'] for x in evaluation),'/20',flush=True)
    (out/'COMPLETE').write_text('All configured pilot runs and evaluations completed. Not a convergence claim.\n')
if __name__=='__main__':main()
