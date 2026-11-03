#!/usr/bin/env python3
"""Engineering-only forward/backward profile, with no optimizer or training."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from torch import nn
from uav_planning.v4_networks import Actor, TwinCritics


def parameter_count(m): return sum(p.numel() for p in m.parameters())


def count_ops(model, call):
    totals={'macs':0,'logical_output_bytes':0}; hooks=[]
    def linear(module, inputs, output):
        totals['macs']+=output.numel()*module.in_features
        totals['logical_output_bytes']+=output.numel()*output.element_size()
    def attn(module, inputs, output):
        x=inputs[0]; b,l,d=x.shape
        totals['macs']+=4*b*l*d*d+2*b*l*l*d
        totals['logical_output_bytes']+=output[0].numel()*output[0].element_size()
    for m in model.modules():
        if isinstance(m,nn.Linear): hooks.append(m.register_forward_hook(linear))
        elif isinstance(m,nn.MultiheadAttention): hooks.append(m.register_forward_hook(attn))
    with torch.no_grad(): call()
    for h in hooks: h.remove()
    totals['approx_flops_2_per_mac']=2*totals['macs']
    return totals


def measure(mode, device, repeats, batch):
    torch.manual_seed(31); actor=Actor(mode).to(device); q=TwinCritics(mode).to(device)
    obs=torch.zeros(batch,3,162,device=device); obs[...,9:12]=.5; obs[...,18]=.5; obs[...,19]=.4
    for sl in (26,61,118): obs[...,sl]=1
    risk=torch.zeros(batch,3,19,11,device=device)
    def sync():
        if device=='mps': torch.mps.synchronize()
    def forward():
        a,lp=actor(obs,risk=risk); q1,q2=q(obs,a,risk=risk); return (q1+q2+lp.mean(-1)).mean()
    class Bundle(nn.Module):
        def __init__(self): super().__init__(); self.actor=actor; self.q=q
    bundle=Bundle(); costs=count_ops(bundle,forward)
    for _ in range(3):
        loss=forward(); loss.backward(); bundle.zero_grad(set_to_none=True)
    sync(); t=time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats): actor(obs,risk=risk)
    sync(); actor_fwd=(time.perf_counter()-t)/repeats
    sync(); t=time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats): forward()
    sync(); fwd=(time.perf_counter()-t)/repeats
    sync(); t=time.perf_counter()
    for _ in range(repeats):
        loss=forward(); loss.backward(); bundle.zero_grad(set_to_none=True)
    sync(); backward=(time.perf_counter()-t)/repeats
    # Synthetic optimizer microbenchmark: random zero inputs, no environment,
    # rollout, replay, checkpoint, or MASAC learning run. Not full SAC update.
    optimizer=torch.optim.Adam(bundle.parameters(),lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True); forward().backward(); optimizer.step()
    sync(); t=time.perf_counter()
    for _ in range(repeats):
        optimizer.zero_grad(set_to_none=True); forward().backward(); optimizer.step()
    sync(); optimizer_step=(time.perf_counter()-t)/repeats
    params=parameter_count(bundle)
    return {'actor_parameters':parameter_count(actor),'twin_q_parameters':parameter_count(q),
            'model_parameter_bytes_fp32':params*4,'gradient_bytes_fp32_if_all_present':params*4,
            'adam_moment_bytes_fp32_if_all_present':params*8,
            'twin_target_parameter_bytes_fp32':parameter_count(q)*4,
            'batch':batch,'agents':3,'object_slots':17,'probe_slots':19,
            'actor_forward_seconds':actor_fwd,'actor_plus_twin_q_forward_seconds':fwd,
            'actor_plus_twin_q_forward_backward_seconds':backward,
            'synthetic_joint_forward_backward_adam_step_seconds':optimizer_step,
            **costs}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True); parser.add_argument('--repeats',type=int,default=10); parser.add_argument('--batch',type=int,default=8)
    args=parser.parse_args(); path=Path(args.out)
    torch.set_num_threads(1)
    devices=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])
    sources=['scripts/profile_masac_v4_networks.py','uav_planning/v4_networks.py','uav_planning/v4_prediction.py']
    data={'purpose':'engineering synthetic forward/backward/Adam microbenchmark only; no environment, replay, checkpoint, or training; not a complete MASAC update',
          'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
          'pytorch':torch.__version__,'threads':torch.get_num_threads(),'mps_available':torch.backends.mps.is_available(),
          'mac_definition':'linear + attention matrix multiplication only; excludes norm, activation, softmax, masks and backward; 2 FLOPs per MAC',
          'logical_output_bytes_definition':'sum of outputs of hooked linear and attention operations; not peak memory',
          'replay_100k_3_agents_fp32_bytes_obs_next_obs_actions_reward_done':100000*(2*3*162+3*3+2)*4,
          'results':{d:{m:measure(m,d,args.repeats,args.batch) for m in ('pool','attention')} for d in devices}}
    with path.open('x') as f: json.dump(data,f,indent=2); f.write('\n')
    print(json.dumps(data,indent=2))

if __name__=='__main__': main()
