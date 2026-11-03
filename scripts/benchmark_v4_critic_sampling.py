#!/usr/bin/env python3
"""Benchmark isolated in-memory V3 double-batch updates on CPU and MPS.

This is an engineering throughput probe from a saved parent, not a training
experiment. Each device/arm gets its own process and unchanged parent files.
"""
import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.continue_masac_v3_paired import restore_snapshot, validate_snapshot_source  # noqa: E402
from uav_planning.critic_sampling_diagnostic import CriticSamplingUpdater  # noqa: E402
from uav_planning.structured import StructuredMission  # noqa: E402


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def measure(arm, device, warmup, steps, seed):
    if warmup < 1 or steps < 1:
        raise ValueError('Warmup and timing counts must be positive')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable')
    torch.set_num_threads(1)
    root=ROOT/'results/masac_v3_paired_300k_20261103'
    protocol_file=ROOT/'configs/masac_v3/paired_300k_protocol.json'
    protocol=json.loads(protocol_file.read_text())
    snapshot=root/f'original_{seed}'/'snapshots/300000'
    validate_snapshot_source(snapshot,protocol,protocol_file)
    manifest=json.loads((snapshot/'complete.json').read_text())
    model, env, obs, exploration, cp=restore_snapshot(snapshot)
    if cp['history_arm'] != 'original' or cp['step'] != 300000:
        raise ValueError('Unexpected benchmark parent checkpoint')
    if device == 'mps':
        # MPS does not share the saved CPU rollout RNG. This benchmark-only
        # deterministic seed is matched across arms, not an exact CPU resume.
        torch.mps.manual_seed(104729*seed+31005)
    updater=CriticSamplingUpdater(model,arm=arm,seed=seed,batch_size=128,done_quota=32,device=device)
    parent_cfg=cp['cfg']; episode=cp['episode']
    durations=[]; sampling=[]; critic=[]; actor=[]; events=0
    mps_current_peak=mps_driver_peak=0
    for i in range(warmup+steps):
        if device == 'mps': torch.mps.synchronize()
        start=time.perf_counter()
        action=updater.act(obs)
        nxt, reward, done, trunc, _=env.step(action)
        # Keep the actual parent Replay fixed for every device/arm.  This
        # excludes FIFO insertion/stratum-maintenance overhead from timing.
        obs=nxt
        metrics=updater.update()
        if done or trunc:
            events+=1; episode+=1
            env=StructuredMission(mode='standard',layout=parent_cfg['layouts'][episode % 3])
            obs=env.reset(parent_cfg['training_map_seed_offset']+seed*1000+episode)
        if device == 'mps': torch.mps.synchronize()
        duration=time.perf_counter()-start
        if i >= warmup:
            durations.append(duration)
            sampling.append(metrics['sampling_s']); critic.append(metrics['critic_s']); actor.append(metrics['actor_alpha_s'])
            if device == 'mps':
                mps_current_peak=max(mps_current_peak,torch.mps.current_allocated_memory())
                mps_driver_peak=max(mps_driver_peak,torch.mps.driver_allocated_memory())
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes=int(peak if sys.platform=='darwin' else peak*1024)
    elapsed=sum(durations)
    return dict(arm=arm,device=device,seed=seed,warmup_steps=warmup,timed_steps=steps,
        timed_seconds=elapsed,environment_steps_and_updates_per_second=steps/elapsed,
        median_step_ms=float(np.median(durations)*1000),p90_step_ms=float(np.percentile(durations,90)*1000),
        mean_sampling_ms=float(np.mean(sampling)*1000),mean_critic_ms=float(np.mean(critic)*1000),
        mean_actor_alpha_ms=float(np.mean(actor)*1000),ended_episodes_during_warmup_and_timing=events,
        process_max_rss_bytes=peak_bytes,mps_peak_current_allocated_bytes=mps_current_peak,
        mps_peak_driver_allocated_bytes=mps_driver_peak,checkpoint_sha256=manifest['checkpoint_sha256'],
        parent_replay_sha256=manifest['replay_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--warmup',type=int,default=50)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--seed',type=int,default=411)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--device',choices=('cpu','mps'))
    parser.add_argument('--arm',choices=('uniform','stratified'))
    args=parser.parse_args()
    if args.worker:
        if not args.device or not args.arm: parser.error('worker requires device and arm')
        print(json.dumps(measure(args.arm,args.device,args.warmup,args.steps,args.seed),allow_nan=False))
        return
    if args.out is None: parser.error('--out is required')
    if args.out.exists(): raise FileExistsError(args.out)
    devices=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])
    measurements=[]
    for device in devices:
        for arm in ('uniform','stratified'):
            command=[sys.executable,str(Path(__file__).resolve()),'--worker','--device',device,'--arm',arm,
                     '--warmup',str(args.warmup),'--steps',str(args.steps),'--seed',str(args.seed)]
            child=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=600)
            if child.returncode:
                measurements.append(dict(device=device,arm=arm,error=child.stderr.strip(),exit_code=child.returncode))
            else:
                measurements.append(json.loads(child.stdout))
    script=ROOT/'uav_planning/critic_sampling_diagnostic.py'
    result=dict(experiment_type='engineering-only in-memory update throughput; not scientific training',
        limitation='Small fixed checkpoint probe; separate child process per device/arm; includes environment steps and update but excludes Replay FIFO insertion/stratum-maintenance. Parent Replay stays fixed in memory, and sampler indices use the same seed for each device per arm. Short sampling, thermal and concurrent-load effects possible. MPS rollout RNG is matched across MPS arms but not a bitwise continuation of CPU rollout RNG. RSS and MPS allocator metrics have different meanings.',
        torch_version=str(torch.__version__),numpy_version=np.__version__,mps_available=torch.backends.mps.is_available(),
        script_sha256=sha(Path(__file__)),updater_source_sha256=sha(script),measurements=measurements)
    with args.out.open('x') as out:
        json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(f'Wrote {args.out}')


if __name__=='__main__':main()
