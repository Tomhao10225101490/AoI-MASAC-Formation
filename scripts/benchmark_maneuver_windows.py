"""Visible end-to-end device/thread benchmark; never changes science defaults."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_planning.v4_maneuver_masac import ManeuverMASAC
from uav_planning.v4_reward_reshape import make_reshape_l0_env, reshape_from_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    cfg = json.loads((ROOT/'configs/masac_v4/maneuver_paper_suite_20261103.json').read_text())
    rs = reshape_from_protocol(cfg['reshape'])
    cases = [('cpu', n) for n in (1, 2, 3)]
    if torch.cuda.is_available():
        cases.append(('cuda', 1))
    results = []
    reference = None
    for device, threads in cases:
        torch.set_num_threads(threads)
        torch.manual_seed(1711)
        env, obs = make_reshape_l0_env('urban_canyon', 830001, reshape=rs)
        model = ManeuverMASAC(capacity=2048, seed=1711, device=device, probe_mode='full')
        if reference is None:
            reference = {k:v.detach().cpu().clone() for k,v in model.actor.state_dict().items()}
        else:
            model.actor.load_state_dict(reference)
        initial_action = model.act(obs, deterministic=True)
        if device == 'cpu' and threads == 1:
            reference_action = initial_action.copy()
            reference_risk = model._risk_numpy(obs)
        action_error = float(np.max(np.abs(initial_action-reference_action)))
        risk_error = float(np.max(np.abs(model._risk_numpy(obs)-reference_risk)))
        if action_error > 1e-4 or risk_error > 1e-4:
            raise AssertionError(f'device parity exceeded tolerance: {action_error}, {risk_error}')
        rng = np.random.default_rng(1711)
        for _ in range(160):
            a=rng.uniform(-1,1,(3,3)).astype(np.float32)
            no,r,done,trunc,_=env.step(a)
            model.push(obs,a,r,no,done)
            obs=no
            if done or trunc:
                env,obs=make_reshape_l0_env('urban_canyon',830001,reshape=rs)

        def tick():
            nonlocal env,obs
            action=model.act(obs)
            no,reward,done,trunc,_=env.step(action)
            model.push(obs,action,reward,no,done)
            metrics=model.update(batch=128)
            if not all(np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError(metrics)
            obs=no
            if done or trunc:
                env,obs=make_reshape_l0_env('urban_canyon',830001,reshape=rs)

        for _ in range(8):
            tick()
        timings=[]
        print(f'BENCH START device={device} threads={threads} batch=128 cache_risk=True full seed=1711 warmup_updates=8',flush=True)
        for repeat in range(args.repeats):
            if device=='cuda': torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(args.steps): tick()
            if device=='cuda': torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            timings.append(elapsed/args.steps)
            print(f'BENCH device={device} threads={threads} repeat={repeat+1} steps={args.steps} seconds={elapsed:.4f} steps_per_s={args.steps/elapsed:.3f}',flush=True)
        replay_bytes=sum(x.nbytes for x in (model.obs,model.next,model.actions,model.rewards,model.done,model.risk_feat,model.next_risk_feat))
        row=dict(device=device,threads=threads,seconds_per_step=timings,median_steps_per_s=1/statistics.median(timings),action_max_abs_error=action_error,risk_max_abs_error=risk_error,replay_bytes_capacity_2048=replay_bytes,parameter_counts=model.parameter_counts())
        results.append(row)
        del model,env
        gc.collect()
        if device=='cuda': torch.cuda.empty_cache()
    record=dict(scope='engineering timing, not learning evidence',torch=torch.__version__,cuda_runtime=torch.version.cuda,cudnn=torch.backends.cudnn.version(),platform=platform.platform(),workload='full probes, act + numpy environment + cached replay + batch128 SAC update; 8 update warmup; no eval/reference audits/checkpoint IO',steps_per_repeat=args.steps,repeats=args.repeats,results=results,default_device_unchanged='cpu')
    args.out.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('BENCH SAVED',args.out.resolve(),flush=True)


if __name__=='__main__': main()
