"""Audit L0 object content, probe constant channels, and paired small-run results."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from uav_planning.v4_prediction import ProbeConfig,predict_probes
from uav_planning.v4_sync_capture import failure_reason


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,value):
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    maps=[]
    folder=ROOT/'local_metrics_restore/results/masac_v4_maneuver_paper_suite_20261103_static_20261103_162345/maneuver_static_1211'
    for p in sorted(folder.glob('eval_80000_*_map.json')):
        d=json.loads(p.read_text())
        maps.append(dict(seed=d['seed'],family=d['config']['family'],static_count=len(d['boxes']),dynamic_count=len(d['dynamic_positions']),conflict_degree=d['config']['conflict_degree'],source=str(p.relative_to(ROOT)),sha256=digest(p)))
    if len(maps)!=12: raise ValueError('expected all twelve archived development maps')
    replay=ROOT/'results/masac_v4_sync_control_cuda_pilot_20261105/maneuver_static_1411/checkpoint_002000/replay.npz'
    with np.load(replay) as z:
        original=z['obs']
        ids=np.linspace(0,len(original)-1,128,dtype=int)
        obs=original[ids].copy()
    with torch.no_grad():
        full=predict_probes(torch.as_tensor(obs),ProbeConfig(world=(240.,240.,90.)),mode='full')
        static=predict_probes(torch.as_tensor(obs),ProbeConfig(world=(240.,240.,90.)),mode='static')
    features=[]
    for col,name in enumerate(('ux','uy','uz','static','moving','peer','boundary','ttc','brake','reserve','positive')):
        f=full[...,col];s=static[...,col]
        features.append(dict(name=name,n_values=f.numel(),full_min=float(f.min()),full_mean=float(f.mean()),full_max=float(f.max()),full_exact_one_count=int((f==1).sum()),full_exact_one_fraction=float((f==1).float().mean()),static_min=float(s.min()),static_mean=float(s.mean()),static_max=float(s.max()),mean_abs_delta=float((f-s).abs().mean()),max_abs_delta=float((f-s).abs().max())))
    content=dict(scope='fixed L0 development-map metadata plus newly collected early static-control replay; not the historical final-policy distribution',maps=maps,replay=str(replay.relative_to(ROOT)),replay_sha256=digest(replay),sample_indices=ids.tolist(),tensor_shape=list(full.shape),features=features,source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__).resolve(),ROOT/'uav_planning/v4_prediction.py')})
    save(args.out/'l0_content_audit.json',content)
    print('CONTENT',json.dumps(dict(maps=len(maps),dynamic_counts=[m['dynamic_count'] for m in maps],tensor_shape=content['tensor_shape'],features=features)),flush=True)
    rows=[]
    for arm,name in [('control','masac_v4_sync_control_cuda_pilot_20261105'),('B','masac_v4_sync_b_cuda_pilot_20261105')]:
        folder=ROOT/'results'/name
        completion=json.loads((folder/'completion.json').read_text())
        if not completion['complete']: raise ValueError(f'incomplete {folder}')
        for run in completion['runs']:
            seed=run['seed']; steps=run['step']
            directory=folder/f'maneuver_static_{seed}'
            final=json.loads((directory/f'evaluation_{steps}.json').read_text())
            metrics=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
            episodes=[x for x in metrics if x['kind']=='episode']
            progress=[x for x in metrics if x['kind']=='progress']
            diagnostics=[x for x in progress if 'critic_loss' in x]
            keys=('critic_loss','actor_loss','alpha','critic_grad','actor_grad','entropy','td_abs')
            if not diagnostics or not all(np.isfinite(x[k]) for x in diagnostics for k in keys):
                raise ValueError('missing or nonfinite logged diagnostic')
            row=dict(arm=arm,seed=seed,steps=steps,updates=progress[-1]['updates'],success=sum(x['success'] for x in final),n_maps=len(final),ever_ball_any=sum(any(x['ever_in_ball_by_uav']) for x in final),ever_ball_all=sum(all(x['ever_in_ball_by_uav']) for x in final),max_simultaneous_capture=max(x['max_simultaneous_capture_occupancy'] for x in final),failures=dict(Counter(failure_reason(x) for x in final if not x['success'])),logged_diagnostics_finite=True,per_update_finite_guard=True,progress_samples=len(diagnostics),sampled_critic_clipping_fraction=float(np.mean([x['critic_clipped'] for x in diagnostics])),last_entropy=diagnostics[-1]['entropy'],last_alpha=diagnostics[-1]['alpha'],completed_training_episodes=len(episodes),base_return_completed_episodes=sum(x['base_return'] for x in episodes),shaping_return_completed_episodes=sum(x['shaping_return'] for x in episodes),runner_elapsed_s_two_seeds=completion['elapsed_s'],checkpoint=str((directory/f'checkpoint_{steps:06d}').relative_to(ROOT)))
            rows.append(row)
            print('PILOT',json.dumps(row),flush=True)
    save(args.out/'summary.json',dict(scope='2k-step two-seed implementation/finite-loss pilot, not method confirmation',rows=rows))
    text=['# B small pilot results — 2026-11-05','','Two arms × two seeds × 2,000 steps = 8,000 transitions and 7,488 SAC updates.','Both arms cold-start on CUDA. These results validate execution only.','', '| Arm | Seed | Success | Any/all ball | Failures | Last entropy | Last alpha |','|---|---:|---:|---:|---|---:|---:|']
    for r in rows: text.append(f'| {r["arm"]} | {r["seed"]} | {r["success"]}/12 | {r["ever_ball_any"]}/{r["ever_ball_all"]} | `{json.dumps(r["failures"])}` | {r["last_entropy"]:.4f} | {r["last_alpha"]:.4f} |')
    text+=['','Every update used the finite-diagnostic stop guard; all sampled logged diagnostics are finite. Critic clipping is frequent, but this does not identify its causal role. Episode shaping/base totals exclude the unfinished final episode and can cancel internally; they are not per-step magnitude estimates.','', 'No evidence of improved joint success at this small budget. Preserve lambda=.5 and tau=.1; a separately frozen 40k study tests B without tuning on these results.','', 'All 12 archived L0 development maps have zero moving objects and conflict_degree=0. The early replay probe audit found moving-clearance full=1 versus static=0, and mostly saturated peer clearance; see l0_content_audit.json. This cannot isolate moving-obstacle prediction and does not characterize all historical final-policy trajectories.']
    (args.out/'PILOT_RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__': main()
