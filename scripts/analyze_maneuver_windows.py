"""Restore trusted repository metric bundles and produce a descriptive suite audit."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
RESTORED=ROOT/'local_metrics_restore'


def restore():
    archives=ROOT/'results_archives'
    for stem in ('maneuver_paper_suite_20261103_metrics','v4_aux_and_v3_core_20261103_metrics'):
        parts=sorted((archives/'parts').glob(stem+'.tar.xz.part*'))
        target=archives/(stem+'.tar.xz')
        payload=b''.join(p.read_bytes() for p in parts)
        if not payload: raise FileNotFoundError(stem)
        if target.exists() and target.read_bytes()!=payload: raise ValueError(target)
        if not target.exists(): target.write_bytes(payload)
        print('ASSEMBLED',target.name,'parts',len(parts),'sha256',hashlib.sha256(payload).hexdigest(),flush=True)
    manifest=[]
    for archive in sorted(archives.glob('*.tar.xz')):
        count=0
        with tarfile.open(archive,'r:xz') as bundle:
            for item in bundle:
                target=(RESTORED/item.name).resolve()
                if not target.is_relative_to(RESTORED.resolve()) or not item.name.startswith('results/'):
                    raise ValueError(f'Unsafe archive path {item.name}')
                if item.isdir():
                    target.mkdir(parents=True,exist_ok=True)
                    continue
                if not item.isfile(): raise ValueError(f'Unsupported archive member {item.name}')
                payload=bundle.extractfile(item).read()
                if target.exists():
                    existing=target.read_bytes()
                    # Git autocrlf changes text bytes on Windows. Preserve that
                    # checkout if newline normalization is the only difference.
                    if existing!=payload and existing.replace(b'\r\n',b'\n')!=payload.replace(b'\r\n',b'\n'):
                        raise ValueError(f'Existing file differs beyond CRLF: {target}')
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(payload)
                count+=1
        row=dict(archive=archive.name,files=count,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        manifest.append(row)
        print('RESTORED',json.dumps(row),flush=True)
    return manifest


def reason(row):
    if row['success']: return 'success'
    collisions=[k for k in ('uav_collision','static_collision','dynamic_collision','boundary_collision') if row.get(k)]
    if collisions: return '+'.join(collisions)
    balls=row.get('ever_in_ball_by_uav',[])
    if row.get('timeout') or (row.get('truncated') and not row.get('terminated')):
        return 'timeout_after_all_ball' if balls and all(balls) else ('timeout_after_any_ball' if any(balls) else 'timeout_without_ball')
    return 'other_failure'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restore',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=restore() if args.restore else []
    rows=[]
    curves={}
    for arm in ('static','full','zero'):
        name=f'masac_v4_maneuver_paper_suite_20261103_{arm}_20261103_162345'
        source=RESTORED/'results'/name
        d=json.loads((ROOT/'results_key'/name/'completion.json').read_text())
        curves[arm]={}
        for run in d['runs']:
            seed=run['seed']
            directories=list(source.glob(f'*{seed}'))
            files=list(source.rglob('evaluation_80000.json'))
            files=[x for x in files if str(seed) in x.parent.name]
            if len(files)!=1: raise ValueError(f'{arm} {seed}: expected one final, found {files}')
            final=json.loads(files[0].read_text())
            n=len(final)
            success=sum(x['success'] for x in final)
            if n!=12 or abs(success/n-run['final_development_success'])>1e-10: raise ValueError('completion/eval mismatch')
            row=dict(arm=arm,seed=seed,success=success,n=n,gate_met=run['gate_met'],ever_ball_any=sum(any(x.get('ever_in_ball_by_uav',[])) for x in final),ever_ball_all=sum(bool(x.get('ever_in_ball_by_uav')) and all(x['ever_in_ball_by_uav']) for x in final),failures=dict(Counter(reason(x) for x in final)))
            rows.append(row)
            curve=[]
            for f in files[0].parent.glob('evaluation_*.json'):
                step=int(f.stem.split('_')[-1]); records=json.loads(f.read_text())
                curve.append([step,sum(x['success'] for x in records)/len(records)])
            curves[arm][str(seed)]=sorted(curve)
            print('FINAL',json.dumps(row),flush=True)
    record=dict(scope='descriptive audit of 3 training seeds on 12 reused development maps; no independent-test or statistical-superiority claim',rows=rows,curves=curves,archive_manifest=manifest,definitions={'any_ball':'at least one UAV ever enters its own distance-only ball','all_ball':'every UAV enters its own distance-only ball at some time; not simultaneous or speed-limited','timeout_categories':'collision/success take priority; disjoint all_ball, any-but-not-all, no-ball timeouts'})
    (args.out/'analysis.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    lines=['# Suite diagnosis — 2026-11-05 (Asia/Shanghai)','','Three independent training seeds per arm; 12 reused development maps per seed. Counts are descriptive.','', '| Arm | Seed | Success | Any ever-ball | All ever-ball | Failure/status counts |','|---|---:|---:|---:|---:|---|']
    for r in rows: lines.append(f'| {r["arm"]} | {r["seed"]} | {r["success"]}/12 | {r["ever_ball_any"]}/12 | {r["ever_ball_all"]}/12 | `{json.dumps(r["failures"],sort_keys=True)}` |')
    lines+=['','`Any ever-ball` does not mean all agents approach their goals. Even `all ever-ball` is historical distance-only entry, not simultaneous capture with speed <2 m/s. B sync-capture is a hypothesis; these counts do not identify a causal failure mechanism.','', 'Standard-C is a separate warm-started learnability floor (different seeds, encoder and budget), excluded from the within-suite ranking.']
    (args.out/'DIAGNOSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=True)
    for ax,(arm,series) in zip(axes,curves.items()):
        for seed,points in series.items(): ax.plot([x[0]/1000 for x in points],[x[1] for x in points],label=seed,linewidth=1)
        ax.axhline(.6,color='black',linestyle='--',linewidth=.7)
        ax.set(title=arm,xlabel='Environment steps (thousands)',ylim=(-.02,1.02))
        ax.legend(title='Train seed',fontsize=7)
    axes[0].set_ylabel('Joint success / 12 development maps')
    fig.tight_layout()
    fig.savefig(args.out/'learning_curves.png',dpi=180)
    plt.close(fig)
    print('ANALYSIS SAVED',args.out.resolve(),flush=True)


if __name__=='__main__': main()
