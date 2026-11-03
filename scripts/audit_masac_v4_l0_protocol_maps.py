#!/usr/bin/env python3
"""Preflight frozen L0 development maps and first training tasks."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from uav_planning.v4_environment import V4Mission
from uav_planning.v4_maps import default_config
from uav_planning.v4_reference import joint_reference

PROTOCOL = ROOT/'configs/masac_v4/l0_protocol.json'
OUT = ROOT/'results/masac_v4_l0_preflight_20261103'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, obj):
    with path.open('x') as out:
        json.dump(obj,out,indent=2,allow_nan=False)
        out.write('\n')


def main():
    cfg=json.loads(PROTOCOL.read_text())
    sets=json.loads((ROOT/cfg['seed_sets']).read_text())
    if cfg['development_maps'] != sets['development']['l0']:
        raise ValueError('L0 development maps differ from frozen seed namespace')
    if cfg['seeds'] != sets['training_seeds']['l0']:
        raise ValueError('L0 training seeds differ from frozen seed namespace')
    items=[dict(kind='development',**entry) for entry in cfg['development_maps']]
    for seed in cfg['seeds']:
        for episode in range(10):
            items.append(dict(kind='training_sample',training_seed=seed,episode=episode,
                seed=cfg['training_map_seed_offset']+seed*cfg['training_seed_multiplier']+episode,
                family=cfg['families'][episode%len(cfg['families'])],level=cfg['level']))
    ids=[item['seed'] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError('Development and sampled training map seeds overlap')
    test_range=sets['final_test']['l0_seed_range']
    if any(test_range[0] <= seed <= test_range[1] for seed in ids):
        raise ValueError('Reserved confirmation seed was included')
    OUT.mkdir(parents=True,exist_ok=False)
    sources=['uav_planning/v4_maps.py','uav_planning/v4_environment.py','uav_planning/v4_reference.py',
             'uav_planning/structured.py','uav_planning/environment.py',
             'docs/design/MASAC_V4_MAP_BENCHMARK.md','scripts/audit_masac_v4_l0_protocol_maps.py',
             'configs/masac_v4/l0_protocol.json','configs/masac_v4/seed_sets.json']
    rows=[]; started=time.perf_counter()
    try:
        for item in items:
            t=time.perf_counter()
            name=f'{item["level"]}_{item["family"]}_{item["seed"]}'
            config=default_config(item['family'],item['level'],sensor_range=cfg['sensor_range'])
            ref=joint_reference(config,item['seed'])
            env=V4Mission(config); obs=env.reset(item['seed'])
            if obs.shape != (3,162) or env.scene.scene_hash != ref['scene']['scene_sha256']:
                raise ValueError(f'L0 actor interface or reference scene mismatch: {name}')
            write_new(OUT/f'{name}_scene.json',ref['scene'])
            attempts=[]
            for index,(summary,trace) in enumerate(zip(ref['attempts'],ref['traces'])):
                trace_file=OUT/f'{name}_attempt_{index+1}.npz'
                with trace_file.open('xb') as f:
                    np.savez_compressed(f,**trace)
                attempts.append({**summary,'trajectory_file':trace_file.name,'trajectory_sha256':digest(trace_file)})
            row={**item,'scene_sha256':env.scene.scene_hash,'geometry':ref['geometry'],
                 'joint_reference_success':ref['joint_reference_success'],'attempts':attempts,
                 'elapsed_s':time.perf_counter()-t}
            rows.append(row)
            print(name, 'joint=', row['joint_reference_success'],flush=True)
        summary=dict(scenes=len(rows),development_scenes=sum(x['kind']=='development' for x in rows),
            training_sample_scenes=sum(x['kind']=='training_sample' for x in rows),
            joint_success=sum(x['joint_reference_success'] for x in rows),
            failed=sum(not x['joint_reference_success'] for x in rows),
            elapsed_s=time.perf_counter()-started)
        result=dict(kind='L0 development and training-sample reference preflight',
            limits='Finite sample only, and simulator feasibility is not policy success or a safety guarantee. Confirmation seeds were not generated.',
            summary=summary,source_sha256={name:digest(ROOT/name) for name in sources},records=rows)
        write_new(OUT/'map_audit.json',result)
        write_new(OUT/'completion.json',dict(complete=summary['failed']==0,summary=summary))
        print(json.dumps(summary,indent=2),flush=True)
        if summary['failed']:
            raise ValueError('At least one requested L0 reference witness failed')
    except BaseException as exc:
        write_new(OUT/'error.json',dict(error=repr(exc),time=time.time()))
        raise


if __name__=='__main__':
    main()
