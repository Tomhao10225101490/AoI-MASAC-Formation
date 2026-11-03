#!/usr/bin/env python3
"""Generate and audit deterministic V4 development scenes and reference traces."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_planning.structured import box_distance
from uav_planning.v4_maps import FAMILIES, LEVELS, default_config
from uav_planning.v4_reference import joint_reference


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, obj):
    with path.open('x') as dst:
        json.dump(obj, dst, indent=2, allow_nan=False)
        dst.write('\n')


def endpoint_stats(scene):
    cfg = scene['config']
    starts, goals = np.asarray(scene['starts']), np.asarray(scene['goals'])
    world = np.asarray(cfg['world'])
    boxes = np.asarray(scene['boxes'], dtype=float).reshape(-1,2,3)
    out = {}
    for label, positions in (('start', starts), ('goal', goals)):
        pair = np.linalg.norm(positions[:,None]-positions[None],axis=-1)
        np.fill_diagonal(pair, np.inf)
        out[label] = dict(boundary_margin_m=float(np.minimum(positions-cfg['uav_radius']-cfg['safety_margin'],
                                  world-cfg['uav_radius']-cfg['safety_margin']-positions).min()),
                          peer_clearance_m=float(pair.min()-2*cfg['uav_radius']-cfg['safety_margin']),
                          inflated_box_clearance_m=(float(np.min(box_distance(positions,
                              boxes[:,0]-cfg['uav_radius']-cfg['safety_margin'],
                              boxes[:,1]+cfg['uav_radius']+cfg['safety_margin']))) if len(boxes) else None))
    out['goal_distance_by_uav_m'] = np.linalg.norm(goals-starts,axis=1).tolist()
    return out


HTML = r'''<!doctype html>
<html lang="zh"><meta charset="utf-8"><title>MASAC V4 场景与参考轨迹</title>
<style>body{margin:20px;background:#111d2c;color:#e8f1f7;font:15px system-ui}p{max-width:960px;line-height:1.5}canvas{max-width:100%;background:#1b2c40;border-radius:10px}select,input,button{margin:7px;padding:4px}</style>
<h1>MASAC V4 · 场景与参考执行</h1>
<p>这是模拟器中真实记录的参考控制器轨迹，不是训练后的策略轨迹。可选择不同场景、难度、种子及成功/失败尝试。完整数值和逐步动作保存在同目录审计文件。</p>
<label>场景 <select id="scene"></select></label><label>尝试 <select id="attempt"></select></label>
<button id="toggle">暂停</button><button id="reset">重播</button>
<label>旋转 <input id="az" type="range" min="0" max="628" value="66"></label>
<label>俯仰 <input id="el" type="range" min="15" max="130" value="65"></label>
<div id="status"></div><canvas id="canvas" width="1100" height="750"></canvas>
<script>
const scenes=SCENE_DATA, sceneEl=document.getElementById('scene'), attemptEl=document.getElementById('attempt');
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
const colors=['#4cc9f0','#ffb46b','#8fdb9d','#f07c9c','#c4a1f7','#e4d26a'];
let frame=0,playing=true;
scenes.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=s.label;sceneEl.appendChild(o)});
function choices(){let s=scenes[+sceneEl.value];attemptEl.innerHTML='';s.attempts.forEach((a,i)=>{let o=document.createElement('option');o.value=i;o.textContent='尝试 '+(i+1)+' · '+(a.success?'全队到达':'未成功')+' · '+a.time_s.toFixed(1)+'s';attemptEl.appendChild(o)});if(!s.attempts.length){let o=document.createElement('option');o.textContent='无参考轨迹';attemptEl.appendChild(o)}frame=0}
sceneEl.onchange=choices;attemptEl.onchange=()=>frame=0;document.getElementById('toggle').onclick=()=>{playing=!playing;document.getElementById('toggle').textContent=playing?'暂停':'播放'};document.getElementById('reset').onclick=()=>frame=0;
function draw(){
 let s=scenes[+sceneEl.value],a=s.attempts[+attemptEl.value],world=s.scene.config.world,az=+document.getElementById('az').value/100,el=+document.getElementById('el').value/100,scale=620/Math.max(world[0],world[1]);
 function P(p){let x=p[0]-world[0]/2,y=p[1]-world[1]/2;return [550+scale*(x*Math.cos(az)-y*Math.sin(az)),520+scale*((x*Math.sin(az)+y*Math.cos(az))*Math.sin(el)-p[2]*Math.cos(el))]}
 ctx.clearRect(0,0,1100,750);ctx.strokeStyle='#456078';ctx.beginPath();[[0,0,0],[world[0],0,0],[world[0],world[1],0],[0,world[1],0],[0,0,0]].forEach((p,i)=>i?ctx.lineTo(...P(p)):ctx.moveTo(...P(p)));ctx.stroke();
 let faces=[];s.scene.boxes.forEach(([l,h])=>{let [x,y,z]=l,[X,Y,Z]=h,v=[[x,y,z],[X,y,z],[X,Y,z],[x,Y,z],[x,y,Z],[X,y,Z],[X,Y,Z],[x,Y,Z]];[[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[4,5,6,7]].forEach(ids=>{let f=ids.map(i=>v[i]);faces.push({f,depth:f.reduce((q,p)=>q+(p[0]*Math.sin(az)+p[1]*Math.cos(az))*Math.cos(el)+p[2]*Math.sin(el),0)/4})})});
 faces.sort((x,y)=>x.depth-y.depth).forEach(({f})=>{ctx.beginPath();f.forEach((p,i)=>i?ctx.lineTo(...P(p)):ctx.moveTo(...P(p)));ctx.closePath();ctx.fillStyle='#739fc3aa';ctx.fill();ctx.strokeStyle='#9db8cb';ctx.stroke()});
 s.scene.starts.forEach((p,i)=>{ctx.fillStyle=colors[i];let q=P(p);ctx.beginPath();ctx.arc(...q,5,0,7);ctx.fill();q=P(s.scene.goals[i]);ctx.font='20px system-ui';ctx.fillText('★',q[0]-9,q[1]+7)});
 if(a){let t=Math.min(frame,a.positions.length-1);for(let i=0;i<s.scene.starts.length;i++){ctx.strokeStyle=colors[i];ctx.lineWidth=2;ctx.beginPath();for(let k=0;k<=t;k++){let q=P(a.positions[k][i]);k?ctx.lineTo(...q):ctx.moveTo(...q)}ctx.stroke();let q=P(a.positions[t][i]);ctx.fillStyle=colors[i];ctx.beginPath();ctx.arc(...q,5,0,7);ctx.fill()}a.dynamic[t].forEach(p=>{let q=P(p);ctx.fillStyle='#ff665f';ctx.beginPath();ctx.arc(...q,5,0,7);ctx.fill()});document.getElementById('status').textContent='参考执行 '+(a.success?'成功':'失败')+' · '+(t*s.scene.config.dt).toFixed(1)+' / '+a.time_s.toFixed(1)+' 秒';if(playing&&frame<a.positions.length-1)frame++}else document.getElementById('status').textContent='未找到联合参考轨迹';
}
choices();setInterval(draw,100);draw();
</script></html>'''


def run(out, seeds):
    if len(set(seeds)) != len(seeds):
        raise ValueError('Development map seeds must be unique')
    out.mkdir(parents=True, exist_ok=False)
    records, visual, started = [], [], time.time()
    try:
        for level in LEVELS:
            for family in FAMILIES:
                for seed in seeds:
                    name = f'{level}_{family}_{seed}'
                    item = dict(level=level, family=family, seed=seed)
                    try:
                        cfg = default_config(family, level)
                        result = joint_reference(cfg, seed)
                        scene = result['scene']
                        write_new(out/f'{name}_scene.json', scene)
                        item.update(scene_sha256=scene['scene_sha256'],
                                    static_objects=len(scene['boxes']),
                                    dynamic_objects=len(scene['dynamic_positions']),
                                    endpoints=endpoint_stats(scene),
                                    geometry=result['geometry'],
                                    joint_reference_success=result['joint_reference_success'],
                                    reference_paths=result['paths'], attempts=result['attempts'])
                        view_attempts = []
                        for i, (attempt, trace) in enumerate(zip(result['attempts'], result['traces'])):
                            path = out/f'{name}_attempt_{i+1}.npz'
                            with path.open('xb') as file:
                                np.savez_compressed(file, **trace)
                            item['attempts'][i]['trajectory_file'] = path.name
                            item['attempts'][i]['trajectory_sha256'] = digest(path)
                            view_attempts.append(dict(success=attempt['success'],time_s=attempt['time_s'],
                                 positions=np.round(trace['positions'],3).tolist(),
                                 dynamic=np.round(trace['dynamic'],3).tolist()))
                        visual.append(dict(label=f'{level} · {family} · seed {seed}',
                                           scene=scene, attempts=view_attempts))
                    except Exception as exc:
                        item.update(generation_or_audit_error=repr(exc), joint_reference_success=False)
                    records.append(item)
                    print(f'{name} joint={item["joint_reference_success"]}'
                          f'{" ERROR="+item["generation_or_audit_error"] if "generation_or_audit_error" in item else ""}',
                          flush=True)
        summary = dict(scenes=len(records), map_seeds=seeds,
                       geometry_all_agents_connected=sum(
                           len(r.get('geometry',[])) == 3 and all(x['status'].startswith('connected_')
                               for x in r['geometry']) for r in records),
                       joint_reference_success=sum(r['joint_reference_success'] for r in records),
                       generation_or_audit_errors=sum('generation_or_audit_error' in r for r in records),
                       attempts=sum(len(r.get('attempts',[])) for r in records),
                       failed_attempts=sum(sum(not x['success'] for x in r.get('attempts',[])) for r in records))
        sources = ['uav_planning/v4_maps.py','uav_planning/v4_environment.py','uav_planning/v4_reference.py',
                   'uav_planning/structured.py','uav_planning/environment.py',
                   'docs/design/MASAC_V4_MAP_BENCHMARK.md','scripts/audit_masac_v4_maps.py']
        write_new(out/'map_audit.json', dict(kind='development map feasibility, not policy evaluation',
                  limits='A successful reference witness is simulator-specific and not a learned-policy safety guarantee; missing witness is unresolved, not a proof of infeasibility. Endpoint clearance and geometric paths do not imply joint feasibility.',
                  summary=summary, source_sha256={s:digest(ROOT/s) for s in sources}, records=records,
                  elapsed_s=time.time()-started))
        (out/'interactive_maps.html').write_text(HTML.replace('SCENE_DATA',
                             json.dumps(visual,separators=(',',':'),allow_nan=False)))
        write_new(out/'completion.json', dict(complete=True, summary=summary, completed_at=time.time()))
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as exc:
        write_new(out/'error.json', dict(error=repr(exc), time=time.time()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[920101,920102,920103])
    args = p.parse_args()
    run(args.out, args.seeds)
