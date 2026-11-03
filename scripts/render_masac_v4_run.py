#!/usr/bin/env python3
"""Render a saved evaluation trajectory; never launches training.

Example: PYTHONPATH=. .venv/bin/python scripts/render_masac_v4_run.py
 --trajectory .../eval_300000_800001.npz --map .../eval_300000_800001_map.json
 --evaluation .../evaluation_300000.json --history .../training.jsonl --out-dir /tmp/render
HTML requires the Plotly CDN; PNG diagnostics work offline. Inputs are hashed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(path):
    if not path: return {}
    with open(path) as f: return json.load(f)


def choose_evaluation(value, seed):
    if isinstance(value, list):
        rows=[r for r in value if isinstance(r,dict) and str(r.get('map_seed',r.get('seed'))) == str(seed)]
        if len(rows)==1: return rows[0]
        if len(value)==1 and isinstance(value[0],dict): return value[0]
        return {}
    if isinstance(value,dict):
        if isinstance(value.get('episodes'),list): return choose_evaluation(value['episodes'],seed)
        return value
    return {}


def get_status(e):
    if not e: return 'UNKNOWN: no matching saved evaluation'
    labels=[]
    if e.get('success') is True: labels.append('SUCCESS')
    elif e.get('success') is False: labels.append('FAILURE')
    else: labels.append('UNKNOWN SUCCESS STATUS')
    for key,name in [('uav_collision','UAV COLLISION'),('static_collision','STATIC COLLISION'),('dynamic_collision','DYNAMIC COLLISION'),('boundary_collision','BOUNDARY / OUT OF BOUNDS')]:
        if e.get(key) is True: labels.append(name)
    if e.get('truncated') is True or e.get('timeout') is True: labels.append('TIMEOUT / TRUNCATED')
    return ' · '.join(labels)


def mesh(box):
    lo,hi=np.asarray(box,dtype=float).tolist()
    return dict(type='mesh3d',x=[lo[0],hi[0],hi[0],lo[0],lo[0],hi[0],hi[0],lo[0]],
        y=[lo[1],lo[1],hi[1],hi[1],lo[1],lo[1],hi[1],hi[1]],
        z=[lo[2],lo[2],lo[2],lo[2],hi[2],hi[2],hi[2],hi[2]],
        i=[0,0,4,4,0,0,1,1,2,2,3,3],j=[1,2,5,6,1,5,2,6,3,7,0,4],
        k=[2,3,6,7,5,4,6,5,7,6,4,7],color='#8995a1',opacity=.28,hoverinfo='skip',showlegend=False)


def calculate_clearance(pos, boxes, dynamic, radii, world, radius):
    values=[]; names=[]
    if boxes is not None and len(boxes):
        p=pos[:,:,None,:]; lo=boxes[None,None,:,0,:]; hi=boxes[None,None,:,1,:]
        q=np.maximum(lo-p,p-hi)
        d=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(np.max(q,axis=-1),0)-radius
        values.append(d.min(axis=(1,2))); names.append('static boundary')
    if world is not None:
        values.append(np.minimum(pos-radius,np.asarray(world)-radius-pos).min(axis=(1,2))); names.append('world boundary')
    if pos.shape[1]>1:
        ds=[]
        for i in range(pos.shape[1]):
            for j in range(i+1,pos.shape[1]): ds.append(np.linalg.norm(pos[:,i]-pos[:,j],axis=-1)-2*radius)
        values.append(np.min(ds,axis=0)); names.append('other UAV')
    if dynamic is not None and radii is not None and len(radii):
        count=min(len(pos),len(dynamic)); d=np.linalg.norm(pos[:count,:,None,:]-dynamic[:count,None,:,:],axis=-1)-radius-np.asarray(radii)[None,None,:]
        val=np.full(len(pos),np.nan); val[:count]=d.min(axis=(1,2)); values.append(val); names.append('dynamic obstacle')
    return names,values


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--trajectory',required=True); parser.add_argument('--map'); parser.add_argument('--evaluation'); parser.add_argument('--history'); parser.add_argument('--out-dir',required=True); parser.add_argument('--dt',type=float,default=.2)
    args=parser.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=False)
    with np.load(args.trajectory,allow_pickle=False) as n: data={k:n[k] for k in n.files}
    meta=metadata(args.map); evaluation=choose_evaluation(metadata(args.evaluation),meta.get('seed'))
    pos=np.asarray(data['positions']); goals=np.asarray(data['goals']) if 'goals' in data else np.asarray(meta.get('goals',[]))
    if pos.ndim!=3 or pos.shape[-1]!=3: raise ValueError('positions must be [time, agents, 3]')
    boxes=data.get('buildings',meta.get('buildings')); boxes=None if boxes is None else np.asarray(boxes)
    config=meta.get('config',{})
    dynamic=data.get('dynamic'); radii=data.get('dynamic_radii',meta.get('dynamic_radii')); world=meta.get('world',config.get('world')); radius=float(meta.get('radius',config.get('uav_radius',1.)))
    status=get_status(evaluation); traces=[]
    if boxes is not None:
        if boxes.ndim!=3 or boxes.shape[1:]!=(2,3): raise ValueError('buildings must be [N,2,3] AABBs')
        traces.extend(mesh(box) for box in boxes)
    for i in range(pos.shape[1]):
        p=pos[:,i,:]; traces.append(dict(type='scatter3d',mode='lines+markers',name=f'UAV {i}',x=p[:,0].tolist(),y=p[:,1].tolist(),z=p[:,2].tolist(),line=dict(width=5),marker=dict(size=1.5)))
        traces.append(dict(type='scatter3d',mode='markers',name=f'UAV {i} start',x=[float(p[0,0])],y=[float(p[0,1])],z=[float(p[0,2])],marker=dict(size=5,symbol='circle')))
        if goals.shape==(pos.shape[1],3):
            g=goals[i]; traces.append(dict(type='scatter3d',mode='markers',name=f'UAV {i} goal',x=[float(g[0])],y=[float(g[1])],z=[float(g[2])],marker=dict(size=6,symbol='diamond')))
    if dynamic is not None and dynamic.ndim==3:
        for i in range(dynamic.shape[1]):
            p=dynamic[:,i]; traces.append(dict(type='scatter3d',mode='lines',name=f'Dynamic {i}',x=p[:,0].tolist(),y=p[:,1].tolist(),z=p[:,2].tolist(),line=dict(width=3,dash='dash')))
    layout={'title':status,'scene':{'xaxis_title':'x (m)','yaxis_title':'y (m)','zaxis_title':'z (m)','aspectmode':'data'},'legend':{'orientation':'h'}}
    html='<!doctype html><meta charset="utf-8"><title>Saved trajectory</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><div id="plot" style="width:100%;height:96vh"></div><script>Plotly.newPlot("plot",'+json.dumps(traces).replace('</','<\\/')+','+json.dumps(layout).replace('</','<\\/')+', {responsive:true});</script>'
    (out/'trajectory.html').write_text(html,encoding='utf-8')
    t=np.arange(len(pos))*args.dt; fig,axs=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
    if goals.shape==(pos.shape[1],3):
        for i in range(pos.shape[1]): axs[0].plot(t,np.linalg.norm(pos[:,i]-goals[i],axis=-1),label=f'UAV {i}')
        axs[0].legend(); axs[0].set_ylabel('distance to goal (m)')
    else: axs[0].text(.5,.5,'Goal positions unavailable',ha='center',transform=axs[0].transAxes)
    names,values=calculate_clearance(pos,boxes,dynamic,radii,world,radius)
    if 'clearances' in data:
        saved=np.asarray(data['clearances'])
        if saved.ndim == 1 and len(saved) == len(pos):
            names.insert(0,'saved clearance'); values.insert(0,saved)
    for name,value in zip(names,values): axs[1].plot(t,value,label=name)
    if names: axs[1].legend(); axs[1].axhline(0,color='red',lw=.7)
    else: axs[1].text(.5,.5,'Clearance inputs unavailable',ha='center',transform=axs[1].transAxes)
    axs[1].set(xlabel='time (s)',ylabel='minimum clearance (m)'); fig.suptitle(status); fig.savefig(out/'trajectory_diagnostics.png',dpi=160); plt.close(fig)
    if 'velocities' in data or 'actions' in data:
        fig,axs=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
        for ax,key,label in ((axs[0],'velocities','speed (m/s)'),(axs[1],'actions','action norm (saved units)')):
            value=data.get(key)
            if value is not None and value.ndim==3 and value.shape[1:]==pos.shape[1:]:
                for i in range(value.shape[1]): ax.plot(np.arange(len(value))*args.dt,np.linalg.norm(value[:,i],axis=-1),label=f'UAV {i}')
                ax.legend(); ax.set_ylabel(label)
            else: ax.text(.5,.5,f'{key} unavailable',ha='center',transform=ax.transAxes)
        axs[-1].set_xlabel('time (s)'); fig.suptitle(status); fig.savefig(out/'velocity_actions.png',dpi=160); plt.close(fig)
    if args.history:
        rows=[]
        with open(args.history) as f:
            for line in f:
                try: row=json.loads(line)
                except json.JSONDecodeError: continue
                if isinstance(row,dict) and isinstance(row.get('step'),(int,float)): rows.append(row)
        fig,axs=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
        for ax,key in zip(axs,('return_','remaining_distance')):
            valid=[r for r in rows if isinstance(r.get(key),(int,float))]
            if valid: ax.plot([r['step'] for r in valid],[r[key] for r in valid],lw=.7); ax.set_ylabel(key)
            else: ax.text(.5,.5,f'No {key} in saved history',ha='center',transform=ax.transAxes)
        axs[-1].set_xlabel('environment steps'); fig.suptitle('Recorded training episodes; no smoothing'); fig.savefig(out/'training_history.png',dpi=160); plt.close(fig)
    files=[p for p in (args.trajectory,args.map,args.evaluation,args.history) if p]
    (out/'manifest.json').write_text(json.dumps({'source_sha256':{p:sha(p) for p in files},'renderer_sha256':sha(__file__),'evaluation_status':status,'success_saved':evaluation.get('success'),'explicit_timeout_saved':evaluation.get('timeout',evaluation.get('truncated')),'time_step_seconds':args.dt,'radius_m_used':radius,'radius_source':'map metadata' if 'radius' in meta or 'uav_radius' in config else 'default assumption','plotly_source':'CDN; online connection required','saved_fields':list(data)},indent=2)+'\n')
    print(status); print(f'Rendered to {out}');
    if evaluation.get('success') is not True: print('No confirmed success in this evaluation; no success figure was created.')

if __name__=='__main__': main()
