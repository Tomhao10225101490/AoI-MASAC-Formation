#!/usr/bin/env python3
import json,sys,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/masac_planning_mpl')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(sys.argv[1]);assert (out/'COMPLETE').exists()
fig,axs=plt.subplots(1,2,figsize=(11,4));summary=[]
for mode in ('standard','distance','prediction','maneuver'):
    returns=[];evaluations=[]
    for seed in (101,202,303):
        d=out/f'{mode}_{seed}';rows=[json.loads(s) for s in (d/'training.jsonl').read_text().splitlines()];eps=[r for r in rows if 'episode_return' in r]
        axs[0].plot([r['step'] for r in eps],[r['episode_return'] for r in eps],alpha=.35,label=mode if seed==101 else None)
        evaluations+=json.loads((d/'evaluation.json').read_text())
    summary.append(dict(mode=mode,evaluations=len(evaluations),successes=sum(r['success'] for r in evaluations),uav_collisions=sum(r['uav_collision'] for r in evaluations),obstacle_collisions=sum(r['obstacle_collision'] for r in evaluations),boundary_collisions=sum(r['boundary_collision'] for r in evaluations)))
axs[0].set(xlabel='Environment steps',ylabel='Episode return',title='Three independent training seeds per arm');axs[0].legend()
axs[1].bar([r['mode'] for r in summary],[r['successes']/r['evaluations'] for r in summary]);axs[1].set(ylabel='Collision-free mission success',ylim=(0,1),title='20 held-out maps per training seed');axs[1].tick_params(axis='x',rotation=20)
fig.tight_layout();fig.savefig(out/'training_and_success.png',dpi=200);plt.close(fig)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
fig=plt.figure(figsize=(11,5))
for i,mode in enumerate(('standard','maneuver')):
    ax=fig.add_subplot(1,2,i+1,projection='3d');data=np.load(out/f'{mode}_101/trajectory_9000.npz');xyz=data['positions']
    for j in range(3):ax.plot(*xyz[:,j].T,label=f'UAV {j+1}');ax.scatter(*data['goals'][j],marker='*',s=60)
    u,v=np.mgrid[0:2*np.pi:16j,0:np.pi:9j]
    for center,r in zip(data['obstacles'],data['radii']):ax.plot_surface(center[0]+r*np.cos(u)*np.sin(v),center[1]+r*np.sin(u)*np.sin(v),center[2]+r*np.cos(v),alpha=.2,color='gray')
    ax.set(xlim=(0,100),ylim=(0,100),zlim=(0,40),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=mode+' | fixed seed 101 / map 9000')
    ax.set_box_aspect((100,100,40));ax.legend()
fig.suptitle('Actual pilot trajectories; failures retained; obstacle positions shown at final tick')
fig.tight_layout();fig.savefig(out/'trajectories.png',dpi=200)
print(json.dumps(summary,indent=2))
