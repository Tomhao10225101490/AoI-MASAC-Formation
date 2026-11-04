#!/usr/bin/env python3
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLCONFIGDIR','/tmp/masac_v3_mpl')
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from uav_planning.structured import StructuredMission
out=ROOT/'results/masac_v3_design';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(15,9));colors=plt.get_cmap('tab10').colors
for idx,layout in enumerate(('urban','warehouse','courtyard')):
    env=StructuredMission(n=6,layout=layout);env.reset(800001);s=env.metadata();ax=fig.add_subplot(2,3,idx+1,projection='3d');top=fig.add_subplot(2,3,idx+4)
    for lo,hi in env.boxes:
        x,y,z=lo;X,Y,Z=hi;v=np.array([[x,y,z],[X,y,z],[X,Y,z],[x,Y,z],[x,y,Z],[X,y,Z],[X,Y,Z],[x,Y,Z]])
        faces=[[v[i] for i in ids] for ids in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]]
        ax.add_collection3d(Poly3DCollection(faces,facecolor=plt.cm.Blues(.25+.5*Z/90),alpha=.45,edgecolor='#51718b',linewidth=.35))
        top.add_patch(Rectangle(lo[:2],X-x,Y-y,facecolor=plt.cm.Blues(.25+.5*Z/90),edgecolor='white',lw=.6))
    for i in range(6):
        ax.scatter(*env.p[i],color=colors[i],s=24);ax.scatter(*env.goal[i],color=colors[i],marker='*',s=55)
        top.scatter(*env.p[i,:2],color=colors[i],s=25);top.scatter(*env.goal[i,:2],color=colors[i],marker='*',s=65)
    ax.scatter(*env.dynamic.T,c='#d14a45',marker='D',s=22)
    top.scatter(*env.dynamic[:,:2].T,c='#d14a45',marker='D',s=25);top.quiver(env.dynamic[:,0],env.dynamic[:,1],env.dv[:,0],env.dv[:,1],color='#d14a45',scale=35)
    ax.set(xlim=(0,240),ylim=(0,240),zlim=(0,90),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=f'{layout.capitalize()} | {len(env.boxes)} buildings')
    ax.set_box_aspect((240,240,90));ax.set_proj_type('ortho');ax.view_init(32,-60);ax.set_xticks([0,120,240]);ax.set_yticks([0,120,240]);ax.set_zticks([0,45,90])
    top.set(xlim=(0,240),ylim=(0,240),xlabel='x (m)',ylabel='y (m)');top.set_aspect('equal');top.set_xticks([0,60,120,180,240]);top.set_yticks([0,60,120,180,240])
fig.suptitle('MASAC structured 3-D mission environments | 6 independent goals | true metric scale\nCircles: starts; stars: goals; red diamonds: moving obstacles; building color: height\nProcedural maps for development — not surveyed city data',fontsize=12)
fig.tight_layout(rect=(0,0,1,.9));fig.savefig(out/'structured_maps.png',dpi=220,bbox_inches='tight');plt.close(fig)
# Static, standalone interactive map preview with no external dependencies.
scenes=[]
for layout in ('urban','warehouse','courtyard'):
    e=StructuredMission(n=6,layout=layout);e.reset(800001);scenes.append(e.metadata())
html='''<!doctype html><meta charset="utf-8"><title>MASAC 3D scene design</title><style>body{background:#101b2b;color:#e5eef7;font:16px system-ui;margin:30px}canvas{background:#17283d;max-width:100%;border-radius:12px}button,input,select{margin:10px}p{max-width:900px;line-height:1.6}</style><h1>MASAC · structured 3D mission scenes</h1><p>Procedural design preview. Buildings are rendered from saved coordinates, circles mark starts, stars mark goals. Red markers move along declared obstacle lanes; UAV trajectories are not fabricated.</p><select id="scene"><option>urban</option><option>warehouse</option><option>courtyard</option></select> Rotate <input id="az" type="range" min="0" max="628" value="66"> Tilt <input id="el" type="range" min="15" max="130" value="65"><br><canvas id="c" width="1100" height="750"></canvas><script>const scenes=DATA,c=document.getElementById('c'),ctx=c.getContext('2d');let tick=0;const colors=['#4cc9f0','#ffab63','#8cda9d','#f07c9c','#c4a1f7','#e4d26a'];function draw(){let s=scenes[document.getElementById('scene').selectedIndex],az=document.getElementById('az').value/100,el=document.getElementById('el').value/100;function P(p){let x=p[0]-120,y=p[1]-120;return [550+2.8*(x*Math.cos(az)-y*Math.sin(az)),500+2.8*((x*Math.sin(az)+y*Math.cos(az))*Math.sin(el)-p[2]*Math.cos(el))]}ctx.clearRect(0,0,1100,750);ctx.strokeStyle='#546882';for(let k=0;k<=240;k+=40){for(let q of [[[0,k,0],[240,k,0]],[[k,0,0],[k,240,0]]]){ctx.beginPath();ctx.moveTo(...P(q[0]));ctx.lineTo(...P(q[1]));ctx.stroke()}}let faces=[];s.buildings.forEach(([l,h])=>{let [x,y,z]=l,[X,Y,Z]=h,v=[[x,y,z],[X,y,z],[X,Y,z],[x,Y,z],[x,y,Z],[X,y,Z],[X,Y,Z],[x,Y,Z]];[[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[4,5,6,7]].forEach(ids=>{let f=ids.map(i=>v[i]);faces.push({f,depth:f.reduce((a,p)=>a+(p[0]*Math.sin(az)+p[1]*Math.cos(az))*Math.cos(el)+p[2]*Math.sin(el),0)/4})})});faces.sort((a,b)=>a.depth-b.depth).forEach(({f})=>{ctx.beginPath();f.forEach((p,i)=>i?ctx.lineTo(...P(p)):ctx.moveTo(...P(p)));ctx.closePath();ctx.fillStyle='#739fc3aa';ctx.fill();ctx.strokeStyle='#9db8cb';ctx.stroke()});s.starts.forEach((p,i)=>{ctx.fillStyle=colors[i];let q=P(p);ctx.beginPath();ctx.arc(...q,6,0,7);ctx.fill();q=P(s.goals[i]);ctx.font='22px system-ui';ctx.fillText('★',q[0]-10,q[1]+8)});s.dynamic.forEach((p,i)=>{let x=p[0]+s.dynamic_velocity[i][0]*tick*.03;let span=210;let u=((x-15)%(2*span)+2*span)%(2*span);let q=P([15+(u<span?u:2*span-u),p[1],p[2]]);ctx.fillStyle='#ff665f';ctx.beginPath();ctx.arc(...q,5,0,7);ctx.fill()});tick++;}setInterval(draw,40);draw();</script>'''.replace('DATA',json.dumps(scenes))
(out/'interactive_maps.html').write_text(html)
