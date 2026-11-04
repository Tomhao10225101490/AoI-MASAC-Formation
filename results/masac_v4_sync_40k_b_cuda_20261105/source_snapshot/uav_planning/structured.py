"""Structured 3D mission scenes and local control-reserve observations (v3)."""
import itertools
import numpy as np
from uav_planning.environment import swept_distance

DIRECTIONS=np.array([x for x in itertools.product((-1.,0.,1.),repeat=3) if np.count_nonzero(x)<=2])
DIRECTIONS/=np.maximum(1,np.linalg.norm(DIRECTIONS,axis=1,keepdims=True))

def limited(v,action,dt,amax,vmax):
    action=action/np.maximum(1,np.linalg.norm(action,axis=-1,keepdims=True))
    w=v+dt*amax*action
    return w*np.minimum(1,vmax/(np.linalg.norm(w,axis=-1,keepdims=True)+1e-12))

def box_distance(points,lo,hi):
    center=(lo+hi)/2;half=(hi-lo)/2;q=np.abs(points[...,None,:]-center)-half
    return np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(np.max(q,axis=-1),0)

def segment_box_hit(start,end,lo,hi):
    # Slab intersection against radius-inflated boxes, conservative at corners.
    a=start[...,None,:];delta=(end-start)[...,None,:];parallel=np.abs(delta)<1e-12
    outside=np.any(parallel&((a<lo)|(a>hi)),axis=-1)
    safe=np.where(parallel,1.,delta);t1=(lo-a)/safe;t2=(hi-a)/safe
    near=np.where(parallel,-np.inf,np.minimum(t1,t2));far=np.where(parallel,np.inf,np.maximum(t1,t2))
    return (~outside)&(np.maximum(0,np.max(near,axis=-1))<=np.minimum(1,np.min(far,axis=-1)))

class StructuredMission:
    obs_dim=162
    def __init__(self,n=3,mode='standard',layout='urban',vmax=10.,amax=4.,horizon=350):
        assert 3<=n<=6 and mode in ('standard','distance','prediction','maneuver')
        self.n=n;self.mode=mode;self.layout=layout;self.vmax=vmax;self.amax=amax;self.horizon=horizon
        self.dt=.2;self.radius=1.;self.world=np.array([240.,240.,90.])
    def reset(self,seed):
        self.seed=seed;rng=np.random.default_rng(seed);boxes=[]
        for i,x in enumerate((40,80,120,160,200)):
            for j,y in enumerate((40,80,120,160,200)):
                if self.layout=='warehouse' and j==2:continue
                if self.layout=='courtyard' and i==j==2:continue
                width=rng.uniform(22,29,2) if self.layout!='warehouse' else rng.uniform(28,32,2)
                height=rng.uniform(30,85) if self.layout=='urban' else rng.uniform(18,55)
                boxes.append([[x-width[0]/2,y-width[1]/2,0],[x+width[0]/2,y+width[1]/2,height]])
        self.boxes=np.array(boxes);self.lo=self.boxes[:,0];self.hi=self.boxes[:,1]
        y=np.linspace(32,208,self.n);left=np.arange(self.n)%2==0
        self.p=np.c_[np.where(left,12.,228.),y,rng.uniform(18,68,self.n)]
        self.goal=self.p.copy();self.goal[:,0]=240-self.p[:,0];self.goal[:,1]=240-self.p[:,1]
        self.v=np.zeros_like(self.p);self.prev_acc=np.zeros_like(self.p);self.captured=np.zeros(self.n,bool)
        self.dynamic=np.c_[rng.uniform(35,205,4),[60,100,140,180],rng.uniform(20,60,4)]
        self.dv=np.zeros((4,3));self.dv[:,0]=rng.choice([-3.,3.],4);self.dr=np.full(4,2.5)
        self.t=0;self.length=0.;self.jerk=0.;self.minimum_separation=float('inf')
        assert np.min(box_distance(self.p,self.lo,self.hi))>self.radius
        assert np.min(box_distance(self.goal,self.lo,self.hi))>self.radius
        self.last_margins=np.zeros((self.n,19));return self.observe()
    def metadata(self):
        return dict(seed=int(self.seed),layout=self.layout,world=self.world.tolist(),buildings=self.boxes.tolist(),goals=self.goal.tolist(),starts=self.p.tolist(),dynamic=self.dynamic.tolist(),dynamic_velocity=self.dv.tolist(),dynamic_radii=self.dr.tolist(),source='procedural structured scene, not measured city',radius=self.radius)
    def local_margins(self,i,indices,controls):
        pos=np.broadcast_to(self.p[i],(len(controls),3)).copy();vel=np.broadcast_to(self.v[i],pos.shape).copy();margin=np.full(len(controls),np.inf)
        peers=np.delete(self.p,i,axis=0);pv=np.delete(self.v,i,axis=0)
        # Point-in-time preview only; not a continuous safety certificate.
        for k in range(11):
            t=k*self.dt
            b=box_distance(pos,self.lo[indices],self.hi[indices]).min(-1)-self.radius
            bounds=np.minimum(pos-self.radius,self.world-self.radius-pos).min(-1)
            u=np.linalg.norm(pos[:,None]-peers[None]-t*pv[None],axis=-1).min(-1)-2*self.radius
            d=(np.linalg.norm(pos[:,None]-self.dynamic[None]-t*self.dv[None],axis=-1)-self.dr-self.radius).min(-1)
            margin=np.minimum(margin,np.minimum(np.minimum(b,bounds),np.minimum(u,d)))
            vel=limited(vel,controls,self.dt,self.amax,self.vmax);pos+=vel*self.dt
        return margin
    def observe(self):
        observations=[]
        for i in range(self.n):
            ids=np.argsort(box_distance(self.p[i],self.lo,self.hi))[:8];peers=[j for j in range(self.n) if j!=i]
            own=np.r_[(self.goal[i]-self.p[i])/240,self.v[i]/self.vmax,self.prev_acc[i]/self.amax,self.p[i]/self.world,self.p[i]/self.world,(self.world-self.p[i])/self.world,self.vmax/20,self.amax/10]
            other=np.zeros((5,7));other[:self.n-1]=np.c_[(self.p[peers]-self.p[i])/240,(self.v[peers]-self.v[i])/self.vmax,np.ones(self.n-1)]
            buildings=np.c_[((self.lo[ids]+self.hi[ids])/2-self.p[i])/240,(self.hi[ids]-self.lo[ids])/480,np.ones(len(ids))]
            moving=np.c_[(self.dynamic-self.p[i])/240,(self.dv-self.v[i])/self.vmax,self.dr/10,np.ones(4)]
            margins=np.zeros(19)
            if self.mode=='distance':
                m=min(float(box_distance(self.p[i],self.lo[ids],self.hi[ids]).min()-1),float(np.linalg.norm(self.p[peers]-self.p[i],axis=-1).min()-2),float((np.linalg.norm(self.dynamic-self.p[i],axis=-1)-self.dr-1).min()),float(np.minimum(self.p[i]-1,self.world-1-self.p[i]).min()))
                margins[:]=m
            elif self.mode=='prediction':margins[:]=self.local_margins(i,ids,np.zeros((1,3)))[0]
            elif self.mode=='maneuver':margins=self.local_margins(i,ids,DIRECTIONS)
            self.last_margins[i]=margins
            observations.append(np.r_[own,other.ravel(),buildings.ravel(),moving.ravel(),np.clip(margins/40,-1,1)])
        return np.asarray(observations,dtype=np.float32)
    def step(self,action):
        old=self.p.copy();oldd=self.dynamic.copy();before=np.linalg.norm(self.goal-self.p,axis=-1);oldv=self.v.copy()
        self.v=limited(self.v,np.clip(action,-1,1),self.dt,self.amax,self.vmax);acc=(self.v-oldv)/self.dt;self.p+=self.dt*self.v
        self.dv[(self.dynamic[:,0]>=225)&(self.dv[:,0]>0),0]*=-1
        self.dv[(self.dynamic[:,0]<=15)&(self.dv[:,0]<0),0]*=-1
        self.dynamic+=self.dt*self.dv
        sep=swept_distance(old[:,None]-old[None],self.p[:,None]-self.p[None]);np.fill_diagonal(sep,np.inf)
        peer=bool(np.any(sep<2));self.minimum_separation=min(self.minimum_separation,float(sep.min()))
        static=bool(segment_box_hit(old,self.p,self.lo-self.radius,self.hi+self.radius).any())
        moving=bool(np.any(swept_distance(old[:,None]-oldd[None],self.p[:,None]-self.dynamic[None])<self.dr+1))
        boundary=bool(np.any(self.p<1) or np.any(self.p>self.world-1));after=np.linalg.norm(self.goal-self.p,axis=-1)
        arrived=(after<4)&(np.linalg.norm(self.v,axis=-1)<2);delta=arrived.astype(float)-self.captured.astype(float);self.captured=arrived
        reward=(before-after)/2-.02-.002*np.sum(acc*acc,axis=-1)+5*delta
        collision=peer or static or moving or boundary
        if collision:reward-=20
        success=bool(arrived.all() and not collision)
        if success:reward+=20
        self.length+=float(np.linalg.norm(self.p-old,axis=-1).sum());self.jerk+=float(np.sum(((acc-self.prev_acc)/self.dt)**2)*self.dt);self.prev_acc=acc;self.t+=1
        info=dict(success=success,uav_collision=peer,static_collision=static,dynamic_collision=moving,boundary_collision=boundary,arrived_fraction=float(arrived.mean()),path_length=self.length,time=self.t*self.dt,min_separation=self.minimum_separation,jerk_integral=self.jerk,remaining_distance=float(after.mean()))
        return self.observe(),float(np.mean(reward)),bool(collision or success),self.t>=self.horizon,info
