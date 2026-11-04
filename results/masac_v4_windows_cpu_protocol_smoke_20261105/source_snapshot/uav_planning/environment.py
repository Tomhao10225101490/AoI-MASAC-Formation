"""Discrete acceleration-limited 3D mission model; continuous segment collision checks.
Velocity updates occur at control ticks, positions move linearly within each tick.
Perfect local state sensing; spherical obstacles. Not a rigid-body flight simulator.
"""
import numpy as np


def swept_distance(a,b):
    delta=b-a;t=np.clip(-np.sum(a*delta,axis=-1)/(np.sum(delta*delta,axis=-1)+1e-12),0,1)
    return np.linalg.norm(a+t[...,None]*delta,axis=-1)

class Mission:
    def __init__(self,n=3,mode='standard',vmax=8.,amax=4.,horizon=150):
        assert 3<=n<=6 and mode in ('standard','distance','prediction','maneuver')
        self.n=n;self.mode=mode;self.vmax=vmax;self.amax=amax;self.horizon=horizon;self.dt=.2;self.radius=1.;self.obs_dim=70
    def reset(self,seed):
        rng=np.random.default_rng(seed);angles=np.arange(self.n)*2*np.pi/self.n+rng.uniform(-.1,.1)
        self.p=np.c_[50+38*np.cos(angles),50+38*np.sin(angles),rng.uniform(14,26,self.n)]
        self.goal=np.c_[100-self.p[:,0],100-self.p[:,1],rng.uniform(14,26,self.n)]
        self.v=np.zeros((self.n,3));self.obstacles=np.array([[35,45,14],[65,55,25],[50,65,18],[50,35,20]],dtype=float)
        self.obstacles[:3]+=rng.uniform(-4,4,(3,3));self.radii=np.array([5.,6.,4.,3.]);self.ov=np.zeros((4,3));self.ov[3,0]=2.
        self.t=0;self.arrived=np.zeros(self.n,bool);self.length=np.zeros(self.n);self.minsep=1e9;self.smooth=0.;self.prev_acc=np.zeros_like(self.v)
        return self.observe()
    def observe(self):
        obs=[]
        for i in range(self.n):
            peers=[j for j in range(self.n) if j!=i];rel=self.p[peers]-self.p[i];vel=self.v[peers]-self.v[i]
            slots=np.zeros((5,6));slots[:len(peers)]=np.c_[rel/100,vel/self.vmax]
            objects=np.c_[(self.obstacles-self.p[i])/100,(self.ov-self.v[i])/self.vmax,self.radii/10]
            features=np.zeros(3)
            # Matched capacity: one summary channel active; standard gets zero.
            if len(rel):
                distance=np.linalg.norm(rel,axis=1);closing=np.maximum(0,-np.sum(rel*vel,axis=1)/(distance+1e-9))
                if self.mode=='distance':features[0]=np.min(distance-2)/100
                elif self.mode=='prediction':
                    tc=np.clip(-np.sum(rel*vel,axis=1)/(np.sum(vel*vel,axis=1)+1e-9),0,2.)
                    features[0]=np.min(np.linalg.norm(rel+tc[:,None]*vel,axis=1)-2)/100
                elif self.mode=='maneuver':features[0]=np.min(distance-2-closing**2/(4*self.amax))/100
            obs.append(np.r_[(self.goal[i]-self.p[i])/100,self.v[i]/self.vmax,self.p[i]/100,slots.ravel(),objects.ravel(),features].astype(np.float32))
        return obs,np.concatenate(obs)
    def step(self,action):
        old=self.p.copy();oldobs=self.obstacles.copy();before=np.linalg.norm(self.goal-self.p,axis=1)
        a=np.clip(np.asarray(action),-1,1);a=a/np.maximum(1,np.linalg.norm(a,axis=1,keepdims=True))*self.amax
        newv=self.v+a*self.dt;newv*=np.minimum(1,self.vmax/(np.linalg.norm(newv,axis=1,keepdims=True)+1e-12))
        # Goal capture must never override acceleration limits.
        acc=(newv-self.v)/self.dt;self.v=newv;self.p+=self.dt*self.v
        if self.obstacles[3,0]>=80:self.ov[3,0]=-2
        if self.obstacles[3,0]<=20:self.ov[3,0]=2
        self.obstacles+=self.dt*self.ov
        dist=swept_distance(old[:,None,:]-old[None,:,:],self.p[:,None,:]-self.p[None,:,:]);np.fill_diagonal(dist,np.inf)
        pair=bool(np.any(dist<2*self.radius));self.minsep=min(self.minsep,float(dist.min()))
        od=swept_distance(old[:,None,:]-oldobs[None,:,:],self.p[:,None,:]-self.obstacles[None,:,:])
        obstacle=bool(np.any(od<self.radius+self.radii))
        boundary=bool(np.any(self.p<np.array([1,1,1])) or np.any(self.p>np.array([99,99,39])))
        after=np.linalg.norm(self.goal-self.p,axis=1);new=(after<3)&(np.linalg.norm(self.v,axis=1)<2)
        capture_delta=new.astype(float)-self.arrived.astype(float)
        self.arrived=new;self.length+=np.linalg.norm(self.p-old,axis=1);self.smooth+=float(np.sum((acc-self.prev_acc)**2));self.prev_acc=acc
        reward=(before-after)-.02-.005*np.sum(a*a,axis=1)+10*capture_delta
        collision=pair or obstacle or boundary
        if collision:reward-=20
        success=bool(np.all(self.arrived) and not collision);terminated=collision or success;self.t+=1
        obs,state=self.observe()
        return obs,state,reward.astype(np.float32),terminated,self.t>=self.horizon,dict(success=success,uav_collision=pair,obstacle_collision=obstacle,boundary_collision=boundary,path_length=float(self.length.sum()),time=self.t*self.dt,min_separation=self.minsep,smoothness=self.smooth)
