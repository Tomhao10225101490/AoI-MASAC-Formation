"""Observation-only directional probes. These are candidate features, not safety guarantees.

Dynamics extrapolate visible objects with constant velocity; no map, simulator
state, future waypoint, or true acceleration is accepted. The trajectory is
sampled discretely at dt, so clearance and TTC may miss between-step contacts.

Clearance aggregation is vectorized over the preview horizon (bit-identical to
the prior per-step loop) to cut CPU cost on multi-arm Mac training.
"""
from dataclasses import dataclass
import torch
from uav_planning.structured import DIRECTIONS


@dataclass(frozen=True)
class ProbeConfig:
    world: tuple = (240.,240.,90.)
    body_radius: float = 1.
    safety_margin: float = 0.
    dt: float = .2
    horizon: float = 2.
    clearance_scale: float = 40.
    mismatch_acceleration_factor: float = .5


def predict_probes(obs, config=ProbeConfig(), mode='full', *, generator=None):
    """Compute (...,19,11) features from current 162D observations alone.

    Columns: canonical direction[3], static/moving/peer/boundary clearance[4],
    earliest sampled collision time, braking distance, current-clearance braking
    reserve, positive minimum preview clearance flag.
    Clearance and reserve use 40 m units and clip [-1,1]; braking distance clips
    [0,1], TTC is divided by horizon. Missing entities have +1 clearance.
    Modes: full, static, zero, shuffle, mismatch. In static mode, dynamic and
    peer output channels are zero and excluded from the safety calculation.
    `shuffle` keeps canonical
    direction columns fixed and permutes the 8 scalar values among probes.
    """
    if mode not in ('full','static','zero','shuffle','mismatch'): raise ValueError(mode)
    if obs.shape[-1] != 162 or not obs.is_floating_point(): raise ValueError('expected floating (...,162)')
    if len(config.world)!=3 or min(config.world)<=0 or config.dt<=0 or config.horizon<=0 or config.clearance_scale<=0 or config.body_radius<0 or config.safety_margin<0:
        raise ValueError('invalid public probe configuration')
    steps=round(config.horizon/config.dt)
    if steps<1 or abs(steps*config.dt-config.horizon)>1e-6: raise ValueError('horizon must be divisible by dt')
    lead=obs.shape[:-1]; o=obs.reshape(-1,162); batch=o.shape[0]
    if mode=='zero': return obs.new_zeros((*lead,19,11))
    world=obs.new_tensor(config.world); scale=float(max(config.world))
    p=o[:,9:12]*world; vmax=o[:,18:19]*20.; amax=o[:,19:20]*10.
    if torch.any(vmax<=0) or torch.any(amax<=0): raise ValueError('observation must encode positive vmax and amax')
    v=o[:,3:6]*vmax
    peers=o[:,20:55].reshape(batch,5,7); boxes=o[:,55:111].reshape(batch,8,7); moving=o[:,111:143].reshape(batch,4,8)
    pm=peers[...,6]>.5; bm=boxes[...,6]>.5; mm=moving[...,7]>.5
    # Clean invalid slots before arithmetic: padded NaNs must be irrelevant.
    peers=torch.where(pm[...,None],peers,torch.zeros_like(peers))
    boxes=torch.where(bm[...,None],boxes,torch.zeros_like(boxes))
    moving=torch.where(mm[...,None],moving,torch.zeros_like(moving))
    pp=p[:,None,:]+peers[...,:3]*scale; pv=v[:,None,:]+peers[...,3:6]*vmax[:,None,:]
    bc=p[:,None,:]+boxes[...,:3]*scale; bh=boxes[...,3:6]*scale
    mp=p[:,None,:]+moving[...,:3]*scale; mv=v[:,None,:]+moving[...,3:6]*vmax[:,None,:]
    mr=moving[...,6]*10.
    dirs=obs.new_tensor(DIRECTIONS).unsqueeze(0).expand(batch,-1,-1)
    pos=p[:,None,:].expand(-1,19,-1).clone(); vel=v[:,None,:].expand(-1,19,-1).clone()
    inflate=config.body_radius+config.safety_margin
    actual_acc=amax*(config.mismatch_acceleration_factor if mode=='mismatch' else 1.)
    # Sequential bounded dynamics (state-dependent); clearances vectorized over time.
    traj=[]
    for step in range(steps+1):
        traj.append(pos)
        if step<steps:
            vel=vel+actual_acc[:,None,:]*dirs*config.dt
            speed=torch.linalg.vector_norm(vel,dim=-1,keepdim=True)
            vel=vel*torch.minimum(torch.ones_like(speed),vmax[:,None,:]/speed.clamp_min(1e-12))
            pos=pos+config.dt*vel
    pos_t=torch.stack(traj,0)  # (T+1,batch,19,3)
    T=pos_t.shape[0]
    times=pos_t.new_tensor([s*config.dt for s in range(T)]).view(T,1,1,1,1)
    q=(pos_t[:,:,:,None,:]-bc[None,:,None,:,:]).abs()-(bh[None,:,None,:,:]+inflate)
    box_sdf=torch.linalg.vector_norm(q.clamp_min(0),dim=-1)+q.amax(-1).clamp_max(0)
    static=box_sdf.masked_fill(~bm[None,:,None,:],config.clearance_scale).amin(-1)
    mc=torch.linalg.vector_norm(pos_t[:,:,:,None,:]-(mp[None,:,None,:,:]+mv[None,:,None,:,:]*times),dim=-1)-mr[None,:,None,:]-inflate
    mov=mc.masked_fill(~mm[None,:,None,:],config.clearance_scale).amin(-1)
    pc=torch.linalg.vector_norm(pos_t[:,:,:,None,:]-(pp[None,:,None,:,:]+pv[None,:,None,:,:]*times),dim=-1)-2*config.body_radius-config.safety_margin
    peer=pc.masked_fill(~pm[None,:,None,:],config.clearance_scale).amin(-1)
    boundary=torch.minimum(pos_t-inflate,world-inflate-pos_t).amin(-1)
    if mode=='static':
        mov=torch.full_like(mov,config.clearance_scale); peer=torch.full_like(peer,config.clearance_scale)
    c=torch.stack((static,mov,peer,boundary),-1)  # (T+1,batch,19,4)
    mins=c.amin(0)
    current_clearance=c[0].amin(-1)
    clearance_t=c.amin(-1)
    hit=clearance_t<=0
    idx=torch.arange(T,device=obs.device).view(T,1,1).expand_as(hit)
    first=torch.where(hit,idx,torch.full_like(idx,T)).amin(0)
    ttc=torch.where(first<T,first.to(obs.dtype)*config.dt,obs.new_full((batch,19),config.horizon))
    brake=(v.square().sum(-1)/(2*amax[:,0])).unsqueeze(-1).expand(-1,19)
    clearance=mins.amin(-1); reserve=current_clearance-brake
    features=torch.cat((dirs,(mins/config.clearance_scale).clamp(-1,1),
                        (ttc/config.horizon).unsqueeze(-1),
                        (brake/config.clearance_scale).clamp(0,1).unsqueeze(-1),
                        (reserve/config.clearance_scale).clamp(-1,1).unsqueeze(-1),
                        (clearance>0).to(obs.dtype).unsqueeze(-1)),-1)
    if mode=='static':
        features=torch.cat((features[...,:4],torch.zeros_like(features[...,4:6]),features[...,6:]),-1)
    if mode=='shuffle':
        order=torch.stack([torch.randperm(19,generator=generator,device='cpu') for _ in range(batch)]).to(obs.device)
        values=torch.gather(features[...,3:],1,order[...,None].expand(-1,-1,8))
        features=torch.cat((dirs,values),-1)
    return features.reshape(*lead,19,11)
