import numpy as np
import torch
from uav_planning.structured import StructuredMission,segment_box_hit,DIRECTIONS
from uav_planning.shared_masac import Actor,TeamQ,SharedMASAC

def test_scene_geometry_and_local_observation():
    for layout,count in [('urban',25),('warehouse',20),('courtyard',24)]:
      for n in (3,4,6):
        e=StructuredMission(n=n,layout=layout);o=e.reset(111)
        assert o.shape==(n,162) and np.isfinite(o).all() and len(e.boxes)==count
        assert np.array_equal(o,e.reset(111))
        for i in range(len(e.boxes)):
            for j in range(i):assert not np.all(np.minimum(e.hi[i],e.hi[j])>np.maximum(e.lo[i],e.lo[j]))
        for _ in range(10):
            old=e.v.copy();e.step(np.ones((n,3)));assert np.linalg.norm(e.v-old,axis=-1).max()<=e.amax*e.dt+1e-8
            assert np.linalg.norm(e.v,axis=-1).max()<=e.vmax+1e-8

def test_thin_building_crossing_and_parallel():
    lo=np.array([[0,-1,-1]]);hi=np.array([[.01,1,1]])
    assert segment_box_hit(np.array([[-2.,0,0]]),np.array([[2.,0,0]]),lo,hi)[0,0]
    assert not segment_box_hit(np.array([[-2.,2,0]]),np.array([[2.,2,0]]),lo,hi)[0,0]

def test_actor_peer_permutation_and_team_q():
    torch.manual_seed(7);a=Actor();q=TeamQ();e=StructuredMission();o=torch.from_numpy(e.reset(25)[None]);p=o.clone()
    p[:,:,20:55]=p[:,:,20:55].reshape(1,3,5,7).flip(-2).reshape(1,3,35)
    assert torch.allclose(a(o,True)[0],a(p,True)[0],atol=1e-6)
    act=torch.zeros(1,3,3);assert torch.allclose(q(o,act),q(o.flip(1),act.flip(1)),atol=1e-6)
    for n in (4,6):assert a(torch.from_numpy(StructuredMission(n=n).reset(1)[None]),True)[0].shape==(1,n,3)

def test_equal_rewards_and_directional_probe():
    rewards=[]
    for mode in ('standard','distance','prediction','maneuver'):
        e=StructuredMission(mode=mode);o=e.reset(20);rewards.append(e.step(np.ones((3,3)))[1]);assert np.isfinite(o).all()
    assert max(rewards)==min(rewards)
    assert DIRECTIONS.shape==(19,3) and np.linalg.norm(DIRECTIONS,axis=-1).max()<=1+1e-9
    e=StructuredMission(mode='maneuver');e.reset(2);e.p[0]=[12,12,25];e.v[0]=[-8,0,0];e.observe()
    assert np.ptp(e.last_margins[0])>1 # directions differ near boundary

def test_update_changes_actor_and_critics():
    torch.set_num_threads(1);torch.manual_seed(4);m=SharedMASAC(capacity=128);e=StructuredMission();o=e.reset(5)
    for _ in range(64):
        a=np.zeros((3,3));no,r,t,tr,_=e.step(a);m.push(o,a,r,no,t);o=no
    old=[x.detach().clone() for x in m.actor.parameters()];loss=m.update(16)
    assert all(np.isfinite(v) for v in loss.values()) and loss['actor_grad']>0 and loss['critic_grad']>0
    assert any(not torch.equal(a,b) for a,b in zip(old,m.actor.parameters()))
