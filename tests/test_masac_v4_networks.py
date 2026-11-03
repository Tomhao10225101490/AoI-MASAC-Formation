import copy
import pytest
import torch
from uav_planning.v4_networks import Actor, TeamQ, TwinCritics, adapt_observation
from uav_planning.v4_prediction import ProbeConfig, predict_probes


def sample(batch=2, agents=3):
    x=torch.zeros(batch,agents,162)
    x[...,9:12]=torch.tensor([.5,.5,.5]); x[...,18]=.5; x[...,19]=.4
    x[...,20:27]=torch.tensor([.1,.05,0., 0.,0.,0., 1.])
    x[...,55:62]=torch.tensor([.2,0.,0.,.05,.05,.05,1.])
    x[...,111:119]=torch.tensor([-.2,0.,0., .1,0.,0., .2,1.])
    return x


@pytest.mark.parametrize('mode',['attention','pool'])
def test_objects_masks_and_variable_count(mode):
    torch.manual_seed(1); actor=Actor(mode).eval(); obs=sample(); ego,tokens,mask=adapt_observation(obs)
    a,_=actor(obs,deterministic=True)
    order=torch.randperm(17); b,_=actor(ego=ego,tokens=tokens[...,order,:],mask=mask[...,order],deterministic=True)
    assert torch.allclose(a,b,atol=2e-6)
    dirty=tokens.clone(); dirty[~mask]=float('nan')
    c,_=actor(ego=ego,tokens=dirty,mask=mask,deterministic=True)
    assert torch.allclose(a,c,atol=1e-6)
    no_tokens=tokens[...,:0,:]; no_mask=mask[...,:0]
    z,_=actor(ego=ego,tokens=no_tokens,mask=no_mask,deterministic=True)
    padded,_=actor(ego=ego,tokens=dirty,mask=torch.zeros_like(mask),deterministic=True)
    assert torch.allclose(z,padded,atol=1e-6)
    small,_=actor(ego=ego,tokens=tokens[...,:3,:],mask=mask[...,:3],deterministic=True)
    assert torch.isfinite(small).all()


@pytest.mark.parametrize('mode',['attention','pool'])
def test_twin_q_agents_gradients_and_matching(mode):
    torch.manual_seed(2); actor=Actor(mode); qs=TwinCritics(mode); obs=sample(); risk=predict_probes(obs)
    actions,lp=actor(obs,risk=risk); q1,q2=qs(obs,actions,risk=risk)
    permutation=torch.tensor([2,0,1]); x=obs[:,permutation]; a=actions[:,permutation]; r=risk[:,permutation]
    assert torch.allclose(q1,qs(x,a,risk=r)[0],atol=2e-6)
    (q1.mean()+q2.mean()+lp.mean()).backward()
    assert actor.mu.weight.grad is not None and torch.isfinite(actor.mu.weight.grad).all()
    assert set(map(id,qs.q1.parameters())).isdisjoint(map(id,qs.q2.parameters()))
    assert set(map(id,qs.q1.parameters())).isdisjoint(map(id,qs.targets().q1.parameters()))


def test_parameter_matched_architectures():
    count=lambda m:sum(p.numel() for p in m.parameters())
    assert count(Actor('pool'))==count(Actor('attention'))
    assert count(TwinCritics('pool'))==count(TwinCritics('attention'))


def test_predictor_scope_modes_and_sensitivity():
    obs=sample(); config=ProbeConfig(); full=predict_probes(obs,config)
    # Reserved features and padded values are not available state and cannot affect probes.
    altered=obs.clone(); altered[...,143:]=torch.randn_like(altered[...,143:])*99
    altered[...,27:55]=float('nan'); altered[...,27:55].reshape(2,3,4,7)[...,6]=0
    assert torch.allclose(full,predict_probes(altered,config))
    assert torch.equal(predict_probes(obs,mode='zero'),torch.zeros_like(full))
    static=predict_probes(obs,mode='static'); assert torch.all(static[...,4:6]==0)
    shuffled=predict_probes(obs,mode='shuffle',generator=torch.Generator().manual_seed(3))
    assert torch.equal(shuffled[...,:3],full[...,:3]); assert not torch.equal(shuffled[...,3:],full[...,3:])
    assert torch.allclose(shuffled[...,3:].sort(dim=-2).values,full[...,3:].sort(dim=-2).values)
    changed=obs.clone(); changed[...,111] += .15
    assert not torch.allclose(full,predict_probes(changed))
    moving=obs.clone(); moving[...,3]=.6
    assert not torch.allclose(predict_probes(moving),predict_probes(moving,mode='mismatch'))
    assert torch.isfinite(full).all()


def test_probe_public_world_geometry_and_initial_collision():
    obs=sample(1,1); obs[...,55:111]=0; obs[...,111:143]=0; obs[...,20:55]=0
    obs[...,9:12]=torch.tensor([.001,.5,.5])
    risk=predict_probes(obs,ProbeConfig(world=(200.,100.,50.),body_radius=1.))
    assert torch.all(risk[...,6]<0) and torch.all(risk[...,7]==0)
