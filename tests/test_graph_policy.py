import pytest
import torch
from algos.graph_policy import LocalGraphGRUActor, CentralSetCritic


def sample(n=4, k=3):
    return (torch.randn(2,n,48), torch.randn(2,n,k,32),
            torch.randn(2,n,k,12), torch.ones(2,n,k,dtype=torch.bool))


def test_local_actor_has_no_neighbor_order_dependence():
    torch.manual_seed(0)
    net = LocalGraphGRUActor()
    s, neighbors, edges, mask = sample()
    a,h = net(s,neighbors,edges,mask)
    order = [2,0,1]
    b,hb = net(s,neighbors[:,:,order],edges[:,:,order],mask[:,:,order])
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(h,hb)


@pytest.mark.parametrize('n',[4,8,16,32])
def test_shared_policy_supports_team_size_and_empty_neighbors(n):
    net=LocalGraphGRUActor()
    s,neighbors,edges,mask=sample(n)
    mask[:]=False
    neighbors[:]=float('nan'); edges[:]=float('nan')
    a,h=net(s,neighbors,edges,mask)
    assert a.shape==(2,n,9) and h.shape==(2,n,128)
    assert torch.isfinite(a).all() and torch.isfinite(h).all()
    a.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_masked_cache_cannot_change_decision_and_reset_clears_memory():
    net=LocalGraphGRUActor()
    s,neighbors,edges,mask=sample()
    mask[:,:,1:]=False
    a,h=net(s,neighbors,edges,mask)
    neighbors[:,:,1:]=1e8; edges[:,:,1:]=-1e8
    reset=torch.ones(2,4,dtype=torch.bool)
    b,hb=net(s,neighbors,edges,mask,hidden=torch.randn_like(h),reset_mask=reset)
    torch.testing.assert_close(a,b);torch.testing.assert_close(h,hb)
    active=~reset
    logits,hn=net(s,neighbors,edges,mask,agent_mask=active)
    assert torch.all(logits.softmax(-1)[...,0]==1.)
    assert torch.count_nonzero(hn)==0


def test_action_mask_and_critic_invariance():
    net=LocalGraphGRUActor(); args=sample()
    mask=torch.ones(2,4,9,dtype=torch.bool);mask[...,3]=False
    logits,_=net(*args,action_mask=mask)
    assert torch.all(logits.softmax(-1)[...,3]==0)
    mask[...,0]=False
    with pytest.raises(ValueError,match='idle'):net(*args,action_mask=mask)
    critic=CentralSetCritic(24,3)
    x=torch.randn(2,4,24);active=torch.tensor([[True,True,False,False]]*2)
    ctx=torch.randn(2,3);a=critic(x,active,ctx)
    x[:,2:]=float('nan')
    order=[2,0,3,1]
    b=critic(x[:,order],active[:,order],ctx)
    torch.testing.assert_close(a,b)
    assert torch.isfinite(critic(x,torch.zeros_like(active),ctx)).all()
