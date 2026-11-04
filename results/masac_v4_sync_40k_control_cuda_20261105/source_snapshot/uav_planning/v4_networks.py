"""V4 set encoders and centralized twin critics; no environment access.

Objects are 13-vectors [type one-hot, relative position, relative velocity,
box half extents, radius]. Lengths use max(world), except radius uses /10,
matching the 162-vector adapter. Masks are True for valid objects. Actor
outputs normalized tanh actions; log probability is in tanh action space,
before the environment's radial acceleration clipping.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from uav_planning.shared_masac import Actor as LegacyActor, TeamQ as LegacyTeamQ


def dense(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o), nn.ReLU())


def adapt_observation(obs):
    """Return ego, typed variable-set-compatible tokens, and valid mask."""
    if obs.shape[-1] != 162:
        raise ValueError('expected observation dimension 162')
    lead = obs.shape[:-1]
    p = obs[..., 20:55].reshape(*lead, 5, 7)
    b = obs[..., 55:111].reshape(*lead, 8, 7)
    m = obs[..., 111:143].reshape(*lead, 4, 8)
    def make(x, typ, pos, vel, extent, radius):
        one = x.new_zeros((*x.shape[:-1], 3)); one[..., typ] = 1
        return torch.cat((one, pos, vel, extent, radius), -1)
    def zeros(x, n): return x.new_zeros((*x.shape[:-1], n))
    tokens = torch.cat((make(p, 0, p[..., :3], p[..., 3:6], zeros(p,3), zeros(p,1)),
                        make(b, 1, b[..., :3], zeros(b,3), b[...,3:6], zeros(b,1)),
                        make(m, 2, m[..., :3], m[...,3:6], zeros(m,3), m[...,6:7])), -2)
    mask = torch.cat((p[..., 6], b[..., 6], m[..., 7]), -1) > .5
    return obs[..., :20], tokens, mask


def masked_pool(x, mask):
    if x.shape[-2] == 0:
        return torch.cat((x.new_zeros((*x.shape[:-2], x.shape[-1])),)*2, -1)
    safe = torch.where(mask[..., None], x, torch.zeros_like(x))
    mean = safe.sum(-2) / mask.sum(-1, keepdim=True).clamp_min(1)
    maximum = x.masked_fill(~mask[..., None], -torch.inf).max(-2).values
    maximum = torch.where(mask.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), -1)


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(128); self.n2 = nn.LayerNorm(128)
        self.attn = nn.MultiheadAttention(128, 4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256,128))
    def forward(self, x, valid):
        y = self.n1(x)
        x = x + self.attn(y, y, y, key_padding_mask=~valid, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class PoolBlock(nn.Module):
    """Token-wise residual control, exactly parameter matched per attention block."""
    def __init__(self):
        super().__init__()
        self.n1=nn.LayerNorm(128); self.n2=nn.LayerNorm(128); self.n3=nn.LayerNorm(128)
        self.fc1=nn.Linear(128,512); self.fc2=nn.Linear(512,128)
    def forward(self,x,valid):
        return self.n3(x + self.n2(self.fc2(F.relu(self.fc1(self.n1(x))))))


class SetEncoder(nn.Module):
    def __init__(self, mode='attention'):
        super().__init__()
        if mode not in ('attention','pool'): raise ValueError(mode)
        self.ego=dense(20,128,128); self.obj=dense(13,128,128)
        self.blocks=nn.ModuleList([AttentionBlock() if mode == 'attention' else PoolBlock() for _ in range(2)])
    def forward(self, ego, tokens, mask):
        if tokens.shape[:-2] != ego.shape[:-1] or tokens.shape[-1] != 13 or mask.shape != tokens.shape[:-1]:
            raise ValueError('incompatible token or mask shape')
        lead=ego.shape[:-1]; count=tokens.shape[-2]
        mask=mask.bool()
        e=self.ego(ego).reshape(-1,1,128)
        clean=torch.where(mask[...,None],tokens,torch.zeros_like(tokens))
        t=self.obj(clean).reshape(e.shape[0],count,128)
        valid=mask.reshape(e.shape[0],count)
        x=torch.cat((e,t),1)
        valid_all=torch.cat((torch.ones((valid.shape[0],1),device=valid.device,dtype=torch.bool),valid),1)
        for block in self.blocks: x=block(x,valid_all)
        return torch.cat((x[:,0],masked_pool(x[:,1:],valid)),-1).reshape(*lead,384)


class RiskEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.net=dense(11,64,64)
    def forward(self, risk):
        if risk.shape[-2:] != (19,11): raise ValueError('risk must have shape (...,19,11)')
        h=self.net(risk)
        return torch.cat((h.mean(-2),h.max(-2).values),-1)


class Features(nn.Module):
    def __init__(self, mode='attention'):
        super().__init__(); self.set=SetEncoder(mode); self.risk=RiskEncoder()
    def forward(self, obs=None, *, ego=None, tokens=None, mask=None, risk=None):
        if obs is not None:
            if any(x is not None for x in (ego,tokens,mask)): raise ValueError('choose obs or tokens')
            ego,tokens,mask=adapt_observation(obs)
        if ego is None or tokens is None or mask is None: raise ValueError('missing inputs')
        if risk is None: risk=ego.new_zeros((*ego.shape[:-1],19,11))
        if risk.shape[:-2] != ego.shape[:-1]: raise ValueError('risk batch mismatch')
        return torch.cat((self.set(ego,tokens,mask),self.risk(risk)),-1)


class Actor(nn.Module):
    def __init__(self,mode='attention'):
        super().__init__(); self.features=Features(mode); self.body=dense(512,256,256)
        self.mu=nn.Linear(256,3); self.logstd=nn.Linear(256,3)
        nn.init.uniform_(self.mu.weight,-.001,.001); nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logstd.weight); nn.init.constant_(self.logstd.bias,-1.)
    def forward(self,obs=None,deterministic=False,*,ego=None,tokens=None,mask=None,risk=None):
        h=self.body(self.features(obs,ego=ego,tokens=tokens,mask=mask,risk=risk))
        mu=self.mu(h); ls=self.logstd(h).clamp(-5,1)
        dist=torch.distributions.Normal(mu,ls.exp()); z=mu if deterministic else dist.rsample()
        action=z.tanh(); logp=(dist.log_prob(z)-2*(math.log(2)-z-F.softplus(-2*z))).sum(-1)
        return action,logp


class TeamQ(nn.Module):
    def __init__(self,mode='attention'):
        super().__init__(); self.features=Features(mode); self.agent=dense(515,256,256)
        self.output=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1))
    def forward(self,obs,action,*,ego=None,tokens=None,mask=None,risk=None):
        h=self.agent(torch.cat((self.features(obs,ego=ego,tokens=tokens,mask=mask,risk=risk),action),-1))
        return self.output(torch.cat((h.mean(-2),h.max(-2).values),-1)).squeeze(-1)


class TwinCritics(nn.Module):
    def __init__(self, mode='attention'):
        super().__init__(); self.q1=TeamQ(mode); self.q2=TeamQ(mode)
    def forward(self,obs,action,*,ego=None,tokens=None,mask=None,risk=None):
        kwargs=dict(ego=ego,tokens=tokens,mask=mask,risk=risk)
        return self.q1(obs,action,**kwargs),self.q2(obs,action,**kwargs)
    def targets(self):
        targets=copy.deepcopy(self); targets.requires_grad_(False); return targets
