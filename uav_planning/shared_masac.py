"""Parameter-shared cooperative MASAC, with mean team entropy and twin team Q.
No action filter, expert controller, message passing or attention is used.
"""
import copy,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def mlp(inp,hidden,out):
    return nn.Sequential(nn.Linear(inp,hidden),nn.ReLU(),nn.Linear(hidden,out),nn.ReLU())

class SetEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.own=mlp(20,64,64);self.peers=mlp(6,32,32);self.buildings=mlp(6,32,32);self.moving=mlp(7,32,32);self.margin=mlp(19,32,32)
    def pool(self,x,module):
        mask=x[...,-1:];h=module(x[...,:-1]);mean=(h*mask).sum(-2)/mask.sum(-2).clamp(min=1)
        maximum=h.masked_fill(mask==0,-1e9).max(-2).values;maximum=torch.where(mask.sum(-2)>0,maximum,0.)
        return torch.cat([mean,maximum],-1)
    def forward(self,x):
        lead=x.shape[:-1]
        return torch.cat([self.own(x[...,:20]),self.pool(x[...,20:55].reshape(*lead,5,7),self.peers),self.pool(x[...,55:111].reshape(*lead,8,7),self.buildings),self.pool(x[...,111:143].reshape(*lead,4,8),self.moving),self.margin(x[...,143:])],-1)

class Actor(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=SetEncoder();self.body=mlp(288,128,128);self.mu=nn.Linear(128,3);self.logstd=nn.Linear(128,3)
        nn.init.uniform_(self.mu.weight,-.001,.001);nn.init.zeros_(self.mu.bias);nn.init.zeros_(self.logstd.weight);nn.init.constant_(self.logstd.bias,-1.)
    def forward(self,obs,deterministic=False):
        h=self.body(self.encoder(obs));mu=self.mu(h);ls=self.logstd(h).clamp(-5,1)
        dist=torch.distributions.Normal(mu,ls.exp());z=mu if deterministic else dist.rsample();a=z.tanh()
        lp=(dist.log_prob(z)-2*(math.log(2)-z-F.softplus(-2*z))).sum(-1)
        return a,lp

class TeamQ(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=SetEncoder();self.agent=mlp(291,128,128);self.output=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,obs,a):
        h=self.agent(torch.cat([self.encoder(obs),a],-1));return self.output(torch.cat([h.mean(-2),h.max(-2).values],-1)).squeeze(-1)

class SharedMASAC:
    def __init__(self,n=3,capacity=100000,seed=0):
        self.n=n;self.capacity=capacity;self.rng=np.random.default_rng(seed);self.size=0;self.index=0
        self.actor=Actor();self.q=nn.ModuleList([TeamQ(),TeamQ()]);self.target=copy.deepcopy(self.q);self.target.requires_grad_(False)
        self.logalpha=torch.tensor(math.log(.1),requires_grad=True);self.ao=torch.optim.Adam(self.actor.parameters(),lr=3e-4);self.qo=torch.optim.Adam(self.q.parameters(),lr=3e-4);self.eo=torch.optim.Adam([self.logalpha],lr=3e-4)
        self.obs=np.empty((capacity,n,162),np.float32);self.next=np.empty_like(self.obs);self.actions=np.empty((capacity,n,3),np.float32);self.rewards=np.empty(capacity,np.float32);self.done=np.empty(capacity,np.float32)
    def act(self,obs,deterministic=False):
        with torch.no_grad():a,_=self.actor(torch.as_tensor(obs).unsqueeze(0),deterministic)
        return a[0].numpy()
    def push(self,o,a,r,no,done):
        k=self.index;self.obs[k]=o;self.actions[k]=a;self.rewards[k]=r;self.next[k]=no;self.done[k]=done;self.index=(k+1)%self.capacity;self.size=min(self.capacity,self.size+1)
    def update(self,batch=128):
        ids=self.rng.integers(self.size,size=batch);o,a,r,no,d=[torch.from_numpy(x[ids]) for x in (self.obs,self.actions,self.rewards,self.next,self.done)]
        with torch.no_grad():
            na,nlp=self.actor(no);tq=torch.minimum(*[q(no,na) for q in self.target]);target=r+.99*(1-d)*(tq-self.logalpha.exp()*nlp.mean(-1))
        qloss=sum(F.mse_loss(q(o,a),target) for q in self.q);self.qo.zero_grad();qloss.backward();qgrad=nn.utils.clip_grad_norm_(self.q.parameters(),5.);self.qo.step()
        self.q.requires_grad_(False);aa,lp=self.actor(o);aloss=(self.logalpha.exp().detach()*lp.mean(-1)-torch.minimum(*[q(o,aa) for q in self.q])).mean()
        self.ao.zero_grad();aloss.backward();agrad=nn.utils.clip_grad_norm_(self.actor.parameters(),5.);self.ao.step();self.q.requires_grad_(True)
        eloss=-(self.logalpha*(lp.detach().mean(-1)-3.)).mean();self.eo.zero_grad();eloss.backward();self.eo.step()
        with torch.no_grad():
            for t,s in zip(self.target.parameters(),self.q.parameters()):t.lerp_(s,.005)
        return dict(actor_loss=float(aloss.detach()),critic_loss=float(qloss.detach()),alpha=float(self.logalpha.exp().detach()),actor_grad=float(agrad),critic_grad=float(qgrad))
    def checkpoint(self):
        return dict(actor=self.actor.state_dict(),critics=self.q.state_dict(),targets=self.target.state_dict(),logalpha=self.logalpha.detach(),actor_optimizer=self.ao.state_dict(),critic_optimizer=self.qo.state_dict(),alpha_optimizer=self.eo.state_dict(),torch_rng=torch.get_rng_state(),replay_rng=self.rng.bit_generator.state,replay_size=self.size,replay_index=self.index)
