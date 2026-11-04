"""Instrumented legacy SAC update: identical CPU algebra and RNG order to V3.

V4 scene qualification uses this learner before architecture selection. Extra
diagnostics reuse already computed tensors and do not draw extra actions.
"""
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F
from uav_planning.shared_masac import SharedMASAC


def grad_norm(parameters):
    return float(torch.linalg.vector_norm(torch.cat([
        p.grad.detach().reshape(-1) for p in parameters if p.grad is not None])))


class InstrumentedLegacyMASAC(SharedMASAC):
    def __init__(self, n=3, capacity=100000, seed=0, device='cpu'):
        super().__init__(n=n, capacity=capacity, seed=seed)
        self.device = torch.device(device)
        self.actor.to(self.device); self.q.to(self.device); self.target.to(self.device)
        self.logalpha = torch.tensor(math.log(.1), device=self.device, requires_grad=True)
        self.ao = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.qo = torch.optim.Adam(self.q.parameters(), lr=3e-4)
        self.eo = torch.optim.Adam([self.logalpha], lr=3e-4)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            action, _ = self.actor(torch.as_tensor(obs, device=self.device).unsqueeze(0), deterministic)
        return action[0].cpu().numpy()

    def update(self, batch=128):
        ids = self.rng.integers(self.size, size=batch)
        o, a, r, no, d = [torch.from_numpy(x[ids]).to(self.device) for x in
                          (self.obs, self.actions, self.rewards, self.next, self.done)]
        with torch.no_grad():
            na, nlp = self.actor(no)
            target_q = torch.minimum(*[q(no, na) for q in self.target])
            target = r + .99*(1-d)*(target_q-self.logalpha.exp()*nlp.mean(-1))
        qs = [q(o, a) for q in self.q]
        qloss = sum(F.mse_loss(q, target) for q in qs)
        self.qo.zero_grad(); qloss.backward()
        qgrad = torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.)
        qpost = grad_norm(self.q.parameters()); self.qo.step()
        self.q.requires_grad_(False)
        aa, lp = self.actor(o)
        policy_q = torch.minimum(*[q(o, aa) for q in self.q])
        aloss = (self.logalpha.exp().detach()*lp.mean(-1)-policy_q).mean()
        self.ao.zero_grad(); aloss.backward()
        agrad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.)
        apost = grad_norm(self.actor.parameters()); self.ao.step(); self.q.requires_grad_(True)
        eloss = -(self.logalpha*(lp.detach().mean(-1)-3.)).mean()
        self.eo.zero_grad(); eloss.backward(); self.eo.step()
        with torch.no_grad():
            for t, s in zip(self.target.parameters(), self.q.parameters()):
                t.lerp_(s, .005)
        return dict(actor_loss=float(aloss.detach()), critic_loss=float(qloss.detach()),
                    alpha=float(self.logalpha.exp().detach()), actor_grad=float(agrad),
                    critic_grad=float(qgrad), actor_grad_post=apost, critic_grad_post=qpost,
                    actor_clipped=int(float(agrad)>5), critic_clipped=int(float(qgrad)>5),
                    q_mean=float(torch.stack(qs).detach().mean()), target_mean=float(target.mean()),
                    td_abs=float(torch.stack([torch.abs(q.detach()-target) for q in qs]).mean()),
                    entropy=float(-lp.detach().mean()), terminal_sample_fraction=float(d.mean()))

    def checkpoint(self):
        cp = super().checkpoint()
        cp['device'] = str(self.device)
        if self.device.type == 'mps': cp['mps_rng'] = torch.mps.get_rng_state()
        return cp

    def load_state(self, cp):
        self.actor.load_state_dict(cp['actor']); self.q.load_state_dict(cp['critics'])
        self.target.load_state_dict(cp['targets'])
        with torch.no_grad(): self.logalpha.copy_(cp['logalpha'].to(self.device))
        for optimizer, key in ((self.ao,'actor_optimizer'),(self.qo,'critic_optimizer'),(self.eo,'alpha_optimizer')):
            optimizer.load_state_dict(cp[key])
            for state in optimizer.state.values():
                for k,v in state.items():
                    if torch.is_tensor(v): state[k] = v.to(self.device)
        self.size, self.index = cp['replay_size'], cp['replay_index']
        self.rng.bit_generator.state = copy.deepcopy(cp['replay_rng'])
        torch.set_rng_state(cp['torch_rng'].cpu())
        if self.device.type=='mps' and 'mps_rng' in cp: torch.mps.set_rng_state(cp['mps_rng'].cpu())
