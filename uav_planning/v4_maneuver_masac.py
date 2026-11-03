"""Predictive Maneuverability-Aware SharedMASAC (V4 pool + 19-probe risk).

Architecture selection is frozen to SetEncoder pool mode (not Attention): the
passed Standard L0 gate used legacy pooling; this candidate attaches the
documented RiskEncoder on the capacity-matched V4 pool backbone.

Probe modes (matched capacity / reward / budget):
  full    — innovation: CV directional probes (static/moving/peer/boundary)
  static  — dynamic/peer channels zeroed
  zero    — zero risk features (control)
  shuffle — direction-mismatched risk scalars (ablation)

Reward and executed actions are identical across probe modes. Probes are
observation-only finite-horizon estimates, not safety certificates.
Warm-start from Legacy SharedMASAC (Path-C L0) is architecture-incompatible;
callers must cold-start or document a partial load.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch.nn import functional as F

from uav_planning.v4_networks import Actor, TwinCritics
from uav_planning.v4_prediction import ProbeConfig, predict_probes


def grad_norm(parameters):
    grads = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(grads)))


class ManeuverMASAC:
    """Centralized twin-team-Q MASAC with predictive maneuverability features."""

    PROBE_MODES = ('full', 'static', 'zero', 'shuffle')

    def __init__(
        self,
        n=3,
        capacity=100000,
        seed=0,
        device='cpu',
        set_mode='pool',
        probe_mode='full',
        probe_config: ProbeConfig | None = None,
        lr=3e-4,
    ):
        if set_mode != 'pool':
            raise ValueError('maneuver suite freezes set_mode=pool (no Attention in this arm)')
        if probe_mode not in self.PROBE_MODES:
            raise ValueError(f'probe_mode must be one of {self.PROBE_MODES}')
        self.n = n
        self.capacity = capacity
        self.device = torch.device(device)
        self.set_mode = set_mode
        self.probe_mode = probe_mode
        self.probe_config = probe_config or ProbeConfig(world=(240.0, 240.0, 90.0))
        self.rng = np.random.default_rng(seed)
        self._probe_torch_gen = torch.Generator(device='cpu')
        self._probe_torch_gen.manual_seed(int(seed) + 17)
        self.size = 0
        self.index = 0
        self.actor = Actor(set_mode).to(self.device)
        self.q = TwinCritics(set_mode).to(self.device)
        self.target = self.q.targets().to(self.device)
        self.logalpha = torch.tensor(math.log(0.1), device=self.device, requires_grad=True)
        self.ao = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.qo = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.eo = torch.optim.Adam([self.logalpha], lr=lr)
        self.obs = np.empty((capacity, n, 162), np.float32)
        self.next = np.empty_like(self.obs)
        self.actions = np.empty((capacity, n, 3), np.float32)
        self.rewards = np.empty(capacity, np.float32)
        self.done = np.empty(capacity, np.float32)

    def risk(self, obs_t: torch.Tensor) -> torch.Tensor:
        """Build (...,19,11) probe features; shuffle uses a dedicated CPU generator."""
        if self.probe_mode == 'shuffle':
            return predict_probes(
                obs_t,
                self.probe_config,
                mode='shuffle',
                generator=self._probe_torch_gen,
            )
        return predict_probes(obs_t, self.probe_config, mode=self.probe_mode)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            risk = self.risk(o)
            action, _ = self.actor(o, deterministic=deterministic, risk=risk)
        return action[0].cpu().numpy()

    def push(self, o, a, r, no, done):
        k = self.index
        self.obs[k] = o
        self.actions[k] = a
        self.rewards[k] = r
        self.next[k] = no
        self.done[k] = done
        self.index = (k + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def update(self, batch=128):
        ids = self.rng.integers(self.size, size=batch)
        o, a, r, no, d = [
            torch.from_numpy(x[ids]).to(self.device)
            for x in (self.obs, self.actions, self.rewards, self.next, self.done)
        ]
        with torch.no_grad():
            risk_n = self.risk(no)
            na, nlp = self.actor(no, risk=risk_n)
            tq1, tq2 = self.target(no, na, risk=risk_n)
            target_q = torch.minimum(tq1, tq2)
            target = r + 0.99 * (1 - d) * (target_q - self.logalpha.exp() * nlp.mean(-1))
        risk_o = self.risk(o)
        q1, q2 = self.q(o, a, risk=risk_o)
        qloss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.qo.zero_grad()
        qloss.backward()
        qgrad = torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        qpost = grad_norm(self.q.parameters())
        self.qo.step()
        self.q.requires_grad_(False)
        aa, lp = self.actor(o, risk=risk_o)
        pq1, pq2 = self.q(o, aa, risk=risk_o)
        policy_q = torch.minimum(pq1, pq2)
        aloss = (self.logalpha.exp().detach() * lp.mean(-1) - policy_q).mean()
        self.ao.zero_grad()
        aloss.backward()
        agrad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        apost = grad_norm(self.actor.parameters())
        self.ao.step()
        self.q.requires_grad_(True)
        eloss = -(self.logalpha * (lp.detach().mean(-1) - 3.0)).mean()
        self.eo.zero_grad()
        eloss.backward()
        self.eo.step()
        with torch.no_grad():
            for t, s in zip(self.target.parameters(), self.q.parameters()):
                t.lerp_(s, 0.005)
        with torch.no_grad():
            risk_stats = {
                'probe_clearance_mean': float(risk_o[..., 3:7].amin(-1).mean()),
                'probe_ttc_mean': float(risk_o[..., 7].mean()),
                'probe_positive_preview_frac': float(risk_o[..., 10].mean()),
            }
        return dict(
            actor_loss=float(aloss.detach()),
            critic_loss=float(qloss.detach()),
            alpha=float(self.logalpha.exp().detach()),
            actor_grad=float(agrad),
            critic_grad=float(qgrad),
            actor_grad_post=apost,
            critic_grad_post=qpost,
            actor_clipped=int(float(agrad) > 5),
            critic_clipped=int(float(qgrad) > 5),
            q_mean=float(torch.stack((q1, q2)).detach().mean()),
            target_mean=float(target.mean()),
            td_abs=float(torch.stack((torch.abs(q1.detach() - target), torch.abs(q2.detach() - target))).mean()),
            entropy=float(-lp.detach().mean()),
            terminal_sample_fraction=float(d.mean()),
            **risk_stats,
        )

    def checkpoint(self):
        return dict(
            actor=self.actor.state_dict(),
            critics=self.q.state_dict(),
            targets=self.target.state_dict(),
            logalpha=self.logalpha.detach().cpu(),
            actor_optimizer=self.ao.state_dict(),
            critic_optimizer=self.qo.state_dict(),
            alpha_optimizer=self.eo.state_dict(),
            torch_rng=torch.get_rng_state(),
            replay_rng=self.rng.bit_generator.state,
            probe_torch_gen=self._probe_torch_gen.get_state(),
            replay_size=self.size,
            replay_index=self.index,
            set_mode=self.set_mode,
            probe_mode=self.probe_mode,
            probe_config=dict(
                world=list(self.probe_config.world),
                body_radius=self.probe_config.body_radius,
                safety_margin=self.probe_config.safety_margin,
                dt=self.probe_config.dt,
                horizon=self.probe_config.horizon,
                clearance_scale=self.probe_config.clearance_scale,
                mismatch_acceleration_factor=self.probe_config.mismatch_acceleration_factor,
            ),
            device=str(self.device),
        )

    def parameter_counts(self):
        count = lambda m: int(sum(p.numel() for p in m.parameters()))
        return {
            'actor': count(self.actor),
            'twin_critics': count(self.q),
            'risk_encoder': count(self.actor.features.risk),
            'set_encoder': count(self.actor.features.set),
            'total_trainable': count(self.actor) + count(self.q) + 1,
        }
