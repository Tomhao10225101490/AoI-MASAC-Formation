"""Two-batch, V3 Standard SAC critic sampling diagnostic.

Both arms use the same separate uniform Actor/temperature batch.  The only
controlled difference is how the Critic batch is drawn and weighted.  This is
an exploratory optimizer diagnostic, not the V4 maneuverability mechanism.
"""
import copy
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F


class ReplayStrata:
    """Maintain exact done/non-done member order across FIFO writes and restores."""

    def __init__(self, model):
        self.model = model
        if not 0 <= model.size <= model.capacity:
            raise ValueError('Invalid Replay size')
        if model.size < model.capacity and model.index != model.size:
            raise ValueError('Nonfull Replay must occupy the initial slots')
        done = model.done[:model.size]
        if not np.isin(done, (0., 1.)).all():
            raise ValueError('Replay done must contain only 0 or 1')
        self.members = [np.flatnonzero(done == i).tolist() for i in (0, 1)]
        self.position = np.full(model.capacity, -1, dtype=np.int64)
        for group in self.members:
            self.position[np.asarray(group, dtype=np.int64)] = np.arange(len(group), dtype=np.int64)

    def on_push(self, index, old_done, new_done):
        if new_done not in (0, 1) or old_done not in (None, 0, 1):
            raise ValueError('Invalid terminal label')
        if old_done == new_done:
            return
        if old_done is not None:
            group = self.members[old_done]
            at = int(self.position[index])
            if at < 0 or at >= len(group) or group[at] != index:
                raise ValueError('Stratum index disagrees with Replay')
            last = group.pop()
            if at < len(group):
                group[at] = last
                self.position[last] = at
        self.position[index] = len(self.members[new_done])
        self.members[new_done].append(index)

    def state_dict(self):
        return dict(members=[np.asarray(g, dtype=np.int64) for g in self.members],
                    position=self.position.copy())

    def load_state_dict(self, state):
        groups = [np.asarray(g, dtype=np.int64) for g in state['members']]
        position = np.asarray(state['position'], dtype=np.int64)
        if len(groups) != 2 or position.shape != (self.model.capacity,):
            raise ValueError('Invalid saved stratum dimensions')
        ids = np.concatenate(groups)
        if len(ids) != self.model.size or not np.array_equal(np.sort(ids), np.arange(self.model.size)):
            raise ValueError('Saved strata do not partition Replay')
        for label, group in enumerate(groups):
            if not np.all(self.model.done[group] == label) or not np.array_equal(position[group], np.arange(len(group))):
                raise ValueError('Saved stratum labels/order disagree with Replay')
        if np.any(position[self.model.size:] != -1):
            raise ValueError('Inactive Replay slots have stratum indices')
        self.members = [g.tolist() for g in groups]
        self.position = position.copy()


def sample_critic(strata, rng, arm, batch_size, done_quota):
    """Return indices and exact p_h/q_h weights for a mean-MSE objective."""
    n0, n1 = map(len, strata.members)
    size = n0 + n1
    if size == 0 or batch_size <= 0 or not 0 < done_quota < batch_size:
        raise ValueError('Invalid Replay or sampling configuration')
    if arm not in ('uniform', 'stratified'):
        raise ValueError('Unknown diagnostic arm')
    fallback = arm == 'stratified' and (n0 == 0 or n1 == 0 or size < batch_size)
    if arm == 'uniform' or fallback:
        ids = rng.integers(size, size=batch_size, dtype=np.int64)
        labels = strata.model.done[ids]
        return ids, np.ones(batch_size, dtype=np.float32), dict(
            population=n0+n1, population_done=n1, sample_size=batch_size,
            sampled_done=int(np.count_nonzero(labels)), fallback_uniform=fallback,
            population_done_fraction=n1/size, sampled_done_fraction=float(np.mean(labels)),
            nonterminal_weight=1., terminal_weight=1.)
    # If one layer is smaller than its desired quota, consume it without
    # replacement and fill from the other; size>=batch guarantees feasibility.
    count1 = min(done_quota, n1)
    count0 = min(batch_size-count1, n0)
    count1 = batch_size-count0
    if not (0 < count0 <= n0 and 0 < count1 <= n1):
        raise ValueError('Cannot allocate both strata')
    selected0 = rng.choice(n0, size=count0, replace=False)
    selected1 = rng.choice(n1, size=count1, replace=False)
    ids0 = np.fromiter((strata.members[0][i] for i in selected0), dtype=np.int64, count=count0)
    ids1 = np.fromiter((strata.members[1][i] for i in selected1), dtype=np.int64, count=count1)
    ids = np.concatenate([ids0, ids1])
    w0, w1 = (n0/size)/(count0/batch_size), (n1/size)/(count1/batch_size)
    weights = np.concatenate([np.full(count0, w0), np.full(count1, w1)]).astype(np.float32)
    return ids, weights, dict(population=size, population_done=n1, sample_size=batch_size,
        sampled_done=count1, fallback_uniform=False, population_done_fraction=n1/size,
        sampled_done_fraction=count1/batch_size, nonterminal_weight=w0, terminal_weight=w1)


@contextmanager
def generator_state(generator, device='cpu'):
    """Use an independent Torch stream without altering the rollout stream."""
    device = torch.device(device)
    if device.type == 'cpu':
        get_state, set_state = torch.random.get_rng_state, torch.random.set_rng_state
    elif device.type == 'mps':
        get_state, set_state = torch.mps.get_rng_state, torch.mps.set_rng_state
    else:
        raise ValueError('Only CPU and MPS diagnostic devices are supported')
    outer = get_state()
    set_state(generator.get_state())
    try:
        yield
    finally:
        generator.set_state(get_state())
        set_state(outer)


def move_model_to(model, device):
    """Move a restored learner and optimizer state without changing values."""
    device = torch.device(device)
    if device.type not in ('cpu', 'mps'):
        raise ValueError('Only CPU and MPS diagnostic devices are supported')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable')
    if next(model.actor.parameters()).device.type == device.type and model.logalpha.device.type == device.type:
        return model
    states = [copy.deepcopy(x.state_dict()) for x in (model.ao, model.qo, model.eo)]
    model.actor.to(device); model.q.to(device); model.target.to(device)
    model.logalpha = model.logalpha.detach().to(device).clone().requires_grad_(True)
    model.ao = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
    model.qo = torch.optim.Adam(model.q.parameters(), lr=3e-4)
    model.eo = torch.optim.Adam([model.logalpha], lr=3e-4)
    for optimizer, state in zip((model.ao, model.qo, model.eo), states):
        optimizer.load_state_dict(state)
    return model


def postclip_norm(parameters):
    grads = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(grads))) if grads else 0.


class CriticSamplingUpdater:
    def __init__(self, model, *, arm, seed, batch_size=128, done_quota=32, device='cpu'):
        if arm not in ('uniform', 'stratified'):
            raise ValueError('Unknown diagnostic arm')
        self.device = torch.device(device)
        self.model, self.arm, self.seed = move_model_to(model, self.device), arm, int(seed)
        self.batch_size, self.done_quota = int(batch_size), int(done_quota)
        base = 104729*self.seed
        self.critic_rng = np.random.default_rng(base+31001)
        self.actor_rng = np.random.default_rng(base+31002)
        self.target_torch_rng = torch.Generator(device=self.device).manual_seed(base+31003)
        self.actor_torch_rng = torch.Generator(device=self.device).manual_seed(base+31004)
        self.strata = ReplayStrata(model)

    def act(self, obs, deterministic=False):
        """Rollout policy uses the device's outer RNG stream."""
        with torch.no_grad():
            action, _ = self.model.actor(torch.as_tensor(obs, device=self.device).unsqueeze(0), deterministic)
        return action[0].cpu().numpy()

    def push(self, obs, action, reward, next_obs, done):
        model = self.model
        index, size = model.index, model.size
        old_done = int(model.done[index]) if index < size else None
        label = int(bool(done))
        model.push(obs, action, reward, next_obs, done)
        self.strata.on_push(index, old_done, label)

    def state_dict(self):
        return dict(version=1, arm=self.arm, seed=self.seed, device=self.device.type, batch_size=self.batch_size,
                    done_quota=self.done_quota, critic_rng=copy.deepcopy(self.critic_rng.bit_generator.state),
                    actor_rng=copy.deepcopy(self.actor_rng.bit_generator.state),
                    target_torch_rng=self.target_torch_rng.get_state().clone(),
                    actor_torch_rng=self.actor_torch_rng.get_state().clone(),
                    rollout_mps_rng=torch.mps.get_rng_state().clone() if self.device.type == 'mps' else None,
                    strata=self.strata.state_dict())

    def load_state_dict(self, state):
        if (state['version'] != 1 or state['arm'] != self.arm or state['seed'] != self.seed or
                state['device'] != self.device.type or
                state['batch_size'] != self.batch_size or state['done_quota'] != self.done_quota):
            raise ValueError('Diagnostic sampling state does not match configuration')
        self.strata.load_state_dict(state['strata'])
        self.critic_rng.bit_generator.state = copy.deepcopy(state['critic_rng'])
        self.actor_rng.bit_generator.state = copy.deepcopy(state['actor_rng'])
        self.target_torch_rng.set_state(state['target_torch_rng'])
        self.actor_torch_rng.set_state(state['actor_torch_rng'])
        if self.device.type == 'mps':
            torch.mps.set_rng_state(state['rollout_mps_rng'])

    def update(self):
        m, batch = self.model, self.batch_size
        start = time.perf_counter()
        ids, weights, sample = sample_critic(self.strata, self.critic_rng, self.arm, batch, self.done_quota)
        actor_ids = self.actor_rng.integers(m.size, size=batch, dtype=np.int64)
        o, a, r, no, d = [torch.from_numpy(x[ids]).to(self.device) for x in (m.obs, m.actions, m.rewards, m.next, m.done)]
        actor_o = torch.from_numpy(m.obs[actor_ids]).to(self.device)
        weight = torch.from_numpy(weights).to(self.device)
        sampling_s = time.perf_counter()-start

        qstart = time.perf_counter()
        with torch.no_grad(), generator_state(self.target_torch_rng, self.device):
            next_action, next_logp = m.actor(no)
            tq = torch.minimum(*[q(no, next_action) for q in m.target])
            target = r + .99*(1.-d)*(tq-m.logalpha.exp()*next_logp.mean(-1))
        q1, q2 = [q(o, a) for q in m.q]
        td1, td2 = target-q1, target-q2
        qloss = (weight*td1.square()).mean() + (weight*td2.square()).mean()
        m.qo.zero_grad()
        qloss.backward()
        q_pre = float(torch.nn.utils.clip_grad_norm_(m.q.parameters(), 5.))
        q_post = postclip_norm(m.q.parameters())
        m.qo.step()
        critic_s = time.perf_counter()-qstart

        astart = time.perf_counter()
        m.q.requires_grad_(False)
        try:
            with generator_state(self.actor_torch_rng, self.device):
                aa, logp = m.actor(actor_o)
            aloss = (m.logalpha.exp().detach()*logp.mean(-1) -
                     torch.minimum(*[q(actor_o, aa) for q in m.q])).mean()
            m.ao.zero_grad()
            aloss.backward()
            a_pre = float(torch.nn.utils.clip_grad_norm_(m.actor.parameters(), 5.))
            a_post = postclip_norm(m.actor.parameters())
            m.ao.step()
        finally:
            m.q.requires_grad_(True)
        eloss = -(m.logalpha*(logp.detach().mean(-1)-3.)).mean()
        m.eo.zero_grad()
        eloss.backward()
        m.eo.step()
        with torch.no_grad():
            for target_q, online_q in zip(m.target.parameters(), m.q.parameters()):
                target_q.lerp_(online_q, .005)
        actor_alpha_s = time.perf_counter()-astart
        result = dict(**sample, actor_batch_done_fraction=float(m.done[actor_ids].mean()),
            critic_loss=float(qloss.detach()), actor_loss=float(aloss.detach()),
            alpha=float(m.logalpha.exp().detach()), entropy=float(-logp.detach().mean()),
            target_mean=float(target.mean()), q1_mean=float(q1.detach().mean()), q2_mean=float(q2.detach().mean()),
            absolute_td_q1_mean=float(td1.detach().abs().mean()), absolute_td_q2_mean=float(td2.detach().abs().mean()),
            critic_grad_preclip=q_pre, critic_grad_postclip=q_post, critic_clipped=bool(q_pre>5.),
            actor_grad_preclip=a_pre, actor_grad_postclip=a_post, actor_clipped=bool(a_pre>5.),
            sampling_s=sampling_s, critic_s=critic_s, actor_alpha_s=actor_alpha_s,
            update_s=time.perf_counter()-start)
        if not all(np.isfinite(x) for x in result.values() if isinstance(x, (float, int))):
            raise ValueError('Nonfinite diagnostic update')
        return result
