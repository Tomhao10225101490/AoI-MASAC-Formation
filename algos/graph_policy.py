"""Shared, local-cache graph/GRU policy building blocks (no MAPPO learner).

No environment object or global state enters the actor. The future adapter
must fill neighbor features from timestamped, delivered messages only.
Forward evaluation is implemented and tested; rollout/GAE/PPO/MAC integration
is still a research milestone, not an implemented training algorithm.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def encoder(n_in: int, hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU())


class LocalGraphGRUActor(nn.Module):
    def __init__(self, self_dim=48, neighbor_dim=32, edge_dim=12,
                 hidden=128, n_actions=9):
        super().__init__()
        self.self_dim, self.neighbor_dim, self.edge_dim = self_dim, neighbor_dim, edge_dim
        self.hidden_dim, self.n_actions = hidden, n_actions
        self.self_encoder = encoder(self_dim, hidden)
        self.neighbor_encoder = encoder(neighbor_dim + edge_dim, hidden)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.memory = nn.GRUCell(2 * hidden, hidden)
        self.policy = nn.Linear(hidden, n_actions)

    def forward(self, self_features, neighbor_features, edge_features,
                neighbor_mask, hidden=None, *, agent_mask=None,
                reset_mask=None, action_mask=None):
        """Return logits and next memory for B environments, N local agents.

        Inputs: [B,N,48], [B,N,K,32], [B,N,K,12], bool [B,N,K].
        Optional memory [B,N,128]; agent/reset masks [B,N]; action mask
        [B,N,9]. Action 0 is always idle and must remain available. Padding
        cannot influence outputs, including when padded input contains NaN.
        """
        b, n, d = self_features.shape
        if d != self.self_dim:
            raise ValueError("wrong self feature dimension")
        k = neighbor_features.shape[-2]
        if neighbor_features.shape != (b, n, k, self.neighbor_dim):
            raise ValueError("wrong neighbor shape")
        if edge_features.shape != (b, n, k, self.edge_dim):
            raise ValueError("wrong edge shape")
        if neighbor_mask.shape != (b, n, k) or neighbor_mask.dtype != torch.bool:
            raise ValueError("neighbor_mask must be bool [B,N,K]")
        active = torch.ones((b, n), dtype=torch.bool, device=self_features.device)
        if agent_mask is not None:
            if agent_mask.shape != (b, n) or agent_mask.dtype != torch.bool:
                raise ValueError("agent_mask must be bool [B,N]")
            active = agent_mask
        if hidden is None:
            hidden = self_features.new_zeros(b, n, self.hidden_dim)
        if hidden.shape != (b, n, self.hidden_dim):
            raise ValueError("wrong hidden shape")
        reset = torch.zeros_like(active) if reset_mask is None else reset_mask
        if reset.shape != (b, n) or reset.dtype != torch.bool:
            raise ValueError("reset_mask must be bool [B,N]")
        hidden = torch.where((active & ~reset).unsqueeze(-1), hidden, 0.)
        own = self.self_encoder(torch.where(active.unsqueeze(-1), self_features, 0.))
        valid = neighbor_mask & active.unsqueeze(-1)
        features = torch.cat((neighbor_features, edge_features), dim=-1)
        messages = self.neighbor_encoder(torch.where(valid.unsqueeze(-1), features, 0.))
        if k:
            scores = (self.query(own).unsqueeze(-2) * self.key(messages)).sum(-1)
            scores = scores / math.sqrt(self.hidden_dim)
            scores = scores.masked_fill(~valid, -torch.inf)
            # Empty neighborhoods have zero context, not NaN softmax(-inf).
            scores = torch.where(valid.any(-1, keepdim=True), scores, 0.)
            weights = scores.softmax(-1).masked_fill(~valid, 0.)
            context = (weights.unsqueeze(-1) * self.value(messages)).sum(-2)
        else:
            context = torch.zeros_like(own)
        next_hidden = self.memory(torch.cat((own, context), -1).reshape(b*n, -1),
                                  hidden.reshape(b*n, -1)).reshape(b, n, -1)
        next_hidden = torch.where(active.unsqueeze(-1), next_hidden, 0.)
        logits = self.policy(next_hidden)
        available = torch.ones_like(logits, dtype=torch.bool)
        if action_mask is not None:
            if action_mask.shape != logits.shape or action_mask.dtype != torch.bool:
                raise ValueError("action_mask must be bool [B,N,n_actions]")
            if not action_mask[..., 0].all():
                raise ValueError("idle action 0 must remain available")
            available = action_mask.clone()
        available[..., 1:] &= active.unsqueeze(-1)
        logits = logits.masked_fill(~available, -torch.inf)
        return logits, next_hidden


class CentralSetCritic(nn.Module):
    """Training-only team value and cost value, invariant to agent order.

    Its explicit privileged feature width is supplied by the future adapter;
    no critic tensors are automatically passed to the decentralized actor.
    """
    def __init__(self, node_dim: int, context_dim: int = 0, hidden: int = 128):
        super().__init__()
        self.node_dim, self.context_dim = node_dim, context_dim
        self.nodes = encoder(node_dim, hidden)
        # Mean and max preserve scale and extremes; count carries team size.
        self.values = nn.Sequential(nn.Linear(2*hidden+context_dim+1, hidden),
                                    nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, nodes, agent_mask, context=None):
        b, n, d = nodes.shape
        if n == 0 or d != self.node_dim or agent_mask.shape != (b, n):
            raise ValueError("wrong critic node/mask shape")
        if agent_mask.dtype != torch.bool:
            raise ValueError("agent_mask must be boolean")
        x = self.nodes(torch.where(agent_mask.unsqueeze(-1), nodes, 0.))
        count = agent_mask.sum(1, keepdim=True)
        mean = x.masked_fill(~agent_mask.unsqueeze(-1), 0.).sum(1) / count.clamp_min(1)
        peak = x.masked_fill(~agent_mask.unsqueeze(-1), -torch.inf).amax(1)
        peak = torch.where(count > 0, peak, 0.)
        if context is None:
            context = nodes.new_zeros(b, self.context_dim)
        if context.shape != (b, self.context_dim):
            raise ValueError("wrong critic context shape")
        return self.values(torch.cat((mean, peak, count.to(nodes.dtype).log1p(), context), -1))
