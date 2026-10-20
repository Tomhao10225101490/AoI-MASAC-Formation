"""Actors and critics used by every off-policy algorithm in this repo.

Actors are decentralized: they read one agent's local observation.
Critics are centralized: they read the privileged global state concatenated
with every agent's action.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class SquashedGaussianActor(nn.Module):
    """SAC actor. Actions are tanh-squashed into [-1, 1]."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        nn.init.uniform_(self.mean.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.mean.bias, -1e-3, 1e-3)
        nn.init.constant_(self.log_std.bias, -0.5)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        # log(1 - tanh(z)^2) = 2 * (log(2) - z - softplus(-2z)), summed over actions.
        log_det = 2.0 * (math.log(2.0) - z - F.softplus(-2.0 * z))
        log_prob = (dist.log_prob(z) - log_det).sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class DeterministicActor(nn.Module):
    """MADDPG / MATD3 actor. Output is already in [-1, 1]."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )
        last = self.net[-2]
        assert isinstance(last, nn.Linear)
        nn.init.uniform_(last.weight, -1e-3, 1e-3)
        nn.init.uniform_(last.bias, -1e-3, 1e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TwinQCritic(nn.Module):
    """Two Q heads. SAC and TD3 use the minimum of the two targets."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.q1 = _mlp(in_dim, hidden, 1)
        self.q2 = _mlp(in_dim, hidden, 1)

    def forward(
        self, state: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, actions], dim=-1)
        return self.q1(x), self.q2(x)


class SingleQCritic(nn.Module):
    """One Q head, used by MADDPG."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.q = _mlp(in_dim, hidden, 1)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, actions], dim=-1)
        return self.q(x)
