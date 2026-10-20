"""Shared helpers for the off-policy multi-agent algorithms."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak average: target <- (1 - tau) * target + tau * source."""
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def clone_module(module: nn.Module) -> nn.Module:
    cloned = copy.deepcopy(module)
    cloned.load_state_dict(module.state_dict())
    return cloned


class OUNoise:
    """Ornstein-Uhlenbeck noise, matching the undergrad MASAC setup.

    The original project added this on top of the action for the first 50
    episodes. SAC already explores through entropy; the noise is optional and
    controlled by ``ou_episodes``.
    """

    def __init__(self, dim: int, theta: float = 0.15, sigma: float = 0.2, seed: int = 0):
        self.dim = dim
        self.theta = theta
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)
        self.state = np.zeros(dim, dtype=np.float64)

    def reset(self) -> None:
        self.state[:] = 0.0

    def sample(self) -> np.ndarray:
        dx = self.theta * (0.0 - self.state) + self.sigma * self._rng.standard_normal(self.dim)
        self.state = self.state + dx
        return self.state.astype(np.float64)


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor
