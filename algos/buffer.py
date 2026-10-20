"""Fixed-size replay buffer for CTDE off-policy updates.

Observations are stored per agent because nothing in the trainer assumes the
agents share one tensor layout, even though this environment currently gives
them the same observation length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    obs: list[torch.Tensor]
    global_state: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: list[torch.Tensor]
    next_global: torch.Tensor
    terminated: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dims: list[int],
        global_dim: int,
        action_dim: int,
        seed: int,
    ):
        self.capacity = int(capacity)
        self.n_agents = len(obs_dims)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(seed)
        self.obs = [
            np.zeros((self.capacity, d), dtype=np.float32) for d in obs_dims
        ]
        self.next_obs = [
            np.zeros((self.capacity, d), dtype=np.float32) for d in obs_dims
        ]
        self.global_state = np.zeros((self.capacity, global_dim), dtype=np.float32)
        self.next_global = np.zeros((self.capacity, global_dim), dtype=np.float32)
        self.actions = np.zeros(
            (self.capacity, self.n_agents, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.capacity, self.n_agents), dtype=np.float32)
        self.terminated = np.zeros((self.capacity, 1), dtype=np.float32)
        self._idx = 0
        self.size = 0

    def add(
        self,
        obs: list[np.ndarray],
        global_state: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: list[np.ndarray],
        next_global: np.ndarray,
        terminated: bool,
    ) -> None:
        i = self._idx
        for a in range(self.n_agents):
            self.obs[a][i] = obs[a]
            self.next_obs[a][i] = next_obs[a]
        self.global_state[i] = global_state
        self.next_global[i] = next_global
        self.actions[i] = actions
        self.rewards[i] = rewards
        self.terminated[i, 0] = float(terminated)
        self._idx = (self._idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        if self.size < batch_size:
            raise ValueError(f"buffer size {self.size} < batch {batch_size}")
        idx = self._rng.integers(0, self.size, size=batch_size)
        def t(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(array, dtype=torch.float32, device=device)

        return Batch(
            obs=[t(self.obs[a][idx]) for a in range(self.n_agents)],
            global_state=t(self.global_state[idx]),
            actions=t(self.actions[idx]),
            rewards=t(self.rewards[idx]),
            next_obs=[t(self.next_obs[a][idx]) for a in range(self.n_agents)],
            next_global=t(self.next_global[idx]),
            terminated=t(self.terminated[idx]),
        )
