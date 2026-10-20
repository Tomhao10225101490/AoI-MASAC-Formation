"""Multi-agent Soft Actor-Critic with a centralized twin critic.

This one class trains four of the paper's methods. They share the update
and differ only in what the environment writes into the actor observation:

* ``belief_masac``      coasted leader state + AoI + slot std
* ``aoi_masac``         last leader measurement + AoI, no coast
* ``vanilla_masac``     trained with a perfect link (see configs)
* ``masac_hold``        stale leader state, AoI feature forced to 0
* ``masac_extrapolate`` constant-velocity coast, AoI and std forced to 0

CTDE:
  * actor i sees only agent i's observation
  * critic i sees the privileged global state and EVERY agent's action
  * the actor update replaces agent i's action with a fresh sample and
    leaves the other agents' actions at the replay-buffer values, so the
    gradient does not flow through teammates' actors

Each agent has its own actor, twin Q, and temperature alpha. Entropy is
adjusted automatically toward target entropy = -action_dim.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from algos.buffer import ReplayBuffer
from algos.common import OUNoise, clone_module, obs_to_tensor, soft_update
from algos.networks import SquashedGaussianActor, TwinQCritic


class MASAC:
    def __init__(
        self,
        obs_dims: list[int],
        global_dim: int,
        action_dim: int,
        n_agents: int,
        train_cfg: dict,
        device: torch.device,
        seed: int = 0,
    ):
        self.obs_dims = list(obs_dims)
        self.global_dim = int(global_dim)
        self.action_dim = int(action_dim)
        self.n_agents = int(n_agents)
        self.device = device
        self.gamma = float(train_cfg.get("gamma", 0.99))
        self.tau = float(train_cfg.get("tau", 0.005))
        self.batch_size = int(train_cfg.get("batch_size", 128))
        hidden = int(train_cfg.get("hidden", 128))
        lr_actor = float(train_cfg.get("lr_actor", 3e-4))
        lr_critic = float(train_cfg.get("lr_critic", 3e-4))
        lr_alpha = float(train_cfg.get("lr_alpha", 3e-4))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.ou_episodes = int(train_cfg.get("ou_episodes", 0))
        self.target_entropy = -float(self.action_dim)
        critic_in = self.global_dim + self.n_agents * self.action_dim

        self.actors = [
            SquashedGaussianActor(obs_dims[i], action_dim, hidden).to(device)
            for i in range(self.n_agents)
        ]
        self.critics = [TwinQCritic(critic_in, hidden).to(device) for _ in range(self.n_agents)]
        self.critic_targets = [clone_module(c).to(device) for c in self.critics]
        for target in self.critic_targets:
            target.requires_grad_(False)

        self.actor_opts = [
            torch.optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors
        ]
        self.critic_opts = [
            torch.optim.Adam(critic.parameters(), lr=lr_critic) for critic in self.critics
        ]
        self.log_alpha = [
            torch.zeros(1, requires_grad=True, device=device) for _ in range(self.n_agents)
        ]
        self.alpha_opts = [
            torch.optim.Adam([self.log_alpha[i]], lr=lr_alpha) for i in range(self.n_agents)
        ]
        self.buffer = ReplayBuffer(
            capacity=int(train_cfg.get("buffer_size", 50_000)),
            obs_dims=self.obs_dims,
            global_dim=self.global_dim,
            action_dim=self.action_dim,
            seed=seed + 17,
        )
        ou_sigma = float(train_cfg.get("ou_sigma", 0.3))
        self._ou = [
            OUNoise(action_dim, sigma=ou_sigma, seed=seed + 100 + i)
            for i in range(self.n_agents)
        ]
        self.algo_name = "masac"

    def reset_noise(self) -> None:
        for noise in self._ou:
            noise.reset()

    @torch.no_grad()
    def act(self, obs_list: list[np.ndarray], explore: bool, episode: int = 0) -> np.ndarray:
        actions = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
        for i, actor in enumerate(self.actors):
            obs = obs_to_tensor(obs_list[i], self.device)
            if explore:
                action, _ = actor.sample(obs)
            else:
                action = actor.deterministic(obs)
            a = action.squeeze(0).cpu().numpy()
            if explore and episode < self.ou_episodes:
                a = a + self._ou[i].sample()
            actions[i] = np.clip(a, -1.0, 1.0).astype(np.float32)
        return actions

    def push(
        self,
        obs: list[np.ndarray],
        global_state: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: list[np.ndarray],
        next_global: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.add(obs, global_state, actions, rewards, next_obs, next_global, terminated)

    def update(self) -> dict[str, float]:
        if self.buffer.size < self.batch_size:
            return {}
        batch = self.buffer.sample(self.batch_size, self.device)
        replay_actions = batch.actions.reshape(self.batch_size, -1)

        next_actions, next_logp = self._policy_actions(batch.next_obs)
        next_cat = torch.cat(next_actions, dim=-1)

        critic_losses = []
        actor_losses = []
        alphas = []
        q_values = []
        for i in range(self.n_agents):
            with torch.no_grad():
                q1_t, q2_t = self.critic_targets[i](batch.next_global, next_cat)
                alpha = self.log_alpha[i].exp()
                q_t = torch.min(q1_t, q2_t) - alpha * next_logp[i]
                y = batch.rewards[:, i : i + 1] + self.gamma * (1.0 - batch.terminated) * q_t

            q1, q2 = self.critics[i](batch.global_state, replay_actions)
            critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            self.critic_opts[i].zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.grad_clip)
            self.critic_opts[i].step()

            new_action, logp = self.actors[i].sample(batch.obs[i])
            mixed = replay_actions.detach().clone()
            start = i * self.action_dim
            mixed[:, start : start + self.action_dim] = new_action
            q1_pi, q2_pi = self.critics[i](batch.global_state, mixed)
            q_pi = torch.min(q1_pi, q2_pi)
            alpha = self.log_alpha[i].exp().detach()
            actor_loss = (alpha * logp - q_pi).mean()
            self.actor_opts[i].zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.grad_clip)
            self.actor_opts[i].step()

            alpha_loss = -(self.log_alpha[i] * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opts[i].zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opts[i].step()

            soft_update(self.critic_targets[i], self.critics[i], self.tau)
            critic_losses.append(float(critic_loss.detach()))
            actor_losses.append(float(actor_loss.detach()))
            alphas.append(float(self.log_alpha[i].exp().detach()))
            q_values.append(float(q_pi.mean().detach()))

        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(np.mean(actor_losses)),
            "alpha": float(np.mean(alphas)),
            "q": float(np.mean(q_values)),
        }

    def _policy_actions(
        self, obs_list: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        actions = []
        logps = []
        for i, actor in enumerate(self.actors):
            action, logp = actor.sample(obs_list[i])
            actions.append(action)
            logps.append(logp)
        return actions, logps

    def state_dict(self) -> dict:
        return {
            "actors": [m.state_dict() for m in self.actors],
            "critics": [m.state_dict() for m in self.critics],
            "critic_targets": [m.state_dict() for m in self.critic_targets],
            "log_alpha": [a.detach().cpu() for a in self.log_alpha],
            "actor_opts": [o.state_dict() for o in self.actor_opts],
            "critic_opts": [o.state_dict() for o in self.critic_opts],
            "alpha_opts": [o.state_dict() for o in self.alpha_opts],
        }

    def load_state_dict(self, payload: dict) -> None:
        for module, state in zip(self.actors, payload["actors"]):
            module.load_state_dict(state)
        for module, state in zip(self.critics, payload["critics"]):
            module.load_state_dict(state)
        for module, state in zip(self.critic_targets, payload["critic_targets"]):
            module.load_state_dict(state)
        for i, value in enumerate(payload["log_alpha"]):
            self.log_alpha[i].data.copy_(value.to(self.device))
        for opt, state in zip(self.actor_opts, payload["actor_opts"]):
            opt.load_state_dict(state)
        for opt, state in zip(self.critic_opts, payload["critic_opts"]):
            opt.load_state_dict(state)
        for opt, state in zip(self.alpha_opts, payload["alpha_opts"]):
            opt.load_state_dict(state)


def iter_modules(algo: MASAC) -> Iterable[torch.nn.Module]:
    return list(algo.actors) + list(algo.critics)
