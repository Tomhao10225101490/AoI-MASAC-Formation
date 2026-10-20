"""MADDPG and MATD3 with the same CTDE critic interface as MASAC.

Both algorithms use a deterministic actor per agent. The critic sees the
global state and every agent's action. The actor update fills in agent i's
action and keeps teammates' actions from the replay buffer.

MATD3 adds the usual three TD3 changes on top of MADDPG:
  * twin critics, target uses the minimum
  * delayed actor updates (``policy_delay``)
  * clipped Gaussian noise on the target actions (target policy smoothing)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from algos.buffer import ReplayBuffer
from algos.common import clone_module, obs_to_tensor, soft_update
from algos.networks import DeterministicActor, SingleQCritic, TwinQCritic


class _DeterministicCTDE:
    def __init__(
        self,
        obs_dims: list[int],
        global_dim: int,
        action_dim: int,
        n_agents: int,
        train_cfg: dict,
        device: torch.device,
        seed: int,
        twin: bool,
    ):
        self.obs_dims = list(obs_dims)
        self.global_dim = int(global_dim)
        self.action_dim = int(action_dim)
        self.n_agents = int(n_agents)
        self.device = device
        self.twin = twin
        self.gamma = float(train_cfg.get("gamma", 0.99))
        self.tau = float(train_cfg.get("tau", 0.005))
        self.batch_size = int(train_cfg.get("batch_size", 128))
        hidden = int(train_cfg.get("hidden", 128))
        lr_actor = float(train_cfg.get("lr_actor", 1e-4))
        lr_critic = float(train_cfg.get("lr_critic", 1e-3))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.expl_noise = float(train_cfg.get("expl_noise", 0.1))
        self.policy_delay = int(train_cfg.get("policy_delay", 2 if twin else 1))
        self.target_noise = float(train_cfg.get("target_noise", 0.2))
        self.noise_clip = float(train_cfg.get("noise_clip", 0.5))
        self._updates = 0
        self._rng = np.random.default_rng(seed + 3)
        critic_in = self.global_dim + self.n_agents * self.action_dim
        critic_cls = TwinQCritic if twin else SingleQCritic

        self.actors = [
            DeterministicActor(obs_dims[i], action_dim, hidden).to(device)
            for i in range(self.n_agents)
        ]
        self.actor_targets = [clone_module(a).to(device) for a in self.actors]
        for target in self.actor_targets:
            target.requires_grad_(False)
        self.critics = [critic_cls(critic_in, hidden).to(device) for _ in range(self.n_agents)]
        self.critic_targets = [clone_module(c).to(device) for c in self.critics]
        for target in self.critic_targets:
            target.requires_grad_(False)
        self.actor_opts = [
            torch.optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors
        ]
        self.critic_opts = [
            torch.optim.Adam(critic.parameters(), lr=lr_critic) for critic in self.critics
        ]
        self.buffer = ReplayBuffer(
            capacity=int(train_cfg.get("buffer_size", 50_000)),
            obs_dims=self.obs_dims,
            global_dim=self.global_dim,
            action_dim=self.action_dim,
            seed=seed + 19,
        )

    def reset_noise(self) -> None:
        return None

    @torch.no_grad()
    def act(self, obs_list: list[np.ndarray], explore: bool, episode: int = 0) -> np.ndarray:
        del episode
        actions = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
        for i, actor in enumerate(self.actors):
            obs = obs_to_tensor(obs_list[i], self.device)
            a = actor(obs).squeeze(0).cpu().numpy()
            if explore and self.expl_noise > 0.0:
                a = a + self._rng.standard_normal(self.action_dim) * self.expl_noise
            actions[i] = np.clip(a, -1.0, 1.0).astype(np.float32)
        return actions

    def push(self, obs, global_state, actions, rewards, next_obs, next_global, terminated) -> None:
        self.buffer.add(obs, global_state, actions, rewards, next_obs, next_global, terminated)

    def update(self) -> dict[str, float]:
        if self.buffer.size < self.batch_size:
            return {}
        batch = self.buffer.sample(self.batch_size, self.device)
        replay_actions = batch.actions.reshape(self.batch_size, -1)
        next_actions = self._target_actions(batch.next_obs)
        next_cat = torch.cat(next_actions, dim=-1)

        critic_losses = []
        q_values = []
        for i in range(self.n_agents):
            with torch.no_grad():
                q_t = self._target_q(i, batch.next_global, next_cat)
                y = batch.rewards[:, i : i + 1] + self.gamma * (1.0 - batch.terminated) * q_t
            pred = self.critics[i](batch.global_state, replay_actions)
            if self.twin:
                critic_loss = F.mse_loss(pred[0], y) + F.mse_loss(pred[1], y)
                q_log = pred[0]
            else:
                critic_loss = F.mse_loss(pred, y)
                q_log = pred
            self.critic_opts[i].zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.grad_clip)
            self.critic_opts[i].step()
            critic_losses.append(float(critic_loss.detach()))
            q_values.append(float(q_log.mean().detach()))

        self._updates += 1
        actor_losses = []
        # TD3 delays the actor and both target networks. MADDPG sets
        # policy_delay=1, so this branch runs on every update.
        do_actor = (self._updates % self.policy_delay) == 0
        if do_actor:
            for i in range(self.n_agents):
                new_action = self.actors[i](batch.obs[i])
                mixed = replay_actions.detach().clone()
                start = i * self.action_dim
                mixed[:, start : start + self.action_dim] = new_action
                q = self._actor_q(i, batch.global_state, mixed)
                actor_loss = -q.mean()
                self.actor_opts[i].zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.grad_clip)
                self.actor_opts[i].step()
                actor_losses.append(float(actor_loss.detach()))
                soft_update(self.actor_targets[i], self.actors[i], self.tau)
                soft_update(self.critic_targets[i], self.critics[i], self.tau)

        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "alpha": 0.0,
            "q": float(np.mean(q_values)),
        }

    def _target_actions(self, next_obs: list[torch.Tensor]) -> list[torch.Tensor]:
        actions = []
        for i, actor in enumerate(self.actor_targets):
            action = actor(next_obs[i])
            if self.twin and self.target_noise > 0.0:
                noise = torch.randn_like(action) * self.target_noise
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                action = (action + noise).clamp(-1.0, 1.0)
            actions.append(action)
        return actions

    def _target_q(self, agent: int, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pred = self.critic_targets[agent](state, actions)
        if self.twin:
            return torch.min(pred[0], pred[1])
        return pred

    def _actor_q(self, agent: int, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pred = self.critics[agent](state, actions)
        if self.twin:
            return pred[0]
        return pred

    def state_dict(self) -> dict:
        return {
            "actors": [m.state_dict() for m in self.actors],
            "actor_targets": [m.state_dict() for m in self.actor_targets],
            "critics": [m.state_dict() for m in self.critics],
            "critic_targets": [m.state_dict() for m in self.critic_targets],
            "actor_opts": [o.state_dict() for o in self.actor_opts],
            "critic_opts": [o.state_dict() for o in self.critic_opts],
            "updates": self._updates,
            "twin": self.twin,
        }

    def load_state_dict(self, payload: dict) -> None:
        if bool(payload.get("twin", self.twin)) != self.twin:
            raise ValueError("checkpoint twin-critic flag does not match this algorithm")
        for module, state in zip(self.actors, payload["actors"]):
            module.load_state_dict(state)
        for module, state in zip(self.actor_targets, payload["actor_targets"]):
            module.load_state_dict(state)
        for module, state in zip(self.critics, payload["critics"]):
            module.load_state_dict(state)
        for module, state in zip(self.critic_targets, payload["critic_targets"]):
            module.load_state_dict(state)
        for opt, state in zip(self.actor_opts, payload["actor_opts"]):
            opt.load_state_dict(state)
        for opt, state in zip(self.critic_opts, payload["critic_opts"]):
            opt.load_state_dict(state)
        self._updates = int(payload.get("updates", 0))


class MADDPG(_DeterministicCTDE):
    def __init__(self, obs_dims, global_dim, action_dim, n_agents, train_cfg, device, seed: int = 0):
        super().__init__(
            obs_dims, global_dim, action_dim, n_agents, train_cfg, device, seed, twin=False
        )
        self.algo_name = "maddpg"
        self.policy_delay = 1


class MATD3(_DeterministicCTDE):
    def __init__(self, obs_dims, global_dim, action_dim, n_agents, train_cfg, device, seed: int = 0):
        super().__init__(
            obs_dims, global_dim, action_dim, n_agents, train_cfg, device, seed, twin=True
        )
        self.algo_name = "matd3"
