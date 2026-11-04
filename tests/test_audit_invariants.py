"""Mathematical regression tests from the independent research audit.

These test target semantics and admissible covariance, not training returns.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Normal

from algos.buffer import Batch
from algos.common import soft_update
from algos.masac import MASAC
from algos.networks import SquashedGaussianActor
from envs.belief import LeaderBelief
from tests.test_relay import _open


def test_squashed_density_matches_change_of_variables():
    torch.manual_seed(12)
    actor = SquashedGaussianActor(4, 2, 8).double()
    obs = torch.randn(7, 4, dtype=torch.float64)
    mean, log_std = actor(obs)
    distribution = Normal(mean, log_std.exp())
    torch.manual_seed(42)
    latent = distribution.rsample()
    expected_action = latent.tanh()
    expected_logp = (distribution.log_prob(latent)
                     - torch.log1p(-expected_action.square())).sum(-1, keepdim=True)
    torch.manual_seed(42)
    action, logp = actor.sample(obs)
    torch.testing.assert_close(action, expected_action)
    torch.testing.assert_close(logp, expected_logp)
    action.sum().backward()
    assert actor.mean.weight.grad.abs().sum() > 0


def test_soft_target_is_exact_convex_interpolation():
    target, source = nn.Linear(2, 1), nn.Linear(2, 1)
    with torch.no_grad():
        for param in target.parameters():
            param.fill_(2.)
        for param in source.parameters():
            param.fill_(10.)
    soft_update(target, source, .125)
    for param in target.parameters():
        torch.testing.assert_close(param, torch.full_like(param, 3.))


def test_soft_q_target_bootstraps_only_nonterminal_samples(monkeypatch):
    algo = MASAC([3], 4, 2, 1,
                 {"batch_size": 2, "hidden": 8, "buffer_size": 2, "gamma": .9},
                 torch.device("cpu"))
    batch = Batch(obs=[torch.zeros(2, 3)], global_state=torch.zeros(2, 4),
                  actions=torch.zeros(2, 1, 2), rewards=torch.tensor([[3.], [3.]]),
                  next_obs=[torch.zeros(2, 3)], next_global=torch.zeros(2, 4),
                  terminated=torch.tensor([[1.], [0.]]))
    algo.buffer.size = 2
    monkeypatch.setattr(algo.buffer, "sample", lambda *args: batch)
    monkeypatch.setattr(algo, "_policy_actions",
                        lambda obs: ([torch.zeros(2, 2)], [torch.full((2, 1), -.5)]))

    class Target(nn.Module):
        def forward(self, state, actions):
            return torch.full((2, 1), 2.), torch.full((2, 1), 4.)

    algo.critic_targets = [Target()]
    import algos.masac as module
    original = module.F.mse_loss
    targets = []

    def recording_loss(prediction, target):
        targets.append(target.detach().clone())
        return original(prediction, target)

    monkeypatch.setattr(module.F, "mse_loss", recording_loss)
    result = algo.update()
    # alpha=1 initially; minQ - alpha*logp = 2.5.
    for actual in targets:
        torch.testing.assert_close(actual, torch.tensor([[3.], [5.25]]))
    assert len(targets) == 2
    assert all(math.isfinite(value) for value in result.values())


def test_long_outage_covariance_remains_positive_semidefinite():
    belief = LeaderBelief(1., .25, .01, .25, 1., 22500.)
    for _ in range(200):
        belief.drift(.2, 0., 0.)
        belief.add_process_noise(1., .01)
        assert np.linalg.eigvalsh(belief.P).min() >= -1e-8
        assert np.diag(belief.P).max() <= 22500. + 1e-8


def test_prior_collision_cannot_receive_success_bonus():
    env = _open(w_form_progress=0., w_formation=0., w_heading=0.,
                formation_bonus=0., w_smooth=0., w_step=0., w_clearance=0.,
                success_bonus=100.)
    env.reset(seed=0)
    env._wp_done = len(env.waypoints)
    env._hit_obstacle_ever = True
    rewards = env._rewards(
        actions=np.zeros((env.n_agents, env.action_dim)),
        errors=np.zeros(env.n_followers), progress=0., captured=0,
        hit_obstacle=False, hit_uav=False,
        oob=np.zeros(env.n_agents, dtype=bool))
    np.testing.assert_array_equal(rewards, np.zeros(env.n_agents))


def test_age_is_uncapped_and_observation_read_is_pure():
    env = _open(channel_mode="independent", p0=0., aoi_cap=4)
    env.reset(seed=11)
    for tick in range(10):
        env._step_count = tick
        env._update_channel()
    assert [link.aoi for link in env.links] == [10, 10]
    state_before = env.global_state().copy()
    obs1 = env._observations()
    obs2 = env._observations()
    for first, second in zip(obs1, obs2):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(state_before, env.global_state())


def test_same_seed_and_actions_reproduce_communication_and_motion():
    one = _open(channel_mode="distance", p0=.6, relay_mode="beacon")
    two = _open(channel_mode="distance", p0=.6, relay_mode="beacon")
    one.reset(seed=91)
    two.reset(seed=91)
    for _ in range(15):
        first = one.step(np.zeros((one.n_agents, one.action_dim)))
        second = two.step(np.zeros((two.n_agents, two.action_dim)))
        np.testing.assert_array_equal(one.global_state(), two.global_state())
        np.testing.assert_array_equal(first[1], second[1])
        assert first[2:4] == second[2:4]
        assert first[4]["communication"] == second[4]["communication"]
