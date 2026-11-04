import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_masac_v3_paired_td import choose_indices, group_gradient, sac_target


class TinyCritic(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(weight, dtype=torch.float32))

    def forward(self, observation, action):
        return self.w[0] * observation[:, 0] + self.w[1] * action[:, 0]


def test_target_uses_entropy_and_masks_only_done():
    reward = torch.tensor([3., -20.])
    done = torch.tensor([0., 1.])
    minimum_q = torch.tensor([4., 10.])
    log_prob = torch.tensor([-2., -3.])
    actual = sac_target(reward, done, minimum_q, torch.tensor(.1), log_prob)
    expected = torch.tensor([3. + .99 * (4. - .1 * -2.), -20.])
    torch.testing.assert_close(actual, expected)


def test_stratified_sampling_is_reproducible_without_replacement():
    done = np.array([0., 0., 0., 0., 1., 1.])
    a, counts = choose_indices(done, seed=1337, budget=3)
    b, _ = choose_indices(done, seed=1337, budget=3)
    assert counts == {0: 4, 1: 2}
    for k in (0, 1):
        assert np.array_equal(a[k], b[k])
        assert len(set(a[k])) == len(a[k])
        assert np.all(done[a[k]] == k)
    assert len(a[0]) == 3 and len(a[1]) == 2


def test_weighted_stratum_gradients_match_direct_weighted_mse():
    critics = nn.ModuleList([TinyCritic([.2, -.3]), TinyCritic([-.5, .4])])
    obs = torch.tensor([[1.], [2.], [3.], [4.]])
    actions = torch.tensor([[.5], [-.5], [.2], [-.3]])
    target = torch.tensor([2., 3., -1., 5.])
    g0 = group_gradient(critics, obs[:3], actions[:3], target[:3])['vector']
    g1 = group_gradient(critics, obs[3:], actions[3:], target[3:])['vector']
    weighted = .8*g0 + .2*g1
    critics.zero_grad(set_to_none=True)
    loss = sum(.8*F.mse_loss(q(obs[:3], actions[:3]), target[:3]) +
               .2*F.mse_loss(q(obs[3:], actions[3:]), target[3:]) for q in critics)
    loss.backward()
    direct = torch.cat([p.grad.reshape(-1) for p in critics.parameters()])
    torch.testing.assert_close(weighted, direct)
