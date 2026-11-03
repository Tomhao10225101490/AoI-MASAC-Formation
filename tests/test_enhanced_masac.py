"""Optional stabilization must preserve the default model and real gradients."""
import numpy as np
import pytest
import torch
from algos.masac import MASAC
from algos.networks import SquashedGaussianActor, TwinQCritic
from evaluate import resolve_test_observation


def test_distance_intercept_one_does_not_leak_true_leader():
    mode, age, note = resolve_test_observation('perfect', False, 'distance', 1., 'auto', None)
    assert mode == 'hold' and not age and note
    assert resolve_test_observation('perfect', False, 'perfect', 1., 'auto', None)[0] == 'perfect'
    assert resolve_test_observation('perfect', False, 'distance', 1., 'perfect', None)[0] == 'perfect'


def test_legacy_network_state_keys_are_preserved():
    actor = SquashedGaussianActor(4, 3, 16)
    assert list(actor.body.state_dict()) == ['0.weight', '0.bias', '2.weight', '2.bias']
    critic = TwinQCritic(10, 16)
    assert list(critic.q1.state_dict()) == ['0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias']


@pytest.mark.parametrize('layer_norm', [False, True])
def test_enhanced_update_changes_actor_and_roundtrips(layer_norm):
    torch.manual_seed(14)
    cfg = dict(hidden=16, batch_size=8, buffer_size=20, layer_norm=layer_norm)
    a = MASAC([4,4], 8, 3, 2, cfg, torch.device('cpu'))
    rng = np.random.default_rng(4)
    for _ in range(12):
        obs = [rng.normal(size=4).astype('float32') for _ in range(2)]
        state = np.concatenate(obs)
        a.push(obs, state, rng.uniform(-1,1,(2,3)), rng.normal(size=2), obs, state, False)
    before = [p.detach().clone() for p in a.actors[0].parameters()]
    stats = a.update()
    assert all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(p,q) for p,q in zip(before,a.actors[0].parameters()))
    b = MASAC([4,4], 8, 3, 2, cfg, torch.device('cpu'))
    b.load_state_dict(a.state_dict())
    np.testing.assert_array_equal(a.act(obs,False),b.act(obs,False))
