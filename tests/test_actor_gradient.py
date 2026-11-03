"""The Q term must improve actions even with residual regularization."""
import pytest
import torch

from algos.masac import sac_actor_loss


@pytest.mark.parametrize("action_l2", [0.0, 0.4])
@pytest.mark.parametrize("slopes", [[1., 1., 1.], [1., 2., 4.]])
def test_actor_moves_toward_larger_q(action_l2, slopes):
    # At zero action the L2 gradient is zero. A positive dQ/da must move
    # the actor positive, even when the batch happens to have identical Q.
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    action = theta.expand(3, 1)
    q = action * torch.tensor(slopes, dtype=torch.float64).reshape(3, 1)
    loss = sac_actor_loss(q, torch.zeros_like(q), action, 0., action_l2)
    loss.backward()
    assert theta.grad.item() == pytest.approx(-sum(slopes) / 3)
    assert (theta - .01 * theta.grad).item() > 0.


def test_regularization_preserves_its_action_scale():
    action = torch.tensor([[.5]], dtype=torch.float64, requires_grad=True)
    loss = sac_actor_loss(action * 0., action * 0., action, 0., .4)
    loss.backward()
    assert action.grad.item() == pytest.approx(.4)
