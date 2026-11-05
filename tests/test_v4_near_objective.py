"""Algebraic contracts for the near-objective replacement, without training."""
import numpy as np
import pytest

from uav_planning.v4_near_objective import near_potential, replace_near_reward


def test_potential_is_bounded_team_mean_and_translation_invariant():
    positions = np.array([[0., 0., 0.], [15., 0., 0.], [0., 0., 30.]])
    goals = np.zeros_like(positions)
    before = positions.copy()
    phi = near_potential(positions, goals)
    assert phi == pytest.approx(0.4 / 3)
    assert 0 <= phi <= 0.4
    shift = np.array([100., -40., 17.])
    assert near_potential(positions + shift, goals + shift) == pytest.approx(phi)
    assert np.array_equal(positions, before)
    assert near_potential(goals, goals) == pytest.approx(0.4)
    assert near_potential(positions, goals, coefficient=0.) == 0.


def test_original_mode_returns_reward_object_unchanged_even_on_terminal():
    reward = np.float64(0.12345678901234567)
    new, correction, detail = replace_near_reward(reward, .2, .3, True, True)
    assert new is reward
    assert new.tobytes() == reward.tobytes()
    assert correction == 0.
    assert detail['removed_near'] == 0.
    assert detail['potential_difference'] == 0.


@pytest.mark.parametrize('last_truncated', [False, True])
def test_true_terminal_telescopes_and_removes_physical_terminal_near(last_truncated):
    gamma = .99
    phi = np.array([.12, .28, .20, .31])
    reward_without_near = np.array([.7, -.2, 50.])
    legacy = reward_without_near + phi[1:]
    rewards = []
    for t, base in enumerate(legacy):
        terminal = t == len(legacy) - 1
        reward, _, detail = replace_near_reward(base, phi[t], phi[t + 1], terminal,
                                                terminal and last_truncated,
                                                gamma=gamma, mode='potential')
        rewards.append(reward)
    weights = gamma ** np.arange(len(legacy))
    # Whole-trajectory identity, including the actual last reward, rather than
    # a mirrored per-transition implementation of the tested formula.
    assert np.dot(weights, rewards) == pytest.approx(np.dot(weights, reward_without_near) - phi[0])
    assert detail['following_phi_observed'] == phi[-1]
    assert detail['removed_near'] == phi[-1]
    assert detail['following_phi_effective_for_potential'] == 0.
    assert detail['bootstrap_allowed_by_flags'] is False


def test_timeout_retains_boundary_and_transformed_bootstrap_cancels_it():
    gamma = .99
    phi = np.array([.12, .28, .20, .31])
    reward_without_near = np.array([.7, -.2, .15])
    rewards = []
    for t, base in enumerate(reward_without_near + phi[1:]):
        reward, _, detail = replace_near_reward(base, phi[t], phi[t + 1], False,
                                                t == len(reward_without_near) - 1,
                                                gamma=gamma, mode='potential')
        rewards.append(reward)
    weights = gamma ** np.arange(len(rewards))
    base_sum, shaped_sum = np.dot(weights, reward_without_near), np.dot(weights, rewards)
    boundary = gamma ** len(rewards)
    assert shaped_sum == pytest.approx(base_sum - phi[0] + boundary * phi[-1])
    unshaped_continuation_value = 2.7
    assert shaped_sum + boundary * (unshaped_continuation_value - phi[-1]) == pytest.approx(
        base_sum + boundary * unshaped_continuation_value - phi[0])
    assert detail['following_phi_effective_for_potential'] == phi[-1]
    assert detail['timeout_without_termination'] is True
    assert detail['bootstrap_allowed_by_flags'] is True


def test_stationary_near_occupancy_changes_from_positive_flow_to_cost():
    phi, living = .2, -.005
    legacy_reward = living + phi
    original, _, _ = replace_near_reward(legacy_reward, phi, phi, False, False)
    replacement, _, _ = replace_near_reward(legacy_reward, phi, phi, False, False, mode='potential')
    assert original > 0
    assert replacement < 0
    assert replacement == pytest.approx(living - .01 * phi)


@pytest.mark.parametrize('positions,goals,kwargs', [
    (np.zeros((0, 3)), np.zeros((0, 3)), {}),
    (np.zeros((2, 3)), np.zeros((3, 3)), {}),
    (np.zeros(3), np.zeros(3), {}),
    (np.full((1, 3), np.nan), np.zeros((1, 3)), {}),
    (np.zeros((1, 3), dtype=complex), np.zeros((1, 3)), {}),
    (np.zeros((1, 3)), np.zeros((1, 3)), {'coefficient': -1}),
    (np.zeros((1, 3)), np.zeros((1, 3)), {'radius_m': 0}),
])
def test_invalid_geometry_and_potential_parameters_are_rejected(positions, goals, kwargs):
    with pytest.raises(ValueError):
        near_potential(positions, goals, **kwargs)


@pytest.mark.parametrize('kwargs', [
    {'base_reward': np.inf}, {'previous_phi': -.1}, {'following_phi': np.nan},
    {'gamma': 1.}, {'gamma': -.1}, {'terminated': 1}, {'truncated': 0},
    {'mode': 'typo'}, {'base_reward': -1.7e308, 'previous_phi': 1.7e308,
                       'following_phi': 1.7e308, 'terminated': True},
])
def test_invalid_transition_data_and_nonfinite_results_are_rejected(kwargs):
    values = dict(base_reward=1., previous_phi=.1, following_phi=.2,
                  terminated=False, truncated=False, mode='potential')
    values.update(kwargs)
    with pytest.raises(ValueError):
        replace_near_reward(**values)
