"""Check the discrete-error witnesses against the unchanged dynamics helper.

These numerical examples support, but do not replace, the exact-arithmetic
argument in docs/MATH_DISCRETE_PROBE_ERROR_20261105.md. They say nothing about
inter-sample collisions, observation masks or learned-policy performance.
"""
import numpy as np
import pytest

from uav_planning.structured import limited


def test_semi_implicit_peer_cv_witness_attains_8_8_metres():
    dt, acceleration, vmax, steps = 0.2, 4.0, 10.0, 10
    velocity = np.zeros((1, 3), dtype=np.float64)
    position = np.zeros_like(velocity)
    action = np.array([[1.0, 0.0, 0.0]])
    for k in range(1, steps + 1):
        velocity = limited(velocity, action, dt, acceleration, vmax)
        position += dt * velocity
        expected = acceleration * dt**2 * k * (k + 1) / 2
        assert np.linalg.norm(position) == pytest.approx(expected, abs=1e-12)
    assert np.linalg.norm(velocity) == pytest.approx(8.0, abs=1e-12)
    assert np.linalg.norm(velocity) < vmax
    assert np.linalg.norm(position) == pytest.approx(8.8, abs=1e-12)
    continuous_time_expression = 0.5 * acceleration * (steps * dt)**2
    assert continuous_time_expression == pytest.approx(8.0)
    assert np.linalg.norm(position) > continuous_time_expression


def test_same_direction_acceleration_mismatch_attains_4_4_metres():
    dt, vmax, steps = 0.2, 10.0, 10
    actual_velocity = np.zeros((1, 3), dtype=np.float64)
    model_velocity = np.zeros_like(actual_velocity)
    actual_position = np.zeros_like(actual_velocity)
    model_position = np.zeros_like(actual_velocity)
    action = np.array([[1.0, 0.0, 0.0]])
    for k in range(1, steps + 1):
        actual_velocity = limited(actual_velocity, action, dt, 4.0, vmax)
        model_velocity = limited(model_velocity, action, dt, 2.0, vmax)
        actual_position += dt * actual_velocity
        model_position += dt * model_velocity
        bound = 2.0 * dt**2 * k * (k + 1) / 2
        error = np.linalg.norm(actual_position - model_position)
        assert error == pytest.approx(bound, abs=1e-12)
    assert error == pytest.approx(4.4, abs=1e-12)


@pytest.mark.parametrize(
    'initial_velocity, action',
    [
        ([10.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 10.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 0.0, 10.0], [1.0, 1.0, 1.0]),
    ],
)
def test_peer_cv_error_bound_with_active_speed_saturation(initial_velocity, action):
    dt, acceleration, vmax = 0.2, 4.0, 10.0
    initial = np.asarray(initial_velocity, dtype=np.float64)[None, :]
    command = np.asarray(action, dtype=np.float64)[None, :]
    unit_command = command / np.maximum(1.0, np.linalg.norm(command, axis=-1, keepdims=True))
    velocity = initial.copy()
    position = np.zeros_like(velocity)
    saturation_seen = False
    for k in range(1, 11):
        requested = velocity + dt * acceleration * unit_command
        saturation_seen |= bool(np.linalg.norm(requested) > vmax)
        velocity = limited(velocity, command, dt, acceleration, vmax)
        position += dt * velocity
        cv_position = k * dt * initial
        bound = acceleration * dt**2 * k * (k + 1) / 2
        # limited uses a 1e-12 denominator regularizer; allow numerical residuals.
        assert np.linalg.norm(position - cv_position) <= bound + 1e-10
        assert np.linalg.norm(velocity) <= vmax + 1e-12
    assert saturation_seen
