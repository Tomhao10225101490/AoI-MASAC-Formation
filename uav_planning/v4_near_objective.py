"""Pure mathematics for the opt-in near-goal objective replacement.

The legacy scalar team reward already contains the current post-step near
bonus, which equals ``following_phi``.  The replacement removes that bonus
and adds a discounted potential difference.  It is PBRS relative to the
legacy reward *minus* its near bonus, intentionally not relative to the full
legacy reward.  No policy improvement or sparse-success guarantee follows.

A true termination (success or collision) has an absorbing next potential of
zero. A timeout alone keeps its observed next potential and bootstrap
semantics. If both flags are true, termination takes precedence. A finite
timeout sum retains its potential boundary term; cancelling it requires the
correspondingly transformed continuation value, not merely this function.

There are no environments, learned models, random draws, or mutable state here.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np

__all__ = ['near_potential', 'replace_near_reward']


def _scalar(value, name, *, minimum=None, strict_minimum=False, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real scalar')
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite real scalar') from exc
    if not math.isfinite(number):
        raise ValueError(f'{name} must be finite')
    if minimum is not None and (number <= minimum if strict_minimum else number < minimum):
        relation = '>' if strict_minimum else '>='
        raise ValueError(f'{name} must be {relation} {minimum}')
    if maximum is not None and number >= maximum:
        raise ValueError(f'{name} must be < {maximum}')
    return number


def _points(value, name):
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0 or
            not np.issubdtype(array.dtype, np.number) or not np.isrealobj(array)):
        raise ValueError(f'{name} must be a nonempty real numeric array of shape (N,3)')
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain only finite coordinates')
    return array


def near_potential(positions, goals, coefficient=0.4, radius_m=15.0):
    """Return ``coefficient * mean(clip(1-distance/radius_m, 0, 1))``.

    Both arrays have the same nonempty ``(N,3)`` shape. This team mean is
    exactly the scalar legacy near reward for the same post-step geometry.
    The result is bounded between zero and the nonnegative coefficient.
    Inputs are never modified. Computation uses float64 coordinates.
    """
    coefficient = _scalar(coefficient, 'coefficient', minimum=0.0)
    radius_m = _scalar(radius_m, 'radius_m', minimum=0.0, strict_minimum=True)
    positions, goals = _points(positions, 'positions'), _points(goals, 'goals')
    if positions.shape != goals.shape:
        raise ValueError('positions and goals must have the same (N,3) shape')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            distance = np.linalg.norm(goals - positions, axis=-1)
            occupancy = np.clip(1.0 - distance / radius_m, 0.0, 1.0)
            # Match the legacy source's per-agent multiply, then team mean.
            potential = float((coefficient * occupancy).mean())
    except FloatingPointError as exc:
        raise ValueError('Near-potential geometry or arithmetic overflowed') from exc
    if not np.isfinite(distance).all() or not math.isfinite(potential):
        raise ValueError('Near-potential geometry or result is nonfinite')
    return potential


def replace_near_reward(base_reward, previous_phi, following_phi, terminated,
                        truncated, gamma=0.99, mode='original'):
    """Return ``(new_reward, correction, term_detail)`` for one transition.

    ``base_reward`` is the scalar legacy Path-C team reward, including the
    physical post-step near bonus ``following_phi`` even on a terminal step.
    ``previous_phi`` and ``following_phi`` must come from the actual pre/post
    transition geometries with the same coefficient/radius/team averaging.

    ``original`` returns the input reward object itself without reward
    arithmetic and a correction of exactly 0.0. ``potential`` computes

        base_reward - following_phi
        + gamma * (0 if terminated else following_phi) - previous_phi.

    Only the added potential uses an absorbing zero on a true termination;
    the legacy near term removed is always the observed ``following_phi``.
    A timeout (``truncated=True, terminated=False``) does not absorb or change
    the learner's bootstrap mask. This function never chooses that mask.
    """
    base = _scalar(base_reward, 'base_reward')
    before = _scalar(previous_phi, 'previous_phi', minimum=0.0)
    after = _scalar(following_phi, 'following_phi', minimum=0.0)
    gamma = _scalar(gamma, 'gamma', minimum=0.0, maximum=1.0)
    if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
        raise ValueError('terminated and truncated must be boolean flags')
    if mode not in ('original', 'potential'):
        raise ValueError("mode must be 'original' or 'potential'")
    terminal, truncation = bool(terminated), bool(truncated)
    effective_after = 0.0 if terminal else after
    if mode == 'original':
        new_reward, correction, removed, added = base_reward, 0.0, 0.0, 0.0
    else:
        removed = after
        added = gamma * effective_after - before
        correction = added - removed
        new_reward = base + correction
        if not all(math.isfinite(value) for value in (added, correction, new_reward)):
            raise ValueError('Reward replacement arithmetic produced a nonfinite result')
    detail = {
        'mode': mode, 'legacy_base_reward': base,
        'previous_phi': before, 'following_phi_observed': after,
        'following_phi_effective_for_potential': effective_after,
        'removed_near': removed, 'potential_difference': added,
        'correction': correction, 'gamma': gamma,
        'terminated': terminal, 'truncated': truncation,
        'absorbing_potential_applied': mode == 'potential' and terminal,
        'bootstrap_allowed_by_flags': not terminal,
        'timeout_without_termination': truncation and not terminal,
    }
    return new_reward, correction, detail
