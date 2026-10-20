"""Packet-success models for the leader → follower link.

The leader's state is the only message that travels on this lossy link.
Obstacle and teammate observations are treated as onboard sensing and do
not use this channel (see the environment docstring).
"""

from __future__ import annotations

import math


def packet_success_probability(
    distance: float,
    p0: float,
    d0: float,
    mode: str,
) -> float:
    """Return P(packet delivered) for one follower on one step.

    Parameters
    ----------
    distance:
        Euclidean distance between that follower and the leader, in meters.
    p0:
        Reference success probability. In ``distance`` mode this is the
        success probability at distance 0. In ``independent`` mode it is
        the success probability at every distance.
    d0:
        Length scale in meters. Used only by ``distance`` mode.
    mode:
        ``distance``    — p = p0 * exp(-distance / d0)
        ``independent`` — p = p0, ignoring distance (ablation)
        ``perfect``     — p = 1

    The result is clipped into [0, 1].
    """
    if mode == "perfect":
        return 1.0
    if p0 < 0.0 or p0 > 1.0:
        raise ValueError(f"p0 must be in [0, 1], got {p0}")
    if mode == "independent":
        return float(p0)
    if mode == "distance":
        if d0 <= 0.0:
            raise ValueError(f"d0 must be positive in distance mode, got {d0}")
        p = float(p0) * math.exp(-float(distance) / float(d0))
        if p < 0.0:
            return 0.0
        if p > 1.0:
            return 1.0
        return p
    raise ValueError(
        f"Unknown channel mode {mode!r}. Use 'distance', 'independent', or 'perfect'."
    )
