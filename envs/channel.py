"""Packet-success model for the leader → follower command link.

Only this link is lossy. Obstacle and teammate observations are onboard
sensing with a finite range; they do not travel on this channel.

The distance law is the standard exponential path-loss success model used as
a simple stand-in for SNR-based packet delivery. Obstacle blockage is the
part that recent "ideal communication" formation papers leave out: a building
that the formation has to fly around also cuts the straight radio path.
"""

from __future__ import annotations

import math


def packet_success_probability(
    distance: float,
    p0: float,
    d0: float,
    mode: str,
    blocked: bool = False,
    nlos_factor: float = 0.2,
) -> float:
    """Return P(one follower's packet arrives this step).

    mode
        ``distance``     p = p0 * exp(-distance / d0), times ``nlos_factor``
                         when ``blocked`` is true.
        ``independent``  p = p0. Distance and blockage are ignored. This is
                         the ablation, and also the x-axis of the loss sweep.
        ``perfect``      p = 1.
    """
    if mode == "perfect":
        return 1.0
    if p0 < 0.0 or p0 > 1.0:
        raise ValueError(f"p0 must be in [0, 1], got {p0}")
    if nlos_factor < 0.0 or nlos_factor > 1.0:
        raise ValueError(f"nlos_factor must be in [0, 1], got {nlos_factor}")
    if mode == "independent":
        return float(p0)
    if mode == "distance":
        if d0 <= 0.0:
            raise ValueError(f"d0 must be positive in distance mode, got {d0}")
        p = float(p0) * math.exp(-float(distance) / float(d0))
        if blocked:
            p *= float(nlos_factor)
        return float(min(1.0, max(0.0, p)))
    raise ValueError(
        f"Unknown channel mode {mode!r}. Use 'distance', 'independent', or 'perfect'."
    )
