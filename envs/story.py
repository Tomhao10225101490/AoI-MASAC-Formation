"""Numbers the motivation figure is allowed to quote.

These are properties of the downlink and of the onboard filter. They are
not the score of a trained policy. The paper's claim is that a lost packet
behind a roof makes the *slot* lie, on the heading lever arm, for long
enough that paying altitude can be worth a fresh measurement.
"""

from __future__ import annotations

import math

import numpy as np

from envs.belief import HEADING, LeaderBelief
from envs.channel import packet_success_probability
from envs.formation_env import EnvConfig
from envs.geometry import make_formation_offsets


def failures_between_successes(success_probability: float) -> float:
    """Expected lost packets between two successful deliveries.

    This is ``(1 - p) / p``. It is the gap the filter has to coast, not a
    conditional streak length.
    """
    p = float(success_probability)
    if p <= 0.0:
        return float("inf")
    if p > 1.0:
        raise ValueError(f"success probability must be in (0, 1], got {p}")
    return (1.0 - p) / p


def representative_link(cfg: EnvConfig | None = None, distance_m: float = 40.0) -> dict[str, float]:
    """Line-of-sight and roof-blocked success at one range.

    Forty meters is the front-row lever arm at the default spacing, and it
    is also a typical leader–follower range while the wedge is intact.
    """
    cfg = cfg or EnvConfig()
    los = packet_success_probability(
        distance_m, cfg.p0, cfg.d0, "distance", blocked=False, nlos_factor=cfg.nlos_factor
    )
    nlos = packet_success_probability(
        distance_m, cfg.p0, cfg.d0, "distance", blocked=True, nlos_factor=cfg.nlos_factor
    )
    return {
        "distance_m": float(distance_m),
        "p0": float(cfg.p0),
        "d0_m": float(cfg.d0),
        "nlos_factor": float(cfg.nlos_factor),
        "p_los": float(los),
        "p_nlos": float(nlos),
        "failures_between_successes_los": failures_between_successes(los),
        "failures_between_successes_nlos": failures_between_successes(nlos),
    }


def _front_row_offset(cfg: EnvConfig) -> tuple[float, float]:
    """Body-frame offset of the first follower in a four-aircraft wedge."""
    return make_formation_offsets(4, spacing=float(cfg.formation_spacing))[0]


def coast_curves(n_steps: int = 12, cfg: EnvConfig | None = None) -> list[dict[str, float | str]]:
    """Slot doubt versus leader-position doubt after each lost packet.

    One row is one coast length and one process-noise scale. ``open`` uses
    scale 1. ``nlos_q`` multiplies every process-noise term by
    ``nlos_q_scale``, which is what the filter does when the predicted
    segment intersects a roof. The heading mean stays at the measurement.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    cfg = cfg or EnvConfig()
    offset = _front_row_offset(cfg)
    q_heading = (cfg.max_yaw_rate * cfg.dt) ** 2 / 3.0
    link = representative_link(cfg)
    rows: list[dict[str, float]] = []
    for scale, regime in ((1.0, "open"), (float(cfg.nlos_q_scale), "nlos_q")):
        belief = LeaderBelief(
            meas_var_pos=cfg.belief_meas_var_pos,
            meas_var_vel=cfg.belief_meas_var_vel,
            meas_var_heading=cfg.belief_meas_var_heading,
            q_pos=cfg.belief_q_pos,
            q_vel=cfg.belief_q_vel,
            var_cap=cfg.belief_var_cap,
        )
        belief.reset_from_measurement(0.0, 0.0, cfg.start_z, cfg.start_speed, 0.0, 0.0, 0.0)
        for lost in range(0, n_steps + 1):
            if lost > 0:
                belief.drift(cfg.dt, 0.0, 0.0)
                belief.add_process_noise(scale, q_heading)
            slot = belief.slot_std(offset)
            position = belief.position_std()
            rows.append(
                {
                    "lost_steps": float(lost),
                    "q_scale": float(scale),
                    "regime": regime,
                    "slot_std_mean_m": float(np.mean(slot)),
                    "slot_std_max_m": float(np.max(slot)),
                    "position_std_mean_m": float(np.mean(position)),
                    "heading_std_rad": float(math.sqrt(max(float(belief.P[HEADING, HEADING]), 0.0))),
                    "heading_mean_rad": float(belief.mean[HEADING]),
                    "lever_arm_m": float(math.hypot(offset[0], offset[1])),
                    "failures_between_successes_los": link["failures_between_successes_los"],
                    "failures_between_successes_nlos": link["failures_between_successes_nlos"],
                    "p_los": link["p_los"],
                    "p_nlos": link["p_nlos"],
                }
            )
    return rows
