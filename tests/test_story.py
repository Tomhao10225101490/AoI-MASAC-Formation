"""The motivation numbers are properties of the channel and the filter."""

from __future__ import annotations

import math

from envs.story import coast_curves, failures_between_successes, representative_link


def test_roof_outage_is_long_and_open_air_outage_is_short():
    link = representative_link(distance_m=40.0)
    assert link["p_los"] == math.exp(-40.0 / 200.0) * 0.95
    assert link["p_nlos"] == link["p_los"] * 0.2
    assert link["failures_between_successes_los"] < 1.0
    assert link["failures_between_successes_nlos"] > 5.0
    assert failures_between_successes(0.5) == 1.0


def test_lost_heading_shows_up_in_the_slot_not_in_the_mean():
    rows = coast_curves(n_steps=6)
    open_6 = next(row for row in rows if row["regime"] == "open" and row["lost_steps"] == 6)
    nlos_6 = next(row for row in rows if row["regime"] == "nlos_q" and row["lost_steps"] == 6)
    assert open_6["heading_mean_rad"] == 0.0
    assert open_6["lever_arm_m"] == 40.0
    assert open_6["slot_std_max_m"] > open_6["position_std_mean_m"] * 4.0
    assert nlos_6["slot_std_max_m"] > open_6["slot_std_max_m"] * 1.5
