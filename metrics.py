"""Paper metrics.

Every rate below is a macro-average over episodes: each episode counts once,
regardless of how long it was. Definitions match ``docs/METRICS.md`` and the
metrics section of the README. Do not rename these keys; the plot scripts
read them.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

# Keys stored on every finished episode (plus return_* added by the rollout).
EPISODE_METRIC_KEYS = (
    "success",
    "collision",
    "obstacle_collision",
    "inter_uav_collision",
    "timeout",
    "formation_keeping_rate",
    "mean_formation_error",
    "mean_aoi",
    "max_aoi",
    "packet_success_rate",
    "mean_outage_length",
    "steps",
    "p0",
    "return_sum",
    "return_mean",
    "waypoint_fraction",
    "nlos_fraction",
    "min_clearance",
    "mean_belief_std",
    "los_mean_formation_error",
    "nlos_mean_formation_error",
    "los_mean_aoi",
    "nlos_mean_aoi",
    "los_mean_slot_std",
    "nlos_mean_slot_std",
    "los_mean_alt_error",
    "nlos_mean_alt_error",
    "los_mean_z",
    "overfly_mean_alt_error",
    "impassable_mean_alt_error",
    "overfly_mean_z",
    "impassable_mean_z",
    "overfly_mean_leader_z",
    "impassable_mean_leader_z",
    "overfly_clear_rate",
    "overfly_link_steps",
    "impassable_link_steps",
    "overfly_inside_steps",
    "near_roof_mean_z",
    "near_roof_mean_alt_error",
    "near_roof_mean_leader_z",
    "near_roof_link_steps",
    "tall_mean_z",
    "tall_mean_alt_error",
    "tall_mean_leader_z",
    "tall_link_steps",
    "far_roof_mean_z",
    "far_roof_mean_alt_error",
    "far_roof_mean_leader_z",
    "far_roof_link_steps",
    "radio_total_airtime_seconds",
    "radio_budget_utilization",
    "radio_broadcast_count",
    "radio_forward_requests",
    "radio_reject_count",
    "relay_decode_at_tx_count",
    "relay_update_count",
)

# Keys that become "<name>" and "<name>_std" after summarize_episodes.
SUMMARY_KEYS = (
    "success_rate",
    "collision_rate",
    "obstacle_collision_rate",
    "inter_uav_collision_rate",
    "timeout_rate",
    "formation_keeping_rate",
    "mean_formation_error",
    "mean_aoi",
    "max_aoi",
    "packet_success_rate",
    "mean_outage_length",
    "mean_steps",
    "mean_return",
    "waypoint_fraction",
    "nlos_fraction",
    "min_clearance",
    "mean_belief_std",
    "los_mean_formation_error",
    "nlos_mean_formation_error",
    "los_mean_aoi",
    "nlos_mean_aoi",
    "los_mean_slot_std",
    "nlos_mean_slot_std",
    "los_mean_alt_error",
    "nlos_mean_alt_error",
    "los_mean_z",
    "overfly_mean_alt_error",
    "impassable_mean_alt_error",
    "overfly_mean_z",
    "impassable_mean_z",
    "overfly_mean_leader_z",
    "impassable_mean_leader_z",
    "overfly_clear_rate",
    "overfly_link_steps",
    "impassable_link_steps",
    "overfly_inside_steps",
    "near_roof_mean_z",
    "near_roof_mean_alt_error",
    "near_roof_mean_leader_z",
    "near_roof_link_steps",
    "tall_mean_z",
    "tall_mean_alt_error",
    "tall_mean_leader_z",
    "tall_link_steps",
    "far_roof_mean_z",
    "far_roof_mean_alt_error",
    "far_roof_mean_leader_z",
    "far_roof_link_steps",
    "radio_total_airtime_seconds",
    "radio_budget_utilization",
    "radio_broadcast_count",
    "radio_forward_requests",
    "radio_reject_count",
    "relay_decode_at_tx_count",
    "relay_update_count",
)


def _mean_std(values: Iterable[float], skip_nonfinite: bool = False) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if skip_nonfinite:
        arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0
    return mean, float(arr.std(ddof=1))


def summarize_episodes(rows: list[dict]) -> dict[str, float]:
    """Aggregate per-episode dicts into the numbers reported in tables.

    success_rate
        Fraction of episodes that ended in success. An episode is a success
        only when every waypoint has been captured AND, on the terminal step,
        every follower is inside ``formation_tol`` of the slot computed from
        the *true* leader pose AND the episode never collided (obstacle or
        inter-UAV).

    collision_rate
        Fraction of episodes with at least one collision. Split further into
        ``obstacle_collision_rate`` and ``inter_uav_collision_rate``. An
        episode can increment both splits; ``collision_rate`` counts it once.

    timeout_rate
        Fraction of episodes that hit ``max_steps`` without success and
        without a collision.

    formation_keeping_rate
        Mean, over episodes, of the within-episode fraction of steps on
        which EVERY follower was inside ``formation_tol``. Steps in crashed
        and timed-out episodes count. This is intentionally stricter than
        "average follower is close".

    mean_formation_error
        Mean, over episodes, of that episode's average slot error in meters.
        Inside an episode the average is: for each step, the mean error over
        followers; then the mean over steps.

    mean_aoi / max_aoi
        ``mean_aoi`` averages, over episodes, the episode-level mean AoI.
        The episode-level mean averages the integer AoI counter over
        followers and over steps, measured *after* that step's packet draw.
        ``max_aoi`` averages each episode's maximum AoI (the worst
        follower-step in that episode).

    packet_success_rate
        Mean, over episodes, of (packets received) / (packets attempted).
        Every follower attempts exactly one packet per step. The free
        delivery at reset is not counted.

    mean_outage_length
        Mean, over episodes, of that episode's mean consecutive-loss streak
        length (in steps). A streak is a maximal run of failed receptions
        for one follower. Followers that never lose a packet contribute no
        streak. Episodes with zero streaks contribute 0.

    mean_return
        Mean over episodes of the sum of all agents' returns.

    waypoint_fraction
        Mean, over episodes, of (waypoints captured) / (waypoints in the
        route). A timed-out or crashed episode keeps the fraction it had
        reached. Success implies 1.

    nlos_fraction
        Mean, over episodes, of the fraction of packet attempts whose
        follower–leader segment intersected a building disk. This is a
        geometric fact about the path. Whether that intersection actually
        multiplies the success probability is controlled by ``use_blockage``
        and is not what this column measures.

    min_clearance
        Mean, over episodes, of the smallest surface gap (meters) between
        any aircraft and any obstacle during that episode. Negative means
        the aircraft penetrated an obstacle on the colliding step.
        For a building this is the horizontal gap while the aircraft is
        beside or below the roof, and the vertical gap while it is inside
        the footprint and above the roof. Traffic uses the 3D gap.

    mean_belief_std
        Mean, over episodes, of that episode's average *slot* standard
        deviation in meters. Inside an episode the average is over followers
        and over steps, taken after the packet draw. The three world axes of
        the slot are averaged first. This is the uncertainty of the slot the
        follower is being told to fly, including heading lever-arm, not the
        leader-position standard deviation alone. Hold and AoI-only
        observations still run the filter, so this column is defined for
        every mode. The actor only *sees* it in ``obs_mode=belief``.

    los_mean_* / nlos_mean_*
        The same quantities restricted to follower-steps whose
        leader–follower segment does, or does not, intersect a building.
        An episode with no samples on one side stores NaN for that side.
        The summary skips those NaNs, so an open-field episode does not
        pull the NLOS average down to zero. ``n_episodes_with_nlos`` counts
        how many episodes contributed to the NLOS columns.

        ``los_mean_alt_error`` / ``nlos_mean_alt_error`` are the absolute
        vertical miss of the true flat slot, in meters. Climbing to clear a
        roof shows up here. A policy that climbs in the open as well is not
        buying a measurement; it is just leaving the formation.

        The non-line-of-sight average mixes roofs the aircraft can fly over
        with the building that is taller than the ceiling. ``overfly_*`` is
        the first kind. ``impassable_*`` is the second. Altitude error is the
        price against the flat slot. ``*_mean_z`` is absolute altitude,
        because a leader that climbs lifts the slot and hides the climb
        inside the error. ``overfly_clear_rate`` is the fraction of
        follower-steps spent inside an overflyable footprint on which the
        aircraft is above that roof. Episodes that never enter that
        footprint store NaN, same as the other split columns.
    """
    if not rows:
        raise ValueError("summarize_episodes received no episodes")

    def col(name: str) -> list[float]:
        return [float(r[name]) for r in rows]

    def optional_col(name: str, default: float = 0.0) -> list[float]:
        return [float(r.get(name, default)) for r in rows]

    mapping = {
        "success_rate": col("success"),
        "collision_rate": col("collision"),
        "obstacle_collision_rate": col("obstacle_collision"),
        "inter_uav_collision_rate": col("inter_uav_collision"),
        "timeout_rate": col("timeout"),
        "formation_keeping_rate": col("formation_keeping_rate"),
        "mean_formation_error": col("mean_formation_error"),
        "mean_aoi": col("mean_aoi"),
        "max_aoi": col("max_aoi"),
        "packet_success_rate": col("packet_success_rate"),
        "mean_outage_length": col("mean_outage_length"),
        "mean_steps": col("steps"),
        "mean_return": col("return_sum"),
        "waypoint_fraction": col("waypoint_fraction"),
        "nlos_fraction": col("nlos_fraction"),
        "min_clearance": col("min_clearance"),
        "mean_belief_std": col("mean_belief_std"),
        "los_mean_formation_error": col("los_mean_formation_error"),
        "nlos_mean_formation_error": col("nlos_mean_formation_error"),
        "los_mean_aoi": col("los_mean_aoi"),
        "nlos_mean_aoi": col("nlos_mean_aoi"),
        "los_mean_slot_std": col("los_mean_slot_std"),
        "nlos_mean_slot_std": col("nlos_mean_slot_std"),
        "los_mean_alt_error": col("los_mean_alt_error"),
        "nlos_mean_alt_error": col("nlos_mean_alt_error"),
        "los_mean_z": col("los_mean_z"),
        "overfly_mean_alt_error": col("overfly_mean_alt_error"),
        "impassable_mean_alt_error": col("impassable_mean_alt_error"),
        "overfly_mean_z": col("overfly_mean_z"),
        "impassable_mean_z": col("impassable_mean_z"),
        "overfly_mean_leader_z": col("overfly_mean_leader_z"),
        "impassable_mean_leader_z": col("impassable_mean_leader_z"),
        "overfly_clear_rate": col("overfly_clear_rate"),
        "overfly_link_steps": col("overfly_link_steps"),
        "impassable_link_steps": col("impassable_link_steps"),
        "overfly_inside_steps": col("overfly_inside_steps"),
        "near_roof_mean_z": col("near_roof_mean_z"),
        "near_roof_mean_alt_error": col("near_roof_mean_alt_error"),
        "near_roof_mean_leader_z": col("near_roof_mean_leader_z"),
        "near_roof_link_steps": col("near_roof_link_steps"),
        "tall_mean_z": col("tall_mean_z"),
        "tall_mean_alt_error": col("tall_mean_alt_error"),
        "tall_mean_leader_z": col("tall_mean_leader_z"),
        "tall_link_steps": col("tall_link_steps"),
        "far_roof_mean_z": col("far_roof_mean_z"),
        "far_roof_mean_alt_error": col("far_roof_mean_alt_error"),
        "far_roof_mean_leader_z": col("far_roof_mean_leader_z"),
        "far_roof_link_steps": col("far_roof_link_steps"),
        "radio_total_airtime_seconds": optional_col("radio_total_airtime_seconds"),
        "radio_budget_utilization": optional_col("radio_budget_utilization"),
        "radio_broadcast_count": optional_col("radio_broadcast_count"),
        "radio_forward_requests": optional_col("radio_forward_requests"),
        "radio_reject_count": optional_col("radio_reject_count"),
        "relay_decode_at_tx_count": optional_col("relay_decode_at_tx_count"),
        "relay_update_count": optional_col("relay_update_count"),
    }
    skip_nan = {
        "los_mean_formation_error",
        "nlos_mean_formation_error",
        "los_mean_aoi",
        "nlos_mean_aoi",
        "los_mean_slot_std",
        "nlos_mean_slot_std",
        "los_mean_alt_error",
        "nlos_mean_alt_error",
        "los_mean_z",
        "overfly_mean_alt_error",
        "impassable_mean_alt_error",
        "overfly_mean_z",
        "impassable_mean_z",
        "overfly_mean_leader_z",
        "impassable_mean_leader_z",
        "overfly_clear_rate",
        "near_roof_mean_z",
        "near_roof_mean_alt_error",
        "near_roof_mean_leader_z",
        "tall_mean_z",
        "tall_mean_alt_error",
        "tall_mean_leader_z",
        "far_roof_mean_z",
        "far_roof_mean_alt_error",
        "far_roof_mean_leader_z",
    }
    out: dict[str, float] = {"n_episodes": float(len(rows))}
    nlos_values = mapping["nlos_mean_formation_error"]
    out["n_episodes_with_nlos"] = float(sum(math.isfinite(v) for v in nlos_values))
    out["n_episodes_with_overfly"] = float(
        sum(math.isfinite(v) for v in mapping["overfly_mean_alt_error"])
    )
    out["n_episodes_with_impassable"] = float(
        sum(math.isfinite(v) for v in mapping["impassable_mean_alt_error"])
    )
    out["n_episodes_with_overfly_inside"] = float(
        sum(math.isfinite(v) for v in mapping["overfly_clear_rate"])
    )
    for key, values in mapping.items():
        mean, std = _mean_std(values, skip_nonfinite=key in skip_nan)
        out[key] = mean
        out[key + "_std"] = std
    # A NaN outside the LOS/NLOS split means an empty or broken input.
    # The LOS/NLOS columns stay NaN when every episode lacked that side
    # (an open-field run has no NLOS samples). That is a reported fact,
    # not a broken aggregate.
    for key, value in out.items():
        if isinstance(value, float) and math.isnan(value):
            # `nlos_mean_slot_std` already ends in `_std`, so also accept
            # the key itself, not only the name with one `_std` stripped.
            base = key[:-4] if key.endswith("_std") else key
            if key in skip_nan or base in skip_nan:
                continue
            raise RuntimeError(f"metric {key} is NaN")
    return out
