"""Constant-velocity coast of one follower's leader state, plus heading uncertainty.

The filter is not a learned module. It is the reconstruction a follower can
run onboard from the last packet, the local wind estimate, and a known
building map.

State is ``[x, y, z, vx, vy, vz, heading]``.

* Position and air-velocity follow a constant-velocity model. Prediction
  adds the sensed wind as a known input on x and y. The mean heading does
  not move: a lost packet does not invent a turn.
* Heading is still in the covariance. While the packet is missing, yaw is
  unknown up to the aircraft's yaw limit, so heading variance grows every
  coast step. The commanded slot is the leader position plus a body-frame
  offset rotated by heading, and that lever arm turns a modest heading
  variance into a large slot variance. That slot variance, not the raw
  leader-position variance, is what the actor is shown.
* When the segment from the follower to the predicted leader intersects a
  building, every process-noise term is multiplied by ``nlos_q_scale``.
  The mean stays a straight coast either way. The scale only says the
  straight coast is a worse description, which is the usual case while the
  leader is going around a roof.

Only the slot standard deviation is read out for the actor. Cross-terms are
still stored, so leader-position variance grows with velocity uncertainty
and slot variance grows with heading uncertainty.
"""

from __future__ import annotations

import math

import numpy as np

HEADING = 6
STATE_DIM = 7


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians into [-pi, pi]."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


class LeaderBelief:
    def __init__(
        self,
        meas_var_pos: float,
        meas_var_vel: float,
        meas_var_heading: float,
        q_pos: float,
        q_vel: float,
        var_cap: float,
    ):
        self.meas_var = np.array(
            [
                meas_var_pos,
                meas_var_pos,
                meas_var_pos,
                meas_var_vel,
                meas_var_vel,
                meas_var_vel,
                meas_var_heading,
            ],
            dtype=np.float64,
        )
        # Heading process variance is supplied per coast step, because it
        # depends on dt and the yaw limit. The diagonal entry here is only
        # a placeholder so Q has the right shape.
        self.Q = np.diag(
            np.array([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, 0.0], dtype=np.float64)
        )
        self.var_cap = float(var_cap)
        self.mean = np.zeros(STATE_DIM, dtype=np.float64)
        self.P = np.diag(self.meas_var.copy())
        self._F = np.eye(STATE_DIM, dtype=np.float64)

    def reset_from_measurement(
        self,
        x: float,
        y: float,
        z: float,
        vx: float,
        vy: float,
        vz: float,
        heading: float,
    ) -> None:
        self.mean[:] = (x, y, z, vx, vy, vz, wrap_angle(heading))
        self.P = np.diag(self.meas_var.copy())

    def drift(self, dt: float, wind_x: float, wind_y: float) -> None:
        """Advance the mean one step. Covariance is applied afterwards.

        Heading's mean stays at the last measurement. Wind is a known input
        on the horizontal position, matching the aircraft kinematics.
        """
        F = np.eye(STATE_DIM, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self._F = F
        self.mean = F @ self.mean
        self.mean[0] += float(wind_x) * dt
        self.mean[1] += float(wind_y) * dt
        self.mean[HEADING] = wrap_angle(float(self.mean[HEADING]))

    def add_process_noise(self, q_scale: float, q_heading: float) -> None:
        """Apply ``P <- F P F^T + q_scale * Q`` after :meth:`drift`.

        ``q_heading`` is the heading variance added on an ordinary coast
        step, before the NLOS scale. ``q_scale`` multiplies every diagonal
        of Q, including heading.
        """
        Q = self.Q.copy()
        Q[HEADING, HEADING] = max(float(q_heading), 0.0)
        scale = max(float(q_scale), 0.0)
        self.P = self._F @ self.P @ self._F.T + scale * Q
        self._clip_covariance()

    def position_std(self) -> np.ndarray:
        """Leader-position standard deviation, meters. Not the slot."""
        return np.sqrt(np.maximum(np.diag(self.P)[:3], 0.0))

    def slot_std(self, offset_xy: tuple[float, float] | np.ndarray) -> np.ndarray:
        """Standard deviation of the flat slot, meters, in world axes.

        The slot is ``leader_xyz + R(heading) @ offset``. With a frozen
        heading mean, a turn does not move the slot the actor is told to
        fly; it shows up here, scaled by the offset's lever arm.
        """
        ox = float(offset_xy[0])
        oy = float(offset_xy[1])
        psi = float(self.mean[HEADING])
        s = math.sin(psi)
        c = math.cos(psi)
        ddx = -ox * s - oy * c
        ddy = ox * c - oy * s
        jx = np.zeros(STATE_DIM, dtype=np.float64)
        jy = np.zeros(STATE_DIM, dtype=np.float64)
        jz = np.zeros(STATE_DIM, dtype=np.float64)
        jx[0] = 1.0
        jx[HEADING] = ddx
        jy[1] = 1.0
        jy[HEADING] = ddy
        jz[2] = 1.0
        variances = [float(j @ self.P @ j) for j in (jx, jy, jz)]
        return np.sqrt(np.maximum(np.asarray(variances, dtype=np.float64), 0.0))

    def _clip_covariance(self) -> None:
        self.P = 0.5 * (self.P + self.P.T)
        np.clip(self.P, -self.var_cap, self.var_cap, out=self.P)
        for i in range(STATE_DIM):
            self.P[i, i] = min(max(self.P[i, i], 0.0), self.var_cap)
