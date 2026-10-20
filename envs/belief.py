"""Diagonal-block Kalman coast of one follower's leader state.

The filter is not a learned module. It is the same constant-velocity
reconstruction a follower can run onboard from the last packet, the local
wind estimate, and a known building map.

State is air-relative: ``[x, y, z, vx, vy, vz]``. Prediction adds the sensed
wind as a known input on x and y. Process noise is diagonal. When the segment
from the follower to the *predicted* leader intersects a building, that noise
is multiplied by ``nlos_q_scale``, because a constant-velocity coast is a
worse model exactly while the leader is likely to be turning around a roof.

Only the diagonal of P is read out (position standard deviation). The
cross-terms are still stored, so position variance grows with velocity
uncertainty instead of only with the position process noise.
"""

from __future__ import annotations

import numpy as np


class LeaderBelief:
    def __init__(
        self,
        meas_var_pos: float,
        meas_var_vel: float,
        q_pos: float,
        q_vel: float,
        var_cap: float,
    ):
        self.meas_var = np.array(
            [meas_var_pos, meas_var_pos, meas_var_pos, meas_var_vel, meas_var_vel, meas_var_vel],
            dtype=np.float64,
        )
        self.Q = np.diag(
            np.array([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel], dtype=np.float64)
        )
        self.var_cap = float(var_cap)
        self.mean = np.zeros(6, dtype=np.float64)
        self.P = np.diag(self.meas_var.copy())

    def reset_from_measurement(
        self, x: float, y: float, z: float, vx: float, vy: float, vz: float
    ) -> None:
        self.mean[:] = (x, y, z, vx, vy, vz)
        self.P = np.diag(self.meas_var.copy())

    def predict(self, dt: float, wind_x: float, wind_y: float, q_scale: float) -> None:
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.mean = F @ self.mean
        self.mean[0] += float(wind_x) * dt
        self.mean[1] += float(wind_y) * dt
        scale = max(float(q_scale), 0.0)
        self.P = F @ self.P @ F.T + scale * self.Q
        self.P = 0.5 * (self.P + self.P.T)
        np.clip(self.P, -self.var_cap, self.var_cap, out=self.P)
        for i in range(6):
            self.P[i, i] = min(max(self.P[i, i], 0.0), self.var_cap)

    def position_std(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.P)[:3], 0.0))
