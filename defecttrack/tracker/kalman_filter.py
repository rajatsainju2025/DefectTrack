"""Constant-velocity Kalman filter for bounding boxes.

Independent implementation of the standard linear Kalman filter used by
SORT-style trackers. The state is

    x = [cx, cy, a, h, vcx, vcy, va, vh]

where (cx, cy) is the box centre, a = w / h its aspect ratio and h its height;
the last four entries are their velocities. Measurements are (cx, cy, a, h).
Process and measurement noise scale with the box height, so small and large
boxes are tracked with comparable relative uncertainty.
"""

import numpy as np
import scipy.linalg

# 95% quantile of the chi-square distribution with N degrees of freedom,
# used as the gate for squared Mahalanobis distances.
chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070,
             6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}

_NDIM = 4
POS_WEIGHT = 1.0 / 20   # position noise, relative to box height
VEL_WEIGHT = 1.0 / 160  # velocity noise, relative to box height


def _noise_std(h, pos_scale, vel_scale, aspect_pos, aspect_vel):
    """Per-state standard deviations for box height(s) ``h``."""
    p, v = pos_scale * POS_WEIGHT * h, vel_scale * VEL_WEIGHT * h
    ones = np.ones_like(h)
    return np.stack([p, p, aspect_pos * ones, p, v, v, aspect_vel * ones, v], axis=-1)


class KalmanFilter:
    def __init__(self, dt=1.0):
        # x_{k+1} = F x_k : position += velocity * dt
        self.F = np.eye(2 * _NDIM)
        self.F[:_NDIM, _NDIM:] = dt * np.eye(_NDIM)
        # z = H x : we observe the position part of the state
        self.H = np.eye(_NDIM, 2 * _NDIM)

    def initiate(self, measurement):
        """Start a track from one (cx, cy, a, h) measurement, at zero velocity."""
        measurement = np.asarray(measurement, dtype=float)
        mean = np.concatenate([measurement, np.zeros(_NDIM)])
        std = _noise_std(measurement[3], 2, 10, 1e-2, 1e-5)
        return mean, np.diag(std ** 2)

    def predict(self, mean, covariance):
        """Propagate one track forward by one time step."""
        mean, covariance = self.multi_predict(mean[None], covariance[None])
        return mean[0], covariance[0]

    def multi_predict(self, mean, covariance):
        """Propagate N tracks at once. ``mean`` is (N, 8), ``covariance`` (N, 8, 8)."""
        std = _noise_std(mean[:, 3], 1, 1, 1e-2, 1e-5)
        Q = np.einsum("ni,ij->nij", std ** 2, np.eye(2 * _NDIM))
        mean = mean @ self.F.T
        covariance = self.F @ covariance @ self.F.T + Q
        return mean, covariance

    def project(self, mean, covariance):
        """Map a state to measurement space, adding measurement noise R."""
        std = POS_WEIGHT * mean[3] * np.ones(_NDIM)
        std[2] = 1e-1
        R = np.diag(std ** 2)
        return self.H @ mean, self.H @ covariance @ self.H.T + R

    def update(self, mean, covariance, measurement):
        """Correct a predicted state with a new measurement."""
        z_mean, S = self.project(mean, covariance)
        cho = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
        # Kalman gain K = P H^T S^-1, computed by solving S K^T = H P^T
        K = scipy.linalg.cho_solve(cho, (covariance @ self.H.T).T, check_finite=False).T
        new_mean = mean + (measurement - z_mean) @ K.T
        new_covariance = covariance - K @ S @ K.T
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric="maha"):
        """Distance from the predicted state to each measurement (rows of an (M, 4) array).

        ``metric="maha"`` returns squared Mahalanobis distances (compare with
        ``chi2inv95``); ``metric="gaussian"`` returns squared Euclidean ones.
        """
        z_mean, S = self.project(mean, covariance)
        measurements = np.asarray(measurements)
        if only_position:
            z_mean, S, measurements = z_mean[:2], S[:2, :2], measurements[:, :2]
        d = measurements - z_mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        if metric == "maha":
            L = np.linalg.cholesky(S)
            z = scipy.linalg.solve_triangular(L, d.T, lower=True,
                                              check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)
        raise ValueError(f"Unknown metric: {metric}")
