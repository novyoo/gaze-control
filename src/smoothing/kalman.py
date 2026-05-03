# src/smoothing/kalman.py
# 2D Kalman filter for gaze coordinate smoothing.
# State vector: [x, y, vx, vy] — position + velocity
# Measurement:  [x, y]         — raw gaze coords

import numpy as np
from collections import deque


class GazeKalmanFilter:

    def __init__(self, config: dict):
        k = config["smoothing"]
        process_noise       = k["process_noise"]
        measurement_noise   = k["measurement_noise"]
        initial_uncertainty = k["initial_uncertainty"]

        # State: [x, y, vx, vy]
        self.x = np.zeros((4, 1))          # initial state

        # State transition matrix — predicts next state from current
        # x_new = x + vx*dt,  y_new = y + vy*dt
        self.F = np.array([
            [1, 0, 1, 0],   # x  = x  + vx
            [0, 1, 0, 1],   # y  = y  + vy
            [0, 0, 1, 0],   # vx = vx
            [0, 0, 0, 1],   # vy = vy
        ], dtype=float)

        # Measurement matrix — we only observe x, y (not velocity)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise covariance — how much we expect state to vary
        self.Q = np.eye(4) * process_noise

        # Measurement noise covariance — how noisy are landmark measurements
        self.R = np.eye(2) * measurement_noise

        # Error covariance matrix — uncertainty in state estimate
        self.P = np.eye(4) * initial_uncertainty

        # For jitter metrics
        self._raw_history      = deque(maxlen=30)
        self._smooth_history   = deque(maxlen=30)
        self._initialized      = False

    def update(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        """
        Feed raw gaze measurement, get smoothed position back.
        """
        measurement = np.array([[raw_x], [raw_y]])

        if not self._initialized:
            # First frame — seed state with measurement
            self.x[0] = raw_x
            self.x[1] = raw_y
            self._initialized = True
            self._raw_history.append((raw_x, raw_y))
            self._smooth_history.append((raw_x, raw_y))
            return raw_x, raw_y

        # ── Predict ───────────────────────────────────────────────────
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ── Update ────────────────────────────────────────────────────
        # Innovation (difference between measurement and prediction)
        y_innov = measurement - self.H @ x_pred

        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R

        # Kalman gain — how much to trust measurement vs prediction
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Updated state
        self.x = x_pred + K @ y_innov

        # Updated error covariance
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        smooth_x = float(self.x[0, 0])
        smooth_y = float(self.x[1, 0])

        # Track for jitter metrics
        self._raw_history.append((raw_x, raw_y))
        self._smooth_history.append((smooth_x, smooth_y))

        return smooth_x, smooth_y

    def reset(self):
        """Call if face is lost — resets velocity."""
        self.x[2] = 0.0   # vx = 0
        self.x[3] = 0.0   # vy = 0
        self.P    = np.eye(4)

    def jitter_reduction_percent(self) -> float:
        """
        Compares frame-to-frame variance of raw vs smoothed signal.
        Returns how much jitter was reduced as a percentage.
        """
        if len(self._raw_history) < 10:
            return 0.0

        raw    = np.array(self._raw_history)
        smooth = np.array(self._smooth_history)

        # Frame-to-frame differences
        raw_diff    = np.diff(raw,    axis=0)
        smooth_diff = np.diff(smooth, axis=0)

        raw_var    = float(np.mean(raw_diff    ** 2))
        smooth_var = float(np.mean(smooth_diff ** 2))

        if raw_var < 1e-9:
            return 0.0

        reduction = (1.0 - smooth_var / raw_var) * 100.0
        return max(0.0, min(reduction, 99.9))

    @property
    def velocity(self) -> tuple[float, float]:
        """Current estimated gaze velocity (vx, vy)."""
        return float(self.x[2, 0]), float(self.x[3, 0])