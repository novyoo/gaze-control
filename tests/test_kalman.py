# tests/test_kalman.py

import numpy as np
import pytest
from src.smoothing.kalman import GazeKalmanFilter


def make_config(process_noise=0.03, measurement_noise=0.5):
    return {
        "smoothing": {
            "process_noise":       process_noise,
            "measurement_noise":   measurement_noise,
            "initial_uncertainty": 1.0,
        }
    }


def test_output_close_to_input_on_first_frame():
    kf = GazeKalmanFilter(make_config())
    x, y = kf.update(0.5, 0.5)
    assert abs(x - 0.5) < 0.01
    assert abs(y - 0.5) < 0.01


def test_smoothing_reduces_noise():
    kf = GazeKalmanFilter(make_config())
    # Feed noisy signal around 0.5
    np.random.seed(42)
    raw_vals, smooth_vals = [], []
    for _ in range(50):
        noisy = 0.5 + np.random.normal(0, 0.05)
        sx, _ = kf.update(noisy, 0.5)
        raw_vals.append(noisy)
        smooth_vals.append(sx)

    raw_var    = float(np.var(np.diff(raw_vals)))
    smooth_var = float(np.var(np.diff(smooth_vals)))
    assert smooth_var < raw_var, "Kalman should reduce variance"


def test_jitter_reduction_positive():
    kf = GazeKalmanFilter(make_config())
    np.random.seed(0)
    for _ in range(40):
        kf.update(0.5 + np.random.normal(0, 0.05),
                  0.5 + np.random.normal(0, 0.05))
    assert kf.jitter_reduction_percent() > 0


def test_reset_zeroes_velocity():
    kf = GazeKalmanFilter(make_config())
    for _ in range(10):
        kf.update(0.6, 0.6)
    kf.reset()
    vx, vy = kf.velocity
    assert vx == 0.0 and vy == 0.0