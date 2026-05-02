"""
tests/test_gaze.py
Unit tests for GazeEstimator logic (no camera needed).
"""

import numpy as np
import pytest
from src.gaze.estimator import GazeEstimator
from src.gaze.detector import FaceData


def make_fake_face(iris_x_ratio=0.5):
    """Build a minimal FaceData for testing."""
    w, h = 640, 480
    # Eye corners: inner at x=200, outer at x=280 (80px wide eye)
    inner = np.array([200.0, 240.0])
    outer = np.array([280.0, 240.0])
    iris = np.array([200.0 + iris_x_ratio * 80.0, 240.0])
    return FaceData(
        landmarks=[],
        left_iris=iris,
        right_iris=iris,
        left_eye_corners=(inner, outer),
        right_eye_corners=(inner, outer),
        confidence=1.0,
        frame_h=h,
        frame_w=w,
    )


def test_gaze_center():
    gx, gy = GazeEstimator().estimate(make_fake_face(0.5))
    assert abs(gx - 0.5) < 0.05, "Center gaze should be ~0.5"


def test_gaze_left():
    gx, _ = GazeEstimator().estimate(make_fake_face(0.0))
    assert gx < 0.1, "Far-left iris should give low gaze_x"


def test_gaze_right():
    gx, _ = GazeEstimator().estimate(make_fake_face(1.0))
    assert gx > 0.9, "Far-right iris should give high gaze_x"