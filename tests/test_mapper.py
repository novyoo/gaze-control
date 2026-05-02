"""
tests/test_mapper.py
Tests for GazeMapper coordinate conversion.
"""

import pytest
from src.cursor.mapper import GazeMapper


def make_config(gx_min=0.35, gx_max=0.65, gy_min=0.38, gy_max=0.62):
    return {
        "screen": {"auto_detect": False, "width": 1920, "height": 1080},
        "mapping": {
            "gaze_x_min": gx_min, "gaze_x_max": gx_max,
            "gaze_y_min": gy_min, "gaze_y_max": gy_max,
            "screen_padding_x": 0, "screen_padding_y": 0,
        }
    }


def test_center_maps_to_center():
    m = GazeMapper(make_config())
    px, py = m.map(0.5, 0.5)
    assert abs(px - 960) < 20, f"Expected ~960, got {px}"
    assert abs(py - 540) < 20, f"Expected ~540, got {py}"


def test_clamp_below_zero():
    m = GazeMapper(make_config())
    px, py = m.map(0.0, 0.0)
    assert px >= 0 and py >= 0


def test_clamp_above_max():
    m = GazeMapper(make_config())
    px, py = m.map(1.0, 1.0)
    assert px <= 1920 and py <= 1080


def test_update_range():
    m = GazeMapper(make_config())
    m.update_range(0.4, 0.6, 0.4, 0.6)
    assert m.gaze_x_min == 0.4