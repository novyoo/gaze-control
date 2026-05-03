# tests/test_gestures.py
# Tests gesture logic without camera or pyautogui

from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from src.gestures.detector import HandData


def make_hand(thumb_pos=(100,100), index_pos=(105,105),
              middle_pos=(150,150), frame_size=(480,640)):
    """Build a fake HandData for testing."""
    h, w = frame_size
    wrist     = np.array([w//2, h//2], dtype=float)
    index_mcp = np.array([w//2, h//2 - 40], dtype=float)
    return HandData(
        landmarks=[],
        thumb_tip=np.array(thumb_pos,  dtype=float),
        index_tip=np.array(index_pos,  dtype=float),
        middle_tip=np.array(middle_pos, dtype=float),
        ring_tip=np.array([200,200],   dtype=float),
        pinky_tip=np.array([220,220],  dtype=float),
        wrist=wrist,
        index_mcp=index_mcp,
        frame_h=h,
        frame_w=w,
    )


@patch("pyautogui.click")
def test_left_click_detected(mock_click):
    from src.gestures.recognizer import GestureRecognizer, Gesture
    config = {"gestures": {
        "click_threshold": 0.06,
        "drag_hold_seconds": 0.4,
        "scroll_sensitivity": 4.0,
        "scroll_deadzone": 0.01,
        "two_finger_threshold": 0.07,
        "cooldown_seconds": 0.0,   # no cooldown for tests
    }}
    rec  = GestureRecognizer(config)

    # Simulate pinch then release
    pinched  = make_hand(thumb_pos=(100,100), index_pos=(103,103))  # close
    released = make_hand(thumb_pos=(100,100), index_pos=(160,160))  # far

    rec.recognize(pinched)    # pinch down
    rec.recognize(released)   # release → click

    mock_click.assert_called_once()


def test_no_gesture_when_hand_open():
    from src.gestures.recognizer import GestureRecognizer, Gesture
    config = {"gestures": {
        "click_threshold": 0.06,
        "drag_hold_seconds": 0.4,
        "scroll_sensitivity": 4.0,
        "scroll_deadzone": 0.01,
        "two_finger_threshold": 0.07,
        "cooldown_seconds": 0.0,
    }}
    rec  = GestureRecognizer(config)
    hand = make_hand(thumb_pos=(100,100), index_pos=(200,200))  # far apart
    result = rec.recognize(hand)
    assert result == Gesture.NONE