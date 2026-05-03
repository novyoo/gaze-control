# tests/test_controller.py
# Note: these tests mock pyautogui so no cursor actually moves

from unittest.mock import patch, MagicMock
import pytest


@patch("pyautogui.moveTo")
@patch("pyautogui.position", return_value=(960, 540))
@patch("pyautogui.size",     return_value=(1920, 1080))
def test_cursor_moves_when_enabled(mock_size, mock_pos, mock_move):
    from src.cursor.controller import CursorController
    config = {"cursor": {"smooth_factor": 1.0, "deadzone_radius": 0}}
    ctrl = CursorController(config)
    ctrl.move(500, 300)
    mock_move.assert_called_once()


@patch("pyautogui.moveTo")
@patch("pyautogui.position", return_value=(960, 540))
@patch("pyautogui.size",     return_value=(1920, 1080))
def test_cursor_does_not_move_when_disabled(mock_size, mock_pos, mock_move):
    from src.cursor.controller import CursorController
    config = {"cursor": {"smooth_factor": 1.0, "deadzone_radius": 0}}
    ctrl = CursorController(config)
    ctrl.disable()
    ctrl.move(500, 300)
    mock_move.assert_not_called()


@patch("pyautogui.moveTo")
@patch("pyautogui.position", return_value=(100, 100))
@patch("pyautogui.size",     return_value=(1920, 1080))
def test_deadzone_blocks_small_movement(mock_size, mock_pos, mock_move):
    from src.cursor.controller import CursorController
    config = {"cursor": {"smooth_factor": 1.0, "deadzone_radius": 50}}
    ctrl = CursorController(config)
    # Move to a point within 50px of current position (100,100)
    ctrl._smooth_x = 100.0
    ctrl._smooth_y = 100.0
    ctrl.move(105, 105)
    mock_move.assert_not_called()