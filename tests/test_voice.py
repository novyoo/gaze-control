# tests/test_voice.py
# Tests commander logic — no mic needed

from unittest.mock import patch, MagicMock
import pytest
from src.voice.commander import VoiceCommander


@patch("webbrowser.open")
def test_search_command(mock_browser):
    cmd = VoiceCommander()
    result = cmd.process("search python tutorials")
    assert result.success
    assert "python+tutorials" in result.action
    mock_browser.assert_called_once()


@patch("subprocess.Popen")
def test_open_notepad(mock_popen):
    cmd = VoiceCommander()
    result = cmd.process("open notepad")
    assert result.success
    mock_popen.assert_called_once()


def test_unknown_command_returns_no_match():
    cmd = VoiceCommander()
    result = cmd.process("blah blah blah")
    assert result is not None
    assert result.action == "no match"


def test_pause_and_resume():
    cmd = VoiceCommander()
    cmd.process("stop listening")
    assert cmd.is_paused
    cmd.process("start listening")
    assert not cmd.is_paused


@patch("pyautogui.press")
def test_volume_up(mock_press):
    cmd = VoiceCommander()
    result = cmd.process("volume up")
    assert result.success
    assert mock_press.call_count == 5