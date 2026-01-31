"""
Tests for Input Utils
"""
import pytest
from unittest.mock import patch, MagicMock
from src.input_utils import click_at, move_to

@patch("src.input_utils.pyautogui")
def test_click_at(mock_gui):
    # Test valid click
    success = click_at(100, 200, 'left')
    assert success is True
    mock_gui.moveTo.assert_called_with(100, 200)
    mock_gui.click.assert_called_with(button='left')

@patch("src.input_utils.pyautogui")
def test_click_at_failure(mock_gui):
    # Test exception handling (e.g. failsafe or permissions)
    mock_gui.moveTo.side_effect = Exception("Failsafe triggered")
    
    success = click_at(0, 0)
    assert success is False

@patch("src.input_utils.pyautogui")
def test_move_to(mock_gui):
    move_to(500, 500)
    mock_gui.moveTo.assert_called_with(500, 500)
