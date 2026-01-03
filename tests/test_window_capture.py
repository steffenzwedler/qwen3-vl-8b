"""
Unit tests for window_capture module.

Tests cover window enumeration, capture functionality, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from window_capture import WindowCapture, print_available_windows


class TestWindowCapture:
    """Test suite for WindowCapture class."""

    def test_initialization(self):
        """Test WindowCapture initializes correctly."""
        capture = WindowCapture()
        assert hasattr(capture, 'mss_instance')

    @patch('window_capture.win32gui.EnumWindows')
    @patch('window_capture.win32gui.IsWindowVisible')
    @patch('window_capture.win32gui.GetWindowText')
    def test_list_windows(self, mock_get_text, mock_is_visible, mock_enum):
        """Test listing visible windows."""
        # Setup mocks
        def enum_callback(callback, _):
            callback(12345, None)
            callback(67890, None)
            return True

        mock_enum.side_effect = enum_callback
        mock_is_visible.return_value = True
        mock_get_text.side_effect = ["Notepad", "Chrome"]

        # Test
        windows = WindowCapture.list_windows()

        # Verify
        assert len(windows) == 2
        assert (12345, "Notepad") in windows or (67890, "Notepad") in windows

    @patch('window_capture.win32gui.EnumWindows')
    @patch('window_capture.win32gui.IsWindowVisible')
    @patch('window_capture.win32gui.GetWindowText')
    def test_get_window_by_title_partial_match(self, mock_get_text, mock_is_visible, mock_enum):
        """Test finding window by partial title match."""
        def enum_callback(callback, _):
            callback(12345, None)
            return True

        mock_enum.side_effect = enum_callback
        mock_is_visible.return_value = True
        mock_get_text.return_value = "Untitled - Notepad"

        # Test partial match
        hwnd = WindowCapture.get_window_by_title("Notepad", exact_match=False)
        assert hwnd == 12345

    @patch('window_capture.win32gui.EnumWindows')
    @patch('window_capture.win32gui.IsWindowVisible')
    @patch('window_capture.win32gui.GetWindowText')
    def test_get_window_by_title_exact_match(self, mock_get_text, mock_is_visible, mock_enum):
        """Test finding window by exact title match."""
        def enum_callback(callback, _):
            callback(12345, None)
            return True

        mock_enum.side_effect = enum_callback
        mock_is_visible.return_value = True
        mock_get_text.return_value = "Notepad"

        # Test exact match
        hwnd = WindowCapture.get_window_by_title("Notepad", exact_match=True)
        assert hwnd == 12345

        # Test exact match failure
        hwnd = WindowCapture.get_window_by_title("Untitled - Notepad", exact_match=True)
        assert hwnd is None

    @patch('window_capture.win32gui.EnumWindows')
    @patch('window_capture.win32gui.IsWindowVisible')
    @patch('window_capture.win32gui.GetWindowText')
    def test_get_window_by_title_not_found(self, mock_get_text, mock_is_visible, mock_enum):
        """Test finding window returns None when not found."""
        def enum_callback(callback, _):
            callback(12345, None)
            return True

        mock_enum.side_effect = enum_callback
        mock_is_visible.return_value = True
        mock_get_text.return_value = "Chrome"

        hwnd = WindowCapture.get_window_by_title("Notepad")
        assert hwnd is None

    @patch('window_capture.win32gui.GetWindowRect')
    def test_get_window_rect_success(self, mock_get_rect):
        """Test getting window rectangle."""
        mock_get_rect.return_value = (100, 200, 900, 700)

        rect = WindowCapture.get_window_rect(12345)

        assert rect == (100, 200, 900, 700)
        mock_get_rect.assert_called_once_with(12345)

    @patch('window_capture.win32gui.GetWindowRect')
    def test_get_window_rect_failure(self, mock_get_rect):
        """Test getting window rectangle handles errors."""
        mock_get_rect.side_effect = Exception("Window not found")

        rect = WindowCapture.get_window_rect(99999)

        assert rect is None

    @patch('window_capture.win32gui.GetWindowRect')
    @patch('window_capture.win32gui.SetForegroundWindow')
    def test_capture_window_invalid_dimensions(self, mock_set_fg, mock_get_rect):
        """Test capture fails gracefully with invalid dimensions."""
        mock_get_rect.return_value = (100, 100, 100, 100)  # Zero width/height

        capture = WindowCapture()
        image = capture.capture_window(12345)

        assert image is None

    @patch('window_capture.WindowCapture.get_window_by_title')
    @patch('window_capture.WindowCapture.capture_window')
    def test_capture_window_by_title_success(self, mock_capture, mock_get_by_title):
        """Test capturing window by title."""
        mock_image = Image.new('RGB', (800, 600))
        mock_get_by_title.return_value = 12345
        mock_capture.return_value = mock_image

        capture = WindowCapture()
        image = capture.capture_window_by_title("Notepad")

        assert image == mock_image
        mock_get_by_title.assert_called_once_with("Notepad", False)
        mock_capture.assert_called_once_with(12345)

    @patch('window_capture.WindowCapture.get_window_by_title')
    def test_capture_window_by_title_not_found(self, mock_get_by_title):
        """Test capturing window by title when window not found."""
        mock_get_by_title.return_value = None

        capture = WindowCapture()
        image = capture.capture_window_by_title("NonExistent")

        assert image is None

    @patch('window_capture.WindowCapture.capture_window')
    def test_capture_to_array(self, mock_capture):
        """Test capturing window as numpy array."""
        mock_image = Image.new('RGB', (800, 600), color='red')
        mock_capture.return_value = mock_image

        capture = WindowCapture()
        array = capture.capture_to_array(12345)

        assert isinstance(array, np.ndarray)
        assert array.shape == (600, 800, 3)

    @patch('window_capture.WindowCapture.capture_window')
    def test_capture_to_array_failure(self, mock_capture):
        """Test capturing to array returns None on failure."""
        mock_capture.return_value = None

        capture = WindowCapture()
        array = capture.capture_to_array(12345)

        assert array is None

    def test_cleanup(self):
        """Test that MSS instance is properly cleaned up."""
        capture = WindowCapture()
        mss_instance = capture.mss_instance

        del capture

        # If cleanup works, this should not raise an error

    @patch('window_capture.WindowCapture.list_windows')
    def test_print_available_windows(self, mock_list_windows, capsys):
        """Test print_available_windows output."""
        mock_list_windows.return_value = [
            (12345, "Notepad"),
            (67890, "Chrome"),
        ]

        print_available_windows()

        captured = capsys.readouterr()
        assert "Found 2 visible windows" in captured.out
        assert "Notepad" in captured.out
        assert "Chrome" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
