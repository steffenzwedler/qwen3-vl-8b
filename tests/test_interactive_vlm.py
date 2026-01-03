"""
Unit tests for interactive_vlm module.

Tests cover interactive interface, command processing, and integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from interactive_vlm import InteractiveVLM


class TestInteractiveVLM:
    """Test suite for InteractiveVLM class."""

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_initialization(self, mock_capture_cls, mock_vlm_cls):
        """Test InteractiveVLM initializes correctly."""
        mock_capture = MagicMock()
        mock_vlm = MagicMock()
        mock_capture_cls.return_value = mock_capture
        mock_vlm_cls.return_value = mock_vlm

        app = InteractiveVLM()

        assert app.capture == mock_capture
        assert app.vlm == mock_vlm
        assert app.current_image is None
        assert app.current_window_title is None

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    @patch('interactive_vlm.print_available_windows')
    def test_list_windows(self, mock_print, mock_capture_cls, mock_vlm_cls):
        """Test list_windows calls print_available_windows."""
        app = InteractiveVLM()
        app.list_windows()

        mock_print.assert_called_once()

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_capture_window_by_title_success(self, mock_capture_cls, mock_vlm_cls):
        """Test capturing window by title successfully."""
        mock_image = Image.new('RGB', (800, 600))
        mock_capture = MagicMock()
        mock_capture.capture_window_by_title.return_value = mock_image
        mock_capture_cls.return_value = mock_capture

        app = InteractiveVLM()
        result = app.capture_window_by_title("Notepad")

        assert result is True
        assert app.current_image == mock_image
        assert app.current_window_title == "Notepad"
        mock_capture.capture_window_by_title.assert_called_once_with("Notepad", False)

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_capture_window_by_title_failure(self, mock_capture_cls, mock_vlm_cls):
        """Test capturing window by title fails gracefully."""
        mock_capture = MagicMock()
        mock_capture.capture_window_by_title.return_value = None
        mock_capture_cls.return_value = mock_capture

        app = InteractiveVLM()
        result = app.capture_window_by_title("NonExistent")

        assert result is False
        assert app.current_image is None

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_save_current_image_success(self, mock_capture_cls, mock_vlm_cls, tmp_path):
        """Test saving current image to file."""
        mock_image = Image.new('RGB', (100, 100))
        app = InteractiveVLM()
        app.current_image = mock_image

        filepath = tmp_path / "test.png"
        app.save_current_image(str(filepath))

        assert filepath.exists()

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_save_current_image_no_image(self, mock_capture_cls, mock_vlm_cls, capsys):
        """Test saving when no image is captured."""
        app = InteractiveVLM()
        app.save_current_image("test.png")

        captured = capsys.readouterr()
        assert "No image captured" in captured.out

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_ask_success(self, mock_capture_cls, mock_vlm_cls):
        """Test asking a question about the image."""
        mock_vlm = MagicMock()
        mock_vlm.generate.return_value = "This is a text editor"
        mock_vlm_cls.return_value = mock_vlm

        app = InteractiveVLM()
        app.current_image = Image.new('RGB', (100, 100))

        response = app.ask("What is this?")

        assert response == "This is a text editor"
        mock_vlm.generate.assert_called_once()

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_ask_no_image(self, mock_capture_cls, mock_vlm_cls, capsys):
        """Test asking when no image is captured."""
        app = InteractiveVLM()

        response = app.ask("What is this?")

        assert response == ""
        captured = capsys.readouterr()
        assert "No image captured" in captured.out

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_describe_success(self, mock_capture_cls, mock_vlm_cls):
        """Test describing the current image."""
        mock_vlm = MagicMock()
        mock_vlm.describe_screen.return_value = "A notepad window"
        mock_vlm_cls.return_value = mock_vlm

        app = InteractiveVLM()
        app.current_image = Image.new('RGB', (100, 100))

        response = app.describe()

        assert response == "A notepad window"
        mock_vlm.describe_screen.assert_called_once()

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_describe_no_image(self, mock_capture_cls, mock_vlm_cls, capsys):
        """Test describing when no image is captured."""
        app = InteractiveVLM()

        response = app.describe()

        assert response == ""
        captured = capsys.readouterr()
        assert "No image captured" in captured.out

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_find_element_success(self, mock_capture_cls, mock_vlm_cls):
        """Test finding a UI element."""
        mock_vlm = MagicMock()
        mock_vlm.analyze_ui.return_value = "Button at top-right"
        mock_vlm_cls.return_value = mock_vlm

        app = InteractiveVLM()
        app.current_image = Image.new('RGB', (100, 100))

        response = app.find_element("OK button")

        assert response == "Button at top-right"
        mock_vlm.analyze_ui.assert_called_once()

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_find_element_no_image(self, mock_capture_cls, mock_vlm_cls, capsys):
        """Test finding element when no image is captured."""
        app = InteractiveVLM()

        response = app.find_element("OK button")

        assert response == ""
        captured = capsys.readouterr()
        assert "No image captured" in captured.out

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_analyze_ui_success(self, mock_capture_cls, mock_vlm_cls):
        """Test analyzing UI elements."""
        mock_vlm = MagicMock()
        mock_vlm.analyze_ui.return_value = "Multiple buttons and text fields"
        mock_vlm_cls.return_value = mock_vlm

        app = InteractiveVLM()
        app.current_image = Image.new('RGB', (100, 100))

        response = app.analyze_ui()

        assert response == "Multiple buttons and text fields"
        mock_vlm.analyze_ui.assert_called_once()

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_analyze_ui_no_image(self, mock_capture_cls, mock_vlm_cls, capsys):
        """Test analyzing UI when no image is captured."""
        app = InteractiveVLM()

        response = app.analyze_ui()

        assert response == ""
        captured = capsys.readouterr()
        assert "No image captured" in captured.out

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_interactive_mode_quit_command(self, mock_capture_cls, mock_vlm_cls, monkeypatch):
        """Test interactive mode quit command."""
        inputs = iter(["quit"])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))

        app = InteractiveVLM()
        app.interactive_mode()

        # Should exit without error

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    @patch('interactive_vlm.print_available_windows')
    def test_interactive_mode_list_command(self, mock_print, mock_capture_cls, mock_vlm_cls, monkeypatch):
        """Test interactive mode list command."""
        inputs = iter(["list", "quit"])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))

        app = InteractiveVLM()
        app.interactive_mode()

        mock_print.assert_called()

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_interactive_mode_help_command(self, mock_capture_cls, mock_vlm_cls, monkeypatch, capsys):
        """Test interactive mode help command."""
        inputs = iter(["help", "quit"])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))

        app = InteractiveVLM()
        app.interactive_mode()

        captured = capsys.readouterr()
        assert "capture" in captured.out
        assert "describe" in captured.out

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_interactive_mode_memory_command_cuda(self, mock_capture_cls, mock_vlm_cls, monkeypatch, capsys):
        """Test interactive mode memory command with CUDA."""
        mock_vlm = MagicMock()
        mock_vlm.get_memory_usage.return_value = {
            "allocated": 2.0,
            "reserved": 3.0,
            "max_allocated": 4.0
        }
        mock_vlm_cls.return_value = mock_vlm

        inputs = iter(["memory", "quit"])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))

        app = InteractiveVLM()
        app.interactive_mode()

        captured = capsys.readouterr()
        assert "2.00 GB" in captured.out

    @patch('interactive_vlm.VLMInference')
    @patch('interactive_vlm.WindowCapture')
    def test_interactive_mode_clear_command(self, mock_capture_cls, mock_vlm_cls, monkeypatch):
        """Test interactive mode clear command."""
        mock_vlm = MagicMock()
        mock_vlm_cls.return_value = mock_vlm

        inputs = iter(["clear", "quit"])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))

        app = InteractiveVLM()
        app.interactive_mode()

        mock_vlm.clear_cache.assert_called_once()


class TestMain:
    """Test suite for main function."""

    @patch('interactive_vlm.InteractiveVLM')
    @patch('sys.argv', ['interactive_vlm.py', '--list-windows'])
    @patch('interactive_vlm.print_available_windows')
    def test_main_list_windows(self, mock_print, mock_app_cls):
        """Test main with --list-windows flag."""
        from interactive_vlm import main

        main()

        mock_print.assert_called_once()
        mock_app_cls.assert_not_called()

    @patch('interactive_vlm.InteractiveVLM')
    @patch('sys.argv', ['interactive_vlm.py'])
    def test_main_default(self, mock_app_cls):
        """Test main with default arguments."""
        from interactive_vlm import main

        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        main()

        mock_app_cls.assert_called_once()
        mock_app.interactive_mode.assert_called_once_with(window_title=None)

    @patch('interactive_vlm.InteractiveVLM')
    @patch('sys.argv', ['interactive_vlm.py', '--window', 'Notepad'])
    def test_main_with_window(self, mock_app_cls):
        """Test main with --window argument."""
        from interactive_vlm import main

        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        main()

        mock_app.interactive_mode.assert_called_once_with(window_title='Notepad')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
