"""
Windows Screen Capture Module

This module provides functionality to capture screenshots of Windows applications
using the Windows API through pywin32 and mss for efficient screen capture.
"""

import ctypes
import ctypes.wintypes
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import mss
import win32gui
import win32con
import win32ui
import logging
import time

logger = logging.getLogger(__name__)


class WindowCapture:
    """
    Captures screenshots of Windows applications efficiently.

    This class provides methods to enumerate windows, capture specific windows,
    and retrieve screenshots as PIL Images or numpy arrays.
    """

    def __init__(self):
        """Initialize the WindowCapture instance."""
        self.mss_instance = mss.mss()

    @staticmethod
    def list_windows() -> List[Tuple[int, str]]:
        """
        List all visible windows with their handles and titles.

        Returns:
            List[Tuple[int, str]]: List of (window_handle, window_title) tuples
        """
        windows = []

        def enum_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    windows.append((hwnd, title))
            return True

        win32gui.EnumWindows(enum_callback, None)
        return windows

    @staticmethod
    def get_window_by_title(title: str, exact_match: bool = False) -> Optional[int]:
        """
        Find a window handle by its title.

        Args:
            title: The window title to search for
            exact_match: If True, requires exact match; otherwise uses substring match

        Returns:
            Optional[int]: Window handle if found, None otherwise
        """
        if not title or not isinstance(title, str):
            logger.error("Invalid title parameter")
            return None

        if len(title) > 256:
            logger.warning(f"Title too long, truncating: {title[:50]}...")
            title = title[:256]

        windows = WindowCapture.list_windows()

        for hwnd, window_title in windows:
            if exact_match:
                if window_title == title:
                    return hwnd
            else:
                if title.lower() in window_title.lower():
                    return hwnd

        return None

    @staticmethod
    def get_window_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the bounding rectangle of a window.

        Args:
            hwnd: Window handle

        Returns:
            Optional[Tuple[int, int, int, int]]: (left, top, right, bottom) or None if failed
        """
        if not isinstance(hwnd, int) or hwnd <= 0:
            logger.error(f"Invalid window handle: {hwnd}")
            return None

        try:
            rect = win32gui.GetWindowRect(hwnd)
            return rect
        except win32gui.error as e:
            logger.error(f"Win32 error getting window rect for handle {hwnd}: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error getting window rect: {e}")
            return None

    def capture_window(self, hwnd: int, bring_to_front: bool = False) -> Optional[Image.Image]:
        """
        Capture a screenshot of a specific window.

        Args:
            hwnd: Window handle to capture
            bring_to_front: If True, bring window to foreground before capture

        Returns:
            Optional[Image.Image]: PIL Image of the window or None if failed
        """
        rect = self.get_window_rect(hwnd)
        if not rect:
            return None

        left, top, right, bottom = rect
        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            logger.error(f"Invalid window dimensions: {width}x{height}")
            return None

        try:
            # Optionally bring window to foreground
            if bring_to_front:
                try:
                    win32gui.SetForegroundWindow(hwnd)
                    time.sleep(0.1)  # Brief delay for window to render
                except Exception as e:
                    logger.warning(f"Could not bring window to foreground: {e}")

            # Use mss for efficient screen capture
            monitor = {
                "left": left,
                "top": top,
                "width": width,
                "height": height
            }

            screenshot = self.mss_instance.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

            return img

        except Exception as e:
            logger.error(f"Error capturing window: {e}")
            return None

    def capture_window_by_title(self, title: str, exact_match: bool = False, bring_to_front: bool = False) -> Optional[Image.Image]:
        """
        Capture a screenshot of a window by its title.

        Args:
            title: The window title to search for
            exact_match: If True, requires exact match; otherwise uses substring match
            bring_to_front: If True, bring window to foreground before capture

        Returns:
            Optional[Image.Image]: PIL Image of the window or None if not found
        """
        hwnd = self.get_window_by_title(title, exact_match)
        if not hwnd:
            logger.warning(f"Window with title '{title}' not found")
            return None

        return self.capture_window(hwnd, bring_to_front=bring_to_front)

    def capture_to_array(self, hwnd: int) -> Optional[np.ndarray]:
        """
        Capture a window and return as numpy array.

        Args:
            hwnd: Window handle to capture

        Returns:
            Optional[np.ndarray]: RGB image as numpy array (H, W, 3) or None if failed
        """
        img = self.capture_window(hwnd)
        if img is None:
            return None

        return np.array(img)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def close(self):
        """Explicitly close resources."""
        if hasattr(self, 'mss_instance') and self.mss_instance:
            try:
                self.mss_instance.close()
            except Exception as e:
                logger.error(f"Error closing mss instance: {e}")
            finally:
                self.mss_instance = None

    def __del__(self):
        """Clean up MSS instance."""
        self.close()


def print_available_windows():
    """Print all available windows for debugging purposes."""
    windows = WindowCapture.list_windows()
    print(f"\nFound {len(windows)} visible windows:")
    print("-" * 80)
    for hwnd, title in windows:
        print(f"Handle: {hwnd:8d} | Title: {title}")
    print("-" * 80)


if __name__ == "__main__":
    print_available_windows()
