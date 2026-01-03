"""
Windows Screen Capture Module

This module provides functionality to capture screenshots of Windows applications
using multiple capture methods for maximum compatibility:
- MSS: Standard GDI windows (browsers, Office, etc.)
- DXCam: DirectX/OpenGL/Vulkan games and GPU-accelerated applications (10-20x faster)
- Fallback chain ensures broad compatibility
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

# Try to import DXCam for GPU-accelerated capture
try:
    import dxcam
    DXCAM_AVAILABLE = True
    logger.info("DXCam available - GPU-accelerated capture enabled for DirectX/OpenGL windows")
except ImportError:
    DXCAM_AVAILABLE = False
    logger.warning("DXCam not available - DirectX/OpenGL capture will use slower MSS fallback")


class WindowCapture:
    """
    Captures screenshots of Windows applications efficiently.

    This class provides methods to enumerate windows, capture specific windows,
    and retrieve screenshots as PIL Images or numpy arrays.

    Supports multiple capture backends:
    - DXCam: GPU-native capture for DirectX/OpenGL/Vulkan (10-20x faster)
    - MSS: Standard GDI capture for regular windows
    """

    def __init__(self, prefer_dxcam: bool = True, dxcam_target_fps: int = 60):
        """
        Initialize the WindowCapture instance.

        Args:
            prefer_dxcam: Prefer DXCam over MSS when available (faster for GPU-rendered windows)
            dxcam_target_fps: Target FPS for DXCam capture (higher = lower latency)
        """
        self.mss_instance = mss.mss()
        self.prefer_dxcam = prefer_dxcam and DXCAM_AVAILABLE
        self.dxcam_camera = None

        # Initialize DXCam if available and preferred
        if self.prefer_dxcam:
            try:
                self.dxcam_camera = dxcam.create(output_idx=0, output_color="RGB")
                if self.dxcam_camera:
                    logger.info("DXCam initialized successfully for GPU-accelerated capture")
            except Exception as e:
                logger.warning(f"Failed to initialize DXCam: {e}")
                self.prefer_dxcam = False

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

    @staticmethod
    def _is_likely_gpu_rendered(hwnd: int) -> bool:
        """
        Heuristic to detect if window likely uses DirectX/OpenGL/Vulkan.

        Args:
            hwnd: Window handle

        Returns:
            bool: True if window likely GPU-rendered
        """
        try:
            class_name = win32gui.GetClassName(hwnd)

            # Known GPU-rendered window classes
            gpu_classes = [
                'UnityWndClass',      # Unity games
                'UnrealWindow',        # Unreal Engine
                'SDL_app',             # SDL applications
                'GLFW',                # GLFW applications
                'CryENGINE',          # CryEngine
                'Qt5',                 # Qt applications (often GPU-accelerated)
                'Chrome_WidgetWin',   # Chrome (uses GPU acceleration)
            ]

            for gpu_class in gpu_classes:
                if gpu_class in class_name:
                    return True

            # Additional heuristics could be added here
            return False

        except Exception:
            return False

    def capture_window(self, hwnd: int, bring_to_front: bool = False, method: str = 'auto') -> Optional[Image.Image]:
        """
        Capture a screenshot of a specific window.

        Args:
            hwnd: Window handle to capture
            bring_to_front: If True, bring window to foreground before capture
            method: Capture method - 'auto', 'dxcam', 'mss'

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

        # Optionally bring window to foreground
        if bring_to_front:
            try:
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.05)  # Reduced delay
            except Exception as e:
                logger.warning(f"Could not bring window to foreground: {e}")

        # Determine capture method
        use_dxcam = False
        if method == 'auto':
            use_dxcam = self.prefer_dxcam and self._is_likely_gpu_rendered(hwnd)
        elif method == 'dxcam':
            use_dxcam = self.prefer_dxcam

        # Try DXCam first if requested and available
        if use_dxcam and self.dxcam_camera:
            try:
                region = (left, top, right, bottom)
                frame = self.dxcam_camera.grab(region=region)

                if frame is not None:
                    img = Image.fromarray(frame)
                    logger.debug(f"Captured using DXCam (GPU-accelerated)")
                    return img
                else:
                    logger.debug("DXCam returned None, falling back to MSS")
            except Exception as e:
                logger.debug(f"DXCam capture failed: {e}, falling back to MSS")

        # Fallback to MSS
        try:
            monitor = {
                "left": left,
                "top": top,
                "width": width,
                "height": height
            }

            screenshot = self.mss_instance.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            logger.debug(f"Captured using MSS")

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
        # Close MSS
        if hasattr(self, 'mss_instance') and self.mss_instance:
            try:
                self.mss_instance.close()
            except Exception as e:
                logger.error(f"Error closing mss instance: {e}")
            finally:
                self.mss_instance = None

        # Close DXCam
        if hasattr(self, 'dxcam_camera') and self.dxcam_camera:
            try:
                self.dxcam_camera.stop()
                self.dxcam_camera = None
            except Exception as e:
                logger.error(f"Error closing dxcam camera: {e}")

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
