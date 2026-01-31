"""
Input Utilities Module

This module provides functions for controlling mouse and keyboard input.
"""

import pyautogui
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Fail-safe mode (moving mouse to corner will throw exception)
pyautogui.FAILSAFE = True

def click_at(x: int, y: int, button: str = 'left'):
    """
    Perform a mouse click at the specified coordinates.
    
    Args:
        x: X coordinate (absolute pixel)
        y: Y coordinate (absolute pixel)
        button: 'left', 'right', or 'middle'
    """
    try:
        # Move first, then click
        pyautogui.moveTo(x, y)
        pyautogui.click(button=button)
        logger.info(f"Clicked {button} at ({x}, {y})")
        return True
    except Exception as e:
        logger.error(f"Failed to click at ({x}, {y}): {e}")
        return False

def move_to(x: int, y: int):
    """Move mouse to coordinates."""
    pyautogui.moveTo(x, y)
    logger.info(f"Moved mouse to ({x}, {y})")
