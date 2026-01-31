"""
Coordinate Utilities Module

This module provides functions for parsing bounding boxes from model output
and converting them between normalized (0-1000) and pixel coordinates.
"""

from typing import List, Tuple, Union
import re
import math

def parse_bounding_boxes(text: str) -> List[Tuple[int, int, int, int]]:
    """
    Parse bounding boxes from text output.
    
    Supports formats:
    - <box>x1 y1 x2 y2</box>
    - [x1, y1, x2, y2]
    
    Args:
        text: The model generated text containing coordinates.
        
    Returns:
        List of (x1, y1, x2, y2) tuples in normalized 0-1000 space.
    """
    boxes = []
    
    # Regex for <box>x1 y1 x2 y2</box>
    # Flexibly handle spaces and potential float values (though usually int)
    box_pattern = r'<box>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</box>'
    matches = re.finditer(box_pattern, text)
    for match in matches:
        try:
            coords = tuple(int(g) for g in match.groups())
            boxes.append(coords)
        except ValueError:
            pass # Should be covered by regex \d+, but safety first
            
    # Regex for [x1, y1, x2, y2]
    list_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    matches_list = re.finditer(list_pattern, text)
    for match in matches_list:
        try:
            coords = tuple(int(g) for g in match.groups())
            boxes.append(coords)
        except ValueError:
            pass
            
    # Regex for (ymin,xmin),(ymax,xmax) - Standard Qwen2-VL format
    # This also handles the case where there are no brackets
    qwen_pattern = r'\((\d+),(\d+)\),\((\d+),(\d+)\)'
    matches_qwen = re.finditer(qwen_pattern, text)
    for match in matches_qwen:
        try:
            coords = tuple(int(g) for g in match.groups())
            boxes.append(coords)
        except ValueError:
            pass
            
    return boxes

def normalize_coordinates(
    box: Tuple[int, int, int, int], 
    width: int, 
    height: int,
    original_scale: int = 1000,
    format: str = 'ymin_xmin_ymax_xmax'
) -> Tuple[int, int, int, int]:
    """
    Convert normalized coordinates (0-1000) to pixel coordinates.
    
    Args:
        box: (v1, v2, v3, v4) in original_scale
        width: Target image width in pixels
        height: Target image height in pixels
        original_scale: The scale of the input box (default 1000)
        format: 'x1_y1_x2_y2' or 'ymin_xmin_ymax_xmax'
        
    Returns:
        (x1, y1, x2, y2) in pixels, clamped to image dimensions.
    """
    v1, v2, v3, v4 = box
    
    if format == 'ymin_xmin_ymax_xmax':
        ymin, xmin, ymax, xmax = v1, v2, v3, v4
    else: # x1, y1, x2, y2
        xmin, ymin, xmax, ymax = v1, v2, v3, v4
    
    # helper for clamping and scaling
    def scale(val, max_size):
        # Clamp to scale
        val = max(0, min(val, original_scale))
        # Scale
        return int((val / original_scale) * max_size)
        
    px1 = scale(xmin, width)
    py1 = scale(ymin, height)
    px2 = scale(xmax, width)
    py2 = scale(ymax, height)
    
    return (px1, py1, px2, py2)

def get_centroid(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate the center point of a bounding box.
    
    Args:
        box: (x1, y1, x2, y2)
        
    Returns:
        (x, y) center point
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx, cy)
