"""
Tests for Coordinate Utilities
"""
import pytest
from src.coord_utils import parse_bounding_boxes, normalize_coordinates, get_centroid

class TestCoordUtils:
    
    def test_parse_qwen_box_format(self):
        # Qwen-VL typically uses <box> and </box> with normalized 0-1000 coords
        # Format often: <box>start_y start_x end_y end_x</box> OR <box>x1 y1 x2 y2</box>
        # We need to be careful about the order. 
        # Standard Qwen-VL: <box>(y1,x1),(y2,x2)</box> ? 
        # Actually Qwen2-VL usually outputs: <|box_start|>(y1,x1),(y2,x2)<|box_end|>
        # But let's assume we prompt for specific format or handle standard regex.
        # Let's target the Plan's regex: <box>x1 y1 x2 y2</box>
        
        text = "Found the button at <box>100 200 300 400</box>."
        boxes = parse_bounding_boxes(text)
        assert len(boxes) == 1
        assert boxes[0] == (100, 200, 300, 400) # x1, y1, x2, y2
        
    def test_parse_bracket_format(self):
        text = "Location: [100, 200, 300, 400]"
        boxes = parse_bounding_boxes(text)
        assert len(boxes) == 1
        assert boxes[0] == (100, 200, 300, 400)

    def test_parse_multiple_boxes(self):
        text = "<box>10 10 20 20</box> and <box>50 50 60 60</box>"
        boxes = parse_bounding_boxes(text)
        assert len(boxes) == 2
        assert boxes[0] == (10, 10, 20, 20)
        assert boxes[1] == (50, 50, 60, 60)

    def test_normalize_coordinates(self):
        # 0-1000 scale
        # Image 1920x1080
        # Point 500, 500 (Center) -> 960, 540
        
        box = (500, 500, 500, 500)
        pixel_box = normalize_coordinates(box, 1920, 1080)
        assert pixel_box == (960, 540, 960, 540)
        
        # Test 0,0
        box = (0, 0, 100, 100)
        pixel_box = normalize_coordinates(box, 1000, 1000) # Square 1000x1000 img
        assert pixel_box == (0, 0, 100, 100)

    def test_clamping(self):
        # Coord > 1000
        box = (1500, 1000, 2000, 1000)
        pixel_box = normalize_coordinates(box, 100, 100)
        assert pixel_box[0] == 100 # Clamped
        assert pixel_box[2] == 100 # Clamped

    def test_get_centroid(self):
        box = (100, 100, 200, 200) # Width 100, Height 100
        center = get_centroid(box)
        assert center == (150, 150)
