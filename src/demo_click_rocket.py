"""
Demo: Click Rocket Icon

This script allows you to enter a window title, and it will attempts to:
1. Capture the window
2. Ask the VLM to find an icon that looks like a rocket
3. Parse the coordinates
4. Click on the center of the icon
"""

import asyncio
import logging
from src.window_capture import WindowCapture
from src.vlm_inference import VLMInference
from src.input_utils import click_at
import win32gui
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rocket_clicker")

async def run_rocket_click(window_title: str):
    print(f"\n--- Rocket Click Demo ---")
    print(f"Target Window: {window_title}")
    
    # 1. Initialize VLM
    # Note: running this script initializes its OWN VLM instance (heavy!). 
    # In a real agent scenario, you'd talk to the running MCP server.
    # But user asked to "run the full solution", which could mean this script acts as the agent.
    
    logger.info("Initializing VLM...")
    # Force 4-bit to prevent OOM on typical dev machines
    # Disable flash_attention since the package is not installed (use SDPA instead)
    vlm = VLMInference(load_in_4bit=True, use_flash_attention=False)
    
    # 2. Capture Window
    capturer = WindowCapture()
    hwnd = capturer.get_window_by_title(window_title)
    if not hwnd:
        logger.error(f"Window '{window_title}' not found!")
        WindowCapture.print_available_windows()
        return

    logger.info(f"Capturing window: {window_title} (HWND: {hwnd})")
    img = capturer.capture_window(hwnd, bring_to_front=True)
    if not img:
        logger.error("Failed to capture window.")
        return
    
    # Debug: save captured image
    img.save("debug_capture.png")
    logger.info("Saved captured image to debug_capture.png")
        
    # 3. Analyze & Find Rocket
    logger.info("Analyzing screen for 'rocket icon'...")
    
    # Test with Source Control icon to check for precision
    result = vlm.detect_element(img, "Source Control icon")
    
    if not result["found"]:
        logger.warning("Could not find a rocket icon on the screen.")
        print(f"Model Raw Output: {result.get('raw_output')}")
        return
        
    print(f"Full Model Output: {result.get('raw_output')}")
    logger.info(f"Rocket found at normalized box: {result['box_normalized']}")
    
    # 4. Calculate Absolute Click Coordinates
    # Coordinates from detect_element are relative to the *captured image*.
    cx, cy = result['center_pixels']
    
    rect = win32gui.GetWindowRect(hwnd)
    win_x, win_y = rect[0], rect[1]
    
    abs_x = win_x + cx
    abs_y = win_y + cy
    
    print(f"Window Pos: ({win_x}, {win_y})")
    print(f"Icon Rel Pos: ({cx}, {cy})")
    print(f"Clicking at Absolute: ({abs_x}, {abs_y})")
    
    # 5. Click
    # Safety delay
    await asyncio.sleep(0.5)
    click_at(abs_x, abs_y)
    print("Click sent!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, help="Window title to capture", default=None)
    args = parser.parse_args()

    target = args.title
    if not target:
        try:
             target = input("Enter partial window title to search for rocket in (e.g. 'Browser'): ")
        except EOFError:
             target = "Code" # Default for automation if input fails

    if target:
        asyncio.run(run_rocket_click(target))
