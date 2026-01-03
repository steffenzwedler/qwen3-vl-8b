"""
Interactive VLM Screen Analyzer

This module provides an interactive command-line interface for analyzing
Windows application screenshots using the Qwen3-VL-8B model.
"""

import argparse
import sys
from typing import Optional
from pathlib import Path
from PIL import Image
import logging
import traceback

try:
    from window_capture import WindowCapture, print_available_windows
    from vlm_inference import VLMInference
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)

logger = logging.getLogger(__name__)


class InteractiveVLM:
    """
    Interactive interface for screen capture and VLM analysis.

    Provides a command-line interface to capture Windows application screenshots
    and interact with the VLM model to analyze, describe, or identify UI elements.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        use_flash_attention: bool = True,
        debug_mode: bool = False,
    ):
        """
        Initialize the interactive VLM interface.

        Args:
            model_name: HuggingFace model identifier
            use_flash_attention: Whether to use Flash Attention 2
            debug_mode: Enable debug output for errors
        """
        logger.info("Initializing Interactive VLM Screen Analyzer...")
        self.capture = WindowCapture()
        self.vlm = VLMInference(
            model_name=model_name,
            use_flash_attention=use_flash_attention
        )
        self.current_image: Optional[Image.Image] = None
        self.current_window_title: Optional[str] = None
        self.debug_mode = debug_mode

    def list_windows(self):
        """List all available windows."""
        print_available_windows()

    def capture_window_by_title(self, title: str, exact_match: bool = False) -> bool:
        """
        Capture a window by its title.

        Args:
            title: Window title to search for
            exact_match: Whether to require exact title match

        Returns:
            bool: True if capture successful, False otherwise
        """
        print(f"Searching for window: '{title}'...")
        image = self.capture.capture_window_by_title(title, exact_match)

        if image is None:
            print(f"Failed to capture window: '{title}'")
            print("\nAvailable windows:")
            self.list_windows()
            return False

        self.current_image = image
        self.current_window_title = title
        print(f"Successfully captured window: '{title}' ({image.size[0]}x{image.size[1]})")
        return True

    def save_current_image(self, filepath: str):
        """
        Save the current captured image to a file with security checks.

        Args:
            filepath: Path where the image should be saved
        """
        if self.current_image is None:
            print("No image captured yet. Capture a window first.")
            return

        try:
            # Convert to Path object
            save_path = Path(filepath).resolve()

            # Validate file extension
            allowed_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
            if save_path.suffix.lower() not in allowed_formats:
                print(f"Unsupported format. Use one of: {', '.join(allowed_formats)}")
                return

            # Check if file exists and confirm overwrite
            if save_path.exists():
                response = input(f"File {save_path.name} exists. Overwrite? (yes/no): ")
                if response.lower() not in ['yes', 'y']:
                    print("Save cancelled")
                    return

            # Create parent directory if needed
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save image
            self.current_image.save(save_path)
            print(f"Image saved to: {save_path}")

        except PermissionError:
            print(f"Permission denied: Cannot write to {filepath}")
        except OSError as e:
            print(f"Error saving image: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error saving image: {e}")
            print(f"Failed to save image: {e}")

    def ask(self, prompt: str) -> str:
        """
        Ask a question about the current captured image.

        Args:
            prompt: Question or instruction for the VLM

        Returns:
            str: VLM response
        """
        if self.current_image is None:
            print("No image captured yet. Capture a window first.")
            return ""

        print(f"\nProcessing prompt: {prompt}")
        print("-" * 80)

        response = self.vlm.generate(
            image=self.current_image,
            prompt=prompt,
        )

        print(f"Response: {response}")
        print("-" * 80)
        return response

    def describe(self) -> str:
        """Describe what's visible in the current screenshot."""
        if self.current_image is None:
            print("No image captured yet. Capture a window first.")
            return ""

        print("\nDescribing screenshot...")
        print("-" * 80)

        response = self.vlm.describe_screen(self.current_image)

        print(f"Description: {response}")
        print("-" * 80)
        return response

    def find_element(self, element_description: str) -> str:
        """
        Find and describe a UI element.

        Args:
            element_description: Description of the UI element to find

        Returns:
            str: Location and description of the element
        """
        if self.current_image is None:
            print("No image captured yet. Capture a window first.")
            return ""

        print(f"\nSearching for UI element: {element_description}")
        print("-" * 80)

        response = self.vlm.analyze_ui(
            image=self.current_image,
            element_description=element_description
        )

        print(f"Element analysis: {response}")
        print("-" * 80)
        return response

    def analyze_ui(self) -> str:
        """Analyze all UI elements in the current screenshot."""
        if self.current_image is None:
            print("No image captured yet. Capture a window first.")
            return ""

        print("\nAnalyzing UI elements...")
        print("-" * 80)

        response = self.vlm.analyze_ui(self.current_image)

        print(f"UI Analysis: {response}")
        print("-" * 80)
        return response

    def interactive_mode(self, window_title: Optional[str] = None):
        """
        Start interactive mode with a command prompt.

        Args:
            window_title: Optional window title to capture initially
        """
        print("\n" + "=" * 80)
        print("Interactive VLM Screen Analyzer")
        print("=" * 80)

        if window_title:
            self.capture_window_by_title(window_title)

        print("\nCommands:")
        print("  capture <title>  - Capture window by title (partial match)")
        print("  list             - List all available windows")
        print("  describe         - Describe the current screenshot")
        print("  ask <prompt>     - Ask a question about the screenshot")
        print("  find <element>   - Find a specific UI element")
        print("  analyze          - Analyze all UI elements")
        print("  save <path>      - Save current screenshot to file")
        print("  memory           - Show GPU memory usage")
        print("  clear            - Clear GPU cache")
        print("  help             - Show this help message")
        print("  quit             - Exit the program")
        print("")

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()

                if command == "quit" or command == "exit":
                    print("Exiting...")
                    break

                elif command == "help":
                    print("\nCommands:")
                    print("  capture <title>  - Capture window by title")
                    print("  list             - List all available windows")
                    print("  describe         - Describe the current screenshot")
                    print("  ask <prompt>     - Ask a question about the screenshot")
                    print("  find <element>   - Find a specific UI element")
                    print("  analyze          - Analyze all UI elements")
                    print("  save <path>      - Save current screenshot")
                    print("  memory           - Show GPU memory usage")
                    print("  clear            - Clear GPU cache")
                    print("  quit             - Exit")

                elif command == "list":
                    self.list_windows()

                elif command == "capture":
                    if len(parts) < 2:
                        print("Usage: capture <window_title>")
                    else:
                        self.capture_window_by_title(parts[1])

                elif command == "describe":
                    self.describe()

                elif command == "ask":
                    if len(parts) < 2:
                        print("Usage: ask <your question>")
                    else:
                        self.ask(parts[1])

                elif command == "find":
                    if len(parts) < 2:
                        print("Usage: find <element description>")
                    else:
                        self.find_element(parts[1])

                elif command == "analyze":
                    self.analyze_ui()

                elif command == "save":
                    if len(parts) < 2:
                        print("Usage: save <filepath>")
                    else:
                        self.save_current_image(parts[1])

                elif command == "memory":
                    memory = self.vlm.get_memory_usage()
                    if memory:
                        print("\nGPU Memory Usage:")
                        print(f"  Allocated: {memory['allocated']:.2f} GB")
                        print(f"  Reserved:  {memory['reserved']:.2f} GB")
                        print(f"  Peak:      {memory['max_allocated']:.2f} GB")
                    else:
                        print("GPU memory tracking not available (using CPU)")

                elif command == "clear":
                    self.vlm.clear_cache()

                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError executing command: {e}")
                if self.debug_mode:
                    traceback.print_exc()
                else:
                    print("Run with --debug flag to see detailed error information")


def main():
    """Main entry point for the interactive VLM application."""
    parser = argparse.ArgumentParser(
        description="Interactive VLM Screen Analyzer for Windows Applications"
    )
    parser.add_argument(
        "--window",
        "-w",
        type=str,
        help="Window title to capture (partial match)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model name or path (default: Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable Flash Attention 2",
    )
    parser.add_argument(
        "--list-windows",
        action="store_true",
        help="List all windows and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed error output",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # If just listing windows, do that and exit
    if args.list_windows:
        print_available_windows()
        return

    # Initialize and run interactive mode
    app = InteractiveVLM(
        model_name=args.model,
        use_flash_attention=not args.no_flash_attention,
        debug_mode=args.debug,
    )

    app.interactive_mode(window_title=args.window)


if __name__ == "__main__":
    main()
