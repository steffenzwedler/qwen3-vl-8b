# Qwen3-VL-8B Windows Screen Analyzer

A powerful Python application for capturing Windows application screenshots and analyzing them using the Qwen3-VL-8B vision-language model. Optimized for NVIDIA GPUs on Windows.

## Features

- **Efficient Windows Screen Capture**: Capture any visible window using native Windows API and MSS
- **GPU-Optimized VLM Inference**: Leverage NVIDIA GPUs with Flash Attention 2 for fast inference
- **Interactive Command Interface**: Ask questions, identify UI elements, and describe screenshots
- **Flexible API**: Use as a library or interactive CLI tool
- **Memory Efficient**: Optimized for Windows with proper GPU memory management

## Requirements

- **Operating System**: Windows 10/11
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **GPU Memory**: Minimum 8GB VRAM recommended

## Installation

⚠️ **IMPORTANT**: For maximum GPU performance, do NOT use `pip install -r requirements.txt` directly!

### Quick Install (GPU-Optimized)

**1. Install PyTorch with CUDA support FIRST:**

```bash
# For CUDA 12.1 (recommended - RTX 30xx/40xx)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2. Install remaining dependencies:**

```bash
pip install transformers accelerate qwen-vl-utils pillow mss pywin32
```

**3. Install Flash Attention 2 for 2-4x speedup (optional):**

```bash
pip install ninja
pip install flash-attn --no-build-isolation
```

### Detailed Installation

See [INSTALL_GPU.md](INSTALL_GPU.md) for comprehensive GPU installation guide including:
- Prerequisites and verification steps
- Troubleshooting common issues
- Performance benchmarks
- Additional optimizations

### Quick Verification

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")  # Should be True
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Quick Start

### Interactive Mode

Launch the interactive interface:

```bash
python src/interactive_vlm.py
```

Or capture a specific window immediately:

```bash
python src/interactive_vlm.py --window "Notepad"
```

### Available Commands

Once in interactive mode:

- `list` - List all visible windows
- `capture <title>` - Capture window by title (supports partial matching)
- `describe` - Get a general description of the current screenshot
- `ask <prompt>` - Ask any question about the screenshot
- `find <element>` - Locate a specific UI element
- `analyze` - Analyze all UI elements in the screenshot
- `save <path>` - Save the current screenshot to a file
- `memory` - Show GPU memory usage statistics
- `clear` - Clear GPU cache
- `quit` - Exit the application

### Example Session

```
> list
Found 42 visible windows:
Handle: 12345678 | Title: Notepad
Handle: 87654321 | Title: Google Chrome

> capture Notepad
Successfully captured window: 'Notepad' (800x600)

> describe
Description: This is a Notepad window showing a text document...

> ask What text is visible in the document?
Response: The document contains the following text: "Hello World"...

> find Save button
Element analysis: The Save button is located in the top menu bar...
```

## Usage as a Library

### Basic Example

```python
from src.window_capture import WindowCapture
from src.vlm_inference import VLMInference

# Initialize components
capture = WindowCapture()
vlm = VLMInference()

# Capture a window
image = capture.capture_window_by_title("Notepad")

# Analyze the screenshot
response = vlm.generate(
    image=image,
    prompt="What text is visible in this window?"
)

print(response)
```

### Advanced Usage

```python
from src.interactive_vlm import InteractiveVLM

# Initialize with custom settings
app = InteractiveVLM(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    use_flash_attention=True
)

# Capture and analyze
app.capture_window_by_title("Calculator")
result = app.analyze_ui()  # Analyze all UI elements

# Find specific elements
button_location = app.find_element("the equals button")

# Ask custom questions
answer = app.ask("What calculation is currently displayed?")
```

## API Reference

### WindowCapture

Main class for Windows screen capture functionality.

```python
capture = WindowCapture()

# List all visible windows
windows = capture.list_windows()  # Returns [(hwnd, title), ...]

# Find window by title
hwnd = capture.get_window_by_title("Notepad", exact_match=False)

# Capture window
image = capture.capture_window(hwnd)  # Returns PIL.Image
image = capture.capture_window_by_title("Notepad")  # Direct capture

# Capture as numpy array
array = capture.capture_to_array(hwnd)  # Returns np.ndarray
```

### VLMInference

Vision-language model inference engine optimized for NVIDIA GPUs.

```python
vlm = VLMInference(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",  # or "cpu", or None for auto-detect
    torch_dtype=torch.bfloat16,
    use_flash_attention=True
)

# Generate response
response = vlm.generate(
    image=image,  # PIL.Image, np.ndarray, or file path
    prompt="What do you see?",
    system_prompt="You are a helpful assistant",  # Optional
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
)

# Specialized methods
description = vlm.describe_screen(image)
ui_analysis = vlm.analyze_ui(image, element_description="the OK button")

# Memory management
memory_stats = vlm.get_memory_usage()
vlm.clear_cache()
```

### InteractiveVLM

High-level interface combining capture and inference.

```python
app = InteractiveVLM()

# Window operations
app.list_windows()
app.capture_window_by_title("Notepad")
app.save_current_image("screenshot.png")

# VLM operations
app.describe()
app.ask("What is this application?")
app.find_element("the File menu")
app.analyze_ui()

# Interactive mode
app.interactive_mode(window_title="Optional initial window")
```

## Performance Optimization

### GPU Settings

The application is optimized for NVIDIA GPUs on Windows:

- **Flash Attention 2**: Enabled by default for faster inference
- **bfloat16 Precision**: Reduces memory usage while maintaining quality
- **Automatic Device Mapping**: Efficiently distributes model across available GPUs

### Memory Management

Monitor GPU memory usage:

```python
memory = vlm.get_memory_usage()
print(f"Allocated: {memory['allocated']:.2f} GB")
print(f"Reserved: {memory['reserved']:.2f} GB")

# Clear cache when needed
vlm.clear_cache()
```

### Best Practices

1. **Reuse VLM Instance**: Initialize once and reuse for multiple inferences
2. **Batch Processing**: Process multiple screenshots in sequence without reinitializing
3. **Clear Cache**: Call `clear_cache()` between large batches
4. **Use Appropriate Temperature**: Lower values (0.3-0.5) for UI analysis, higher (0.7-0.9) for creative tasks

## Project Structure

```
qwen3-vl-8b/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── window_capture.py        # Windows screen capture module
│   ├── vlm_inference.py         # VLM inference engine
│   └── interactive_vlm.py       # Interactive CLI interface
├── tests/
│   ├── test_window_capture.py   # Window capture tests
│   ├── test_vlm_inference.py    # VLM inference tests
│   └── test_interactive_vlm.py  # Integration tests
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_window_capture.py
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:

1. Close other GPU applications
2. Use smaller `max_new_tokens` values
3. Clear cache regularly: `vlm.clear_cache()`
4. Consider using CPU mode for testing: `VLMInference(device="cpu")`

### Window Capture Issues

If window capture fails:

1. Ensure the window is visible (not minimized)
2. Check window title with `list` command
3. Try with `exact_match=True` for precise matching
4. Some applications may block screen capture

### Flash Attention Installation

If Flash Attention fails to install:

1. Ensure you have Visual Studio Build Tools installed
2. Check CUDA version compatibility
3. The application will work without it (slightly slower)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) - Vision-language model by Alibaba
- [Transformers](https://github.com/huggingface/transformers) - HuggingFace Transformers library
- [MSS](https://github.com/BoboTiG/python-mss) - Fast cross-platform screen capture
