# GPU Installation Guide for Windows

This guide ensures you get the **fastest possible performance** with NVIDIA GPUs on Windows.

## Prerequisites

1. **NVIDIA GPU** with compute capability ≥ 7.0 (RTX 20xx series or newer recommended)
2. **CUDA Toolkit 12.4 or 12.6** (Recommended for最新的 GPUs)
3. **Visual Studio 2019/2022** with C++ build tools - Required for optimized kernels
4. **Python 3.10+** (3.11/3.12 recommended)
5. **12GB+ GPU VRAM** recommended for Qwen3-VL-8B (8GB minimum with INT4)

## Step-by-Step Installation

### 1. Verify CUDA Installation

```bash
nvidia-smi
nvcc --version
```

Both commands should work. Note your CUDA version (11.8 or 12.1).

### 2. Install PyTorch with CUDA Support (CRITICAL!)

**For CUDA 12.4 / 12.6 (Recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify GPU support:**
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output: `CUDA available: True`

### 3. Install Core Dependencies

```bash
pip install transformers>=4.37.0
pip install accelerate>=0.25.0
pip install qwen-vl-utils
pip install pillow pillow-simd
pip install sentencepiece protobuf
```

### 4. Install Windows-Specific Dependencies

```bash
pip install mss pywin32
```

### 5. Install Flash Attention 2 (Optional but Highly Recommended)

**Flash Attention 2 provides 2-4x speedup!**

**Prerequisites:**
- Visual Studio with C++ build tools installed
- CUDA Toolkit installed

**Installation:**
```bash
pip install ninja
pip install flash-attn --no-build-isolation
```

**If installation fails:**
- Ensure Visual Studio C++ tools are in PATH
- Try pre-built wheels from: https://github.com/Dao-AILab/flash-attention/releases
- The application will work without it, just slower

### 6. Install Optional Optimizations

**For Quantization (CRITICAL for Windows stability):**

To avoid inference hangs on Windows, use the specifically compiled BitsAndBytes wheels:

```bash
pip install bitsandbytes --index-url https://jllllll.github.io/bitsandbytes-windows-webui
```

### 7. Install Development Dependencies (Optional)

```bash
pip install pytest pytest-cov pytest-mock black flake8 mypy
```

## Quick Install Script

Save as `install_gpu.bat`:

```batch
@echo off
echo Installing GPU-accelerated Qwen3-VL-8B...

REM Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Install core dependencies
pip install transformers>=4.37.0 accelerate>=0.25.0 qwen-vl-utils
pip install pillow pillow-simd sentencepiece protobuf numpy

REM Install Windows dependencies
pip install mss pywin32

REM Install Flash Attention (may fail if no C++ compiler)
pip install ninja
pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed - continuing without it"

echo Installation complete!
echo.
echo Verifying CUDA...
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Verify Installation

Run this to check everything is working:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test Flash Attention
try:
    import flash_attn
    print("Flash Attention 2: ✓ Available")
except ImportError:
    print("Flash Attention 2: ✗ Not available (will use standard attention)")

# Test transformers
from transformers import AutoProcessor
print("Transformers: ✓ Available")

# Test qwen-vl-utils
from qwen_vl_utils import process_vision_info
print("Qwen VL Utils: ✓ Available")
```

## Performance Benchmarks

Expected inference times on RTX 4090 (24GB VRAM):

| Configuration | Time per inference |
|--------------|-------------------|
| CPU only | ~30-60 seconds |
| CUDA without Flash Attention | ~3-5 seconds |
| **CUDA + Flash Attention 2** | **~1-2 seconds** |
| CUDA + Flash Attention + bfloat16 | **~0.8-1.5 seconds** |

## Troubleshooting

### "CUDA out of memory"
- Close other GPU applications
- Reduce `max_new_tokens` in prompts
- Use 8-bit quantization: `VLMInference(load_in_8bit=True)`

### "Flash Attention installation failed"
- Ensure Visual Studio C++ build tools installed
- Check CUDA Toolkit is in PATH
- The app works without it (just slower)

### "DLL load failed" errors
- Reinstall Visual C++ Redistributable
- Check CUDA bin directory in PATH

### Slow performance even with GPU
- Verify CUDA is actually being used: `nvidia-smi` should show python process
- Check torch is using GPU: `torch.cuda.is_available()` should be True
- Ensure you installed torch with `--index-url` flag

## Additional Optimizations

### Enable TensorFloat-32 (Ampere GPUs and newer)
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Use torch.compile() (PyTorch 2.0+)
The VLMInference class supports compilation for additional speedup on Ampere+ GPUs.

### Monitor GPU Usage
```bash
# Watch GPU usage in real-time
nvidia-smi -l 1
```

## Support

- CUDA issues: https://docs.nvidia.com/cuda/
- PyTorch issues: https://pytorch.org/get-started/locally/
- Flash Attention: https://github.com/Dao-AILab/flash-attention
