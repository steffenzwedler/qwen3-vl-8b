# Quantization Reality: Memory vs Speed

## ⚠️ Important Clarification

**Quantization is primarily for MEMORY reduction, NOT speed improvement.**

## The Truth About INT8/INT4 Quantization

### What Quantization ACTUALLY Does

| Feature | BF16 (Default) | INT8 | INT4 |
|---------|----------------|------|------|
| **VRAM Usage** | 16GB | 8GB (50% less) | 4GB (75% less) |
| **Inference Speed** | 1.0x (baseline) | 0.8-1.2x (often slower) | 0.6-0.9x (usually slower) |
| **Quality** | 100% | 98-99% | 95-98% |

### Why Quantization Can Be SLOWER

1. **Dequantization Overhead**
   - Weights stored as INT8/INT4
   - Must convert to FP16/FP32 for computation
   - Conversion adds latency

2. **Windows Compatibility Layer Hacks**
   - Standard `bitsandbytes` often hangs on Windows in 4-bit/8-bit modes
   - Custom DLLs are often required to fix the "inference hang" issue
   - **Crucial Fix**: Use supervised Windows wheels (e.g., from jllllll) to ensure stability.

3. **Model Not Memory-Bound**
   - Qwen3-VL-8B (~16GB) fits on RTX 3090/4090
   - Computation is the bottleneck, not memory bandwidth
   - Reducing memory doesn't help when compute-bound

4. **Mixed Precision Overhead**
   ```
   Quantized: INT8 → dequantize → FP16 → compute → quantize → INT8
   Standard:  BF16 → compute → BF16
   ```

## When to Use Quantization

### ✅ USE Quantization When:

1. **Out of Memory Errors**
   ```python
   # Model doesn't fit
   vlm = VLMInference()  # OOM!

   # Quantization saves the day
   vlm = VLMInference(load_in_8bit=True)  # Works!
   ```

2. **Small GPU (8-12GB VRAM)**
   - RTX 3060 (12GB) → INT8 recommended
   - RTX 3070 (8GB) → INT8 or INT4 required
   - RTX 4060 (8GB) → INT8 or INT4 required

3. **Larger Batch Sizes**
   ```python
   # More VRAM freed = bigger batches
   vlm = VLMInference(load_in_8bit=True)
   # Can now do batch_size=4 instead of 2
   ```

4. **Multiple Models Simultaneously**
   ```python
   # Load 2 quantized models vs 1 full precision
   vlm1 = VLMInference(load_in_8bit=True)  # 8GB
   vlm2 = VLMInference(load_in_8bit=True)  # 8GB
   # Total: 16GB (fits on RTX 4090)
   ```

### ❌ DON'T Use Quantization For:

1. **Maximum Speed** - You'll get SLOWER inference
2. **Large GPUs with Sufficient VRAM** - No benefit
3. **Maximum Quality** - Slight degradation
4. **When VRAM isn't the bottleneck**

## Realistic Performance Expectations

### RTX 4090 (24GB VRAM) - VRAM Not Limited

```python
# Fastest (if model fits)
vlm = VLMInference()  # BF16
# ~300-400ms per query

# Slower but uses less VRAM
vlm = VLMInference(load_in_8bit=True)
# ~350-500ms per query (10-25% slower)
```

**Recommendation: Don't quantize on RTX 4090 unless running multiple models**

### RTX 3080 (10GB VRAM) - VRAM Limited

```python
# May OOM on complex images
vlm = VLMInference()  # BF16
# Risk of OOM

# Safer, fits comfortably
vlm = VLMInference(load_in_8bit=True)
# ~350-500ms per query
# Worth the slight slowdown to avoid OOM
```

**Recommendation: Use INT8 on RTX 3080 for stability**

### RTX 3060 (8GB VRAM) - VRAM Very Limited

```python
# Will likely OOM
vlm = VLMInference()  # BF16
# OOM!

# INT8 might still be tight
vlm = VLMInference(load_in_8bit=True)
# ~350-500ms, may still OOM on large images

# INT4 safest
vlm = VLMInference(load_in_4bit=True)
# ~450-700ms (slower but works)
```

**Recommendation: Use INT4 on RTX 3060**

## When INT8 IS Actually Faster (Advanced)

### TensorRT INT8 (Not Implemented)

```python
# Requires TensorRT engine conversion
# Properly optimized INT8 kernels
# Can be 1.5-3x faster than FP16
# Complex setup, not in current implementation
```

### ONNX Runtime INT8 (Not Implemented)

```python
# Requires ONNX export + quantization calibration
# Optimized INT8 inference
# Can be 1.3-2x faster
# Not in current implementation
```

### Very Large Models (70B+)

```python
# For 70B+ parameter models:
# Memory bandwidth becomes bottleneck
# Less data = faster
# Not applicable to Qwen3-VL-8B
```

## Actual Speed Optimizations

These **DO** make inference faster:

1. ✅ **torch.compile()** - 30-50% faster (actually faster!)
2. ✅ **Flash Attention 2** - 2-4x faster (actually faster!)
3. ✅ **TF32** - 10-20% faster on Ampere+ (actually faster!)
4. ✅ **KV cache reuse** - 3-5x faster for follow-ups (actually faster!)
5. ✅ **Non-blocking transfers** - 15-25% faster (actually faster!)

## Decision Tree

```
Do you have OOM errors?
├─ Yes → Use INT8 or INT4 quantization
└─ No
   └─ Do you have <12GB VRAM?
      ├─ Yes → Consider INT8 for safety
      └─ No → Don't quantize, use BF16 for max speed
```

## Summary

**Quantization Trade-off:**
- ✅ Reduces VRAM usage significantly
- ✅ Enables running on smaller GPUs
- ✅ Allows larger batch sizes
- ❌ May reduce inference speed by 10-40%
- ❌ Slight quality degradation

**Use quantization for VRAM, not for speed. On Windows, use specialized wheels to prevent hangs.**

For maximum speed with sufficient VRAM, use:
```python
vlm = VLMInference(
    load_in_8bit=False,     # No quantization
    use_compile=True,        # 30-50% speedup
    use_flash_attention=True # 2-4x speedup
)
```
