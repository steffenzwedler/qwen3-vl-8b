# K-Cache and V-Cache Explained

## What is KV Cache?

**KV Cache (Key-Value Cache)** is a performance optimization for transformer models that stores intermediate computation results to avoid redundant calculations during text generation.

## The Transformer Attention Mechanism

### Standard Self-Attention (No Cache)

In transformer models, the self-attention mechanism computes three matrices for each token:

```
Query (Q) = Token × W_query
Key (K)   = Token × W_key
Value (V) = Token × W_value

Attention = softmax(Q × K^T / √d) × V
```

**Example:** Generating the response "The cat is on the mat"

```
Step 1: Generate "The"
- Process prompt: "What is this?"
- Compute Q, K, V for: ["What", "is", "this", "?"]
- Generate: "The"

Step 2: Generate "cat"
- Process: "What is this? The"
- Compute Q, K, V for: ["What", "is", "this", "?", "The"]  ← REDUNDANT!
- Generate: "cat"

Step 3: Generate "is"
- Process: "What is this? The cat"
- Compute Q, K, V for: ["What", "is", "this", "?", "The", "cat"]  ← REDUNDANT!
- Generate: "is"
```

**Problem:** We recompute K and V for previous tokens at every step!

### With KV Cache (Optimized)

```
Step 1: Generate "The"
- Compute Q, K, V for: ["What", "is", "this", "?"]
- STORE K and V in cache
- Generate: "The"

Step 2: Generate "cat"
- Compute Q, K, V only for: ["The"]
- REUSE cached K, V for ["What", "is", "this", "?"]
- Concatenate: K_cached + K_new, V_cached + V_new
- Generate: "cat"

Step 3: Generate "is"
- Compute Q, K, V only for: ["cat"]
- REUSE cached K, V for ["What", "is", "this", "?", "The"]
- Generate: "is"
```

**Benefit:** Only compute new tokens, reuse previous computations!

## Why Only Cache K and V?

**Query (Q) is NOT cached** because:
- Q only needs to attend to previous tokens
- Q is computed from the current token being generated
- Q changes at every generation step

**Key (K) and Value (V) are cached** because:
- K and V for previous tokens never change
- K is used to match against Q (attention scores)
- V is used to aggregate information based on attention

## Memory Layout

```python
# Shape of cached tensors
past_key_values = [
    # For each transformer layer (e.g., 32 layers for Qwen3-VL-8B)
    (
        key_cache,    # Shape: [batch_size, num_heads, seq_len, head_dim]
        value_cache   # Shape: [batch_size, num_heads, seq_len, head_dim]
    ),
    ...
]
```

**Example sizes for Qwen3-VL-8B:**
- 32 layers
- 28 attention heads
- 128 head dimension
- Sequence length: varies (grows with conversation)

**Memory calculation:**
```
Single layer KV cache = 2 × (num_heads × seq_len × head_dim) × dtype_size
                      = 2 × (28 × seq_len × 128) × 2 bytes (FP16)
                      = 7,168 × seq_len bytes

Total cache (32 layers) = 32 × 7,168 × seq_len
                        = 229,376 × seq_len bytes
                        = ~0.23 MB per token

For 1000 tokens: ~230 MB VRAM
For 4096 tokens: ~942 MB VRAM
```

## Implementation in VLMInference

### Initialization

```python
class VLMInference:
    def __init__(self, use_kv_cache=True):
        self.use_kv_cache = use_kv_cache
        self.past_key_values = None  # Stores KV cache
```

### During Generation

```python
def generate(self, image, prompt, use_kv_cache=True):
    # Prepare generation arguments
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }

    # REUSE cached K and V from previous inferences
    if use_kv_cache and self.past_key_values is not None:
        gen_kwargs["past_key_values"] = self.past_key_values
        logging.info("Reusing KV cache from previous inference")

    # Generate response
    outputs = self.model.generate(
        **inputs,
        **gen_kwargs,
        return_dict_in_generate=True  # Required to get past_key_values
    )

    # STORE K and V for next inference
    if use_kv_cache and hasattr(outputs, 'past_key_values'):
        self.past_key_values = outputs.past_key_values
        logging.info("Stored KV cache for next inference")

    return decoded_text
```

### Cache Clearing

```python
def clear_kv_cache(self):
    """Clear the KV cache to start a fresh conversation."""
    self.past_key_values = None
    logging.info("KV cache cleared")
```

## Performance Impact

### Without KV Cache (Baseline)

```python
vlm = VLMInference(use_kv_cache=False)

# First question: 1000ms
result1 = vlm.generate(image, "What is this?")

# Second question: 1000ms (recomputes everything)
result2 = vlm.generate(image, "What color is it?")

# Third question: 1000ms (recomputes everything)
result3 = vlm.generate(image, "How many are there?")

Total time: 3000ms
```

### With KV Cache (Optimized)

```python
vlm = VLMInference(use_kv_cache=True)

# First question: 1000ms (nothing cached yet)
result1 = vlm.generate(image, "What is this?")

# Second question: 250ms (reuses cached K, V)
result2 = vlm.generate(image, "What color is it?")

# Third question: 250ms (reuses cached K, V)
result3 = vlm.generate(image, "How many are there?")

Total time: 1500ms (2x faster!)
```

**Speedup:** 3-5x for follow-up questions

## Efficient Usage Patterns

### ✅ GOOD: Conversation about Same Image

```python
vlm = VLMInference()

# Capture image once
image = capture.capture_active_window()

# Ask multiple questions - cache is reused
answer1 = vlm.generate(image, "What is this application?")
# KV cache stores: image tokens + "What is this application?" + answer

answer2 = vlm.generate(image, "What buttons are visible?")
# Reuses cache, only processes new prompt + generates new answer
# 3-5x FASTER!

answer3 = vlm.generate(image, "What is the main menu?")
# Even faster! Cache keeps growing
```

### ✅ GOOD: Analyzing Different Regions with Context

```python
vlm = VLMInference()

# First analysis
image1 = capture.capture_window_region(hwnd, x=0, y=0, w=800, h=600)
answer1 = vlm.generate(image1, "Describe the top-left region")

# Related follow-up (reuses context)
answer2 = vlm.generate(image1, "What icons are there?")
# Fast - reuses KV cache!
```

### ❌ BAD: Switching Images Without Clearing Cache

```python
vlm = VLMInference()

# First image
image1 = capture.capture_window(hwnd1)
answer1 = vlm.generate(image1, "What is this?")

# Different image - should clear cache but doesn't!
image2 = capture.capture_window(hwnd2)
answer2 = vlm.generate(image2, "What is this?")
# BAD: Model is confused, has cached K,V from image1!
# Response may be incorrect or hallucinate
```

**Fix:**
```python
vlm.clear_kv_cache()  # Clear before new image!
answer2 = vlm.generate(image2, "What is this?")
```

### ❌ BAD: Long Conversations Without Clearing

```python
vlm = VLMInference()

# Very long conversation (100+ exchanges)
for i in range(100):
    answer = vlm.generate(image, f"Question {i}")
    # Cache keeps growing...
    # VRAM usage: ~230 MB per 1000 tokens
    # After 100 exchanges: Several GB of VRAM!
```

**Fix:**
```python
for i in range(100):
    if i % 10 == 0:
        vlm.clear_kv_cache()  # Clear every 10 questions
    answer = vlm.generate(image, f"Question {i}")
```

## Best Practices

### 1. Clear Cache When Changing Context

```python
# New image = new context
vlm.clear_kv_cache()
image2 = capture.capture_window(new_hwnd)
answer = vlm.generate(image2, prompt)
```

### 2. Monitor VRAM Usage

```python
import torch

# Check VRAM before and after
print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Long conversation
for i in range(50):
    answer = vlm.generate(image, f"Question {i}")

    # Clear if VRAM usage is high
    if torch.cuda.memory_allocated() > 20e9:  # 20GB threshold
        vlm.clear_kv_cache()
```

### 3. Use Cache for Follow-Up Questions

```python
# First question: slow (no cache)
answer1 = vlm.generate(image, "What is the main window?")

# Follow-ups: fast (uses cache)
answer2 = vlm.generate(image, "What buttons are visible?")
answer3 = vlm.generate(image, "What is the title?")
answer4 = vlm.generate(image, "What colors are used?")
```

### 4. Batch Processing: Clear Between Items

```python
images = [img1, img2, img3, img4]

for image in images:
    vlm.clear_kv_cache()  # Fresh cache for each image
    answer = vlm.generate(image, "Describe this")
```

### 5. Disable Cache for One-Shot Queries

```python
# Single question, no follow-ups planned
answer = vlm.generate(image, prompt, use_kv_cache=False)
# Saves VRAM, no performance benefit since no follow-ups
```

## Cache Lifecycle Example

```python
vlm = VLMInference()

# === Scenario 1: Analyzing a code editor ===
image = capture.capture_window_by_title("Visual Studio Code")

# Q1: Initial question (cache empty)
answer1 = vlm.generate(image, "What file is open?")
# Cache: [image_tokens, "What file is open?", answer1_tokens]
# VRAM: ~200 MB

# Q2: Follow-up (cache reused)
answer2 = vlm.generate(image, "What is the file path?")
# Cache: [image_tokens, Q1, A1, "What is the file path?", answer2_tokens]
# VRAM: ~400 MB
# Speed: 3-5x faster than Q1

# Q3: Another follow-up (cache reused)
answer3 = vlm.generate(image, "What errors are shown?")
# Cache: [image_tokens, Q1, A1, Q2, A2, "What errors are shown?", answer3_tokens]
# VRAM: ~600 MB
# Speed: 3-5x faster

# === Scenario 2: Switch to different window ===
vlm.clear_kv_cache()  # IMPORTANT: Clear cache for new context!
# Cache: []
# VRAM: ~50 MB (model only)

image2 = capture.capture_window_by_title("Chrome")
answer4 = vlm.generate(image2, "What website is this?")
# Cache: [image2_tokens, "What website is this?", answer4_tokens]
# VRAM: ~200 MB
```

## Common Pitfalls

### Pitfall 1: Forgetting to Clear Cache

```python
# WRONG
answer1 = vlm.generate(screenshot1, "What is this?")
answer2 = vlm.generate(screenshot2, "What is this?")  # Uses cache from screenshot1!
```

**Symptoms:**
- Model mentions elements from previous image
- Hallucinations or confused responses
- Incorrect object counts

**Fix:**
```python
answer1 = vlm.generate(screenshot1, "What is this?")
vlm.clear_kv_cache()  # Clear before new image!
answer2 = vlm.generate(screenshot2, "What is this?")
```

### Pitfall 2: OOM from Unbounded Cache Growth

```python
# WRONG: Infinite conversation without clearing
while True:
    user_question = input("Ask: ")
    answer = vlm.generate(image, user_question)
    # Cache grows indefinitely → OOM after ~30-50 exchanges
```

**Fix:**
```python
conversation_count = 0
while True:
    user_question = input("Ask: ")
    answer = vlm.generate(image, user_question)

    conversation_count += 1
    if conversation_count % 10 == 0:
        vlm.clear_kv_cache()  # Periodic clearing
        print("(Cache cleared for memory)")
```

### Pitfall 3: Disabling Cache for Conversations

```python
# WRONG: Disables cache for every call
for question in questions:
    answer = vlm.generate(image, question, use_kv_cache=False)
    # Each call is slow, no speedup!
```

**Fix:**
```python
# Enable cache for conversation
for question in questions:
    answer = vlm.generate(image, question, use_kv_cache=True)
    # 3-5x faster for Q2, Q3, Q4...
```

## Advanced: Cache Management in Interactive CLI

```python
class InteractiveVLM:
    def __init__(self):
        self.vlm = VLMInference()
        self.current_image_hash = None

    def ask_question(self, image, question):
        # Compute image hash to detect changes
        import hashlib
        image_hash = hashlib.md5(image.tobytes()).hexdigest()

        # Clear cache if image changed
        if image_hash != self.current_image_hash:
            self.vlm.clear_kv_cache()
            self.current_image_hash = image_hash
            print("New image detected - cache cleared")

        # Generate with cache
        answer = self.vlm.generate(image, question)
        return answer

# Usage
cli = InteractiveVLM()

# Same image - cache is reused
cli.ask_question(screenshot, "What is this?")
cli.ask_question(screenshot, "What color?")  # Fast!

# New image - cache auto-cleared
new_screenshot = capture.capture_active_window()
cli.ask_question(new_screenshot, "What is this?")  # Cache cleared automatically
```

## Summary

### What KV Cache Is
- Stores Key and Value matrices from previous tokens
- Avoids recomputing attention for previous context
- Query is NOT cached (changes each step)

### Performance Benefits
- **First question:** Normal speed (cache empty)
- **Follow-up questions:** 3-5x faster (reuses cache)
- **Memory cost:** ~230 MB per 1000 tokens

### When to Use
- ✅ Multiple questions about same image
- ✅ Conversational interactions
- ✅ Follow-up clarifications
- ✅ Iterative analysis

### When to Clear
- ✅ Switching to different image
- ✅ Context change (new window, new task)
- ✅ After 10-20 exchanges (prevent OOM)
- ✅ VRAM usage approaching limit

### Implementation Pattern
```python
# Initialize once
vlm = VLMInference(use_kv_cache=True)

# Conversation about image A
answer1 = vlm.generate(imageA, "Question 1")  # Slow (cache empty)
answer2 = vlm.generate(imageA, "Question 2")  # Fast (cache reused)
answer3 = vlm.generate(imageA, "Question 3")  # Fast (cache reused)

# Switch to image B
vlm.clear_kv_cache()  # CRITICAL!
answer4 = vlm.generate(imageB, "Question 4")  # Slow (cache empty)
answer5 = vlm.generate(imageB, "Question 5")  # Fast (cache reused)
```

### Key Takeaway
**KV Cache is like short-term memory for the model** - it remembers the conversation context to answer follow-ups faster, but you must clear it when changing topics (images) to avoid confusion.
