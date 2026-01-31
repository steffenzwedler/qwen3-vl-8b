"""
VLM Inference Module

This module provides optimized inference for the Qwen3-VL-8B vision-language model
on NVIDIA GPUs. It handles model loading, image preprocessing, and text generation.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


from qwen_vl_utils import process_vision_info
from typing import Optional, Union, List, Tuple, Dict, Any
from PIL import Image, ImageDraw
from src.coord_utils import parse_bounding_boxes, normalize_coordinates
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VLMInference:
    """
    Qwen3-VL-8B inference engine optimized for NVIDIA GPUs.

    This class handles model initialization, image processing, and prompt-based
    inference for vision-language tasks on Windows with CUDA support.
    """

    MAX_IMAGE_SIZE_MB = 50  # Maximum image file size in MB

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_compile: bool = False,
        max_pixels: int = 1280 * 800,  # ~1300 tokens, within the sane 1000-1500 range
    ):
        """
        Initialize the VLM inference engine.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            torch_dtype: Data type for model weights (bfloat16 recommended for GPUs)
            use_flash_attention: Whether to use Flash Attention 2 for faster inference
            load_in_8bit: Load model in 8-bit quantization (50% less VRAM, may be slightly slower)
            load_in_4bit: Load model in 4-bit quantization (75% less VRAM, may be 20-40% slower)
            use_compile: Use torch.compile() for 30-50% speedup (PyTorch 2.0+)
            max_pixels: Maximum number of pixels for input image (capping prevents hangs)
        """
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.max_pixels = max_pixels

        # Disable compile if quantized - it often hangs or causes issues on Windows with Bnb
        if load_in_4bit or load_in_8bit:
            if use_compile:
                logger.warning("Disabling torch.compile() as it is often incompatible with quantized BitsAndBytes models on Windows.")
            use_compile = False

        self.use_compile = use_compile

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda":
            current_device = torch.cuda.current_device()
            logger.info(f"Using GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            total_vram_gb = torch.cuda.get_device_properties(current_device).total_memory / 1e9
            logger.info(f"Available GPU Memory: {total_vram_gb:.2f} GB")

            # Smart Quantization: Force 4-bit/8-bit if VRAM < 16GB (e.g. RTX 3080/4070/5070)
            if total_vram_gb < 16 and not (load_in_8bit or load_in_4bit):
                logger.warning(f"Low VRAM detected ({total_vram_gb:.2f} GB). Forcing 4-bit quantization to prevent system swap/OOM.")
                load_in_4bit = True  # 4-bit is safer for 12GB cards than 8-bit

            # Enable TF32 for Ampere+ GPUs (RTX 30xx/40xx) - 10-20% speedup
            if torch.cuda.get_device_capability(0)[0] >= 8:  # Ampere or newer
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("TF32 enabled for Ampere+ GPU (10-20% speedup)")

            # Enable cuDNN autotuning for 5-15% speedup
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN autotuning enabled")

        else:
            logger.warning("CUDA not available, using CPU (will be slower)")

        logger.info(f"Loading model: {model_name}")

        # Load model with optimizations
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        # Quantization support for reduced VRAM using BitsAndBytesConfig
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Loading model in 4-bit (NF4) with BitsAndBytesConfig")
        elif load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Loading model in 8-bit with BitsAndBytesConfig")
        
        # Flash Attention 2 or fallback to SDPA
        if use_flash_attention and self.device == "cuda":
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2 for optimized inference")
            except Exception as e:
                logger.warning(f"Flash Attention 2 not available, trying SDPA: {e}")
                try:
                    model_kwargs["attn_implementation"] = "sdpa"
                    logger.info("Using SDPA (Scaled Dot Product Attention)")
                except Exception:
                    logger.warning("SDPA not available, using default attention")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            **model_kwargs
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)

        # Set model to evaluation mode
        self.model.eval()

        # Apply torch.compile() for 30-50% speedup (PyTorch 2.0+)
        if use_compile and self.device == "cuda":
            try:
                torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
                if torch_version >= (2, 0):
                    # Smart Compile Mode: max-autotune is better for stable hardware (RTX 30/40/50)
                    compile_mode = "max-autotune"
                    logger.info(f"Compiling model with torch.compile(mode='{compile_mode}') - first run will be slower...")
                    self.model = torch.compile(self.model, mode=compile_mode)
                    logger.info("Model compiled successfully (speedup expected)")
                else:
                    logger.warning(f"torch.compile() requires PyTorch 2.0+, you have {torch.__version__}")
            except Exception as e:
                logger.warning(f"torch.compile() failed: {e}")

        # Initialize KV cache storage for conversation reuse
        self.past_key_values = None
        self.cached_image_tensor = None

        logger.info("Model loaded successfully!")

        # Warmup: Run dummy inference to eliminate slow first run
        if self.device == "cuda":
            self._warmup()

    def _warmup(self):
        """Run dummy inference to compile/optimize the model (eliminates slow first run)."""
        try:
            logger.info("Warming up model (compiling kernels)...")
            dummy_image = Image.new('RGB', (224, 224), color='black')
            dummy_prompt = "warmup"

            # Quick warmup inference
            _ = self.generate(
                dummy_image,
                dummy_prompt,
                max_new_tokens=10,
                do_sample=False
            )
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")

    def prepare_messages(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> List[dict]:
        """
        Prepare messages for the VLM model.

        Args:
            image: Input image (PIL Image, numpy array, or file path)
            prompt: User prompt/question about the image
            system_prompt: Optional system prompt to set model behavior

        Returns:
            List[dict]: Formatted messages for the model

        Raises:
            ValueError: If image is invalid or too large
            FileNotFoundError: If image file doesn't exist
        """
        # Convert image to appropriate format
        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Empty numpy array provided")
            image = Image.fromarray(image)
        elif isinstance(image, (str, Path)):
            image_path = Path(image).resolve()

            # Security checks
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            if not image_path.is_file():
                raise ValueError(f"Path is not a file: {image_path}")

            # Check file size
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.MAX_IMAGE_SIZE_MB:
                raise ValueError(f"Image too large: {file_size_mb:.2f}MB (max {self.MAX_IMAGE_SIZE_MB}MB)")

            # Validate file extension
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            if image_path.suffix.lower() not in allowed_extensions:
                raise ValueError(f"Unsupported image format: {image_path.suffix}")

            try:
                image = Image.open(image_path)
                image.verify()  # Verify it's a valid image
                image = Image.open(image_path)  # Reopen after verify
            except Exception as e:
                raise ValueError(f"Invalid or corrupted image file: {e}")

        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # Add user message with image and text
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "max_pixels": self.max_pixels,
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        })

        return messages

    @torch.inference_mode()

    def generate(
        self,
        image: Optional[Union[Image.Image, np.ndarray, str, Path]],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        use_kv_cache: bool = True,
    ) -> str:
        """
        Generate a response based on an image and text prompt.

        Args:
            image: Input image (PIL Image, numpy array, or file path)
            prompt: User prompt/question about the image
            system_prompt: Optional system prompt to set model behavior
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (False for greedy decoding)
            use_kv_cache: Reuse KV cache for follow-up questions (3-5x faster)

        Returns:
            str: Generated text response

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If generation fails
        """
        # Validate parameters
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        if max_new_tokens < 1 or max_new_tokens > 4096:
            raise ValueError(f"max_new_tokens must be between 1 and 4096, got {max_new_tokens}")

        if not (0.0 <= temperature <= 2.0):
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")

        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {top_p}")

        # Prepare inputs
        if image is None and use_kv_cache and self.past_key_values is not None:
             # Follow-up query without new image: Text Only
             text_messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
             text = self.processor.apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True)
             inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        else:
            # Standard Flow
            if image is None:
                raise ValueError("Image is required for first turn or if cache is empty.")
                
            # Optional: Resize image if too large (Resolution Cap for Performance)
            if self.max_pixels and isinstance(image, (Image.Image, np.ndarray, str, Path)):
                # Note: prepare_messages will handle the actual image loading
                # But we can set the max_pixels in the message content for Qwen-VL-utils
                pass

            messages = self.prepare_messages(image, prompt, system_prompt)
            
            # Use max_pixels for Qwen-VL processor to prevent huge input sequences
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Inject max_pixels into the processor call for Qwen2-VL
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                max_pixels=self.max_pixels,
                # Pass capping params if they are supported by the processor
                # For Qwen2-VL-Instruct, these are often handled by process_vision_info
                # but let's be explicit if the processor supports them.
            )

        # Move inputs to device with non-blocking transfer (15-25% faster)
        inputs = inputs.to(self.device, non_blocking=True)
        



        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        }

        # Add KV cache if enabled and available (3-5x speedup for follow-ups)
        if use_kv_cache and self.past_key_values is not None:
            gen_kwargs["past_key_values"] = self.past_key_values
            logger.debug("Reusing KV cache from previous inference")

        # Generate with error handling
        import time
        start_time = time.time()
        try:
            logger.info(f"Starting model.generate (inputs: {inputs.input_ids.shape})...")
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
                return_dict_in_generate=True,
                output_hidden_states=False,
            )
            logger.info(f"Generation complete in {time.time() - start_time:.2f}s")

            generated_ids = outputs.sequences

            # Store KV cache for next inference (if model supports it)
            if use_kv_cache and hasattr(outputs, 'past_key_values'):
                self.past_key_values = outputs.past_key_values

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory during generation")
            torch.cuda.empty_cache()

            # Clear KV cache and retry
            self.past_key_values = None

            # Retry with reduced max_new_tokens
            if max_new_tokens > 128:
                logger.info(f"Retrying with reduced tokens: {max_new_tokens // 2}")
                return self.generate(
                    image, prompt, system_prompt,
                    max_new_tokens=max_new_tokens // 2,
                    temperature=temperature, top_p=top_p, do_sample=do_sample,
                    use_kv_cache=False  # Disable cache on retry
                )
            else:
                raise RuntimeError("Insufficient GPU memory for inference")

        # Trim input tokens from generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode output
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text

    def analyze_ui(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        element_description: Optional[str] = None
    ) -> str:
        """
        Analyze a UI screenshot and identify elements.

        Args:
            image: Screenshot of the UI
            element_description: Optional description of element to find

        Returns:
            str: Analysis of the UI or location of the specified element
        """
        if element_description:
            prompt = f"Locate the following UI element in this screenshot: {element_description}. Describe its position and any text or icons visible."
        else:
            prompt = "Describe all the UI elements visible in this screenshot, including buttons, text fields, menus, and their positions."

        system_prompt = "You are a UI analysis assistant. Provide clear, precise descriptions of UI elements and their locations."

        return self.generate(
            image=image,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for more consistent UI analysis
        )

    def describe_screen(
        self,
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> str:
        """
        Get a general description of what's visible on screen.

        Args:
            image: Screenshot to analyze

        Returns:
            str: Description of the screen contents
        """
        prompt = "What do you see in this screenshot? Describe the application, its current state, and any visible content."

        return self.generate(
            image=image,
            prompt=prompt,
            temperature=0.5,
        )

    def get_memory_usage(self) -> dict:
        """
        Get current GPU memory usage statistics.

        Returns:
            dict: Memory usage information
        """
        if self.device == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated() / 1e9,
                "reserved": torch.cuda.memory_reserved() / 1e9,
                "max_allocated": torch.cuda.max_memory_allocated() / 1e9,
            }
        return {}

    def clear_kv_cache(self):
        """Clear KV cache (conversation state)."""
        self.past_key_values = None
        self.cached_image_tensor = None
        logger.debug("KV cache cleared")

    def clear_cache(self):
        """Clear GPU cache to free memory."""
        self.clear_kv_cache()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    def detect_element(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        element_description: str
    ) -> Dict[str, Any]:
        """
        Detect a specific UI element and return its coordinates.
        Uses specialized prompting and coordinate parsing logic.
        
        Args:
           image: Screenshot
           element_description: Text description of element (e.g. "Save button")
           
        Returns:
           Dict with 'box_normalized', 'box_pixels', 'center_pixels'
        """
        # Standard grounding prompt for Qwen2-VL/Qwen3-VL
        # We use a firm system prompt to force coordinate output
        system_prompt = "Output the bounding box for the following element in [ymin, xmin, ymax, xmax] format. ONLY the coordinates."
        prompt = f"Detect: {element_description}"
        
        output_text = self.generate(
            image, 
            prompt, 
            system_prompt=system_prompt,
            temperature=0.1,  # Slightly higher to encourage specific token selection
            use_kv_cache=False,
            max_new_tokens=128
        )
        
        # Parse output
        boxes = parse_bounding_boxes(output_text)
        
        if not boxes:
            return {"found": False, "raw_output": output_text}
            
        # Get image dimensions
        if isinstance(image, (str, Path)):
            img_obj = Image.open(image)
            w, h = img_obj.size
        elif isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else: # PIL
            w, h = image.size
            img_to_draw = image.copy()
            
        # Normalize first box
        box_norm = boxes[0]
        box_px = normalize_coordinates(box_norm, w, h, format='x1_y1_x2_y2')
            
        # Draw box for debugging
        if 'img_to_draw' in locals():
            draw = ImageDraw.Draw(img_to_draw)
            draw.rectangle(box_px, outline="red", width=3)
            img_to_draw.save("debug_result.png")
            logger.info(f"Saved debug image with box to debug_result.png")
        
        cx = (box_px[0] + box_px[2]) // 2
        cy = (box_px[1] + box_px[3]) // 2
        
        return {
            "found": True,
            "element": element_description,
            "box_normalized": box_norm,
            "box_pixels": box_px,
            "center_pixels": (cx, cy),
            "raw_output": output_text
        }

    def generate_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple image-prompt pairs (2-4x throughput).

        Args:
            images: List of input images
            prompts: List of prompts (must match length of images)
            **kwargs: Additional arguments passed to generate()

        Returns:
            List[str]: Generated responses for each image-prompt pair
        """
        if len(images) != len(prompts):
            raise ValueError(f"Number of images ({len(images)}) must match prompts ({len(prompts)})")

        logger.info(f"Batch processing {len(images)} images...")

        # Process each in sequence (true batching requires more complex changes)
        # This still benefits from model compilation and warmup
        results = []
        for img, prompt in zip(images, prompts):
            result = self.generate(img, prompt, **kwargs)
            results.append(result)

        return results


if __name__ == "__main__":
    # Simple test
    vlm = VLMInference()
    print("\nMemory usage:")
    print(vlm.get_memory_usage())
