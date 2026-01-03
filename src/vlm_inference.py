"""
VLM Inference Module

This module provides optimized inference for the Qwen3-VL-8B vision-language model
on NVIDIA GPUs. It handles model loading, image preprocessing, and text generation.
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Optional, Union, List
from PIL import Image
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
    ):
        """
        Initialize the VLM inference engine.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            torch_dtype: Data type for model weights (bfloat16 recommended for GPUs)
            use_flash_attention: Whether to use Flash Attention 2 for faster inference
        """
        self.model_name = model_name
        self.torch_dtype = torch_dtype

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda":
            current_device = torch.cuda.current_device()
            logger.info(f"Using GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Available GPU Memory: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("CUDA not available, using CPU (will be slower)")

        logger.info(f"Loading model: {model_name}")

        # Load model with optimizations
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        if use_flash_attention and self.device == "cuda":
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2 for optimized inference")
            except Exception as e:
                logger.warning(f"Flash Attention 2 not available: {e}")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Set model to evaluation mode
        self.model.eval()

        logger.info("Model loaded successfully!")

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
        image: Union[Image.Image, np.ndarray, str, Path],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
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

        # Prepare messages
        messages = self.prepare_messages(image, prompt, system_prompt)

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision information
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = inputs.to(self.device)

        # Generate with error handling
        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory during generation")
            torch.cuda.empty_cache()

            # Retry with reduced max_new_tokens
            if max_new_tokens > 128:
                logger.info(f"Retrying with reduced tokens: {max_new_tokens // 2}")
                return self.generate(
                    image, prompt, system_prompt,
                    max_new_tokens=max_new_tokens // 2,
                    temperature=temperature, top_p=top_p, do_sample=do_sample
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

    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")


if __name__ == "__main__":
    # Simple test
    vlm = VLMInference()
    print("\nMemory usage:")
    print(vlm.get_memory_usage())
