"""
Unit tests for vlm_inference module.

Tests cover model initialization, inference, and memory management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vlm_inference import VLMInference


class TestVLMInference:
    """Test suite for VLMInference class."""

    @patch('vlm_inference.torch.cuda.is_available')
    @patch('vlm_inference.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('vlm_inference.AutoProcessor.from_pretrained')
    def test_initialization_cuda(self, mock_processor, mock_model, mock_cuda):
        """Test VLMInference initializes with CUDA."""
        mock_cuda.return_value = True
        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        vlm = VLMInference(device=None)

        assert vlm.device == "cuda"
        assert vlm.model_name == "Qwen/Qwen3-VL-8B-Instruct"
        mock_model.assert_called_once()
        mock_processor.assert_called_once()

    @patch('vlm_inference.torch.cuda.is_available')
    @patch('vlm_inference.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('vlm_inference.AutoProcessor.from_pretrained')
    def test_initialization_cpu(self, mock_processor, mock_model, mock_cuda):
        """Test VLMInference initializes with CPU."""
        mock_cuda.return_value = False
        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        vlm = VLMInference(device=None)

        assert vlm.device == "cpu"

    @patch('vlm_inference.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('vlm_inference.AutoProcessor.from_pretrained')
    def test_initialization_custom_device(self, mock_processor, mock_model):
        """Test VLMInference initializes with custom device."""
        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        vlm = VLMInference(device="cpu")

        assert vlm.device == "cpu"

    @patch('vlm_inference.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('vlm_inference.AutoProcessor.from_pretrained')
    def test_prepare_messages_with_pil_image(self, mock_processor, mock_model):
        """Test preparing messages with PIL Image."""
        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        vlm = VLMInference(device="cpu")
        image = Image.new('RGB', (100, 100))
        prompt = "What do you see?"

        messages = vlm.prepare_messages(image, prompt)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][1]["text"] == prompt

    @patch('vlm_inference.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('vlm_inference.AutoProcessor.from_pretrained')
    def test_prepare_messages_with_numpy_array(self, mock_processor, mock_model):
        """Test preparing messages with numpy array."""
        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        vlm = VLMInference(device="cpu")
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        prompt = "Describe this."

        messages = vlm.prepare_messages(array, prompt)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch('vlm_inference.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('vlm_inference.AutoProcessor.from_pretrained')
    def test_prepare_messages_with_system_prompt(self, mock_processor, mock_model):
        """Test preparing messages with system prompt."""
        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        vlm = VLMInference(device="cpu")
        image = Image.new('RGB', (100, 100))
        prompt = "What do you see?"
        system_prompt = "You are a helpful assistant."

        messages = vlm.prepare_messages(image, prompt, system_prompt)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_prompt
        assert messages[1]["role"] == "user"

    @patch('vlm_inference.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('vlm_inference.AutoProcessor.from_pretrained')
    @patch('vlm_inference.process_vision_info')
    def test_generate(self, mock_process_vision, mock_processor_cls, mock_model_cls):
        """Test generate method."""
        # Setup mocks
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_processor_cls.return_value = mock_processor

        # Mock processor methods
        mock_processor.apply_chat_template.return_value = "formatted_text"
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
        }
        mock_processor.batch_decode.return_value = ["Generated response"]

        # Mock vision info processing
        mock_process_vision.return_value = (["image"], None)

        # Mock model generation
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        vlm = VLMInference(device="cpu")
        image = Image.new('RGB', (100, 100))
        prompt = "What do you see?"

        response = vlm.generate(image, prompt)

        assert response == "Generated response"
        mock_model.generate.assert_called_once()

    @patch('vlm_inference.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('vlm_inference.AutoProcessor.from_pretrained')
    @patch('vlm_inference.VLMInference.generate')
    def test_analyze_ui_with_element(self, mock_generate, mock_processor, mock_model):
        """Test analyze_ui with element description."""
        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()
        mock_generate.return_value = "Element found at top-left"

        vlm = VLMInference(device="cpu")
        image = Image.new('RGB', (100, 100))

        result = vlm.analyze_ui(image, element_description="OK button")

        assert result == "Element found at top-left"
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]
        assert "OK button" in call_kwargs["prompt"]

    @patch('vlm_inference.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('vlm_inference.AutoProcessor.from_pretrained')
    @patch('vlm_inference.VLMInference.generate')
    def test_analyze_ui_without_element(self, mock_generate, mock_processor, mock_model):
        """Test analyze_ui without element description."""
        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()
        mock_generate.return_value = "UI contains buttons and text"

        vlm = VLMInference(device="cpu")
        image = Image.new('RGB', (100, 100))

        result = vlm.analyze_ui(image)

        assert result == "UI contains buttons and text"
        mock_generate.assert_called_once()

    @patch('vlm_inference.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('vlm_inference.AutoProcessor.from_pretrained')
    @patch('vlm_inference.VLMInference.generate')
    def test_describe_screen(self, mock_generate, mock_processor, mock_model):
        """Test describe_screen method."""
        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()
        mock_generate.return_value = "This is a text editor"

        vlm = VLMInference(device="cpu")
        image = Image.new('RGB', (100, 100))

        result = vlm.describe_screen(image)

        assert result == "This is a text editor"
        mock_generate.assert_called_once()

    @patch('vlm_inference.torch.cuda.is_available')
    @patch('vlm_inference.torch.cuda.memory_allocated')
    @patch('vlm_inference.torch.cuda.memory_reserved')
    @patch('vlm_inference.torch.cuda.max_memory_allocated')
    @patch('vlm_inference.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('vlm_inference.AutoProcessor.from_pretrained')
    def test_get_memory_usage_cuda(self, mock_processor, mock_model,
                                   mock_max_alloc, mock_reserved, mock_allocated, mock_cuda):
        """Test get_memory_usage with CUDA."""
        mock_cuda.return_value = True
        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()
        mock_allocated.return_value = 2e9  # 2 GB
        mock_reserved.return_value = 3e9   # 3 GB
        mock_max_alloc.return_value = 4e9  # 4 GB

        vlm = VLMInference(device="cuda")
        memory = vlm.get_memory_usage()

        assert memory["allocated"] == pytest.approx(2.0, 0.1)
        assert memory["reserved"] == pytest.approx(3.0, 0.1)
        assert memory["max_allocated"] == pytest.approx(4.0, 0.1)

    @patch('vlm_inference.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('vlm_inference.AutoProcessor.from_pretrained')
    def test_get_memory_usage_cpu(self, mock_processor, mock_model):
        """Test get_memory_usage with CPU."""
        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        vlm = VLMInference(device="cpu")
        memory = vlm.get_memory_usage()

        assert memory == {}

    @patch('vlm_inference.torch.cuda.is_available')
    @patch('vlm_inference.torch.cuda.empty_cache')
    @patch('vlm_inference.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('vlm_inference.AutoProcessor.from_pretrained')
    def test_clear_cache_cuda(self, mock_processor, mock_model, mock_empty_cache, mock_cuda):
        """Test clear_cache with CUDA."""
        mock_cuda.return_value = True
        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        vlm = VLMInference(device="cuda")
        vlm.clear_cache()

        mock_empty_cache.assert_called_once()

    @patch('vlm_inference.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('vlm_inference.AutoProcessor.from_pretrained')
    @patch('vlm_inference.torch.cuda.empty_cache')
    def test_clear_cache_cpu(self, mock_empty_cache, mock_processor, mock_model):
        """Test clear_cache with CPU (should not call empty_cache)."""
        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        vlm = VLMInference(device="cpu")
        vlm.clear_cache()

        mock_empty_cache.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
