"""
Tests for VLM Inference Logic
"""
import pytest
from unittest.mock import MagicMock, patch, ANY
import torch
from src.vlm_inference import VLMInference

# Mock torch.cuda
@pytest.fixture
def mock_cuda():
    with patch("torch.cuda") as mock:
        mock.is_available.return_value = True
        mock.current_device.return_value = 0
        mock.get_device_name.return_value = "NVIDIA GeForce RTX 5070"
        
        # Mock device properties (Total Memory)
        props = MagicMock()
        props.total_memory = 12 * 1024**3 # 12 GB
        mock.get_device_properties.return_value = props
        
        mock.get_device_capability.return_value = (8, 9) # Ampere/Ada/Blackwell
        
        # Patch compile to avoid analyzing mocks
        with patch("torch.compile", side_effect=lambda m, **k: m):
            yield mock

@pytest.fixture
def mock_qwen_classes():
    with patch("src.vlm_inference.Qwen2VLForConditionalGeneration") as mock_model_cls, \
         patch("src.vlm_inference.AutoProcessor") as mock_proc_cls, \
         patch("src.vlm_inference.process_vision_info") as mock_process:
        
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        
        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        
        yield mock_model_cls, mock_proc_cls, mock_process

def test_smart_quantization_low_vram(mock_cuda, mock_qwen_classes):
    # 12GB VRAM -> Should force 4bit or 8bit
    # Our logic: if < 16GB, force 4-bit (safer)
    
    vlm = VLMInference(
        load_in_8bit=False,
        load_in_4bit=False
    )
    
    model_cls, _, _ = mock_qwen_classes
    # Check assertions on from_pretrained call
    args, kwargs = model_cls.from_pretrained.call_args
    
    # We expect load_in_4bit=True
    assert kwargs.get("load_in_4bit") is True
    assert kwargs.get("load_in_8bit") is None

def test_smart_quantization_high_vram(mock_cuda, mock_qwen_classes):
    # 24GB VRAM -> Should NOT force quantization
    mock_cuda.get_device_properties.return_value.total_memory = 24 * 1024**3
    
    vlm = VLMInference(load_in_8bit=False, load_in_4bit=False)
    
    model_cls, _, _ = mock_qwen_classes
    args, kwargs = model_cls.from_pretrained.call_args
    
    assert kwargs.get("load_in_4bit") is None
    assert kwargs.get("load_in_8bit") is None

def test_generate_image_none_cache(mock_cuda, mock_qwen_classes):
    # Test that valid cache allows image=None
    vlm = VLMInference(device="cpu") # Avoid CUDA logic for this test part
    
    # Mock cache existence
    vlm.past_key_values = MagicMock()
    
    # Mock processor and model behavior
    vlm.processor.apply_chat_template.return_value = "Prompt Text"
    vlm.processor.return_value = MagicMock() # inputs
    vlm.model.generate.return_value = MagicMock(sequences=[[1, 2, 3]])
    
    # Should NOT raise ValueError
    vlm.generate(image=None, prompt="Follow up", use_kv_cache=True)
    
    # Verify processor called with text ONLY (no images arg)
    call_args = vlm.processor.call_args
    assert "images" not in call_args.kwargs
    assert call_args.kwargs.get("text") == ["Prompt Text"]

def test_generate_image_none_no_cache_error(mock_cuda, mock_qwen_classes):
    vlm = VLMInference(device="cpu")
    vlm.past_key_values = None
    
    with pytest.raises(ValueError, match="Image is required"):
        vlm.generate(image=None, prompt="Fail me")
