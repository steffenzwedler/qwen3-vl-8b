"""
Integration Tests for Qwen3-VL System
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import json
import os

# We want to test the FLOW:
# API -> MCP Server -> VLM Inference (Mocked Logic but Real Helper calls) -> Response

@pytest.fixture
def integration_client():
    os.environ["MCP_USER"] = "admin"
    os.environ["MCP_PASS"] = "password"
    
    # We patch VLMInference but keep its methods 'real' where possible?
    # No, VLMInference is too heavy (requires Model download).
    # We will Mock VLMInference but verify the CALL flow matches what we expect from the Server.
    
    with patch("src.mcp_server.VLMInference") as MockVLM:
        mock_instance = MockVLM.return_value
        # Setup mock behavior
        mock_instance.generate.return_value = "I see a Start Button."
        
        from src.mcp_server import app
        with TestClient(app) as c:
            yield c, mock_instance

def test_full_flow_capture_analyze(integration_client):
    client, mock_vlm = integration_client
    
    # 1. Capture and Analyze Request
    # We also need to patch WindowCapture because we don't have real windows in CI environment usually
    with patch("src.mcp_server.WindowCapture") as MockCapture:
        mock_cap_instance = MockCapture.return_value
        # Return a dummy PIL image
        from PIL import Image
        mock_cap_instance.capture_window_by_title.return_value = Image.new('RGB', (100, 100))
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "capture_and_analyze",
                "arguments": {
                    "window_title": "Notepad",
                    "prompt": "Describe this."
                }
            },
            "id": 1
        }
        
        headers = {"Authorization": "Basic YWRtaW46cGFzc3dvcmQ="} # admin:password
        response = client.post("/messages", json=payload, headers=headers)
        
        assert response.status_code == 200
        result = response.json()["result"]
        
        # Verify VLM was called
        mock_vlm.generate.assert_called_once()
        args = mock_vlm.generate.call_args
        # Check that an image was passed
        assert args.kwargs.get('image') is not None
        assert args.kwargs.get('prompt') == "Describe this."
        
        # Verify result content
        assert "I see a Start Button" in result["content"][0]["text"]

def test_full_flow_ask_current(integration_client):
    client, mock_vlm = integration_client
    
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "ask_current_image",
            "arguments": {
                "prompt": "Where is the file menu?"
            }
        },
        "id": 2
    }
    
    headers = {"Authorization": "Basic YWRtaW46cGFzc3dvcmQ="}
    response = client.post("/messages", json=payload, headers=headers)
    
    # Verify VLM called with image=None and use_kv_cache=True
    mock_vlm.generate.assert_called()
    args = mock_vlm.generate.call_args
    assert args.kwargs.get('image') is None
    assert args.kwargs.get('use_kv_cache') is True

def test_full_flow_get_coordinates(integration_client):
    client, mock_vlm = integration_client
    
    # Mock VLM to return a box string
    mock_vlm.generate.return_value = "<box>100 200 300 400</box>"
    
    with patch("src.mcp_server.WindowCapture") as MockCapture:
        mock_cap_instance = MockCapture.return_value
        from PIL import Image
        # 1000x1000 image for easy math (matches default scale)
        mock_cap_instance.capture_window_by_title.return_value = Image.new('RGB', (1000, 1000))
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "get_element_coordinates",
                "arguments": {
                    "window_title": "Notepad",
                    "element_description": "the cursor"
                }
            },
            "id": 3
        }
        
        headers = {"Authorization": "Basic YWRtaW46cGFzc3dvcmQ="}
        response = client.post("/messages", json=payload, headers=headers)
        
        data = response.json()
        result_text = data["result"]["content"][0]["text"]
        result_json = json.loads(result_text)
        
        # Verify parsed coordinates
        # Input <box>100 200 300 400</box> (y1 x1 y2 x2) or (x1 y1...)?
        # Our regex in coord_utils is <box>d d d d</box>. 
        # And normalize_coordinates assumes (x1, y1, x2, y2).
        # We need to verify the mocks align with parsing.
        
        assert "box_pixels" in result_json
        # Expected: 100, 200, 300, 400 (if scale is 1000 and img is 1000)
        assert result_json["box_pixels"] == [100, 200, 300, 400]
