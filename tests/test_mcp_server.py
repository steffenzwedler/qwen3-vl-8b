"""
Tests for MCP Server API
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os

# We will import app from src.mcp_server once created
# For now we can structure the test to import safely
# or just assume it will exist.

@pytest.fixture
def client():
    # Mock credentials in env
    os.environ["MCP_USER"] = "testuser"
    os.environ["MCP_PASS"] = "testpass"
    
    # Patch the VLMInference class to avoid loading the model
    with patch("src.mcp_server.VLMInference") as MockVLM:
        from src.mcp_server import app
        # Determine if we need to set the global instance manually or if the startup will use the mock
        # startup uses VLMInference(), which is now MockVLM()
        
        with TestClient(app) as c:
            # We can also access the global instance if needed, but the mock should handle it
            # The lifespan context runs on entering TestClient context
            yield c

def test_auth_missing_header(client):
    response = client.post("/messages", json={})
    assert response.status_code == 401

def test_auth_invalid_creds(client):
    response = client.post(
        "/messages", 
        json={}, 
        auth=("wrong", "wrong")
    )
    assert response.status_code == 401

def test_auth_success(client):
    # Valid JSON-RPC request
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1
    }
    response = client.post(
        "/messages", 
        json=payload, 
        auth=("testuser", "testpass")
    )
    assert response.status_code == 200
    # Should perform JSON-RPC response validation here
    data = response.json()
    assert data["result"] is not None

def test_list_windows_tool(client):
    # Mock window capture
    with patch("src.mcp_server.WindowCapture.list_windows") as mock_list:
        mock_list.return_value = [(123, "Notepad"), (456, "Calculator")]
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "list_windows",
                "arguments": {}
            },
            "id": 2
        }
        response = client.post(
            "/messages", 
            json=payload, 
            auth=("testuser", "testpass")
        )
        assert response.status_code == 200
        result = response.json()["result"]
        # FastMCP structure might verify content
        assert "Notepad" in str(result)
        assert "Calculator" in str(result)

def test_capture_and_analyze_tool(client):
    with patch("src.mcp_server.WindowCapture.capture_window_by_title") as mock_cap:
        mock_cap.return_value = MagicMock() # PIL Image
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "capture_and_analyze",
                "arguments": {
                    "window_title": "Notepad",
                    "prompt": "What do you see?"
                }
            },
            "id": 3
        }
        response = client.post(
            "/messages", 
            json=payload, 
            auth=("testuser", "testpass")
        )
        assert response.status_code == 200
        # Check result has content
