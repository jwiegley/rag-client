#!/usr/bin/env python3
"""Tests for API endpoints.

This module tests the FastAPI server endpoints to ensure
they match the OpenAI API specification and handle errors properly.
"""

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from rag_client.api.server import api as app
from rag_client.config.models import Config, RetrievalConfig


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_workflow():
    """Create mock RAG workflow."""
    workflow = Mock()
    workflow.query = Mock(return_value="Test response")
    workflow.chat = Mock(return_value="Chat response")
    workflow.get_chat_engine = Mock()
    return workflow


@pytest.fixture
def auth_headers():
    """Create authorization headers."""
    return {"Authorization": "Bearer test-api-key"}


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health endpoint returns correct status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_root_redirect(self, client):
        """Test root path redirects to docs."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/docs"


class TestModelsEndpoint:
    """Tests for models listing endpoint."""
    
    def test_list_models_no_auth(self, client):
        """Test models endpoint without authentication."""
        with patch("rag_client.api.server.API_KEY", "test-key"):
            response = client.get("/v1/models")
            assert response.status_code == 401
            assert "Unauthorized" in response.json()["detail"]
    
    def test_list_models_with_auth(self, client, auth_headers):
        """Test models endpoint with authentication."""
        with patch("rag_client.api.server.API_KEY", "test-api-key"):
            response = client.get("/v1/models", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            assert "object" in data
            assert data["object"] == "list"
            assert "data" in data
            assert len(data["data"]) > 0
            assert data["data"][0]["id"] == "rag-model"


class TestChatCompletionsEndpoint:
    """Tests for chat completions endpoint."""
    
    def test_chat_completion_no_auth(self, client):
        """Test chat endpoint without authentication."""
        with patch("rag_client.api.server.API_KEY", "test-key"):
            response = client.post("/v1/chat/completions", json={
                "model": "rag-model",
                "messages": [{"role": "user", "content": "Hello"}]
            })
            assert response.status_code == 401
    
    def test_chat_completion_invalid_model(self, client, auth_headers, mock_workflow):
        """Test chat with invalid model name."""
        with patch("rag_client.api.server.API_KEY", "test-api-key"):
            with patch("rag_client.api.server.rag_workflow", mock_workflow):
                response = client.post(
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={
                        "model": "invalid-model",
                        "messages": [{"role": "user", "content": "Hello"}]
                    }
                )
                assert response.status_code == 400
                assert "Model not found" in response.json()["detail"]
    
    def test_chat_completion_empty_messages(self, client, auth_headers):
        """Test chat with empty messages."""
        with patch("rag_client.api.server.API_KEY", "test-api-key"):
            response = client.post(
                "/v1/chat/completions",
                headers=auth_headers,
                json={
                    "model": "rag-model",
                    "messages": []
                }
            )
            assert response.status_code == 400
            assert "No messages provided" in response.json()["detail"]
    
    def test_chat_completion_success(self, client, auth_headers, mock_workflow):
        """Test successful chat completion."""
        with patch("rag_client.api.server.API_KEY", "test-api-key"):
            with patch("rag_client.api.server.rag_workflow", mock_workflow):
                response = client.post(
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={
                        "model": "rag-model",
                        "messages": [{"role": "user", "content": "Hello"}]
                    }
                )
                assert response.status_code == 200
                data = response.json()
                
                # Verify OpenAI API format
                assert "id" in data
                assert data["object"] == "chat.completion"
                assert "created" in data
                assert data["model"] == "rag-model"
                assert "choices" in data
                assert len(data["choices"]) == 1
                assert data["choices"][0]["message"]["role"] == "assistant"
                assert data["choices"][0]["message"]["content"] == "Test response"
                assert data["choices"][0]["finish_reason"] == "stop"
                assert "usage" in data
    
    def test_chat_completion_with_temperature(self, client, auth_headers, mock_workflow):
        """Test chat completion with temperature parameter."""
        with patch("rag_client.api.server.API_KEY", "test-api-key"):
            with patch("rag_client.api.server.rag_workflow", mock_workflow):
                response = client.post(
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={
                        "model": "rag-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "temperature": 0.5
                    }
                )
                assert response.status_code == 200
    
    def test_chat_completion_with_max_tokens(self, client, auth_headers, mock_workflow):
        """Test chat completion with max_tokens parameter."""
        with patch("rag_client.api.server.API_KEY", "test-api-key"):
            with patch("rag_client.api.server.rag_workflow", mock_workflow):
                response = client.post(
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={
                        "model": "rag-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 100
                    }
                )
                assert response.status_code == 200
    
    def test_chat_completion_streaming(self, client, auth_headers, mock_workflow):
        """Test chat completion with streaming."""
        with patch("rag_client.api.server.API_KEY", "test-api-key"):
            with patch("rag_client.api.server.rag_workflow", mock_workflow):
                mock_workflow.stream_chat = Mock(return_value=iter([
                    "Hello", " ", "world"
                ]))
                
                response = client.post(
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={
                        "model": "rag-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True
                    }
                )
                assert response.status_code == 200
                
                # Check SSE format
                content = response.text
                assert "data: " in content
                assert "[DONE]" in content
    
    def test_chat_completion_error_handling(self, client, auth_headers, mock_workflow):
        """Test error handling in chat completion."""
        with patch("rag_client.api.server.API_KEY", "test-api-key"):
            with patch("rag_client.api.server.rag_workflow", mock_workflow):
                mock_workflow.query.side_effect = Exception("Test error")
                
                response = client.post(
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={
                        "model": "rag-model",
                        "messages": [{"role": "user", "content": "Hello"}]
                    }
                )
                assert response.status_code == 500
                assert "Internal server error" in response.json()["detail"]


class TestCompletionsEndpoint:
    """Tests for completions endpoint."""
    
    def test_completion_success(self, client, auth_headers, mock_workflow):
        """Test successful completion."""
        with patch("rag_client.api.server.API_KEY", "test-api-key"):
            with patch("rag_client.api.server.rag_workflow", mock_workflow):
                response = client.post(
                    "/v1/completions",
                    headers=auth_headers,
                    json={
                        "model": "rag-model",
                        "prompt": "Complete this"
                    }
                )
                assert response.status_code == 200
                data = response.json()
                
                # Verify OpenAI API format
                assert "id" in data
                assert data["object"] == "text_completion"
                assert "created" in data
                assert data["model"] == "rag-model"
                assert "choices" in data
                assert len(data["choices"]) == 1
                assert data["choices"][0]["text"] == "Test response"
                assert data["choices"][0]["finish_reason"] == "stop"
    
    def test_completion_invalid_model(self, client, auth_headers):
        """Test completion with invalid model."""
        with patch("rag_client.api.server.API_KEY", "test-api-key"):
            response = client.post(
                "/v1/completions",
                headers=auth_headers,
                json={
                    "model": "invalid-model",
                    "prompt": "Complete this"
                }
            )
            assert response.status_code == 400
            assert "Model not found" in response.json()["detail"]


class TestEmbeddingsEndpoint:
    """Tests for embeddings endpoint."""
    
    def test_embeddings_success(self, client, auth_headers, mock_workflow):
        """Test successful embedding generation."""
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_workflow.get_embedding = Mock(return_value=mock_embedding)
        
        with patch("rag_client.api.server.API_KEY", "test-api-key"):
            with patch("rag_client.api.server.rag_workflow", mock_workflow):
                response = client.post(
                    "/v1/embeddings",
                    headers=auth_headers,
                    json={
                        "model": "rag-embedding",
                        "input": "Test text"
                    }
                )
                assert response.status_code == 200
                data = response.json()
                
                # Verify OpenAI API format
                assert data["object"] == "list"
                assert "data" in data
                assert len(data["data"]) == 1
                assert data["data"][0]["object"] == "embedding"
                assert data["data"][0]["embedding"] == mock_embedding
                assert data["data"][0]["index"] == 0
                assert "usage" in data
                assert data["usage"]["prompt_tokens"] > 0
    
    def test_embeddings_batch(self, client, auth_headers, mock_workflow):
        """Test batch embedding generation."""
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_workflow.get_embedding = Mock(side_effect=mock_embeddings)
        
        with patch("rag_client.api.server.API_KEY", "test-api-key"):
            with patch("rag_client.api.server.rag_workflow", mock_workflow):
                response = client.post(
                    "/v1/embeddings",
                    headers=auth_headers,
                    json={
                        "model": "rag-embedding",
                        "input": ["Text 1", "Text 2"]
                    }
                )
                assert response.status_code == 200
                data = response.json()
                assert len(data["data"]) == 2


class TestErrorHandling:
    """Tests for error handling across endpoints."""
    
    def test_malformed_json(self, client, auth_headers):
        """Test handling of malformed JSON."""
        with patch("rag_client.api.server.API_KEY", "test-api-key"):
            response = client.post(
                "/v1/chat/completions",
                headers={**auth_headers, "Content-Type": "application/json"},
                content="invalid json"
            )
            assert response.status_code == 422
    
    def test_missing_required_fields(self, client, auth_headers):
        """Test handling of missing required fields."""
        with patch("rag_client.api.server.API_KEY", "test-api-key"):
            response = client.post(
                "/v1/chat/completions",
                headers=auth_headers,
                json={
                    "model": "rag-model"
                    # Missing messages field
                }
            )
            assert response.status_code == 422
    
    def test_workflow_initialization_error(self, client, auth_headers):
        """Test handling of workflow initialization errors."""
        with patch("rag_client.api.server.rag_workflow", None):
            with patch("rag_client.api.server.API_KEY", "test-api-key"):
                response = client.post(
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={
                        "model": "rag-model",
                        "messages": [{"role": "user", "content": "Hello"}]
                    }
                )
                assert response.status_code == 500


class TestMiddleware:
    """Tests for API middleware."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/v1/models")
        assert "access-control-allow-origin" in response.headers
    
    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting if implemented."""
        # This would test rate limiting if implemented
        pass
    
    def test_request_id_header(self, client):
        """Test request ID header is added."""
        response = client.get("/health")
        # Check if request ID header is present if implemented
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])