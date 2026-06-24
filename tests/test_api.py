"""
tests/test_api.py
---------------------
Tests for the FastAPI endpoints using TestClient.
Note: requires a fine-tuned model already present at the configured
model_output_dir to actually load successfully -- these tests mock
the heavy model loading so they run fast and don't need a trained model.

Run:
    pytest tests/test_api.py -v
"""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def test_health_endpoint():
    with patch("app_fastapi.PredictionPipeline"):
        from app_fastapi import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_chat_endpoint_rejects_empty_message():
    with patch("app_fastapi.PredictionPipeline"):
        from app_fastapi import app
        client = TestClient(app)
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 400


def test_chat_endpoint_returns_reply():
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.generate_reply.return_value = "I'm here for you."

    with patch("app_fastapi.PredictionPipeline", return_value=mock_pipeline_instance):
        from app_fastapi import app
        client = TestClient(app)
        # Manually trigger the same startup logic the lifespan event would
        with client:
            response = client.post("/chat", json={"message": "I feel anxious"})
            assert response.status_code == 200
            assert "reply" in response.json()
