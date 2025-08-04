"""Test health check endpoint."""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["app_name"] == "MoneyPilot"
        assert "version" in data
        assert "timestamp" in data
        assert "debug" in data

    def test_health_check_response_format(self, client):
        """Test health check response format."""
        response = client.get("/api/v1/health")
        data = response.json()

        timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        assert isinstance(timestamp, datetime)

        required_fields = {"status", "app_name", "version", "timestamp", "debug"}
        assert set(data.keys()) == required_fields

    def test_health_check_debug_mode(self, client, monkeypatch):
        """Test health check reflects debug mode."""
        monkeypatch.setattr("app.utils.config.app_settings.DEBUG", True)
        response = client.get("/api/v1/health")
        assert response.json()["debug"] is True

        monkeypatch.setattr("app.utils.config.app_settings.DEBUG", False)
        response = client.get("/api/v1/health")
        assert response.json()["debug"] is False
