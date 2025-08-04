"""Shared pytest fixtures and configuration."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "APP_NAME": "MoneyPilot Test",
        "VERSION": "0.1.0-test",
        "DEBUG": True,
        "API_PREFIX": "/api/v1",
        "LOG_LEVEL": "DEBUG",
    }
