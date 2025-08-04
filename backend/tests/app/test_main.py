"""Test main FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app, lifespan
from app.utils.config import app_settings


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestMainApp:
    """Test main FastAPI application configuration."""

    def test_app_configuration(self):
        """Test FastAPI app is configured correctly."""
        assert app.title == "MoneyPilot"
        assert app.version == "0.1.0"
        assert "/api/v1/docs" in app.docs_url
        assert "/api/v1/redoc" in app.redoc_url
        assert "/api/v1/openapi.json" in app.openapi_url

    def test_cors_middleware_configured(self):
        """Test CORS middleware is configured."""
        middlewares = [str(m) for m in app.user_middleware]
        assert any("CORSMiddleware" in m for m in middlewares)

    def test_health_router_included(self):
        """Test health router is included."""
        routes = [route.path for route in app.routes]
        assert "/api/v1/health" in routes

    @pytest.mark.asyncio
    async def test_lifespan_startup_shutdown(self):
        """Test lifespan context manager for startup/shutdown."""
        with patch("app.main.logger") as mock_logger:
            mock_app = MagicMock()

            async with lifespan(mock_app):
                mock_logger.info.assert_any_call(
                    "Starting MoneyPilot application",
                    extra={"version": app_settings.VERSION},
                )
                mock_logger.info.assert_any_call(
                    "Configuration loaded", extra=app_settings.get_feature_summary()
                )

            mock_logger.info.assert_called_with("Shutting down MoneyPilot application")

    @pytest.mark.asyncio
    async def test_global_exception_handler(self):
        """Test global exception handler."""
        from app.main import global_exception_handler
        from starlette.requests import Request
        from starlette.datastructures import URL

        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock(spec=URL)
        mock_request.url.path = "/test-path"
        mock_request.method = "GET"

        with patch("app.main.logger") as mock_logger:
            test_exception = ValueError("Test error")

            response = await global_exception_handler(mock_request, test_exception)

            assert response.status_code == 500
            assert response.body == b'{"detail":"Internal server error"}'

            mock_logger.error.assert_called_once_with(
                "Unhandled exception",
                extra={
                    "path": "/test-path",
                    "method": "GET",
                    "exception": "Test error",
                },
                exc_info=True,
            )

    def test_api_prefix_applied(self, client):
        """Test API prefix is correctly applied to routes."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        response = client.get("/health")
        assert response.status_code == 404

    def test_openapi_schema_available(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert schema["info"]["title"] == "MoneyPilot"
        assert schema["info"]["version"] == "0.1.0"
        assert "paths" in schema
        assert "/api/v1/health" in schema["paths"]
