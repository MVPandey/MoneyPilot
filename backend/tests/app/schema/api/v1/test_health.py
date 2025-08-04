"""Test health check response schema."""
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schema.api.v1.health import HealthCheckResponse


class TestHealthCheckResponse:
    """Test HealthCheckResponse schema."""

    def test_valid_health_check_response(self):
        """Test creating a valid health check response."""
        response = HealthCheckResponse(
            status="healthy",
            app_name="TestApp",
            version="1.0.0",
            timestamp=datetime.now(timezone.utc),
            debug=True,
        )
        
        assert response.status == "healthy"
        assert response.app_name == "TestApp"
        assert response.version == "1.0.0"
        assert response.debug is True
        assert isinstance(response.timestamp, datetime)

    def test_health_check_response_model_dump(self):
        """Test model dump for JSON serialization."""
        timestamp = datetime.now(timezone.utc)
        response = HealthCheckResponse(
            status="healthy",
            app_name="TestApp",
            version="1.0.0",
            timestamp=timestamp,
            debug=False,
        )
        
        dumped = response.model_dump()
        assert dumped["status"] == "healthy"
        assert dumped["app_name"] == "TestApp"
        assert dumped["version"] == "1.0.0"
        assert dumped["debug"] is False
        assert dumped["timestamp"] == timestamp

    def test_health_check_response_json_serialization(self):
        """Test JSON serialization of the response."""
        response = HealthCheckResponse(
            status="healthy",
            app_name="TestApp",
            version="1.0.0",
            timestamp=datetime.now(timezone.utc),
            debug=True,
        )
        
        json_str = response.model_dump_json()
        assert isinstance(json_str, str)
        assert "healthy" in json_str
        assert "TestApp" in json_str

    def test_health_check_response_missing_fields(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError) as exc_info:
            HealthCheckResponse(
                status="healthy",
                app_name="TestApp",
            )
        
        errors = exc_info.value.errors()
        assert len(errors) == 3
        missing_fields = {error["loc"][0] for error in errors}
        assert missing_fields == {"version", "timestamp", "debug"}
