from datetime import datetime

from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: str
    app_name: str
    version: str
    timestamp: datetime
    debug: bool
