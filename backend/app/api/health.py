from datetime import datetime, timezone

from fastapi import APIRouter

from app.schema.api.v1.health import HealthCheckResponse
from app.utils.config import app_settings
from app.utils.logger import logger


router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.

    Returns basic application information and status.
    """
    logger.debug("Health check requested")

    return HealthCheckResponse(
        status="healthy",
        app_name=app_settings.APP_NAME,
        version=app_settings.VERSION,
        timestamp=datetime.now(timezone.utc),
        debug=app_settings.DEBUG,
    )
