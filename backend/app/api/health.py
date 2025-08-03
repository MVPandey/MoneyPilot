from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from app.utils.config import app_settings
from app.utils.logger import logger


# Response models
class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    app_name: str
    version: str
    timestamp: datetime
    debug: bool


# Create router
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
        timestamp=datetime.utcnow(),
        debug=app_settings.DEBUG,
    )