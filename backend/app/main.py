from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.utils.config import app_settings
from app.utils.logger import logger
from app.api import health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    # Startup
    logger.info("Starting MoneyPilot application", extra={"version": app_settings.VERSION})
    logger.info("Configuration loaded", extra=app_settings.get_feature_summary())
    
    yield
    
    # Shutdown
    logger.info("Shutting down MoneyPilot application")


# Create FastAPI app
app = FastAPI(
    title=app_settings.APP_NAME,
    version=app_settings.VERSION,
    debug=app_settings.DEBUG,
    lifespan=lifespan,
    docs_url=f"{app_settings.API_PREFIX}/docs",
    redoc_url=f"{app_settings.API_PREFIX}/redoc",
    openapi_url=f"{app_settings.API_PREFIX}/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix=app_settings.API_PREFIX, tags=["health"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        extra={
            "path": request.url.path,
            "method": request.method,
            "exception": str(exc),
        },
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=app_settings.DEBUG,
        log_level=app_settings.LOG_LEVEL.lower(),
    )