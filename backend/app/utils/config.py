from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        str_strip_whitespace=True,
        strict=False,
        extra="ignore",
    )

    # ================================================================
    # Application settings
    # ================================================================
    APP_NAME: str = Field(default="MoneyPilot", description="Application name")
    VERSION: str = Field(default="0.1.0", description="Application version")
    DEBUG: bool = Field(default=False, description="Debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # ================================================================
    # API settings
    # ================================================================
    API_PREFIX: str = Field(default="/api/v1", description="API prefix")
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    # ================================================================
    # LLM Configuration (Optional for now)
    # ================================================================
    LLM_API_KEY: str | None = Field(default=None, description="LLM API key (optional)")
    LLM_API_BASE_URL: str = Field(default="https://api.openai.com/v1", description="LLM API base URL")
    LLM_MODEL_NAME: str = Field(default="gpt-4", description="LLM model name")
    LLM_TIMEOUT_SECONDS: int = Field(default=600, description="LLM request timeout")
    
    # ================================================================
    # Security settings
    # ================================================================
    SECRET_KEY: str = Field(
        default="your-secret-key-here-please-change-in-production",
        description="Secret key for JWT and other security features"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiry time in minutes")
    
    def get_feature_summary(self) -> dict:
        """Get summary of enabled features."""
        return {
            "app_name": self.APP_NAME,
            "version": self.VERSION,
            "debug": self.DEBUG,
            "llm_configured": bool(self.LLM_API_KEY),
            "api_prefix": self.API_PREFIX,
        }


app_settings = Config()