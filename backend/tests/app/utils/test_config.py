"""Test configuration module."""
from unittest.mock import patch

from app.utils.config import Config


class TestConfig:
    """Test configuration settings."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.APP_NAME == "MoneyPilot"
        assert config.VERSION == "0.1.0"
        assert config.DEBUG is False
        assert config.LOG_LEVEL == "INFO"
        assert config.API_PREFIX == "/api/v1"
        assert config.CORS_ORIGINS == ["http://localhost:3000"]
        assert config.LLM_TIMEOUT_SECONDS == 600
        assert config.ACCESS_TOKEN_EXPIRE_MINUTES == 30

    def test_get_feature_summary(self):
        """Test feature summary generation."""
        config = Config()
        summary = config.get_feature_summary()
        
        assert summary["app_name"] == "MoneyPilot"
        assert summary["version"] == "0.1.0"
        assert summary["debug"] is False
        assert summary["llm_configured"] is False
        assert summary["api_prefix"] == "/api/v1"

    def test_config_with_llm_key(self):
        """Test config with LLM API key."""
        config = Config(LLM_API_KEY="test-key")
        summary = config.get_feature_summary()
        
        assert summary["llm_configured"] is True

    @patch.dict("os.environ", {"DEBUG": "true", "LOG_LEVEL": "DEBUG"})
    def test_config_from_env(self):
        """Test loading config from environment variables."""
        config = Config()
        
        assert config.DEBUG is True
        assert config.LOG_LEVEL == "DEBUG"

    @patch.dict("os.environ", {
        "APP_NAME": "TestApp",
        "VERSION": "2.0.0",
        "API_PREFIX": "/api/v2",
        "LLM_MODEL_NAME": "gpt-4-turbo"
    })
    def test_config_override_from_env(self):
        """Test overriding multiple config values from environment."""
        config = Config()
        
        assert config.APP_NAME == "TestApp"
        assert config.VERSION == "2.0.0"
        assert config.API_PREFIX == "/api/v2"
        assert config.LLM_MODEL_NAME == "gpt-4-turbo"

    def test_cors_origins_as_list(self):
        """Test CORS origins is always a list."""
        config = Config()
        assert isinstance(config.CORS_ORIGINS, list)
        assert len(config.CORS_ORIGINS) > 0

    def test_secret_key_default(self):
        """Test secret key has a default value."""
        config = Config()
        assert config.SECRET_KEY == "your-secret-key-here-please-change-in-production"
        assert len(config.SECRET_KEY) > 0

    @patch.dict("os.environ", {"SECRET_KEY": "super-secret-production-key"})
    def test_secret_key_from_env(self):
        """Test loading secret key from environment."""
        config = Config()
        assert config.SECRET_KEY == "super-secret-production-key"
