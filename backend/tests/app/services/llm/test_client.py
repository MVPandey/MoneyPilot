"""Test LLM client functionality."""
from unittest.mock import patch, MagicMock

from app.services.llm.client import LLMClient


class TestLLMClient:
    """Test LLM client functionality."""

    @patch("app.services.llm.client.AsyncOpenAI")
    def test_llm_client_initialization(self, mock_openai_class):
        """Test LLM client initialization."""
        client = LLMClient(
            base_url="https://api.test.com",
            api_key="test-key",
            timeout=30.0,
            max_retries=5
        )
        
        assert client.base_url == "https://api.test.com"
        assert client.api_key == "test-key"
        assert client.timeout == 30.0
        assert client.max_retries == 5

    @patch("app.services.llm.client.app_settings")
    @patch("app.services.llm.client.AsyncOpenAI")
    def test_llm_client_default_values(self, mock_openai_class, mock_settings):
        """Test LLM client with default values from config."""
        mock_settings.LLM_API_BASE_URL = "https://default.api.com"
        mock_settings.LLM_API_KEY = "default-key"
        mock_settings.LLM_TIMEOUT_SECONDS = 600
        
        client = LLMClient()
        
        assert client.base_url == "https://default.api.com"
        assert client.api_key == "default-key"
        assert client.timeout == 600.0

    @patch("app.services.llm.client.AsyncOpenAI")
    def test_get_client(self, mock_openai_class):
        """Test getting the client instance."""
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance
        
        client = LLMClient(api_key="test-key")
        result = client.get_client()
        
        mock_openai_class.assert_called_once_with(
            base_url=client.base_url,
            api_key="test-key",
            timeout=client.timeout,
            max_retries=client.max_retries
        )
        assert result == mock_client_instance

    def test_get_retry_decorator(self):
        """Test retry decorator creation."""
        decorator = LLMClient.get_retry_decorator()
        
        assert decorator is not None
        assert callable(decorator)

    @patch("app.services.llm.client.logger")
    def test_log_retry_attempt(self, mock_logger):
        """Test retry attempt logging."""
        mock_retry_state = MagicMock()
        mock_retry_state.attempt_number = 2
        mock_retry_state.outcome = MagicMock()
        mock_retry_state.outcome.exception.return_value = Exception("Test error")
        
        LLMClient._log_retry_attempt(mock_retry_state)
        
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "Retrying LLM request" in call_args[0][0]
        assert call_args[1]["extra"]["attempt"] == 2
        assert "Test error" in str(call_args[1]["extra"]["exception"])

    @patch("app.services.llm.client.logger")
    def test_log_retry_attempt_first_try(self, mock_logger):
        """Test that first attempt is not logged."""
        mock_retry_state = MagicMock()
        mock_retry_state.attempt_number = 1
        
        LLMClient._log_retry_attempt(mock_retry_state)
        
        mock_logger.warning.assert_not_called()
