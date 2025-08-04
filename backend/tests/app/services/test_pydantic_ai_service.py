"""Test PydanticAI service functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from app.services.pydantic_ai_service import PydanticAIService, SimpleAgent
from app.utils.exceptions import AgentException


class OutputModel(BaseModel):
    """Test output model."""

    value: str
    score: float


class WeatherInfo(BaseModel):
    """Weather information model for testing."""

    temperature: float
    condition: str
    humidity: int


class TestPydanticAIService:
    """Test PydanticAI service functionality."""

    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    @patch("app.services.pydantic_ai_service.app_settings")
    def test_init_with_defaults(
        self, mock_settings, mock_model_class, mock_provider_class
    ):
        """Test initialization with default values."""
        mock_settings.LLM_MODEL_NAME = "gpt-4"
        mock_settings.LLM_API_BASE_URL = "https://api.test.com"
        mock_settings.LLM_API_KEY = "test-key"

        service = PydanticAIService()

        mock_provider_class.assert_called_once_with(
            base_url="https://api.test.com", api_key="test-key"
        )

        mock_model_class.assert_called_once_with(
            model_name="gpt-4", provider=mock_provider_class.return_value
        )

        assert service.model_name == "gpt-4"
        assert service.base_url == "https://api.test.com"
        assert service.api_key == "test-key"
        assert service.system_prompt is None
        assert service.output_type is None
        assert service.tools == []

    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    def test_init_with_custom_values(self, mock_model_class, mock_provider_class):
        """Test initialization with custom values."""
        service = PydanticAIService(
            output_type=OutputModel,
            system_prompt="You are a test assistant",
            model_name="custom-model",
            base_url="https://custom.api.com",
            api_key="custom-key",
            tools=["tool1", "tool2"],
        )

        mock_provider_class.assert_called_once_with(
            base_url="https://custom.api.com", api_key="custom-key"
        )

        mock_model_class.assert_called_once_with(
            model_name="custom-model", provider=mock_provider_class.return_value
        )

        assert service.model_name == "custom-model"
        assert service.base_url == "https://custom.api.com"
        assert service.api_key == "custom-key"
        assert service.system_prompt == "You are a test assistant"
        assert service.output_type == OutputModel
        assert service.tools == ["tool1", "tool2"]

    @patch.dict(
        "os.environ",
        {"LLM_API_BASE_URL": "https://env.api.com", "LLM_API_KEY": "env-key"},
    )
    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    @patch("app.services.pydantic_ai_service.app_settings")
    def test_init_with_env_vars(
        self, mock_settings, mock_model_class, mock_provider_class
    ):
        """Test initialization with environment variables."""
        mock_settings.LLM_MODEL_NAME = "gpt-4"
        mock_settings.LLM_API_BASE_URL = "https://default.api.com"
        mock_settings.LLM_API_KEY = "default-key"

        service = PydanticAIService()

        mock_provider_class.assert_called_once_with(
            base_url="https://env.api.com", api_key="env-key"
        )

        assert service.base_url == "https://env.api.com"
        assert service.api_key == "env-key"

    @patch("app.services.pydantic_ai_service.Agent")
    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    def test_create_agent_default(
        self, mock_model_class, mock_provider_class, mock_agent_class
    ):
        """Test create_agent with default values."""
        service = PydanticAIService(
            output_type=OutputModel,
            system_prompt="Default prompt",
            tools=["tool1", "tool2"],
        )

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        agent = service.create_agent()

        mock_agent_class.assert_called_once_with(
            model=service.model, system_prompt="Default prompt", output_type=OutputModel
        )

        assert mock_agent_instance.tool.call_count == 2
        mock_agent_instance.tool.assert_any_call("tool1")
        mock_agent_instance.tool.assert_any_call("tool2")

        assert agent == mock_agent_instance

    @patch("app.services.pydantic_ai_service.Agent")
    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    def test_create_agent_with_overrides(
        self, mock_model_class, mock_provider_class, mock_agent_class
    ):
        """Test create_agent with override values."""
        service = PydanticAIService(
            output_type=OutputModel, system_prompt="Default prompt", tools=["tool1"]
        )

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        service.create_agent(
            system_prompt="Override prompt",
            output_type=WeatherInfo,
            tools=["tool3", "tool4"],
        )

        mock_agent_class.assert_called_once_with(
            model=service.model,
            system_prompt="Override prompt",
            output_type=WeatherInfo,
        )

        assert mock_agent_instance.tool.call_count == 2
        mock_agent_instance.tool.assert_any_call("tool3")
        mock_agent_instance.tool.assert_any_call("tool4")

    @pytest.mark.asyncio
    @patch("app.services.pydantic_ai_service.Agent")
    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    async def test_run_success(
        self, mock_model_class, mock_provider_class, mock_agent_class
    ):
        """Test successful run operation."""
        service = PydanticAIService(output_type=OutputModel)

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        mock_result = MagicMock()
        mock_result.data = OutputModel(value="test", score=0.9)
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        result = await service.run("Test prompt", context={"key": "value"})

        mock_agent_instance.run.assert_called_once_with(
            "Test prompt", deps={"key": "value"}
        )
        assert result == mock_result.data

    @pytest.mark.asyncio
    @patch("app.services.pydantic_ai_service.Agent")
    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    async def test_run_without_context(
        self, mock_model_class, mock_provider_class, mock_agent_class
    ):
        """Test run without context."""
        service = PydanticAIService()

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        mock_result = MagicMock()
        mock_result.data = "String response"
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        result = await service.run("Test prompt")

        mock_agent_instance.run.assert_called_once_with("Test prompt", deps=None)
        assert result == "String response"

    @pytest.mark.asyncio
    @patch("app.services.pydantic_ai_service.Agent")
    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    async def test_run_with_exception(
        self, mock_model_class, mock_provider_class, mock_agent_class
    ):
        """Test run with exception."""
        service = PydanticAIService()

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance
        mock_agent_instance.run = AsyncMock(side_effect=RuntimeError("Test error"))

        with pytest.raises(AgentException) as exc_info:
            await service.run("Test prompt")

        assert "Agent execution failed: Test error" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("app.services.pydantic_ai_service.Agent")
    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    async def test_run_stream_success(
        self, mock_model_class, mock_provider_class, mock_agent_class
    ):
        """Test successful run_stream operation."""
        service = PydanticAIService()

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        mock_response = AsyncMock()
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_agent_instance.run_stream.return_value = mock_stream_context

        async def mock_stream_text():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        mock_response.stream_text = mock_stream_text

        chunks = []
        async for chunk in service.run_stream("Test prompt", context={"key": "value"}):
            chunks.append(chunk)

        mock_agent_instance.run_stream.assert_called_once_with(
            "Test prompt", deps={"key": "value"}
        )
        assert chunks == ["chunk1", "chunk2", "chunk3"]

    @pytest.mark.asyncio
    @patch("app.services.pydantic_ai_service.Agent")
    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    async def test_run_stream_with_exception(
        self, mock_model_class, mock_provider_class, mock_agent_class
    ):
        """Test run_stream with exception."""
        service = PydanticAIService()

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance
        mock_agent_instance.run_stream.side_effect = RuntimeError("Stream error")

        with pytest.raises(AgentException) as exc_info:
            async for _ in service.run_stream("Test prompt"):
                pass

        assert "Agent streaming failed: Stream error" in str(exc_info.value)


class TestSimpleAgent:
    """Test SimpleAgent functionality."""

    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    def test_init(self, mock_model_class, mock_provider_class):
        """Test SimpleAgent initialization."""
        agent = SimpleAgent(
            system_prompt="Test prompt",
            output_type=OutputModel,
            model_name="test-model",
        )

        assert agent.system_prompt == "Test prompt"
        assert agent.output_type == OutputModel
        assert agent.model_name == "test-model"

    @patch("app.services.pydantic_ai_service.Agent")
    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    def test_create_agent_with_parameters(
        self, mock_model_class, mock_provider_class, mock_agent_class
    ):
        """Test create_agent with parameters."""
        agent = SimpleAgent(system_prompt="Hello {name}, you are in {location}")

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        agent.create_agent(parameters={"name": "John", "location": "New York"})

        mock_agent_class.assert_called_once_with(
            model=agent.model,
            system_prompt="Hello John, you are in New York",
            output_type=None,
        )

    @patch("app.services.pydantic_ai_service.Agent")
    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    def test_create_agent_without_parameters(
        self, mock_model_class, mock_provider_class, mock_agent_class
    ):
        """Test create_agent without parameters."""
        agent = SimpleAgent(system_prompt="Static prompt")

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        agent.create_agent()

        mock_agent_class.assert_called_once_with(
            model=agent.model, system_prompt="Static prompt", output_type=None
        )

    @pytest.mark.asyncio
    @patch("app.services.pydantic_ai_service.Agent")
    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    async def test_run_with_parameters(
        self, mock_model_class, mock_provider_class, mock_agent_class
    ):
        """Test run with parameters."""
        agent = SimpleAgent(
            system_prompt="Assistant for {user_type}", output_type=OutputModel
        )

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        mock_result = MagicMock()
        mock_result.data = OutputModel(value="test", score=0.8)
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        result = await agent.run("Test prompt", parameters={"user_type": "developer"})

        expected_prompt = "Assistant for developer"
        mock_agent_class.assert_called_with(
            model=agent.model, system_prompt=expected_prompt, output_type=OutputModel
        )

        assert result == mock_result.data

    @pytest.mark.asyncio
    @patch("app.services.pydantic_ai_service.Agent")
    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    async def test_run_stream_with_parameters(
        self, mock_model_class, mock_provider_class, mock_agent_class
    ):
        """Test run_stream with parameters."""
        agent = SimpleAgent(system_prompt="Stream for {mode}")

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        mock_response = AsyncMock()
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_agent_instance.run_stream.return_value = mock_stream_context

        async def mock_stream_text():
            yield "data1"
            yield "data2"

        mock_response.stream_text = mock_stream_text

        chunks = []
        async for chunk in agent.run_stream("Test prompt", parameters={"mode": "fast"}):
            chunks.append(chunk)

        mock_agent_class.assert_called_with(
            model=agent.model, system_prompt="Stream for fast", output_type=None
        )

        assert chunks == ["data1", "data2"]

    @pytest.mark.asyncio
    @patch("app.services.pydantic_ai_service.Agent")
    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    async def test_backward_compatibility(
        self, mock_model_class, mock_provider_class, mock_agent_class
    ):
        """Test backward compatibility without output_type."""
        agent = SimpleAgent("You are a helpful assistant")

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        mock_result = MagicMock()
        mock_result.data = "Plain text response"
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        result = await agent.run("What is the weather?")

        assert result == "Plain text response"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    @patch("app.services.pydantic_ai_service.Agent")
    @patch("app.services.pydantic_ai_service.OpenAIProvider")
    @patch("app.services.pydantic_ai_service.OpenAIModel")
    async def test_run_with_kwargs(
        self, mock_model_class, mock_provider_class, mock_agent_class
    ):
        """Test run with additional kwargs."""
        agent = SimpleAgent()

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        mock_result = MagicMock()
        mock_result.data = "Response"
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        await agent.run("Test prompt", temperature=0.7, max_tokens=100)

        mock_agent_instance.run.assert_called_once_with(
            "Test prompt", deps=None, temperature=0.7, max_tokens=100
        )
