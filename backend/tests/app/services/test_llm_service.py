"""Test LLM service functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from app.services.llm_service import LLMService
from app.schema.llm.message import Message
from app.utils.exceptions import LLMException


class TestLLMService:
    """Test LLM service functionality."""

    def create_mock_completion(self, content="Test response", tool_calls=None):
        """Helper to create mock ChatCompletion."""
        message = ChatCompletionMessage(
            role="assistant", content=content, tool_calls=tool_calls
        )

        choice = Choice(index=0, message=message, finish_reason="stop")

        usage = CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        return ChatCompletion(
            id="test_completion",
            model="test-model",
            object="chat.completion",
            created=1234567890,
            choices=[choice],
            usage=usage,
        )

    @patch("app.services.llm_service.LLMClient")
    def test_llm_service_initialization(self, mock_client_class):
        """Test LLM service initialization."""
        service = LLMService(
            base_url="https://test.api.com",
            api_key="test-key",
            model_name="test-model",
            timeout=30.0,
        )

        assert service.model_name == "test-model"
        mock_client_class.assert_called_once_with(
            base_url="https://test.api.com", api_key="test-key", timeout=30.0
        )

    @pytest.mark.asyncio
    @patch("app.services.llm_service.LLMClient")
    async def test_query_llm_simple_text(self, mock_client_class):
        """Test simple text query to LLM."""
        mock_client = MagicMock()
        mock_async_client = AsyncMock()
        mock_client.get_client.return_value = mock_async_client
        mock_client_class.return_value = mock_client

        mock_completion = self.create_mock_completion("Hello, world!")
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )

        service = LLMService()
        service.retry_decorator = lambda f: f
        message = await service.query_llm(Message(role="user", content="Say hello"))

        assert isinstance(message, ChatCompletionMessage)
        assert message.content == "Hello, world!"

    @pytest.mark.asyncio
    @patch("app.services.llm_service.LLMClient")
    async def test_query_llm_json_response(self, mock_client_class):
        """Test JSON response query."""
        mock_client = MagicMock()
        mock_async_client = AsyncMock()
        mock_client.get_client.return_value = mock_async_client
        mock_client_class.return_value = mock_client

        json_content = '{"status": "success", "value": 42}'
        mock_completion = self.create_mock_completion(json_content)
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )

        service = LLMService()
        service.retry_decorator = lambda f: f

        result = await service.query_llm(
            Message(role="user", content="Get status"), json_response=True
        )

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["value"] == 42

    @pytest.mark.asyncio
    @patch("app.services.llm_service.LLMClient")
    async def test_query_llm_with_messages_list(self, mock_client_class):
        """Test query with multiple messages."""
        mock_client = MagicMock()
        mock_async_client = AsyncMock()
        mock_client.get_client.return_value = mock_async_client
        mock_client_class.return_value = mock_client

        mock_completion = self.create_mock_completion("Response")
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )

        service = LLMService()
        service.retry_decorator = lambda f: f

        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
        ]

        await service.query_llm(messages)

        call_args = mock_async_client.chat.completions.create.call_args
        assert len(call_args[1]["messages"]) == 2
        assert call_args[1]["messages"][0]["role"] == "system"
        assert call_args[1]["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    @patch("app.services.llm_service.LLMClient")
    async def test_query_llm_validation_errors(self, mock_client_class):
        """Test parameter validation."""
        service = LLMService()

        with pytest.raises(ValueError) as exc_info:
            await service.query_llm(Message(role="user", content="Test"), max_tokens=-1)
        assert "max_tokens must be positive" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            await service.query_llm(
                Message(role="user", content="Test"), temperature=3.0
            )
        assert "temperature must be between 0 and 2" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            await service.query_llm(Message(role="user", content="Test"), top_p=1.5)
        assert "top_p must be between 0 and 1" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("app.services.llm_service.LLMClient")
    async def test_query_llm_json_parsing_error(self, mock_client_class):
        """Test JSON parsing error handling."""
        mock_client = MagicMock()
        mock_async_client = AsyncMock()
        mock_client.get_client.return_value = mock_async_client
        mock_client_class.return_value = mock_client

        mock_completion = self.create_mock_completion("Not valid JSON")
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )

        service = LLMService()
        service.retry_decorator = lambda f: f

        with pytest.raises(LLMException) as exc_info:
            await service.query_llm(
                Message(role="user", content="Test"), json_response=True
            )

        assert "Failed to parse JSON response" in str(exc_info.value)

    def test_normalize_messages(self):
        """Test message normalization."""
        service = LLMService()

        single = Message(role="user", content="Test")
        result = service._normalize_messages(single)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == single

        messages = [
            Message(role="user", content="Test1"),
            Message(role="assistant", content="Test2"),
        ]
        result = service._normalize_messages(messages)
        assert result == messages

    def test_prepare_tools(self):
        """Test tool preparation."""
        service = LLMService()

        with patch.object(
            service.tool_executor.registry, "get_tool_schemas"
        ) as mock_get_schemas:
            mock_get_schemas.return_value = [
                {"type": "function", "function": {"name": "tool1"}},
                {"type": "function", "function": {"name": "tool2"}},
            ]

            service._prepare_tools("tool1")
            mock_get_schemas.assert_called_with(["tool1"])

            service._prepare_tools(["tool1", "tool2"])
            mock_get_schemas.assert_called_with(["tool1", "tool2"])

    @pytest.mark.asyncio
    @patch("app.services.llm_service.LLMClient")
    async def test_query_llm_with_custom_parameters(self, mock_client_class):
        """Test query with custom parameters."""
        mock_client = MagicMock()
        mock_async_client = AsyncMock()
        mock_client.get_client.return_value = mock_async_client
        mock_client_class.return_value = mock_client

        mock_completion = self.create_mock_completion("Response")
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )

        service = LLMService()
        service.retry_decorator = lambda f: f

        await service.query_llm(
            Message(role="user", content="Test"),
            temperature=0.7,
            top_p=0.9,
            max_tokens=500,
        )

        call_args = mock_async_client.chat.completions.create.call_args
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["top_p"] == 0.9
        assert call_args[1]["max_tokens"] == 500

    @pytest.mark.asyncio
    @patch("app.services.llm_service.LLMClient")
    async def test_query_llm_empty_response_json(self, mock_client_class):
        """Test handling empty content for JSON response."""
        mock_client = MagicMock()
        mock_async_client = AsyncMock()
        mock_client.get_client.return_value = mock_async_client
        mock_client_class.return_value = mock_client

        mock_completion = self.create_mock_completion(content=None)
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )

        service = LLMService()
        service.retry_decorator = lambda f: f

        with pytest.raises(LLMException) as exc_info:
            await service.query_llm(
                Message(role="user", content="Test"), json_response=True
            )

        assert "returned None content" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("app.services.llm_service.LLMClient")
    async def test_query_llm_with_tools(self, mock_client_class):
        """Test query with tools enabling function calling."""
        mock_client = MagicMock()
        mock_async_client = AsyncMock()
        mock_client.get_client.return_value = mock_async_client
        mock_client_class.return_value = mock_client

        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
        )
        from openai.types.chat.chat_completion_message_tool_call import Function

        tool_call = ChatCompletionMessageToolCall(
            id="call_123",
            type="function",
            function=Function(name="test_tool", arguments='{"arg": "value"}'),
        )

        first_completion = self.create_mock_completion("", tool_calls=[tool_call])
        final_completion = self.create_mock_completion("Tool executed successfully")

        mock_async_client.chat.completions.create = AsyncMock(
            side_effect=[first_completion, final_completion]
        )

        with patch.object(LLMService, "_prepare_tools") as mock_prepare:
            mock_prepare.return_value = [
                {"type": "function", "function": {"name": "test_tool"}}
            ]

            service = LLMService()
            service.retry_decorator = lambda f: f

            tool_result = MagicMock()
            tool_result.model_dump.return_value = {
                "role": "tool",
                "content": '{"result": "success"}',
                "tool_call_id": "call_123",
                "name": "test_tool",
            }
            service.tool_executor.execute_tool_calls = AsyncMock(
                return_value=[tool_result]
            )

            result = await service.query_llm(
                Message(role="user", content="Use the tool"), tools=["test_tool"]
            )

            assert result.content == "Tool executed successfully"
            service.tool_executor.execute_tool_calls.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_tool_calls_backward_compatibility(self):
        """Test handle_tool_calls method for backward compatibility."""
        from app.schema.llm.tool import ToolCall, ToolCallFunction

        service = LLMService()

        tool_call = ToolCall(
            id="test_id",
            type="function",
            function=ToolCallFunction(name="test", arguments="{}"),
        )

        mock_result = MagicMock()
        service.tool_executor.execute_tool_calls = AsyncMock(return_value=[mock_result])

        result = await service.handle_tool_calls([tool_call])

        assert result == [mock_result]
        service.tool_executor.execute_tool_calls.assert_called_once_with([tool_call])

    @pytest.mark.asyncio
    async def test_extract_json_from_response(self):
        """Test JSON extraction method."""
        service = LLMService()

        result = await service._extract_json_from_response('{"test": "value"}')
        assert result == {"test": "value"}

        with patch("app.services.llm_service.clean_json_response") as mock_clean:
            mock_clean.return_value = {"cleaned": True}

            result = await service._extract_json_from_response("not json")

            assert result == {"cleaned": True}
            mock_clean.assert_called_once_with("not json")

    def test_process_tool_calls(self):
        """Test processing tool calls into LLM format."""
        from app.schema.llm.tool import ToolCall, ToolCallFunction

        service = LLMService()

        tool_calls = [
            ToolCall(
                id="call_1",
                type="function",
                function=ToolCallFunction(name="tool1", arguments='{"arg1": "value1"}'),
            ),
            ToolCall(
                id="call_2",
                type="function",
                function=ToolCallFunction(name="tool2", arguments='{"arg2": "value2"}'),
            ),
        ]

        result = service._process_tool_calls(tool_calls)

        assert len(result) == 2
        assert result[0] == {
            "id": "call_1",
            "type": "function",
            "function": {"name": "tool1", "arguments": '{"arg1": "value1"}'},
        }
        assert result[1] == {
            "id": "call_2",
            "type": "function",
            "function": {"name": "tool2", "arguments": '{"arg2": "value2"}'},
        }

    @pytest.mark.asyncio
    @patch("app.services.llm_service.LLMClient")
    async def test_query_llm_empty_string_response_json(self, mock_client_class):
        """Test handling empty string content for JSON response."""
        mock_client = MagicMock()
        mock_async_client = AsyncMock()
        mock_client.get_client.return_value = mock_async_client
        mock_client_class.return_value = mock_client

        mock_completion = self.create_mock_completion(content="   ")
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )

        service = LLMService()
        service.retry_decorator = lambda f: f

        with pytest.raises(LLMException) as exc_info:
            await service.query_llm(
                Message(role="user", content="Test"), json_response=True
            )

        assert "returned empty content" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("app.services.llm_service.LLMClient")
    async def test_query_llm_handles_json_decode_fallback(self, mock_client_class):
        """Test query_llm falls back to clean_json_response on decode error."""
        mock_client = MagicMock()
        mock_async_client = AsyncMock()
        mock_client.get_client.return_value = mock_async_client
        mock_client_class.return_value = mock_client

        mock_completion = self.create_mock_completion(
            '```json\n{"result": "cleaned"}\n```'
        )
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )

        service = LLMService()
        service.retry_decorator = lambda f: f

        result = await service.query_llm(
            Message(role="user", content="Test"), json_response=True
        )

        assert result == {"result": "cleaned"}

    def test_process_response_non_json(self):
        """Test _process_response with non-JSON response."""
        service = LLMService()

        completion = self.create_mock_completion("Regular text response")

        result = service._process_response(
            completion, json_response=False, request_id="test"
        )

        assert result.content == "Regular text response"

    def test_process_response_json_success(self):
        """Test _process_response with successful JSON response."""
        service = LLMService()

        completion = self.create_mock_completion('{"status": "ok"}')

        result = service._process_response(
            completion, json_response=True, request_id="test"
        )

        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    @patch("app.services.llm_service.LLMClient")
    async def test_query_llm_general_exception(self, mock_client_class):
        """Test handling of general exceptions during query."""
        mock_client = MagicMock()
        mock_async_client = AsyncMock()
        mock_client.get_client.return_value = mock_async_client
        mock_client_class.return_value = mock_client

        mock_async_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("Unexpected error")
        )

        service = LLMService()
        service.retry_decorator = lambda f: f

        with pytest.raises(LLMException) as exc_info:
            await service.query_llm(Message(role="user", content="Test"))

        assert "Failed to query LLM" in str(exc_info.value)
        assert "Unexpected error" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("app.services.llm_service.LLMClient")
    async def test_query_llm_with_json_response_format(self, mock_client_class):
        """Test that json_response adds response_format parameter."""
        mock_client = MagicMock()
        mock_async_client = AsyncMock()
        mock_client.get_client.return_value = mock_async_client
        mock_client_class.return_value = mock_client

        mock_completion = self.create_mock_completion('{"result": "json"}')
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )

        service = LLMService()
        service.retry_decorator = lambda f: f

        await service.query_llm(
            Message(role="user", content="Test"), json_response=True
        )

        call_args = mock_async_client.chat.completions.create.call_args
        assert "response_format" in call_args[1]
        assert call_args[1]["response_format"] == {"type": "json_object"}
