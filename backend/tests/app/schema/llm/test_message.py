"""Test LLM message schemas."""

import pytest
from pydantic import ValidationError

from app.schema.llm.message import Message, ToolMessage


class TestMessage:
    """Test Message schema."""

    def test_valid_message(self):
        """Test creating a valid message."""
        message = Message(
            role="user",
            content="Hello, world!",
        )

        assert message.role == "user"
        assert message.content == "Hello, world!"

    def test_message_roles(self):
        """Test different message roles."""
        user_msg = Message(role="user", content="User message")
        assert user_msg.role == "user"

        assistant_msg = Message(role="assistant", content="Assistant message")
        assert assistant_msg.role == "assistant"

        system_msg = Message(role="system", content="System message")
        assert system_msg.role == "system"

    def test_message_invalid_role(self):
        """Test invalid role raises error."""
        with pytest.raises(ValidationError):
            Message(role="invalid", content="Test")

    def test_message_model_dump(self):
        """Test model dump for API calls."""
        message = Message(
            role="system",
            content="You are a helpful assistant.",
        )

        dumped = message.model_dump()
        assert dumped["role"] == "system"
        assert dumped["content"] == "You are a helpful assistant."

    def test_message_required_fields(self):
        """Test that role and content are required."""
        with pytest.raises(ValidationError) as exc_info:
            Message(role="user")
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("content",)

    def test_message_empty_content_allowed(self):
        """Test that empty content is allowed."""
        message = Message(role="assistant", content="")
        assert message.content == ""


class TestToolMessage:
    """Test ToolMessage schema."""

    def test_valid_tool_message(self):
        """Test creating a valid tool message."""
        message = ToolMessage(
            role="tool",
            tool_call_id="test_123",
            name="test_tool",
            content="Tool result",
        )

        assert message.role == "tool"
        assert message.tool_call_id == "test_123"
        assert message.name == "test_tool"
        assert message.content == "Tool result"

    def test_tool_message_model_dump(self):
        """Test model dump for tool messages."""
        message = ToolMessage(
            role="tool",
            tool_call_id="test_789",
            name="calculator",
            content="Success",
        )

        dumped = message.model_dump()
        assert dumped["role"] == "tool"
        assert dumped["tool_call_id"] == "test_789"
        assert dumped["name"] == "calculator"
        assert dumped["content"] == "Success"

    def test_tool_message_role_is_always_tool(self):
        """Test that role is always 'tool' for ToolMessage."""
        message = ToolMessage(
            role="tool", tool_call_id="test", name="test_tool", content="Result"
        )
        assert message.role == "tool"

    def test_tool_message_required_fields(self):
        """Test required fields for ToolMessage."""
        with pytest.raises(ValidationError) as exc_info:
            ToolMessage(role="tool", name="test", content="Result")

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("tool_call_id",) for error in errors)

        with pytest.raises(ValidationError) as exc_info:
            ToolMessage(role="tool", tool_call_id="123", content="Result")

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("name",) for error in errors)

    def test_tool_message_inherits_from_message(self):
        """Test that ToolMessage inherits from Message."""
        assert issubclass(ToolMessage, Message)
