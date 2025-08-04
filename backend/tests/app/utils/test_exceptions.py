"""Test custom exceptions."""

import pytest

from app.utils.exceptions import (
    LLMException,
    ToolException,
    WorkflowException,
    AgentException,
)


class TestExceptions:
    """Test custom exceptions."""

    def test_llm_exception(self):
        """Test LLMException."""
        with pytest.raises(LLMException) as exc_info:
            raise LLMException("Test LLM error")

        assert str(exc_info.value) == "Test LLM error"
        assert isinstance(exc_info.value, Exception)

    def test_llm_exception_inheritance(self):
        """Test LLMException inherits from Exception."""
        exc = LLMException("Test")
        assert isinstance(exc, Exception)

    def test_tool_exception(self):
        """Test ToolException."""
        with pytest.raises(ToolException) as exc_info:
            raise ToolException("Test tool error")

        assert str(exc_info.value) == "Test tool error"
        assert isinstance(exc_info.value, Exception)

    def test_tool_exception_inheritance(self):
        """Test ToolException inherits from Exception."""
        exc = ToolException("Test")
        assert isinstance(exc, Exception)

    def test_workflow_exception(self):
        """Test WorkflowException."""
        with pytest.raises(WorkflowException) as exc_info:
            raise WorkflowException("Test workflow error")

        assert str(exc_info.value) == "Test workflow error"
        assert isinstance(exc_info.value, Exception)

    def test_workflow_exception_inheritance(self):
        """Test WorkflowException inherits from Exception."""
        exc = WorkflowException("Test")
        assert isinstance(exc, Exception)

    def test_agent_exception(self):
        """Test AgentException."""
        with pytest.raises(AgentException) as exc_info:
            raise AgentException("Test agent error")

        assert str(exc_info.value) == "Test agent error"
        assert isinstance(exc_info.value, Exception)

    def test_agent_exception_inheritance(self):
        """Test AgentException inherits from Exception."""
        exc = AgentException("Test")
        assert isinstance(exc, Exception)

    def test_exceptions_are_different_types(self):
        """Test that exceptions are different types."""
        llm_exc = LLMException("llm")
        tool_exc = ToolException("tool")
        workflow_exc = WorkflowException("workflow")
        agent_exc = AgentException("agent")

        assert type(llm_exc) is not type(tool_exc)
        assert type(llm_exc) is not type(workflow_exc)
        assert type(llm_exc) is not type(agent_exc)
        assert type(tool_exc) is not type(workflow_exc)
        assert type(tool_exc) is not type(agent_exc)
        assert type(workflow_exc) is not type(agent_exc)

    def test_exception_with_cause(self):
        """Test raising exception with cause."""
        original_error = ValueError("Original error")

        with pytest.raises(LLMException) as exc_info:
            try:
                raise original_error
            except ValueError as e:
                raise LLMException("Wrapped error") from e

        assert exc_info.value.__cause__ == original_error
