"""Test tool executor functionality."""
import json
import pytest
from unittest.mock import patch, AsyncMock

from app.services.llm.tool_executor import ToolExecutor
from app.schema.llm.tool import ToolCall, ToolCallFunction


class TestToolExecutor:
    """Test tool executor functionality."""

    @pytest.mark.asyncio
    async def test_execute_tool_calls_success(self):
        """Test successful tool execution."""
        executor = ToolExecutor()
        
        mock_tool_func = AsyncMock(return_value={"result": "success"})
        
        with patch.object(executor.registry, 'get_tool_function', return_value=mock_tool_func):
            tool_call = ToolCall(
                id="test_123",
                type="function",
                function=ToolCallFunction(
                    name="test_tool",
                    arguments='{"param": "value"}'
                )
            )
            
            results = await executor.execute_tool_calls([tool_call])
            
            assert len(results) == 1
            assert results[0].role == "tool"
            assert results[0].tool_call_id == "test_123"
            assert results[0].name == "test_tool"
            assert json.loads(results[0].content) == {"result": "success"}
            
            mock_tool_func.assert_called_once_with(param="value")

    @pytest.mark.asyncio
    async def test_execute_tool_calls_with_error(self):
        """Test tool execution with error."""
        executor = ToolExecutor()
        
        mock_tool_func = AsyncMock(side_effect=ValueError("Tool failed"))
        
        with patch.object(executor.registry, 'get_tool_function', return_value=mock_tool_func):
            tool_call = ToolCall(
                id="test_456",
                type="function",
                function=ToolCallFunction(
                    name="failing_tool",
                    arguments="{}"
                )
            )
            
            results = await executor.execute_tool_calls([tool_call])
            
            assert len(results) == 1
            assert results[0].role == "tool"
            assert results[0].tool_call_id == "test_456"
            
            error_content = json.loads(results[0].content)
            assert "error" in error_content
            assert "Tool failed" in error_content["error"]

    @pytest.mark.asyncio
    async def test_execute_tool_calls_invalid_json(self):
        """Test tool execution with invalid JSON arguments."""
        executor = ToolExecutor()
        
        tool_call = ToolCall(
            id="test_789",
            type="function",
            function=ToolCallFunction(
                name="test_tool",
                arguments="invalid json"
            )
        )
        
        results = await executor.execute_tool_calls([tool_call])
        
        assert len(results) == 1
        error_content = json.loads(results[0].content)
        assert "error" in error_content
        assert "Invalid JSON" in error_content["error"]

    @pytest.mark.asyncio
    async def test_execute_multiple_tool_calls(self):
        """Test executing multiple tool calls."""
        executor = ToolExecutor()
        
        mock_tool_func1 = AsyncMock(return_value={"result": "first"})
        mock_tool_func2 = AsyncMock(return_value={"result": "second"})
        
        def get_tool_side_effect(name):
            if name == "tool1":
                return mock_tool_func1
            elif name == "tool2":
                return mock_tool_func2
        
        with patch.object(executor.registry, 'get_tool_function', side_effect=get_tool_side_effect):
            tool_calls = [
                ToolCall(
                    id="call_1",
                    type="function",
                    function=ToolCallFunction(name="tool1", arguments="{}")
                ),
                ToolCall(
                    id="call_2",
                    type="function",
                    function=ToolCallFunction(name="tool2", arguments="{}")
                )
            ]
            
            results = await executor.execute_tool_calls(tool_calls)
            
            assert len(results) == 2
            assert results[0].tool_call_id == "call_1"
            assert results[1].tool_call_id == "call_2"
            assert json.loads(results[0].content) == {"result": "first"}
            assert json.loads(results[1].content) == {"result": "second"}

    @pytest.mark.asyncio
    async def test_execute_tool_calls_with_execution_id(self):
        """Test tool calls with execution ID for logging."""
        executor = ToolExecutor()
        
        mock_tool_func = AsyncMock(return_value={"status": "ok"})
        
        with patch.object(executor.registry, 'get_tool_function', return_value=mock_tool_func):
            tool_call = ToolCall(
                id="test_exec_id",
                type="function",
                function=ToolCallFunction(name="test", arguments="{}")
            )
            
            results = await executor.execute_tool_calls(
                [tool_call],
                execution_id="exec_12345"
            )
            
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_empty_list(self):
        """Test executing empty list of tool calls."""
        executor = ToolExecutor()
        
        results = await executor.execute_tool_calls([])
        
        assert results == []

    @pytest.mark.asyncio
    async def test_tool_result_serialization(self):
        """Test that tool results are properly serialized."""
        executor = ToolExecutor()
        
        test_cases = [
            {"dict": "result"},
            ["list", "result"],
            "string result",
            42,
            True,
            None
        ]
        
        async def make_tool_func(result):
            return result
        
        for i, expected_result in enumerate(test_cases):
            mock_func = AsyncMock(return_value=expected_result)
            
            with patch.object(executor.registry, 'get_tool_function', return_value=mock_func):
                tool_call = ToolCall(
                    id=f"call_{i}",
                    type="function",
                    function=ToolCallFunction(name=f"tool_{i}", arguments="{}")
                )
                
                results = await executor.execute_tool_calls([tool_call])
                
                assert len(results) == 1
                assert json.loads(results[0].content) == expected_result

    @pytest.mark.asyncio
    async def test_tool_execution_with_complex_arguments(self):
        """Test tool execution with complex nested arguments."""
        executor = ToolExecutor()
        
        complex_args = {
            "user": {
                "name": "John",
                "age": 30,
                "preferences": ["option1", "option2"]
            },
            "settings": {
                "enabled": True,
                "threshold": 0.8
            }
        }
        
        mock_tool_func = AsyncMock(return_value={"processed": True})
        
        with patch.object(executor.registry, 'get_tool_function', return_value=mock_tool_func):
            tool_call = ToolCall(
                id="complex_test",
                type="function",
                function=ToolCallFunction(
                    name="complex_tool",
                    arguments=json.dumps(complex_args)
                )
            )
            
            await executor.execute_tool_calls([tool_call])
            
            mock_tool_func.assert_called_once_with(**complex_args)

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        """Test handling when tool is not found in registry."""
        executor = ToolExecutor()
        
        with patch.object(executor.registry, 'get_tool_function', side_effect=ValueError("Tool not found")):
            tool_call = ToolCall(
                id="test_not_found",
                type="function",
                function=ToolCallFunction(
                    name="nonexistent_tool",
                    arguments="{}"
                )
            )
            
            results = await executor.execute_tool_calls([tool_call])
            
            assert len(results) == 1
            error_content = json.loads(results[0].content)
            assert "error" in error_content
            assert "Tool not found" in error_content["error"]

    @pytest.mark.asyncio
    async def test_tool_arguments_not_dict(self):
        """Test handling when tool arguments are not a dictionary."""
        executor = ToolExecutor()
        
        mock_tool_func = AsyncMock()
        
        with patch.object(executor.registry, 'get_tool_function', return_value=mock_tool_func):
            tool_call = ToolCall(
                id="test_bad_args",
                type="function",
                function=ToolCallFunction(
                    name="test_tool",
                    arguments='["not", "a", "dict"]'                )
            )
            
            results = await executor.execute_tool_calls([tool_call])
            
            assert len(results) == 1
            error_content = json.loads(results[0].content)
            assert "error" in error_content
            assert "Tool arguments must be a dictionary" in error_content["error"]
            assert "got list" in error_content["error"]

    def test_serialize_result_with_json_method(self):
        """Test serialization of result with json() method."""
        executor = ToolExecutor()
        
        class JsonObject:
            def json(self):
                return '{"from": "json_method"}'
        
        result = executor._serialize_result(JsonObject())
        assert result == '{"from": "json_method"}'

    def test_serialize_result_with_model_dump(self):
        """Test serialization of result with model_dump() method."""
        executor = ToolExecutor()
        
        class ModelObject:
            def model_dump(self):
                return {"from": "model_dump"}
        
        result = executor._serialize_result(ModelObject())
        assert result == '{"from": "model_dump"}'
