"""Test edge cases for tool executor."""
import json
import pytest
from unittest.mock import patch, AsyncMock
import asyncio

from app.services.llm.tool_executor import ToolExecutor
from app.schema.llm.tool import ToolCall, ToolCallFunction


class TestToolExecutorEdgeCases:
    """Test edge cases for tool executor functionality."""

    @pytest.mark.asyncio
    async def test_execute_with_none_result(self):
        """Test tool execution that returns None."""
        executor = ToolExecutor()
        
        mock_tool_func = AsyncMock(return_value=None)
        
        with patch.object(executor.registry, 'get_tool_function', return_value=mock_tool_func):
            tool_call = ToolCall(
                id="test_none",
                type="function",
                function=ToolCallFunction(
                    name="none_tool",
                    arguments='{}'
                )
            )
            
            results = await executor.execute_tool_calls([tool_call])
            
            assert len(results) == 1
            assert results[0].role == "tool"
            assert results[0].tool_call_id == "test_none"
            assert json.loads(results[0].content) is None

    @pytest.mark.asyncio
    async def test_execute_with_sync_function(self):
        """Test tool execution with synchronous function wrapped as async."""
        executor = ToolExecutor()
        
        async def async_wrapped_sync_func(**kwargs):
            return {"sync": "result"}
        
        with patch.object(executor.registry, 'get_tool_function', return_value=async_wrapped_sync_func):
            tool_call = ToolCall(
                id="test_sync",
                type="function",
                function=ToolCallFunction(
                    name="sync_tool",
                    arguments='{}'
                )
            )
            
            results = await executor.execute_tool_calls([tool_call])
            
            assert len(results) == 1
            assert json.loads(results[0].content) == {"sync": "result"}

    @pytest.mark.asyncio
    async def test_execute_with_concurrent_calls(self):
        """Test concurrent tool execution."""
        executor = ToolExecutor()
        
        call_order = []
        
        async def slow_tool(name):
            call_order.append(f"{name}_start")
            await asyncio.sleep(0.01)
            call_order.append(f"{name}_end")
            return {"name": name}
        
        tool_funcs = {
            "tool1": lambda: slow_tool("tool1"),
            "tool2": lambda: slow_tool("tool2"),
            "tool3": lambda: slow_tool("tool3")
        }
        
        def get_tool_side_effect(name):
            return tool_funcs[name]
        
        with patch.object(executor.registry, 'get_tool_function', side_effect=get_tool_side_effect):
            tool_calls = [
                ToolCall(
                    id=f"call_{i}",
                    type="function",
                    function=ToolCallFunction(name=f"tool{i}", arguments='{}')
                )
                for i in range(1, 4)
            ]
            
            results = await executor.execute_tool_calls(tool_calls)
            
            assert "tool1_start" in call_order
            assert "tool2_start" in call_order
            assert "tool3_start" in call_order
            
            assert len(results) == 3
            for i, result in enumerate(results):
                content = json.loads(result.content)
                assert content["name"] == f"tool{i+1}"

    @pytest.mark.asyncio
    async def test_serialization_with_complex_types(self):
        """Test JSON serialization of complex return types."""
        executor = ToolExecutor()
        
        class CustomObject:
            def __init__(self, value):
                self.value = value
            
            def __str__(self):
                return f"CustomObject({self.value})"
        
        mock_tool_func = AsyncMock(return_value=CustomObject("test"))
        
        with patch.object(executor.registry, 'get_tool_function', return_value=mock_tool_func):
            tool_call = ToolCall(
                id="test_custom",
                type="function",
                function=ToolCallFunction(
                    name="custom_tool",
                    arguments='{}'
                )
            )
            
            results = await executor.execute_tool_calls([tool_call])
            
            assert len(results) == 1
            assert "CustomObject(test)" in results[0].content

    @pytest.mark.asyncio
    async def test_execute_with_logging(self):
        """Test that execution logs are generated."""
        executor = ToolExecutor()
        
        mock_tool_func = AsyncMock(return_value={"logged": True})
        
        with patch.object(executor.registry, 'get_tool_function', return_value=mock_tool_func):
            with patch("app.services.llm.tool_executor.logger") as mock_logger:
                tool_call = ToolCall(
                    id="test_log",
                    type="function",
                    function=ToolCallFunction(
                        name="logged_tool",
                        arguments='{}'
                    )
                )
                
                await executor.execute_tool_calls([tool_call], execution_id="exec_123")
                
                assert mock_logger.info.called
                log_calls = mock_logger.info.call_args_list
                assert any(
                    "exec_123" in str(call)
                    for call in log_calls
                )
