"""Test tool registry functionality."""

import pytest
from unittest.mock import patch

from app.utils.tool_registry import ToolRegistry
from app.schema.llm.tool import (
    AbstractTool,
    ToolSchema,
    ToolFunction,
    ToolFunctionParameters,
    ToolParameterProperty,
)


class MockTool(AbstractTool):
    """Mock tool for testing."""

    tool_schema = ToolSchema(
        type="function",
        function=ToolFunction(
            name="mock_tool",
            description="A mock tool for testing",
            parameters=ToolFunctionParameters(
                type="object",
                properties={
                    "test_param": ToolParameterProperty(
                        type="string", description="Test parameter"
                    )
                },
                required=["test_param"],
            ),
        ),
    )

    @classmethod
    def tool_function(cls):
        """Return mock function."""
        return lambda test_param: f"Mock result: {test_param}"


class TestToolRegistry:
    """Test tool registry functionality."""

    def test_tool_registry_initialization(self):
        """Test that tool registry initializes correctly."""
        registry = ToolRegistry()

        assert registry._tools is None
        assert registry._initialized is False

    def test_ensure_initialized(self):
        """Test lazy initialization."""
        registry = ToolRegistry()

        assert registry._initialized is False

        registry._ensure_initialized()

        assert registry._initialized is True
        assert registry._tools is not None
        assert isinstance(registry._tools, dict)

    @patch("app.schema.llm.tool.AbstractTool.__subclasses__")
    def test_collect_tools(self, mock_subclasses):
        """Test tool collection from subclasses."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()
        registry._tools = {}
        registry._collect_tools()

        assert "MockTool" in registry._tools
        assert registry._tools["MockTool"]["class"] == MockTool
        assert registry._tools["MockTool"]["schema"] == MockTool.tool_schema
        assert callable(registry._tools["MockTool"]["function"])

    @patch("app.schema.llm.tool.AbstractTool.__subclasses__")
    def test_tools_property(self, mock_subclasses):
        """Test accessing tools property."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()
        tools = registry.tools

        assert isinstance(tools, dict)
        assert "MockTool" in tools

    @patch("app.schema.llm.tool.AbstractTool.__subclasses__")
    def test_get_tool(self, mock_subclasses):
        """Test getting a specific tool."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()

        tool = registry.get_tool("MockTool")
        assert tool is not None
        assert tool["class"] == MockTool

        tool = registry.get_tool("NonExistentTool")
        assert tool is None

    @patch("app.schema.llm.tool.AbstractTool.__subclasses__")
    def test_get_tool_schemas(self, mock_subclasses):
        """Test getting tool schemas."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()
        schemas = registry.get_tool_schemas(["MockTool"])

        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "mock_tool"

    @patch("app.schema.llm.tool.AbstractTool.__subclasses__")
    def test_get_tool_schemas_unknown_tool(self, mock_subclasses):
        """Test getting schemas with unknown tool."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()

        with pytest.raises(ValueError) as exc_info:
            registry.get_tool_schemas(["UnknownTool"])

        assert "Unknown tool: UnknownTool" in str(exc_info.value)
        assert "Available tools: ['MockTool']" in str(exc_info.value)

    @patch("app.schema.llm.tool.AbstractTool.__subclasses__")
    def test_get_tool_function(self, mock_subclasses):
        """Test getting tool function."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()
        func = registry.get_tool_function("MockTool")

        assert callable(func)
        result = func("test")
        assert result == "Mock result: test"

    @patch("app.schema.llm.tool.AbstractTool.__subclasses__")
    def test_get_tool_function_not_found(self, mock_subclasses):
        """Test getting function for non-existent tool."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()

        with pytest.raises(ValueError) as exc_info:
            registry.get_tool_function("NonExistentTool")

        assert "Tool 'NonExistentTool' not found" in str(exc_info.value)

    @patch("app.schema.llm.tool.AbstractTool.__subclasses__")
    def test_list_tool_names(self, mock_subclasses):
        """Test listing tool names."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()
        names = registry.list_tool_names()

        assert "MockTool" in names
        assert isinstance(names, list)

    def test_reset(self):
        """Test resetting the registry."""
        registry = ToolRegistry()

        registry._ensure_initialized()
        assert registry._initialized is True

        registry.reset()

        assert registry._tools is None
        assert registry._initialized is False

    @patch("app.schema.llm.tool.AbstractTool.__subclasses__")
    def test_collect_tools_with_error(self, mock_subclasses):
        """Test tool collection with error in one tool."""

        class BrokenTool(AbstractTool):
            @classmethod
            def tool_function(cls):
                raise ValueError("Broken tool")

        mock_subclasses.return_value = [MockTool, BrokenTool]

        registry = ToolRegistry()
        registry._tools = {}
        registry._collect_tools()

        assert "MockTool" in registry._tools
        assert "BrokenTool" not in registry._tools

    def test_singleton_instance(self):
        """Test that tool_registry is a singleton instance."""
        from app.utils.tool_registry import tool_registry

        assert isinstance(tool_registry, ToolRegistry)
