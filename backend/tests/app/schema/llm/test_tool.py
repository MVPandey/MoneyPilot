"""Test tool-related schemas."""

import pytest
from pydantic import ValidationError

from app.schema.llm.tool import (
    ToolSchema,
    ToolFunction,
    ToolFunctionParameters,
    ToolParameterProperty,
    ToolCall,
    ToolCallFunction,
    AbstractTool,
)


class TestToolSchema:
    """Test ToolSchema schema."""

    def test_valid_tool_schema(self):
        """Test creating a valid tool schema."""
        params = ToolFunctionParameters(
            type="object",
            properties={
                "location": ToolParameterProperty(
                    type="string", description="The city and country"
                )
            },
            required=["location"],
        )

        func = ToolFunction(
            name="get_weather", description="Get current weather", parameters=params
        )

        tool = ToolSchema(type="function", function=func)

        assert tool.type == "function"
        assert tool.function.name == "get_weather"
        assert tool.function.description == "Get current weather"
        assert tool.function.parameters.properties["location"].type == "string"

    def test_tool_schema_model_dump(self):
        """Test model dump for tool schema."""
        params = ToolFunctionParameters(
            type="object",
            properties={
                "test": ToolParameterProperty(type="string", description="Test param")
            },
        )

        func = ToolFunction(name="test", description="Test tool", parameters=params)

        tool = ToolSchema(type="function", function=func)

        dumped = tool.model_dump()
        assert dumped["type"] == "function"
        assert dumped["function"]["name"] == "test"

    def test_tool_schema_frozen(self):
        """Test that ToolSchema is frozen."""
        params = ToolFunctionParameters(type="object", properties={})

        func = ToolFunction(name="test", description="Test", parameters=params)

        tool = ToolSchema(type="function", function=func)

        with pytest.raises(ValidationError):
            tool.type = "other"


class TestToolCallFunction:
    """Test ToolCallFunction schema."""

    def test_valid_tool_call_function(self):
        """Test creating a valid tool call function."""
        func = ToolCallFunction(
            name="calculator", arguments='{"operation": "add", "a": 5, "b": 3}'
        )

        assert func.name == "calculator"
        assert func.arguments == '{"operation": "add", "a": 5, "b": 3}'

    def test_tool_call_function_empty_arguments(self):
        """Test tool call function with empty arguments."""
        func = ToolCallFunction(name="no_args_tool", arguments="{}")

        assert func.name == "no_args_tool"
        assert func.arguments == "{}"

    def test_tool_call_function_model_dump(self):
        """Test model dump for tool call function."""
        func = ToolCallFunction(name="test_func", arguments='{"key": "value"}')

        dumped = func.model_dump()
        assert dumped["name"] == "test_func"
        assert dumped["arguments"] == '{"key": "value"}'

    def test_tool_call_function_required_fields(self):
        """Test required fields for ToolCallFunction."""
        with pytest.raises(ValidationError) as exc_info:
            ToolCallFunction(name="test")
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("arguments",)


class TestToolCall:
    """Test ToolCall schema."""

    def test_valid_tool_call(self):
        """Test creating a valid tool call."""
        func = ToolCallFunction(name="test_function", arguments='{"param": "value"}')

        tool_call = ToolCall(id="call_123", type="function", function=func)

        assert tool_call.id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "test_function"
        assert tool_call.function.arguments == '{"param": "value"}'

    def test_tool_call_with_inline_function(self):
        """Test creating tool call with inline function definition."""
        tool_call = ToolCall(
            id="call_456",
            type="function",
            function=ToolCallFunction(name="inline_func", arguments="{}"),
        )

        assert tool_call.id == "call_456"
        assert tool_call.function.name == "inline_func"

    def test_tool_call_model_dump(self):
        """Test model dump for tool call."""
        tool_call = ToolCall(
            id="call_789",
            type="function",
            function=ToolCallFunction(name="dump_test", arguments='{"test": true}'),
        )

        dumped = tool_call.model_dump()
        assert dumped["id"] == "call_789"
        assert dumped["type"] == "function"
        assert dumped["function"]["name"] == "dump_test"
        assert dumped["function"]["arguments"] == '{"test": true}'

    def test_tool_call_required_fields(self):
        """Test required fields for ToolCall."""
        with pytest.raises(ValidationError):
            ToolCall(id="test", type="function")

        with pytest.raises(ValidationError):
            ToolCall(
                type="function", function=ToolCallFunction(name="test", arguments="{}")
            )


class TestToolFunction:
    """Test ToolFunction model."""

    def test_valid_tool_function(self):
        """Test creating a valid ToolFunction."""
        func = ToolFunction(
            name="calculate",
            description="Calculate something",
            parameters=ToolFunctionParameters(
                type="object",
                properties={
                    "x": ToolParameterProperty(
                        type="number", description="First number"
                    ),
                    "y": ToolParameterProperty(
                        type="number", description="Second number"
                    ),
                },
                required=["x", "y"],
            ),
        )

        assert func.name == "calculate"
        assert func.description == "Calculate something"
        assert func.parameters.type == "object"
        assert "x" in func.parameters.properties
        assert "y" in func.parameters.properties
        assert func.parameters.required == ["x", "y"]

    def test_tool_function_model_dump(self):
        """Test model dump for ToolFunction."""
        func = ToolFunction(
            name="test_func",
            description="Test function",
            parameters=ToolFunctionParameters(
                type="object",
                properties={
                    "param": ToolParameterProperty(type="string", description="A param")
                },
                required=["param"],
            ),
        )

        dumped = func.model_dump()

        assert dumped["name"] == "test_func"
        assert dumped["description"] == "Test function"
        assert dumped["parameters"]["type"] == "object"
        assert "param" in dumped["parameters"]["properties"]


class TestToolParameterProperty:
    """Test ToolParameterProperty model."""

    def test_valid_parameter_property(self):
        """Test creating valid parameter property."""
        prop = ToolParameterProperty(
            type="string", description="A string parameter", enum=["option1", "option2"]
        )

        assert prop.type == "string"
        assert prop.description == "A string parameter"
        assert prop.enum == ["option1", "option2"]

    def test_parameter_property_types(self):
        """Test different parameter types."""
        string_prop = ToolParameterProperty(type="string", description="String param")
        assert string_prop.type == "string"

        number_prop = ToolParameterProperty(type="number", description="Number param")
        assert number_prop.type == "number"

        bool_prop = ToolParameterProperty(type="boolean", description="Bool param")
        assert bool_prop.type == "boolean"

        array_prop = ToolParameterProperty(type="array", description="Array param")
        assert array_prop.type == "array"

    def test_parameter_property_model_dump(self):
        """Test model dump for parameter property."""
        prop = ToolParameterProperty(
            type="string",
            description="A string choice",
            enum=["option1", "option2", "option3"],
        )

        dumped = prop.model_dump()

        assert dumped["type"] == "string"
        assert dumped["description"] == "A string choice"
        assert dumped["enum"] == ["option1", "option2", "option3"]


class TestAbstractTool:
    """Test AbstractTool base class."""

    def test_abstract_tool_interface(self):
        """Test that AbstractTool defines required interface."""

        class ConcreteTool(AbstractTool):
            tool_schema = ToolSchema(
                type="function",
                function=ToolFunction(
                    name="concrete_tool",
                    description="A concrete tool",
                    parameters=ToolFunctionParameters(
                        type="object", properties={}, required=[]
                    ),
                ),
            )

            @classmethod
            def tool_function(cls):
                return lambda: "Tool executed"

        tool = ConcreteTool()
        assert tool.tool_schema.function.name == "concrete_tool"
        assert callable(ConcreteTool.tool_function())

        func = ConcreteTool.tool_function()
        assert func() == "Tool executed"
