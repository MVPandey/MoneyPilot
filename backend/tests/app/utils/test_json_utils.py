"""Test JSON utility functions."""

import pytest

from app.utils.json_utils import clean_json_response, safe_json_dumps
from app.utils.exceptions import LLMException


class TestJsonUtils:
    """Test JSON utility functions."""

    def test_clean_json_response_valid_json(self):
        """Test cleaning valid JSON."""
        valid_json = '{"key": "value", "number": 42}'
        result = clean_json_response(valid_json)

        assert result == {"key": "value", "number": 42}

    def test_clean_json_response_with_markdown(self):
        """Test cleaning JSON wrapped in markdown code blocks."""
        markdown_json = '```json\n{"key": "value"}\n```'
        result = clean_json_response(markdown_json)

        assert result == {"key": "value"}

    def test_clean_json_response_with_extra_text(self):
        """Test cleaning JSON with surrounding text."""
        messy_json = 'Here is the JSON:\n{"result": true}\nEnd of response.'
        result = clean_json_response(messy_json)

        assert result == {"result": True}

    def test_clean_json_response_nested_objects(self):
        """Test cleaning nested JSON objects."""
        nested_json = '{"outer": {"inner": {"value": 123}}}'
        result = clean_json_response(nested_json)

        assert result == {"outer": {"inner": {"value": 123}}}

    def test_clean_json_response_array(self):
        """Test cleaning JSON arrays."""
        array_json = '[{"id": 1}, {"id": 2}]'
        result = clean_json_response(array_json)

        assert result == [{"id": 1}, {"id": 2}]

    def test_clean_json_response_with_backticks(self):
        """Test cleaning JSON with various backtick formats."""
        json_with_backticks = '`{"test": "value"}`'
        result = clean_json_response(json_with_backticks)
        assert result == {"test": "value"}

        json_with_triple = '```\n{"test": "value"}\n```'
        result = clean_json_response(json_with_triple)
        assert result == {"test": "value"}

    def test_clean_json_response_complex_markdown(self):
        """Test cleaning JSON from complex markdown."""
        complex_markdown = """
        Here's the response:
        
        ```json
        {
            "status": "success",
            "data": {
                "items": [1, 2, 3],
                "count": 3
            }
        }
        ```
        
        That's the JSON output.
        """
        result = clean_json_response(complex_markdown)
        assert result == {"status": "success", "data": {"items": [1, 2, 3], "count": 3}}

    def test_clean_json_response_invalid_json(self):
        """Test handling of invalid JSON."""
        with pytest.raises(LLMException) as exc_info:
            clean_json_response("This is not JSON at all")

        assert "Failed to parse JSON response" in str(exc_info.value)

    def test_clean_json_response_empty_string(self):
        """Test handling of empty string."""
        with pytest.raises(LLMException) as exc_info:
            clean_json_response("")

        assert "Empty response received" in str(exc_info.value)

    def test_clean_json_response_whitespace_only(self):
        """Test handling of whitespace-only string."""
        with pytest.raises(LLMException) as exc_info:
            clean_json_response("   \n\t   ")

        assert "Failed to parse JSON response after all attempts" in str(exc_info.value)

    def test_clean_json_response_with_comments(self):
        """Test cleaning JSON with comment-like text."""
        json_with_comments = """
        // This is a comment
        {"key": "value"}
        // Another comment
        """
        result = clean_json_response(json_with_comments)
        assert result == {"key": "value"}

    def test_clean_json_response_boolean_values(self):
        """Test handling of boolean values."""
        json_with_bools = '{"active": true, "deleted": false, "value": null}'
        result = clean_json_response(json_with_bools)

        assert result["active"] is True
        assert result["deleted"] is False
        assert result["value"] is None

    def test_clean_json_response_numeric_values(self):
        """Test handling of numeric values."""
        json_with_numbers = (
            '{"int": 42, "float": 3.14, "negative": -10, "exp": 1.5e-10}'
        )
        result = clean_json_response(json_with_numbers)

        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["negative"] == -10
        assert result["exp"] == 1.5e-10

    def test_clean_json_response_escaped_characters(self):
        """Test handling of escaped characters."""
        json_with_escapes = (
            '{"path": "C:\\\\Users\\\\test", "quote": "He said \\"Hello\\""}'
        )
        result = clean_json_response(json_with_escapes)

        assert result["path"] == "C:\\Users\\test"
        assert result["quote"] == 'He said "Hello"'

    def test_clean_json_response_multiple_json_blocks(self):
        """Test handling multiple JSON blocks in text."""
        text_with_multiple_json = """
        Here's the first JSON:
        ```json
        {"first": true}
        ```
        
        And here's another:
        {"second": true}
        """

        result = clean_json_response(text_with_multiple_json)
        # Should extract the first valid JSON found
        assert result == {"first": True}

    def test_clean_json_response_malformed_json_recovery(self):
        """Test recovery from various malformed JSON patterns."""
        # Trailing comma
        malformed = '{"key": "value",}'
        result = clean_json_response(malformed)
        assert result == {"key": "value"}

    def test_clean_json_response_with_unicode(self):
        """Test handling of unicode characters."""
        unicode_json = '{"emoji": "🚀", "text": "Hello 世界"}'
        result = clean_json_response(unicode_json)

        assert result["emoji"] == "🚀"
        assert result["text"] == "Hello 世界"

    def test_clean_json_response_large_numbers(self):
        """Test handling of large numbers and scientific notation."""
        number_json = '{"big": 1e100, "negative": -42.5, "int": 999999999999}'
        result = clean_json_response(number_json)

        assert result["big"] == 1e100
        assert result["negative"] == -42.5
        assert result["int"] == 999999999999

    def test_clean_json_response_brace_extraction(self):
        """Test extraction of JSON from curly braces."""
        # Test simple extraction
        text_with_json = 'Some text before {"key": "value"} and after'
        result = clean_json_response(text_with_json)
        assert result == {"key": "value"}

        # Test nested braces
        nested_json = '{"outer": {"inner": "value"}, "array": [1, 2, 3]}'
        result = clean_json_response(nested_json)
        assert result == {"outer": {"inner": "value"}, "array": [1, 2, 3]}

    def test_clean_json_response_bracket_extraction(self):
        """Test extraction of JSON arrays."""
        # Test array extraction
        array_text = 'Response: [{"id": 1}, {"id": 2}] Done.'
        result = clean_json_response(array_text)
        assert result == [{"id": 1}, {"id": 2}]

    def test_clean_json_response_all_extraction_attempts_fail(self):
        """Test when all extraction attempts fail."""
        # Text with no valid JSON
        no_json = "This has no JSON at all, just plain text"

        with pytest.raises(LLMException) as exc_info:
            clean_json_response(no_json)

        assert "Failed to parse JSON response after all attempts" in str(exc_info.value)

    def test_clean_json_response_markdown_json_decode_error(self):
        """Test when JSON in markdown block is invalid."""
        # Invalid JSON in markdown block that will fail to parse
        invalid_markdown_json = '```json\n{"key": "value", invalid}\n```'

        with pytest.raises(LLMException) as exc_info:
            clean_json_response(invalid_markdown_json)

        assert "Failed to parse JSON response after all attempts" in str(exc_info.value)

    def test_clean_json_response_cleaned_json_decode_error(self):
        """Test when cleaned JSON still fails to parse."""
        # JSON-like text that looks extractable but is invalid
        invalid_json_like = 'Response: {"key": "value", "nested": {"bad": syntax}}'

        with pytest.raises(LLMException) as exc_info:
            clean_json_response(invalid_json_like)

        assert "Failed to parse JSON response after all attempts" in str(exc_info.value)


class TestSafeJsonDumps:
    """Test safe_json_dumps function."""

    def test_safe_json_dumps_valid_object(self):
        """Test serializing valid JSON objects."""
        # Simple dict
        result = safe_json_dumps({"key": "value", "number": 42})
        assert result == '{"key": "value", "number": 42}'

        # List
        result = safe_json_dumps([1, 2, 3])
        assert result == "[1, 2, 3]"

        # Nested structure
        result = safe_json_dumps({"nested": {"array": [1, 2, 3]}})
        assert result == '{"nested": {"array": [1, 2, 3]}}'

    def test_safe_json_dumps_with_kwargs(self):
        """Test safe_json_dumps with additional kwargs."""
        obj = {"key": "value", "number": 42}

        # Test with indent
        result = safe_json_dumps(obj, indent=2)
        assert '"key": "value"' in result
        assert '"number": 42' in result

        # Test with sort_keys
        result = safe_json_dumps(obj, sort_keys=True)
        assert result == '{"key": "value", "number": 42}'

    def test_safe_json_dumps_non_serializable_object(self):
        """Test serializing non-JSON-serializable objects."""

        # Create a non-serializable object
        class CustomObject:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"CustomObject({self.value})"

        obj = CustomObject("test")
        result = safe_json_dumps(obj)

        # Should serialize the string representation
        assert result == '"CustomObject(test)"'

    def test_safe_json_dumps_circular_reference(self):
        """Test handling circular references."""
        # Create circular reference
        obj = {"key": "value"}
        obj["self"] = obj

        # Should handle gracefully by converting to string
        result = safe_json_dumps(obj)
        assert isinstance(result, str)
        # Should contain the string representation
        assert "{'key': 'value', 'self': {...}}" in result or result.startswith('"')

    def test_safe_json_dumps_with_set(self):
        """Test serializing a set (non-serializable by default)."""
        obj = {"items": {1, 2, 3}}

        # Should handle by converting to string
        result = safe_json_dumps(obj)
        assert isinstance(result, str)
        # The set will be part of the string representation
        assert "items" in result
