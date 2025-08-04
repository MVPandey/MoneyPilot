"""Test constants module."""

from app.utils.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_RETRIES,
    REQUEST_ID_PREFIX_LLM,
    REQUEST_ID_PREFIX_TOOL,
)


class TestConstants:
    """Test constants module."""

    def test_default_max_tokens(self):
        """Test default max tokens value."""
        assert DEFAULT_MAX_TOKENS == 250
        assert isinstance(DEFAULT_MAX_TOKENS, int)
        assert DEFAULT_MAX_TOKENS > 0

    def test_default_max_retries(self):
        """Test default max retries value."""
        assert DEFAULT_MAX_RETRIES == 3
        assert isinstance(DEFAULT_MAX_RETRIES, int)
        assert DEFAULT_MAX_RETRIES > 0

    def test_request_id_prefixes(self):
        """Test request ID prefixes."""
        assert REQUEST_ID_PREFIX_LLM == "llm"
        assert REQUEST_ID_PREFIX_TOOL == "tool"
        assert isinstance(REQUEST_ID_PREFIX_LLM, str)
        assert isinstance(REQUEST_ID_PREFIX_TOOL, str)

    def test_prefixes_are_different(self):
        """Test that prefixes are unique."""
        assert REQUEST_ID_PREFIX_LLM != REQUEST_ID_PREFIX_TOOL
