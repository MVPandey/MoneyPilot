"""Test logger configuration and functionality."""

import logging
import json
from unittest.mock import patch, MagicMock

from app.utils.logger import (
    InterceptHandler,
    format_record,
    format_record_json,
    get_logger,
    logger,
)


class TestInterceptHandler:
    """Test InterceptHandler functionality."""

    def test_emit_filters_uvicorn_debug(self):
        """Test that uvicorn debug logs are filtered."""
        handler = InterceptHandler()

        record = logging.LogRecord(
            name="uvicorn",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=10,
            msg="Debug message",
            args=(),
            exc_info=None,
        )

        with patch("app.utils.logger.logger") as mock_logger:
            handler.emit(record)
            mock_logger.opt.assert_not_called()

    def test_emit_allows_uvicorn_info(self):
        """Test that uvicorn info logs are allowed."""
        handler = InterceptHandler()

        record = logging.LogRecord(
            name="uvicorn",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info message",
            args=(),
            exc_info=None,
        )

        with patch("app.utils.logger.logger") as mock_logger:
            mock_opt = MagicMock()
            mock_logger.opt.return_value = mock_opt
            mock_logger.level.return_value = MagicMock(name="INFO")

            handler.emit(record)

            mock_logger.opt.assert_called_once()
            mock_opt.log.assert_called_once()

    def test_emit_with_exception(self):
        """Test emit with exception info."""
        handler = InterceptHandler()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )

        with patch("app.utils.logger.logger") as mock_logger:
            mock_opt = MagicMock()
            mock_logger.opt.return_value = mock_opt
            mock_logger.level.return_value = MagicMock(name="ERROR")

            handler.emit(record)

            mock_logger.opt.assert_called_once()
            assert mock_logger.opt.call_args[1]["exception"] == record.exc_info


class TestFormatRecord:
    """Test format_record functionality."""

    def test_format_record_without_extra(self):
        """Test formatting record without extra fields."""
        record = {
            "time": MagicMock(),
            "level": MagicMock(name="INFO"),
            "name": "test.module",
            "function": "test_func",
            "line": 42,
            "message": "Test message",
        }

        result = format_record(record)

        assert "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>" in result
        assert "<level>{level: <8}</level>" in result
        assert (
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>" in result
        )
        assert "<level>{message}</level>" in result

    def test_format_record_with_extra(self):
        """Test formatting record with extra fields."""
        record = {
            "time": MagicMock(),
            "level": MagicMock(name="INFO"),
            "name": "test.module",
            "function": "test_func",
            "line": 42,
            "message": "Test message",
            "extra": {"request_id": "123", "user_id": "456", "_internal": "hidden"},
        }

        result = format_record(record)

        assert "request_id" in result
        assert "123" in result
        assert "user_id" in result
        assert "456" in result
        assert "_internal" not in result

    def test_format_record_truncates_long_strings(self):
        """Test that long string values are truncated."""
        long_string = "x" * 200
        record = {
            "time": MagicMock(),
            "level": MagicMock(name="INFO"),
            "name": "test",
            "function": "func",
            "line": 1,
            "message": "Test",
            "extra": {"long_field": long_string},
        }

        result = format_record(record)

        assert "..." in result
        assert len(long_string) > 100

    def test_format_record_with_non_serializable_extra(self):
        """Test formatting record with non-serializable extra field."""
        circular_dict = {"key": "value"}
        circular_dict["self"] = circular_dict

        record = {
            "time": MagicMock(),
            "level": MagicMock(name="INFO"),
            "name": "test",
            "function": "func",
            "line": 1,
            "message": "Test",
            "extra": {"circular": circular_dict},
        }

        result = format_record(record)

        assert "circular" in result
        assert "..." in result or "dict" in result


class TestFormatRecordJson:
    """Test format_record_json functionality."""

    def test_format_record_json_basic(self):
        """Test JSON formatting of basic log record."""

        mock_time = MagicMock()
        mock_time.isoformat.return_value = "2024-01-01T00:00:00"

        mock_level = MagicMock()
        mock_level.name = "INFO"

        record = {
            "time": mock_time,
            "level": mock_level,
            "name": "test.module",
            "function": "test_func",
            "line": 42,
            "message": "Test message",
            "module": "test_module",
            "exception": None,
        }

        result = format_record_json(record)

        parsed = json.loads(result.strip())

        assert parsed["timestamp"] == "2024-01-01T00:00:00"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.module"
        assert parsed["function"] == "test_func"
        assert parsed["line"] == 42
        assert parsed["message"] == "Test message"
        assert parsed["module"] == "test_module"
        assert "exception" not in parsed

    def test_format_record_json_with_extra(self):
        """Test JSON formatting with extra fields."""

        mock_time = MagicMock()
        mock_time.isoformat.return_value = "2024-01-01T00:00:00"

        mock_level = MagicMock()
        mock_level.name = "INFO"

        record = {
            "time": mock_time,
            "level": mock_level,
            "name": "test",
            "function": "func",
            "line": 1,
            "message": "Test",
            "module": "mod",
            "exception": None,
            "extra": {
                "request_id": "123",
                "user_id": "456",
                "_internal": "should_be_excluded",
            },
        }

        result = format_record_json(record)

        parsed = json.loads(result.strip())

        assert parsed["request_id"] == "123"
        assert parsed["user_id"] == "456"
        assert "_internal" not in parsed

    def test_format_record_json_with_exception(self):
        """Test JSON formatting with exception info."""

        mock_time = MagicMock()
        mock_time.isoformat.return_value = "2024-01-01T00:00:00"

        mock_level = MagicMock()
        mock_level.name = "ERROR"

        mock_exc_type = type("TestException", (Exception,), {})
        mock_exc_value = mock_exc_type("Test error")

        mock_traceback = MagicMock()
        mock_traceback.raw = "Raw traceback data"

        mock_exception = MagicMock()
        mock_exception.type = mock_exc_type
        mock_exception.value = mock_exc_value
        mock_exception.traceback = mock_traceback

        record = {
            "time": mock_time,
            "level": mock_level,
            "name": "test",
            "function": "func",
            "line": 1,
            "message": "Error occurred",
            "module": "mod",
            "exception": mock_exception,
        }

        result = format_record_json(record)

        parsed = json.loads(result.strip())

        assert "exception" in parsed
        assert parsed["exception"]["type"] == "TestException"
        assert parsed["exception"]["value"] == "Test error"
        assert parsed["exception"]["traceback"] == "Raw traceback data"

    def test_format_record_json_with_exception_no_raw(self):
        """Test JSON formatting with exception info without raw traceback."""

        mock_time = MagicMock()
        mock_time.isoformat.return_value = "2024-01-01T00:00:00"

        mock_level = MagicMock()
        mock_level.name = "ERROR"

        mock_exc_type = type("TestException", (Exception,), {})
        mock_exc_value = mock_exc_type("Test error")
        mock_traceback = "String traceback"

        mock_exception = MagicMock()
        mock_exception.type = mock_exc_type
        mock_exception.value = mock_exc_value
        mock_exception.traceback = mock_traceback

        record = {
            "time": mock_time,
            "level": mock_level,
            "name": "test",
            "function": "func",
            "line": 1,
            "message": "Error occurred",
            "module": "mod",
            "exception": mock_exception,
        }

        result = format_record_json(record)

        parsed = json.loads(result.strip())

        assert parsed["exception"]["traceback"] == "String traceback"


class TestGetLogger:
    """Test get_logger functionality."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        test_logger = get_logger(request_id="123")

        assert hasattr(test_logger, "debug")
        assert hasattr(test_logger, "info")
        assert hasattr(test_logger, "warning")
        assert hasattr(test_logger, "error")

    def test_get_logger_with_context(self):
        """Test get_logger with context fields."""
        test_logger = get_logger(request_id="123", user_id="456")

        assert test_logger is not None


class TestLoggerModule:
    """Test logger module functionality."""

    def test_logger_instance(self):
        """Test that logger instance is available."""
        assert logger is not None

    def test_logger_has_methods(self):
        """Test that logger has logging methods."""
        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")

    def test_logger_is_wrapper(self):
        """Test that logger is wrapped correctly."""
        assert logger.__class__.__name__ == "LoggerWrapper"

    def test_logger_critical_method(self):
        """Test logger critical method with extra."""
        test_logger = get_logger(test_id="123")

        with patch.object(test_logger._logger, "critical") as mock_critical:
            test_logger.critical("Critical error")
            mock_critical.assert_called_once_with("Critical error")

            with patch.object(test_logger._logger, "bind") as mock_bind:
                bound_logger = MagicMock()
                mock_bind.return_value = bound_logger

                test_logger.critical("Critical with extra", extra={"error_code": 500})

                mock_bind.assert_called_once_with(error_code=500)
                bound_logger.critical.assert_called_once_with("Critical with extra")

    def test_logger_opt_method(self):
        """Test logger opt method pass-through."""
        test_logger = get_logger()

        mock_opt_result = MagicMock()
        with patch.object(
            test_logger._logger, "opt", return_value=mock_opt_result
        ) as mock_opt:
            result = test_logger.opt(depth=3, exception=True)

            mock_opt.assert_called_once_with(depth=3, exception=True)
            assert result == mock_opt_result

    def test_logger_getattr_fallback(self):
        """Test logger __getattr__ for unknown attributes."""
        test_logger = get_logger()

        test_logger._logger.custom_attribute = "custom_value"

        assert test_logger.custom_attribute == "custom_value"

        mock_method = MagicMock(return_value="result")
        test_logger._logger.custom_method = mock_method

        result = test_logger.custom_method("arg1", keyword="arg2")

        mock_method.assert_called_once_with("arg1", keyword="arg2")
        assert result == "result"

    def test_logger_warning_method(self):
        """Test logger warning method with extra."""
        test_logger = get_logger(test_id="456")

        with patch.object(test_logger._logger, "warning") as mock_warning:
            test_logger.warning("Warning message")
            mock_warning.assert_called_once_with("Warning message")

            with patch.object(test_logger._logger, "bind") as mock_bind:
                bound_logger = MagicMock()
                mock_bind.return_value = bound_logger

                test_logger.warning("Warning with extra", extra={"warning_code": 404})

                mock_bind.assert_called_once_with(warning_code=404)
                bound_logger.warning.assert_called_once_with("Warning with extra")


class TestInterceptHandlerEdgeCases:
    """Test edge cases for InterceptHandler."""

    def test_emit_with_invalid_level(self):
        """Test emit with invalid log level."""
        handler = InterceptHandler()

        record = logging.LogRecord(
            name="test",
            level=999,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        with patch("app.utils.logger.logger") as mock_logger:
            mock_opt = MagicMock()
            mock_logger.opt.return_value = mock_opt
            mock_logger.level.side_effect = ValueError("Invalid level")

            handler.emit(record)

            mock_opt.log.assert_called_once_with(999, "Test")

    def test_emit_with_no_frame(self):
        """Test emit when frame is None."""
        handler = InterceptHandler()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        with patch("app.utils.logger.logger") as mock_logger:
            with patch("app.utils.logger.logging.currentframe", return_value=None):
                mock_opt = MagicMock()
                mock_logger.opt.return_value = mock_opt
                mock_logger.level.return_value = MagicMock(name="INFO")

                handler.emit(record)

                mock_logger.opt.assert_called_once_with(depth=2, exception=None)

    def test_emit_with_logging_module_frame(self):
        """Test emit when frame is from logging module."""
        handler = InterceptHandler()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=logging.__file__,
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        with patch("app.utils.logger.logger") as mock_logger:
            mock_opt = MagicMock()
            mock_logger.opt.return_value = mock_opt
            mock_logger.level.return_value = MagicMock(name="INFO")

            mock_frame = MagicMock()
            mock_frame.f_code.co_filename = "test.py"
            mock_frame.f_back = None

            with patch("app.utils.logger.logging.currentframe") as mock_currentframe:
                logging_frame = MagicMock()
                logging_frame.f_code.co_filename = logging.__file__
                logging_frame.f_back = mock_frame

                mock_currentframe.return_value = logging_frame

                handler.emit(record)

                mock_logger.opt.assert_called_once()
                call_args = mock_logger.opt.call_args
                assert call_args[1]["depth"] > 2
