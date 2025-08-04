from typing import Any


class LLMException(Exception):
    """Exception raised for errors in LLM operations.

    Attributes:
        message: Explanation of the error
        details: Additional error details or context
    """

    def __init__(self, message: str, details: Any = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class WorkflowException(Exception):
    """Base exception for workflow errors"""

    pass


class AgentException(Exception):
    """Base exception for agent errors"""

    pass


class ToolException(Exception):
    """Exception raised for tool execution errors"""

    pass
