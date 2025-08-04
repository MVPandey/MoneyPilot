"""PydanticAI Agent Service for handling AI agent interactions."""

import os
from typing import Any, TypeVar, Generic, Type
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..utils.config import app_settings
from ..utils.logger import logger
from ..utils.exceptions import AgentException


T = TypeVar("T", bound=BaseModel)


class PydanticAIService(Generic[T]):
    """
    Factory service for creating and managing PydanticAI agents.

    This service acts as a factory that creates fresh agent instances for each request,
    enabling thread safety and agent collaboration. Each call to run() or run_stream()
    creates a new agent instance, avoiding shared mutable state.

    This service provides:
    - Agent factory pattern for creating agents on demand
    - Configuration for OpenAI-compatible endpoints
    - Type-safe agent responses
    - Support for streaming and tool calling
    - Integration with existing configuration

    Example:
        ```python
        from pydantic import BaseModel

        class StockAnalysis(BaseModel):
            ticker: str
            recommendation: str
            confidence: float

        service = PydanticAIService(
            output_type=StockAnalysis,
            system_prompt="You are a stock analysis expert."
        )

        # Each run creates a new agent instance
        result = await service.run("Analyze AAPL stock")
        print(result.ticker)  # Type-safe access

        # Can also create agents directly for collaboration
        agent1 = service.create_agent()
        agent2 = service.create_agent(system_prompt="You are a risk analyst")
        ```
    """

    def __init__(
        self,
        output_type: Type[T] | None = None,
        system_prompt: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        tools: list[Any] | None = None,
    ):
        """
        Initialize PydanticAI service.

        Args:
            output_type: Pydantic model for structured responses
            system_prompt: System prompt for the agent
            model_name: Model name (defaults to config)
            base_url: API base URL (defaults to config/env)
            api_key: API key (defaults to config/env)
            tools: List of tools available to the agent
        """
        self.model_name = model_name or app_settings.LLM_MODEL_NAME
        self.base_url = base_url or os.getenv(
            "LLM_API_BASE_URL", app_settings.LLM_API_BASE_URL
        )
        self.api_key = api_key or os.getenv("LLM_API_KEY", app_settings.LLM_API_KEY)

        provider = OpenAIProvider(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        self.model = OpenAIModel(
            model_name=self.model_name,
            provider=provider,
        )

        self.system_prompt = system_prompt
        self.output_type = output_type
        self.tools = tools or []

        logger.info(
            "Initialized PydanticAI agent",
            extra={
                "model": self.model_name,
                "base_url": self.base_url,
                "has_output_type": output_type is not None,
                "tool_count": len(self.tools),
            },
        )

    def create_agent(
        self,
        system_prompt: str | None = None,
        output_type: Type[T] | None = None,
        tools: list[Any] | None = None,
    ) -> Agent:
        """
        Create a new agent instance.

        Args:
            system_prompt: Override default system prompt
            output_type: Override default output type
            tools: Override default tools

        Returns:
            Configured Agent instance
        """
        agent = Agent(
            model=self.model,
            system_prompt=system_prompt or self.system_prompt,
            output_type=output_type or self.output_type,
        )

        tools_to_register = tools if tools is not None else self.tools
        for tool in tools_to_register:
            agent.tool(tool)

        return agent

    async def run(
        self, prompt: str, context: dict[str, Any] | None = None, **kwargs
    ) -> T | str:
        """
        Run the agent with a prompt.

        Args:
            prompt: User prompt
            context: Optional context dictionary
            **kwargs: Additional arguments for the agent

        Returns:
            Structured response (if output_type) or string

        Raises:
            AgentException: If agent execution fails
        """
        try:
            agent = self.create_agent()
            result = await agent.run(prompt, deps=context, **kwargs)

            logger.info(
                "Agent execution completed",
                extra={
                    "prompt_length": len(prompt),
                    "has_context": context is not None,
                },
            )

            return result.data

        except Exception as e:
            logger.error(
                "Agent execution failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise AgentException(f"Agent execution failed: {e}") from e

    async def run_stream(
        self, prompt: str, context: dict[str, Any] | None = None, **kwargs
    ):
        """
        Run the agent with streaming response.

        Args:
            prompt: User prompt
            context: Optional context dictionary
            **kwargs: Additional arguments for the agent

        Yields:
            Streaming response chunks

        Raises:
            AgentException: If agent execution fails
        """
        try:
            agent = self.create_agent()
            async with agent.run_stream(prompt, deps=context, **kwargs) as response:
                async for chunk in response.stream_text():
                    yield chunk

        except Exception as e:
            logger.error(
                "Agent streaming failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise AgentException(f"Agent streaming failed: {e}") from e


class SimpleAgent(PydanticAIService[T]):
    """
    Enhanced agent that supports both structured outputs and parameters.

    This agent extends the base PydanticAIService to provide:
    - Support for structured outputs using Pydantic models
    - Support for plain text output (when output_type is None)
    - Dynamic parameters that can be passed at runtime
    - Backward compatibility with the original SimpleAgent

    Example with structured output:
        ```python
        from pydantic import BaseModel

        class WeatherInfo(BaseModel):
            temperature: float
            condition: str
            humidity: int

        agent = SimpleAgent(
            output_type=WeatherInfo,
            system_prompt="You are a weather information assistant"
        )
        response = await agent.run("What is the weather?")
        print(response.temperature)  # Type-safe access
        ```

    Example with parameters:
        ```python
        agent = SimpleAgent(
            system_prompt="You are a helpful assistant. User location: {location}"
        )
        response = await agent.run(
            "What is the weather?",
            parameters={"location": "New York"}
        )
        ```

    Example without structured output (backward compatible):
        ```python
        agent = SimpleAgent("You are a helpful assistant")
        response = await agent.run("What is the weather?")
        print(response)  # String response
        ```
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        output_type: Type[T] | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        tools: list[Any] | None = None,
    ):
        super().__init__(
            output_type=output_type,
            system_prompt=system_prompt,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            tools=tools,
        )

    def create_agent(
        self,
        system_prompt: str | None = None,
        output_type: Type[T] | None = None,
        tools: list[Any] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Agent:
        """
        Create a new agent instance with optional parameters.

        Args:
            system_prompt: Override default system prompt
            output_type: Override default output type
            tools: Override default tools
            parameters: Parameters to format the system prompt

        Returns:
            Configured Agent instance
        """
        prompt = system_prompt or self.system_prompt

        if prompt and parameters and isinstance(prompt, str):
            prompt = prompt.format(**parameters)

        return super().create_agent(
            system_prompt=prompt,
            output_type=output_type,
            tools=tools,
        )

    async def run(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | str:
        """
        Run the agent with a prompt and optional parameters.

        Args:
            prompt: User prompt
            context: Optional context dictionary for agent dependencies
            parameters: Optional parameters to format the system prompt
            **kwargs: Additional arguments for the agent

        Returns:
            Structured response (if output_type) or string

        Raises:
            AgentException: If agent execution fails
        """
        try:
            agent = self.create_agent(parameters=parameters)
            result = await agent.run(prompt, deps=context, **kwargs)

            logger.info(
                "Agent execution completed",
                extra={
                    "prompt_length": len(prompt),
                    "has_context": context is not None,
                    "has_parameters": parameters is not None,
                },
            )

            return result.data

        except Exception as e:
            logger.error(
                "Agent execution failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise AgentException(f"Agent execution failed: {e}") from e

    async def run_stream(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Run the agent with streaming response and optional parameters.

        Args:
            prompt: User prompt
            context: Optional context dictionary for agent dependencies
            parameters: Optional parameters to format the system prompt
            **kwargs: Additional arguments for the agent

        Yields:
            Streaming response chunks

        Raises:
            AgentException: If agent execution fails
        """
        try:
            agent = self.create_agent(parameters=parameters)
            async with agent.run_stream(prompt, deps=context, **kwargs) as response:
                async for chunk in response.stream_text():
                    yield chunk

        except Exception as e:
            logger.error(
                "Agent streaming failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise AgentException(f"Agent streaming failed: {e}") from e
