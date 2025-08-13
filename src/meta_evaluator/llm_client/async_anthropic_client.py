"""Async Anthropic client implementation.

This module provides an asynchronous implementation of the LLM client interface
specifically for Anthropic's Claude API.
"""

from typing import TypeVar

import instructor
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from pydantic import BaseModel

from .async_client import AsyncLLMClient, AsyncLLMClientConfig
from .enums import AsyncLLMClientEnum, RoleEnum
from .exceptions import LLMAPIError
from .models import LLMUsage, Message
from .serialization import AnthropicSerializedState, LLMClientSerializedState

T = TypeVar("T", bound=BaseModel)


class AsyncAnthropicConfig(AsyncLLMClientConfig):
    """Configuration settings for async Anthropic LLMClient."""

    supports_structured_output: bool = True
    supports_logprobs: bool = False

    def _prevent_instantiation(self) -> None:
        """Required abstract method - allows instantiation."""
        pass

    def serialize(self) -> AnthropicSerializedState:
        """Serialize Anthropic config to typed state object (excluding API key for security).

        Returns:
            AnthropicSerializedState: Configuration as typed state object without sensitive data.
        """
        return AnthropicSerializedState(
            default_model=self.default_model,
            supports_structured_output=self.supports_structured_output,
            supports_logprobs=self.supports_logprobs,
            supports_instructor=True,  # OpenAI always supports instructor
        )

    @classmethod
    def deserialize(
        cls, state: LLMClientSerializedState, api_key: str, **kwargs
    ) -> "AsyncAnthropicConfig":
        """Deserialize config from typed state object and API key.

        Args:
            state: Serialized configuration state object.
            api_key: API key to use for the config.
            **kwargs: Additional configuration parameters that may be needed.

        Returns:
            AsyncAnthropicConfig: Reconstructed configuration instance.
        """
        return cls(
            api_key=api_key,
            default_model=state.default_model,
            supports_structured_output=state.supports_structured_output,
            supports_logprobs=state.supports_logprobs,
        )


class AsyncAnthropicClient(AsyncLLMClient):
    """Async Anthropic LLM client implementation using Claude models."""

    def __init__(self, config: AsyncAnthropicConfig):
        """Initialize the async Anthropic client.

        Args:
            config (AsyncAnthropicConfig): Configuration settings for the Anthropic client.
        """
        super().__init__(config)
        self._anthropic_client = AsyncAnthropic(api_key=config.api_key)
        self._instructor_client = instructor.from_anthropic(self._anthropic_client)
        self.logger.info("Async Anthropic client initialized successfully")

    @property
    def enum_value(self) -> AsyncLLMClientEnum:
        """Return the unique AsyncLLMClientEnum value associated with this client.

        Returns:
            AsyncLLMClientEnum: The ANTHROPIC enumeration value.
        """
        return AsyncLLMClientEnum.ANTHROPIC

    def _convert_message_to_anthropic_format(self, message: Message) -> MessageParam:
        """Convert a Message object to Anthropic-compatible format.

        Args:
            message: The message to convert.

        Returns:
            MessageParam: Anthropic-compatible message format.

        Raises:
            LLMAPIError: If the message role is not supported by Anthropic.
        """
        if message.role == RoleEnum.USER:
            return MessageParam(role="user", content=message.content)
        elif message.role == RoleEnum.ASSISTANT:
            return MessageParam(role="assistant", content=message.content)
        else:
            raise LLMAPIError(
                f"Unsupported role: {message.role}. Anthropic only supports user and assistant messages.",
                self.enum_value,
                ValueError(f"Unsupported role: {message.role}"),
            )

    async def _prompt(
        self, model: str, messages: list[Message], get_logprobs: bool
    ) -> tuple[str, LLMUsage]:
        """Send messages to Anthropic Claude and return response content and usage (async).

        Args:
            model: The model name to use for this request.
            messages: List of Message objects representing the conversation.
            get_logprobs: Whether to get logprobs (not supported by Anthropic).

        Returns:
            tuple[str, LLMUsage]: Response content and usage statistics.

        Raises:
            LLMAPIError: If the API request fails or if multiple system messages are provided.
        """
        try:
            system_message = None
            anthropic_messages = []

            for message in messages:
                if message.role == RoleEnum.SYSTEM:
                    if system_message is not None:
                        raise LLMAPIError(
                            "Multiple system messages are not supported by Anthropic",
                            self.enum_value,
                            ValueError(
                                "Multiple system messages are not supported by Anthropic"
                            ),
                        )
                    system_message = message.content
                else:
                    anthropic_messages.append(
                        self._convert_message_to_anthropic_format(message)
                    )

            kwargs = {
                "model": model,
                "max_tokens": 4096,
                "messages": anthropic_messages,
            }

            if system_message:
                kwargs["system"] = system_message

            response = await self._anthropic_client.messages.create(**kwargs)

            content = response.content[0].text if response.content else ""
            usage = LLMUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )

            return content, usage

        except Exception as e:
            raise LLMAPIError(
                "Failed to get response from Anthropic", self.enum_value, e
            )

    async def _get_embedding(
        self, text_list: list[str], model: str
    ) -> list[list[float]]:
        """Get embeddings for a list of prompts (async).

        Note: Anthropic does not provide embedding models. This method raises an error.

        Args:
            text_list: List of text prompts to generate embeddings for.
            model: The embedding model to use.

        Raises:
            LLMAPIError: Always raised as Anthropic doesn't support embeddings.
        """
        raise LLMAPIError(
            "Anthropic does not support embeddings",
            self.enum_value,
            NotImplementedError("Anthropic does not support embeddings"),
        )

    async def _prompt_with_structured_response(
        self, messages: list[Message], response_model: type[T], model: str
    ) -> tuple[T, LLMUsage]:
        """Send a prompt to Anthropic and return a structured response (async).

        Args:
            messages: List of Message objects representing the conversation.
            response_model: Pydantic model class for structured output.
            model: The model name to use for this request.

        Returns:
            tuple[T, LLMUsage]: Structured response object and usage statistics.

        Raises:
            LLMAPIError: If the API request fails or if multiple system messages are provided.
        """
        try:
            system_message = None
            anthropic_messages = []

            for message in messages:
                if message.role == RoleEnum.SYSTEM:
                    if system_message is not None:
                        raise LLMAPIError(
                            "Multiple system messages are not supported by Anthropic",
                            self.enum_value,
                            ValueError(
                                "Multiple system messages are not supported by Anthropic"
                            ),
                        )
                    system_message = message.content
                else:
                    anthropic_messages.append(
                        self._convert_message_to_anthropic_format(message)
                    )

            kwargs = {
                "model": model,
                "max_tokens": 4096,
                "messages": anthropic_messages,
                "response_model": response_model,
            }

            if system_message:
                kwargs["system"] = system_message

            response = await self._instructor_client.messages.create(**kwargs)

            usage = LLMUsage(
                prompt_tokens=0,  # Instructor doesn't provide detailed usage
                completion_tokens=0,
                total_tokens=0,
            )

            return response, usage

        except Exception as e:
            raise LLMAPIError(
                "Failed to get structured response from Anthropic", self.enum_value, e
            )
