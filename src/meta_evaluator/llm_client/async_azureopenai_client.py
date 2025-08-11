"""Async Azure OpenAI client implementation.

This module provides an asynchronous implementation of the LLM client interface
specifically for Azure OpenAI's API, including GPT models and embeddings.
"""

from typing import TypeVar

import instructor
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .async_client import AsyncLLMClient, AsyncLLMClientConfig
from .enums import LLMClientEnum, RoleEnum
from .exceptions import LLMAPIError
from .models import LLMUsage, Message
from .serialization import AzureOpenAISerializedState, LLMClientSerializedState

T = TypeVar("T", bound=BaseModel)


class AsyncAzureOpenAIConfig(AsyncLLMClientConfig):
    """Configuration for async Azure OpenAI client."""

    supports_structured_output: bool = True
    supports_logprobs: bool = True
    endpoint: str
    api_version: str

    def _prevent_instantiation(self) -> None:
        """Required abstract method - allows instantiation."""
        pass

    def serialize(self) -> AzureOpenAISerializedState:
        """Serialize Azure OpenAI config to typed state object (excluding API key for security).

        Returns:
            AzureOpenAISerializedState: Configuration as typed state object without sensitive data.
        """
        return AzureOpenAISerializedState(
            endpoint=self.endpoint,
            api_version=self.api_version,
            default_model=self.default_model,
            default_embedding_model=self.default_embedding_model,
            supports_structured_output=self.supports_structured_output,
            supports_logprobs=self.supports_logprobs,
            supports_instructor=True,
        )

    @classmethod
    def deserialize(
        cls,
        state: LLMClientSerializedState,
        api_key: str,
        **kwargs,
    ) -> "AsyncAzureOpenAIConfig":
        """Reconstruct Azure OpenAI config from serialized state with new API key.

        Args:
            state: Serialized configuration state.
            api_key: New API key to use.
            **kwargs: Additional configuration parameters.

        Returns:
            AsyncAzureOpenAIConfig: Reconstructed configuration instance.

        Raises:
            TypeError: If state is not the correct type.
        """
        if not isinstance(state, AzureOpenAISerializedState):
            raise TypeError(
                f"Expected AzureOpenAISerializedState, got {type(state).__name__}"
            )

        return cls(
            api_key=api_key,
            endpoint=state.endpoint,
            api_version=state.api_version,
            default_model=state.default_model,
            default_embedding_model=state.default_embedding_model,
            supports_structured_output=state.supports_structured_output,
            supports_logprobs=state.supports_logprobs,
        )


class AsyncAzureOpenAIClient(AsyncLLMClient):
    """Async Azure OpenAI client implementation."""

    def __init__(self, config: AsyncAzureOpenAIConfig):
        """Initialize async Azure OpenAI client."""
        super().__init__(config)
        self.config: AsyncAzureOpenAIConfig = config

        # Initialize clients
        self.client = AsyncAzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.endpoint,
            api_version=config.api_version,
        )
        self.instructor_client = instructor.from_openai(self.client)

    @property
    def enum_value(self) -> LLMClientEnum:
        """Return Azure OpenAI client enum."""
        return LLMClientEnum.AZURE_OPENAI

    # ================================
    # Helper Methods
    # ================================

    def _convert_messages_to_openai_format(
        self, messages: list[Message]
    ) -> list[ChatCompletionMessageParam]:
        """Convert internal Message objects to Azure OpenAI ChatCompletionMessageParam format.

        Args:
            messages: List of internal Message objects to convert.

        Returns:
            list[ChatCompletionMessageParam]: Messages in Azure OpenAI API format.
        """
        openai_messages: list[ChatCompletionMessageParam] = []

        for message in messages:
            if message.role == RoleEnum.USER:
                openai_messages.append(
                    {
                        "role": "user",
                        "content": message.content,
                    }
                )
            elif message.role == RoleEnum.SYSTEM:
                openai_messages.append(
                    {
                        "role": "system",
                        "content": message.content,
                    }
                )
            elif message.role == RoleEnum.ASSISTANT:
                openai_messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                    }
                )

        return openai_messages

    def _extract_usage_from_response(self, response) -> LLMUsage:
        """Extract usage statistics from Azure OpenAI response.

        Args:
            response: Azure OpenAI API response object.

        Returns:
            LLMUsage: Extracted token usage information.

        Raises:
            LLMAPIError: If usage data is missing from the response.
        """
        usage_data = response.usage
        if not usage_data:
            raise LLMAPIError(
                "Expected usage data from Azure OpenAI response but got None",
                provider=LLMClientEnum.AZURE_OPENAI,
                original_error=ValueError("Missing usage data in response"),
            )

        return LLMUsage(
            prompt_tokens=usage_data.prompt_tokens,
            completion_tokens=usage_data.completion_tokens,
            total_tokens=usage_data.total_tokens,
        )

    # ================================
    # Core API Methods
    # ================================

    async def _prompt(
        self, model: str, messages: list[Message], get_logprobs: bool
    ) -> tuple[str, LLMUsage]:
        """Send prompt to Azure OpenAI and return response content and usage.

        Args:
            model: The model name to use for this request.
            messages: List of Message objects for the conversation.
            get_logprobs: Whether to include log probabilities in response.

        Returns:
            tuple[str, LLMUsage]: Response content and token usage information.

        Raises:
            LLMAPIError: If response content or usage data is missing.
        """
        openai_messages = self._convert_messages_to_openai_format(messages)

        # Make API call with or without logprobs
        if get_logprobs and self.config.supports_logprobs:
            response = await self.client.chat.completions.create(
                model=model, messages=openai_messages, logprobs=True, top_logprobs=5
            )
        else:
            response = await self.client.chat.completions.create(
                model=model, messages=openai_messages
            )

        # Extract content
        content = response.choices[0].message.content
        if not content:
            raise LLMAPIError(
                f"Expected non-empty content from Azure OpenAI response but got: {content}",
                provider=LLMClientEnum.AZURE_OPENAI,
                original_error=ValueError("Empty content in response"),
            )

        usage = self._extract_usage_from_response(response)
        return content, usage

    async def _get_embedding(
        self, text_list: list[str], model: str
    ) -> list[list[float]]:
        """Get text embeddings from Azure OpenAI.

        Args:
            text_list: List of text strings to embed.
            model: The embedding model to use.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        response = await self.client.embeddings.create(input=text_list, model=model)
        return [embedding.embedding for embedding in response.data]

    async def _prompt_with_structured_response(
        self, messages: list[Message], response_model: type[T], model: str
    ) -> tuple[T, LLMUsage]:
        """Get structured response using instructor.

        Args:
            messages: List of Message objects for the conversation.
            response_model: Pydantic model class for structured response.
            model: The model name to use for this request.

        Returns:
            tuple[T, LLMUsage]: Parsed response object and token usage information.
        """
        openai_messages = self._convert_messages_to_openai_format(messages)

        # Use instructor for structured response
        (
            structured_response,
            completion,
        ) = await self.instructor_client.chat.completions.create_with_completion(
            model=model,
            response_model=response_model,
            messages=openai_messages,
        )

        usage = self._extract_usage_from_response(completion)
        return structured_response, usage
