"""Concrete implementation of LLMClient for OpenAI."""

from typing import TypeVar

import instructor
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from .client import LLMClient, LLMClientConfig
from .exceptions import LLMAPIError
from .models import LLMClientEnum, LLMUsage, Message, RoleEnum
from .serialization import LLMClientSerializedState, OpenAISerializedState

T = TypeVar("T", bound=BaseModel)


class OpenAIConfig(LLMClientConfig):
    """Configuration settings for OpenAI LLMClient."""

    supports_structured_output: bool = True
    supports_logprobs: bool = True

    def _prevent_instantiation(self) -> None:
        """Allow instantiation.

        This is a dummy method that must be implemented according to the
        abstract base class. It is not intended to be used, and calling it
        will have no effect.
        """
        pass

    def serialize(self) -> OpenAISerializedState:
        """Serialize OpenAI config to typed state object (excluding API key for security).

        Returns:
            OpenAISerializedState: Configuration as typed state object without sensitive data.
        """
        return OpenAISerializedState(
            default_model=self.default_model,
            default_embedding_model=self.default_embedding_model,
            supports_structured_output=self.supports_structured_output,
            supports_logprobs=self.supports_logprobs,
            supports_instructor=True,  # OpenAI always supports instructor
        )

    @classmethod
    def deserialize(
        cls,
        state: LLMClientSerializedState,
        api_key: str,
        **kwargs,
    ) -> "OpenAIConfig":
        """Deserialize OpenAI config from typed state object and API key.

        Args:
            state: Serialized configuration state object.
            api_key: API key to use for the config.
            **kwargs: Additional configuration parameters (unused for OpenAI).

        Returns:
            OpenAIConfig: Reconstructed configuration instance.

        Raises:
            TypeError: If state is not an OpenAISerializedState instance.
        """
        if not isinstance(state, OpenAISerializedState):
            raise TypeError(
                f"Expected OpenAISerializedState, got {type(state).__name__}"
            )

        return cls(
            api_key=api_key,
            default_model=state.default_model,
            default_embedding_model=state.default_embedding_model,
            supports_structured_output=state.supports_structured_output,
            supports_logprobs=state.supports_logprobs,
        )


class OpenAIClient(LLMClient):
    """Concrete implementation of LLMClient for OpenAI."""

    def __init__(self, config: OpenAIConfig):
        """Initialize the OpenAI client with the given configuration."""
        super().__init__(config)
        self.config: OpenAIConfig = config

        # Initialize OpenAI client
        self.client = OpenAI(api_key=config.api_key)

        # Initialize instructor client for structured responses
        self.instructor_client = instructor.from_openai(self.client)

    def _convert_messages_to_openai_format(
        self, messages: list[Message]
    ) -> list[ChatCompletionMessageParam]:
        """Convert Message objects to OpenAI ChatCompletionMessageParam format.

        Returns:
            list[ChatCompletionMessageParam]: List of OpenAI-formatted message parameters.
        """
        openai_messages: list[ChatCompletionMessageParam] = []

        for message in messages:
            if message.role == RoleEnum.USER:
                user_message: ChatCompletionUserMessageParam = {
                    "role": "user",
                    "content": message.content,
                }
                openai_messages.append(user_message)
            elif message.role == RoleEnum.SYSTEM:
                system_message: ChatCompletionSystemMessageParam = {
                    "role": "system",
                    "content": message.content,
                }
                openai_messages.append(system_message)
            elif message.role == RoleEnum.ASSISTANT:
                assistant_message: ChatCompletionAssistantMessageParam = {
                    "role": "assistant",
                    "content": message.content,
                }
                openai_messages.append(assistant_message)

        return openai_messages

    def _prompt(
        self, model: str, messages: list[Message], get_logprobs: bool
    ) -> tuple[str, LLMUsage]:
        """Send messages to OpenAI and return response content and usage.

        Returns:
            tuple[str, LLMUsage]: A tuple containing the response content and usage statistics.

        Raises:
            LLMAPIError: If the response content is empty or usage data is missing.
        """
        openai_messages = self._convert_messages_to_openai_format(messages)

        # Make API call with or without logprobs
        if get_logprobs and self.config.supports_logprobs:
            response = self.client.chat.completions.create(
                model=model, messages=openai_messages, logprobs=True, top_logprobs=5
            )
        else:
            response = self.client.chat.completions.create(
                model=model, messages=openai_messages
            )

        # Extract content
        content = response.choices[0].message.content
        if not content:
            raise LLMAPIError(
                f"Expected non-empty content from OpenAI response but got: {content}",
                provider=LLMClientEnum.OPENAI,
                original_error=ValueError("Empty content in response"),
            )

        # Extract usage information
        usage_data = response.usage
        if not usage_data:
            raise LLMAPIError(
                "Expected usage data from OpenAI response but got None",
                provider=LLMClientEnum.OPENAI,
                original_error=ValueError("Missing usage data in response"),
            )

        usage = LLMUsage(
            prompt_tokens=usage_data.prompt_tokens,
            completion_tokens=usage_data.completion_tokens,
            total_tokens=usage_data.total_tokens,
        )

        return content, usage

    def _prompt_with_structured_response(
        self, messages: list[Message], response_model: type[T], model: str
    ) -> tuple[T, LLMUsage]:
        """Send messages to OpenAI and return structured response and usage.

        Returns:
            tuple[T, LLMUsage]: A tuple containing the structured response and usage statistics.
        """
        openai_messages = self._convert_messages_to_openai_format(messages)

        # Use instructor for structured response
        response, completion = (
            self.instructor_client.chat.completions.create_with_completion(
                model=model,
                response_model=response_model,
                messages=openai_messages,
            )
        )

        usage = LLMUsage(
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            total_tokens=completion.usage.total_tokens,
        )

        return response, usage

    @property
    def enum_value(self) -> LLMClientEnum:
        """Return the unique LLMClientEnum value for OpenAI."""
        return LLMClientEnum.OPENAI

    def _get_embedding(self, text_list: list[str], model: str) -> list[list[float]]:
        """Get embeddings for a list of prompts using OpenAI.

        Args:
            text_list: List of text prompts to generate embeddings for.
            model: The embedding model to use for this request.

        Returns:
            list[list[float]]: List of embedding vectors, one for each input prompt.
        """
        embedding_response = self.client.embeddings.create(input=text_list, model=model)

        return [embedding.embedding for embedding in embedding_response.data]
