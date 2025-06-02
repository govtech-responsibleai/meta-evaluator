"""Concrete implementation of LLMClient for Azure OpenAI."""

from typing import TypeVar
import warnings

import instructor
from openai import AzureOpenAI
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

from .LLM_client import LLMClientConfig, LLMClient
from .models import LLMClientEnum, Message, LLMUsage, RoleEnum

T = TypeVar("T", bound=BaseModel)


class AzureOpenAIConfig(LLMClientConfig):
    """Configuration settings for Azure OpenAI LLMClient."""

    supports_structured_output: bool = True
    supports_logprobs: bool = True
    endpoint: str
    api_version: str

    def _prevent_instantiation(self) -> None:
        """Allow instantiation.

        This is a dummy method that must be implemented according to the
        abstract base class. It is not intended to be used, and calling it
        will have no effect.
        """
        pass


class AzureOpenAIClient(LLMClient):
    """Concrete implementation of LLMClient for Azure OpenAI."""

    def __init__(self, config: AzureOpenAIConfig):
        """Initialize the Azure OpenAI client with the given configuration."""
        super().__init__(config)
        self.config: AzureOpenAIConfig = config

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.endpoint,
            api_version=config.api_version,
        )

        # Initialize instructor client for structured responses
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Client should be an instance of openai.OpenAI or openai.AsyncOpenAI.*",
            )
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
        """Send messages to Azure OpenAI and return response content and usage.

        Returns:
            tuple[str, LLMUsage]: A tuple containing the response content and usage statistics.

        Raises:
            ValueError: If the response content is empty or usage data is missing.
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
            raise ValueError(
                f"Expected non-empty content from Azure OpenAI response but got: {content}"
            )

        # Extract usage information
        usage_data = response.usage
        if not usage_data:
            raise ValueError(
                "Expected usage data from Azure OpenAI response but got None"
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
        """Send messages to Azure OpenAI and return structured response and usage.

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
        """Return the unique LLMClientEnum value for Azure OpenAI."""
        return LLMClientEnum.AZURE_OPENAI

    def _get_embedding(self, text_list: list[str], model: str) -> list[list[float]]:
        """Get embeddings for a list of prompts using Azure OpenAI.

        Args:
            text_list: List of text prompts to generate embeddings for.
            model: The embedding model to use for this request.

        Returns:
            list[list[float]]: List of embedding vectors, one for each input prompt.
        """
        embedding_response = self.client.embeddings.create(input=text_list, model=model)

        return [embedding.embedding for embedding in embedding_response.data]
