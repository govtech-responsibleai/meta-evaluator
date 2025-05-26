"""Unified interface for Large Language Model (LLM) providers with comprehensive logging.

This package provides a standardized abstraction layer for interacting with multiple
LLM providers including OpenAI, Anthropic, Gemini, and Azure OpenAI. It normalizes
API differences, provides consistent logging and usage tracking, and enables easy
switching between providers.

Key Features:
- Provider-agnostic interface with consistent Message/Response models
- Comprehensive logging of requests, responses, and token usage
- Automatic model fallback and configuration management
- Type-safe enums and Pydantic validation
- Extensible design for adding new LLM providers

Supported Providers:
- OpenAI
- Azure OpenAI
- Gemini
- Anthropic

Usage:
    >>> config = SomeProviderConfig(api_key="...", default_model="gpt-4")
    >>> client = SomeProviderClient(config)
    >>> messages = [Message(role=RoleEnum.USER, content="Hello")]
    >>> response = client.prompt(messages)
    >>> print(response.content)

"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel
from .models import Message, LLMClientEnum, LLMResponse
from .exceptions import LLMClientError


class LLMClientConfig(ABC, BaseModel):
    """Configuration settings for an LLM client.

    Attributes:
        api_key (str): API key for authenticating requests to the LLM service.
        supports_instructor (bool): Indicates whether the client supports instructor-led models.
        default_model (str): The default language model to use when none is specified.
        default_embedding_model (str): The default embedding model for generating vector representations.
    """

    api_key: str
    supports_instructor: bool
    default_model: str
    default_embedding_model: str


class LLMClient(ABC):
    """Abstract base class for all LLM clients.

    This class serves as a base class for all LLM clients. It provides a common
    interface for interacting with different LLM services. The class is
    abstract and cannot be instantiated directly. Instead, one of its concrete
    subclasses should be used.

    Attributes:
        config (LLMClientConfig): The configuration settings for the LLM client.
        logger (logging.Logger): Logger instance for logging messages and errors.
    """

    def __init__(self, config: LLMClientConfig):
        """Initializes the LLMClient.

        This is an abstract base class for all LLM clients. It sets up the
        client with the given configuration and logger.

        Args:
            config (LLMClientConfig): The configuration settings for the LLM client.

        Attributes:
            config (LLMClientConfig): Stores the configuration settings for the LLM client.
            logger (logging.Logger): Logger instance for logging messages and errors.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__module__)

    @abstractmethod
    def _prompt(self, model: str, messages: list[Message]) -> LLMResponse:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def enum_value(self) -> LLMClientEnum:
        """Return the unique LLMClientEnum value associated with this client.

        This method returns the unique enumeration value associated with this LLM
        client. The enumeration value is used to identify the client in logging and
        other contexts.

        Returns:
            LLMClientEnum: The unique enumeration value associated with this client.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def prompt(
        self, messages: list[Message], model: Optional[str] = None
    ) -> LLMResponse:
        """Send a prompt to the underlying LLM client with comprehensive logging.

        This method handles model selection, logging, and delegation to the provider-specific
        implementation. If no model is specified, falls back to the client's configured
        default model.

        Args:
            messages (list[Message]): Complete conversation history including system, user,
                and assistant messages. The conversation context is preserved and sent to
                the underlying provider.
            model (Optional[str], optional): Model to use for this request. If None,
                uses self.config.default_model. Defaults to None.

        Returns:
            LLMResponse: Response object containing the full conversation (input + new
                assistant response), usage statistics, and metadata. Access the latest
                response via response.latest_response or response.content.

        Raises:
            LLMClientError: If the request to the underlying LLM client fails.

        Note:
            All requests are logged including:
            - Model selection (chosen vs default)
            - Complete input message payload
            - Assistant response content
            - Token usage statistics

        Example:
            >>> messages = [Message(role=RoleEnum.USER, content="Hello")]
            >>> response = client.prompt(messages, model="gpt-4")
            >>> print(response.content)  # Assistant's response
            >>> print(len(response.messages))  # Original + assistant response
        """
        model_used = model or self.config.default_model
        self.logger.info(f"Using model: {model_used}")
        self.logger.info(f"Input Payload: {messages}")
        try:
            output = self._prompt(model_used, messages)
        except Exception as e:
            raise LLMClientError(
                f"Failed to get response from {self.enum_value}", self.enum_value, e
            )
        self.logger.info(f"Latest response: {output.latest_response}")
        self.logger.info(f"Output usage: {output.usage}")

        return output
