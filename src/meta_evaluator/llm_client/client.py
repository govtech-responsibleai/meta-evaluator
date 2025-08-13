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

from abc import abstractmethod
from typing import Optional, TypeVar

from pydantic import BaseModel

from .base_client import (
    _FAILED_EMBEDDING_ERROR_TEMPLATE,
    _FAILED_RESPONSE_ERROR_TEMPLATE,
    _LOGPROBS_NOT_SUPPORTED_ERROR_TEMPLATE,
    BaseLLMClient,
    BaseLLMClientConfig,
)
from .exceptions import LLMAPIError, LLMValidationError
from .models import (
    LLMResponse,
    LLMUsage,
    Message,
    ParseResult,
    TagConfig,
)
from .serialization import LLMClientSerializedState

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound="LLMClientSerializedState")
ConfigT = TypeVar("ConfigT", bound="LLMClientConfig")


class LLMClientConfig(BaseLLMClientConfig):
    """Configuration settings for a synchronous LLM client.

    Inherits all configuration from BaseLLMClientConfig including API key,
    model settings, and capability flags.
    """

    @classmethod
    @abstractmethod
    def deserialize(
        cls, state: LLMClientSerializedState, api_key: str, **kwargs
    ) -> "LLMClientConfig":
        """Deserialize config from typed state object and API key.

        Args:
            state: Serialized configuration state object.
            api_key: API key to use for the config.
            **kwargs: Additional configuration parameters that may be needed.

        Returns:
            LLMClientConfig: Reconstructed configuration instance.
        """
        pass


class LLMClient(BaseLLMClient):
    """Abstract base class for all synchronous LLM clients.

    This class inherits from BaseLLMClient and provides synchronous implementations
    of the prompt and embedding methods. It serves as the base for all synchronous
    LLM client implementations.

    Attributes:
        config (LLMClientConfig): The configuration settings for the LLM client.
        logger (logging.Logger): Logger instance for logging messages and errors.
    """

    def __init__(self, config: LLMClientConfig):
        """Initialize the synchronous LLM client.

        Args:
            config (LLMClientConfig): The configuration settings for the LLM client.
        """
        super().__init__(config)

    @abstractmethod
    def _prompt(
        self, model: str, messages: list[Message], get_logprobs: bool
    ) -> tuple[str, LLMUsage]:
        """Send messages to the LLM and return response content and usage.

        Args:
            model: The model name to use for this request.
            messages: List of Message objects representing the conversation.
            get_logprobs: Whether to get logprobs from the underlying LLM client.

        Returns:
            tuple[str, LLMUsage]: Response content and usage statistics.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _get_embedding(self, text_list: list[str], model: str) -> list[list[float]]:
        """Get embeddings for a list of prompts using the underlying LLM client.

        Args:
            text_list: List of text prompts to generate embeddings for.
            model: The embedding model to use for this request.

        Returns:
            List of embedding vectors, one for each input prompt.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def prompt(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        get_logprobs: bool = False,
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
            get_logprobs (bool, optional): Whether to get logprobs from the underlying LLM client.
                Defaults to False.

        Returns:
            LLMResponse: Response object containing the full conversation (input + new
                assistant response), usage statistics, and metadata. Access the latest
                response via response.latest_response or response.content.

        Raises:
            LLMAPIError: If the request to the underlying LLM client fails.
            LLMValidationError: If no messages are provided or if get_logprobs is True and logprobs are not supported by the underlying LLM client

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
        # Re-raise validation errors to make them visible to ruff's DOC501 rule.
        # Without this, ruff can't see exceptions from private methods and won't
        # enforce docstring accuracy. DO NOT remove this try/catch.
        try:
            self._validate_messages(messages)
        except LLMValidationError:
            raise  # Ruff sees this explicit raise

        self.logger.info(f"Using model: {model_used}")
        self.logger.info(f"Input Payload: {messages}")
        if get_logprobs and not self.config.supports_logprobs:
            raise LLMValidationError(
                _LOGPROBS_NOT_SUPPORTED_ERROR_TEMPLATE.format(self.enum_value),
                self.enum_value,
            )
        try:
            raw_text, usage = self._prompt(
                model=model_used, messages=messages, get_logprobs=get_logprobs
            )
            output = self._construct_llm_response(raw_text, usage, messages, model_used)
        except Exception as e:
            raise LLMAPIError(
                _FAILED_RESPONSE_ERROR_TEMPLATE.format(self.enum_value),
                self.enum_value,
                e,
            )
        self.logger.info(f"Latest response: {output.latest_response}")
        self.logger.info(f"Output usage: {output.usage}")

        return output

    def get_embedding(
        self, text_list: list[str], model: Optional[str] = None
    ) -> list[list[float]]:
        """Get embeddings for a list of prompts with comprehensive logging.

        This method handles model selection, logging, and delegation to the provider-specific
        implementation. If no model is specified, falls back to the client's configured
        default embedding model.

        Args:
            text_list (list[str]): List of text prompts to generate embeddings for.
                Must contain at least one prompt.
            model (Optional[str], optional): Embedding model to use for this request. If None,
                uses self.config.default_embedding_model. Defaults to None.

        Returns:
            list[list[float]]: List of embedding vectors, one for each input prompt.
                Each embedding is represented as a list of floating-point numbers.

        Raises:
            LLMAPIError: If the request to the underlying LLM client fails.
            LLMValidationError: If no prompts are provided.

        Note:
            All requests are logged including:
            - Model selection (chosen vs default embedding model)
            - Number of prompts processed
            - Success confirmation

        Example:
            >>> prompts = ["Hello world", "How are you?"]
            >>> embeddings = client.get_embedding(prompts, model="text-embedding-3-large")
            >>> print(len(embeddings))  # 2
            >>> print(len(embeddings[0]))  # Embedding dimension (e.g., 1536)
        """
        model_used = model or self.config.default_embedding_model

        if model_used is None:
            raise LLMAPIError(
                f"No embedding model specified and {self.enum_value} has no default embedding model",
                self.enum_value,
                ValueError("No embedding model available"),
            )

        # Re-raise validation errors to make them visible to ruff's DOC501 rule.
        # Without this, ruff can't see exceptions from private methods and won't
        # enforce docstring accuracy. DO NOT remove this try/catch.
        try:
            self._validate_prompts(text_list)
        except LLMValidationError:
            raise  # Ruff sees this explicit raise

        self.logger.info(f"Using embedding model: {model_used}")
        self.logger.info(f"Processing {len(text_list)} prompts for embeddings")

        try:
            embeddings = self._get_embedding(text_list=text_list, model=model_used)
        except Exception as e:
            raise LLMAPIError(
                _FAILED_EMBEDDING_ERROR_TEMPLATE.format(self.enum_value),
                self.enum_value,
                e,
            )

        self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings

    @abstractmethod
    def _prompt_with_structured_response(
        self, messages: list[Message], response_model: type[T], model: str
    ) -> tuple[T, LLMUsage]:
        raise NotImplementedError("Subclasses must implement this method")

    def prompt_with_structured_response(
        self,
        messages: list[Message],
        response_model: type[T],
        model: Optional[str] = None,
    ) -> tuple[T, LLMResponse]:
        """Send a prompt to the underlying LLM client with comprehensive logging.

        This method handles model selection, logging, and delegation to the provider-specific
        implementation. If no model is specified, falls back to the client's configured
        default model.

        Args:
            model (Optional[str], optional): Model to use for this request. If None,
                uses self.config.default_model.
            messages (list[Message]): Complete conversation history including system, user,
                and assistant messages. The conversation context is preserved and sent to
                the underlying provider.
            response_model (T): Response model to use for this request, has to be a pydantic model


        Returns:
            LLMResponse: Response object containing the full conversation (input + new
                assistant response), usage statistics, and metadata. Access the latest
                response via response.latest_response or response.content.

        Raises:
            LLMAPIError: If the request to the underlying LLM client fails.
            LLMValidationError: If no messages are provided

        Note:
            All requests are logged including:
            - Model selection (chosen vs default)
        """
        model_used = model or self.config.default_model
        # Re-raise validation errors to make them visible to ruff's DOC501 rule.
        # Without this, ruff can't see exceptions from private methods and won't
        # enforce docstring accuracy. DO NOT remove this try/catch.
        try:
            self._validate_messages(messages)
        except LLMValidationError:
            raise  # Ruff sees this explicit raise

        self.logger.info(f"Using model: {model_used}")
        self.logger.info(f"Input Payload: {messages}")
        try:
            structured_response, usage = self._prompt_with_structured_response(
                messages=messages, response_model=response_model, model=model_used
            )

        except Exception as e:
            raise LLMAPIError(
                _FAILED_RESPONSE_ERROR_TEMPLATE.format(self.enum_value),
                self.enum_value,
                e,
            )
        output = self._construct_llm_response_with_structured_response(
            structured_response, usage, messages
        )
        self.logger.info(f"Latest response: {output.latest_response}")
        self.logger.info(f"Output usage: {output.usage}")

        return structured_response, output

    def prompt_with_xml_tags(
        self,
        messages: list[Message],
        tag_configs: list[TagConfig],
        model: Optional[str] = None,
        get_logprobs: bool = False,
    ) -> tuple[ParseResult, LLMResponse]:
        """Send a prompt to the LLM and parse XML tags from the response with comprehensive error handling.

        This method combines LLM prompting with structured XML tag parsing. It sends the provided
        messages to the underlying LLM client, then parses the response text according to the
        specified tag configurations. Unlike structured output methods, this approach works with
        any LLM by parsing XML tags from natural language responses.

        The parsing follows a "partial success" approach where individual tag parsing failures
        do not prevent the method from returning successfully. Instead, parsing errors are
        collected and returned alongside successfully parsed data, allowing callers to handle
        data quality issues gracefully while still accessing valid portions of the response.

        Args:
            messages (list[Message]): Complete conversation history including system, user,
                and assistant messages. The conversation context is preserved and sent to
                the underlying LLM provider. Must contain at least one message.
            tag_configs (list[TagConfig]): List of XML tag parsing configurations that define
                how to extract and validate content from the LLM response. Each configuration
                specifies the tag name, allowed values, cardinality expectations (single vs
                multiple instances), and error handling behavior for multiple values.
            model (Optional[str], optional): Specific model to use for this request. If None,
                uses the client's configured default model from self.config.default_model.
                Defaults to None.
            get_logprobs (bool, optional): Whether to get logprobs from the underlying LLM client.
                Defaults to False.

        Returns:
            tuple[ParseResult, LLMResponse]: A two-element tuple containing:
                - ParseResult: Structured parsing results with both successfully extracted
                data and detailed error information. The `data` dict maps tag names to
                their parsed values (str for cardinality="one", list[str] for cardinality="many").
                The `errors` list contains ParseError objects detailing any validation failures.
                Use `result.success` or `result.partial_success` properties to assess parsing outcomes.
                - LLMResponse: Complete response object containing the full conversation
                (input messages + raw LLM response), token usage statistics, and metadata.
                The raw LLM response text is available via `response.content`.

        Raises:
            LLMValidationError: If the messages list is empty or otherwise invalid for
                sending to the LLM provider. This prevents the LLM request from being made.
            LLMAPIError: If the request to the underlying LLM provider fails due to network
                issues, authentication problems, or other API-level errors. The original
                exception is wrapped and available via the `original_error` attribute.

        Note:
            XML parsing errors (invalid tag values, cardinality mismatches, etc.) do NOT
            raise exceptions. Instead, they are collected in the returned ParseResult's
            `errors` list, allowing robust handling of partially malformed LLM responses.

            The method logs all requests and responses following the same patterns as other
            prompt methods, including model selection, input payload, parsed results, and
            any parsing errors encountered.

            Tag extraction uses case-insensitive matching and handles both self-closing
            and standard XML tag formats. Whitespace is automatically trimmed from
            extracted values.

        Examples:
            Basic usage with single value extraction:
                >>> tag_config = TagConfig(
                ...     name="sentiment",
                ...     allowed_values=["positive", "negative", "neutral"],
                ...     cardinality="one"
                ... )
                >>> messages = [Message(role=RoleEnum.USER, content="Analyze: Great product!")]
                >>> result, response = client.prompt_with_xml_tags(messages, [tag_config])
                >>> if result.success:
                ...     print(f"Sentiment: {result.data['sentiment']}")
                >>> else:
                ...     print(f"Errors: {[str(e) for e in result.errors]}")

            Multiple tags with error handling:
                >>> configs = [
                ...     TagConfig(name="category", allowed_values=["tech", "sports"], cardinality="one"),
                ...     TagConfig(name="keywords", cardinality="many")  # No validation, multiple allowed
                ... ]
                >>> result, response = client.prompt_with_xml_tags(messages, configs, model="gpt-4")
                >>> print(f"Parsed {len(result.data)} tags with {len(result.errors)} errors")
                >>> if result.partial_success:
                ...     # Handle successful data even if some tags failed
                ...     for tag_name, value in result.data.items():
                ...         if value is not None:  # Skip failed single-value tags
                ...             print(f"{tag_name}: {value}")

            Accessing raw response alongside parsed data:
                >>> result, response = client.prompt_with_xml_tags(messages, configs)
                >>> print(f"Raw LLM response: {response.content}")
                >>> print(f"Token usage: {response.usage.total_tokens}")
                >>> print(f"Successfully parsed: {result.data}")
        """
        model_used = model or self.config.default_model

        # Re-raise validation errors to make them visible to ruff's DOC501 rule.
        # Without this, ruff can't see exceptions from private methods and won't
        # enforce docstring accuracy. DO NOT remove this try/catch.
        try:
            self._validate_messages(messages)
        except LLMValidationError:
            raise  # Ruff sees this explicit raise

        if get_logprobs and not self.config.supports_logprobs:
            raise LLMValidationError(
                _LOGPROBS_NOT_SUPPORTED_ERROR_TEMPLATE.format(self.enum_value),
                self.enum_value,
            )

        self.logger.info(f"Using model: {model_used}")
        self.logger.info(f"Input Payload: {messages}")
        self.logger.info(
            f"XML Tag Configurations: {[config.name for config in tag_configs]}"
        )

        try:
            raw_text, usage = self._prompt(
                model=model_used, messages=messages, get_logprobs=get_logprobs
            )
            llm_response = self._construct_llm_response(
                raw_text, usage, messages, model_used
            )
        except Exception as e:
            raise LLMAPIError(
                _FAILED_RESPONSE_ERROR_TEMPLATE.format(self.enum_value),
                self.enum_value,
                e,
            )

        # Parse XML tags from the raw response
        parse_result = self._construct_xml_tag_parsing(tag_configs, raw_text)

        # Log parsing results
        self.logger.info(f"Latest response: {llm_response.latest_response}")
        self.logger.info(f"Output usage: {llm_response.usage}")
        self.logger.info(f"Successfully parsed {len(parse_result.data)} XML tags")

        if parse_result.errors:
            self.logger.warning(
                f"XML parsing encountered {len(parse_result.errors)} errors:"
            )
            for error in parse_result.errors:
                self.logger.warning(f"  - {error}")

        if parse_result.success:
            self.logger.info("XML parsing completed successfully")
        elif parse_result.partial_success:
            self.logger.info("XML parsing completed with partial success")
        else:
            self.logger.warning("XML parsing failed for all configured tags")

        return parse_result, llm_response
