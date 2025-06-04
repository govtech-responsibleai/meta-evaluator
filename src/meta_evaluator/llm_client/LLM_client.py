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
from typing import Optional, TypeVar

from pydantic import BaseModel, StrictBool
from .models import (
    ErrorType,
    Message,
    LLMClientEnum,
    LLMResponse,
    LLMUsage,
    ParseError,
    ParseResult,
    RoleEnum,
    TagConfig,
)
from .exceptions import LLMAPIError, LLMValidationError
import re


T = TypeVar("T", bound=BaseModel)
_NO_MESSAGES_ERROR = "No messages provided"
_NO_PROMPTS_ERROR = "No prompts provided"
_FAILED_RESPONSE_ERROR_TEMPLATE = "Failed to get response from {}"
_FAILED_EMBEDDING_ERROR_TEMPLATE = "Failed to get embeddings from {}"
_LOGPROBS_NOT_SUPPORTED_ERROR_TEMPLATE = "Logprobs not supported by {}"


class LLMClientConfig(ABC, BaseModel):
    """Configuration settings for an LLM client.

    Attributes:
        api_key (str): API key for authenticating requests to the LLM service.
        supports_structured_output (bool): Indicates whether the client supports structured output.
        default_model (str): The default language model to use when none is specified.
        default_embedding_model (str): The default embedding model for generating vector representations.
    """

    api_key: str
    supports_structured_output: StrictBool
    default_model: str
    default_embedding_model: str
    supports_logprobs: bool

    @abstractmethod
    def _prevent_instantiation(self) -> None:
        pass


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
    def _prompt(
        self, model: str, messages: list[Message], get_logprobs: bool
    ) -> tuple[str, LLMUsage]:
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

    def _validate_messages(self, messages: list[Message]) -> None:
        if len(messages) == 0:
            raise LLMValidationError(_NO_MESSAGES_ERROR, self.enum_value)

    def _validate_prompts(self, text_list: list[str]) -> None:
        if len(text_list) == 0:
            raise LLMValidationError(_NO_PROMPTS_ERROR, self.enum_value)

    def _construct_llm_response(
        self, new_response: str, usage: LLMUsage, messages: list[Message], model: str
    ) -> LLMResponse:
        new_message = Message(role=RoleEnum.ASSISTANT, content=new_response)
        new_message_list = messages + [new_message]

        return LLMResponse(
            provider=self.enum_value,
            model=model,
            messages=new_message_list,
            usage=usage,
        )

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

    def _construct_llm_response_with_structured_response(
        self, new_response: BaseModel, usage: LLMUsage, messages: list[Message]
    ) -> LLMResponse:
        new_message = Message(
            role=RoleEnum.ASSISTANT, content=new_response.model_dump_json()
        )
        new_message_list = messages + [new_message]

        return LLMResponse(
            provider=self.enum_value,
            model=self.config.default_model,
            messages=new_message_list,
            usage=usage,
        )

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

    def _extract_tag_values(self, tag_name: str, raw_text: str) -> list[str]:
        """Extract all content values from XML tags with the given name.

        Args:
            tag_name: The XML tag name to search for (e.g., "user_id", "status")
            raw_text: Raw XML text to search within

        Returns:
            List of string values found within the specified XML tags
        """
        # Pattern matches <tag_name>content</tag_name> or <tag_name attr="value">content</tag_name>
        pattern = rf"<{re.escape(tag_name)}(?:\s[^>]*)?>(.*?)</{re.escape(tag_name)}>"
        matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)

        # Strip whitespace from each match and filter out empty strings
        values = [match.strip() for match in matches if match.strip()]

        return values

    def _construct_xml_tag_parsing(
        self, tag_config_list: list[TagConfig], raw_text: str
    ) -> ParseResult:
        """Parse XML tags from raw text according to configuration rules.

        Extracts content from XML tags specified in the configuration list and validates
        the results according to value constraints, cardinality rules, and error handling
        preferences. Returns both successfully parsed data and detailed error information
        for any validation failures.

        The parsing process follows these steps for each configured tag:
        1. Extract all instances of the tag from raw text using regex matching
        2. If no instances found, record TAG_NOT_FOUND error and skip to next tag
        3. Validate extracted values against allowed_values if specified, recording
        INVALID_VALUE errors for non-matching values while keeping valid ones
        4. Apply cardinality constraints to valid values:
        - cardinality="one": Requires exactly 1 valid value
        - cardinality="many": Requires 1+ valid values (empty is an error)
        5. Handle multiple values for cardinality="one" according to multiple_handling rules
        6. Add successfully parsed values to data dict, failures only go to errors list

        This method uses a "clean separation" design where the data dict contains only
        successfully parsed values, while all failure information is recorded in the
        errors list. This avoids redundant None values and empty lists in the data.

        Args:
            tag_config_list: List of TagConfig objects defining parsing rules for each
                XML tag. Each config specifies the tag name, allowed values, cardinality
                expectations, and behavior when multiple values are found.
            raw_text: Raw XML text content to parse. Should contain the XML tags
                specified in the tag configurations.

        Returns:
            ParseResult containing:
                - data: Dictionary mapping tag names to successfully parsed values.
                Keys only exist for tags that were successfully parsed and satisfied
                all validation rules. Values are strings for cardinality="one" or
                lists of strings for cardinality="many".
                - errors: List of ParseError objects detailing any validation failures.
                Includes errors for missing tags, invalid values, cardinality mismatches,
                and multiple value conflicts.

        Note:
            Failed parsing attempts do not create entries in the data dictionary.
            Use `"tag_name" in result.data` or `result.data.get("tag_name")` to check
            for successful parsing. Use `result.get_errors_by_tag("tag_name")` to get
            detailed error information for failed tags.

            Tag extraction uses case-insensitive regex matching and handles both
            self-closing and standard XML tag formats. Whitespace is automatically
            trimmed from extracted values, and empty values are filtered out.

        Examples:
            Successful parsing with mixed results:
                >>> configs = [
                ...     TagConfig(name="status", allowed_values=["active"], cardinality="one"),
                ...     TagConfig(name="missing", cardinality="one")
                ... ]
                >>> result = self._construct_xml_tag_parsing(configs, "<status>active</status>")
                >>> result.data  # {"status": "active"} - no "missing" key
                >>> len(result.errors)  # 1 - TAG_NOT_FOUND for "missing"

            All parsing failures:
                >>> configs = [TagConfig(name="invalid", allowed_values=["valid"], cardinality="one")]
                >>> result = self._construct_xml_tag_parsing(configs, "<invalid>bad</invalid>")
                >>> result.data  # {} - empty dict, no successful parses
                >>> result.errors[0].error_type  # ErrorType.INVALID_VALUE
        """
        data = {}
        errors = []

        for config in tag_config_list:
            # Step 1: Extract all instances of this tag from the raw text
            raw_values = self._extract_tag_values(config.name, raw_text)

            # Step 2: Handle case where no tag instances were found
            if not raw_values:
                errors.append(
                    ParseError(
                        error_type=ErrorType.TAG_NOT_FOUND,
                        tag_name=config.name,
                        message=f"Required tag '{config.name}' not found in text",
                    )
                )
                # Skip to next config - no data entry for missing tags
                continue

            # Step 3: Validate extracted values against allowed_values constraint
            if config.allowed_values is not None:
                valid_values = []
                allowed_values_lower = [v.lower() for v in config.allowed_values]

                for value in raw_values:
                    if value in allowed_values_lower:
                        valid_values.append(value)
                    else:
                        # Record invalid value error but continue processing other values
                        errors.append(
                            ParseError(
                                error_type=ErrorType.INVALID_VALUE,
                                tag_name=config.name,
                                message=f"Invalid value '{value}' for tag '{config.name}'. "
                                f"Allowed values: {config.allowed_values}",
                                found_values=[value],
                                expected_values=config.allowed_values,
                            )
                        )
            else:
                # No value restrictions - all extracted values are considered valid
                valid_values = raw_values

            # Step 4: Check if we have any valid values after filtering
            if not valid_values:
                # All values were invalid - no data entry, errors already recorded above
                continue

            # Step 5: Apply cardinality constraints to valid values
            if config.cardinality == "one":
                # Expect exactly one valid value
                if len(valid_values) == 1:
                    # Perfect - exactly one valid value
                    data[config.name] = valid_values[0]
                else:
                    # Multiple valid values found - handle according to multiple_handling strategy
                    if config.multiple_handling == "error":
                        errors.append(
                            ParseError(
                                error_type=ErrorType.CARDINALITY_MISMATCH,
                                tag_name=config.name,
                                message=f"Expected exactly 1 value for '{config.name}', "
                                f"found {len(valid_values)}",
                                found_values=valid_values,
                            )
                        )
                        # No data entry - cardinality constraint violated
                    elif config.multiple_handling == "allow_both":
                        # Accept multiple values as a list despite cardinality="one"
                        data[config.name] = valid_values
                    elif config.multiple_handling == "error_if_different":
                        # Check if all values are identical
                        unique_values = list(set(valid_values))
                        if len(unique_values) == 1:
                            # All values are the same - accept the common value
                            data[config.name] = unique_values[0]
                        else:
                            # Multiple different values - this is an error
                            errors.append(
                                ParseError(
                                    error_type=ErrorType.MULTIPLE_VALUES_CONFLICT,
                                    tag_name=config.name,
                                    message=f"Multiple different values for '{config.name}': "
                                    f"{valid_values}",
                                    found_values=valid_values,
                                )
                            )
                            # No data entry - conflicting values

            elif config.cardinality == "many":
                # Expect one or more valid values (empty list after filtering is an error)
                # Note: We already checked `not valid_values` above, so we have 1+ values here
                data[config.name] = valid_values

        return ParseResult(data=data, errors=errors)

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
