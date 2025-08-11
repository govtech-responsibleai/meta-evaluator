"""Shared base class for LLM clients with common functionality.

This module provides the shared base class that both synchronous and asynchronous
LLM clients inherit from. It contains all the common functionality including
validation, XML parsing, response construction, and configuration management.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, StrictBool

from .enums import ErrorType, LLMClientEnum, RoleEnum
from .exceptions import LLMValidationError
from .models import (
    LLMResponse,
    LLMUsage,
    Message,
    ParseError,
    ParseResult,
    TagConfig,
)
from .serialization import LLMClientSerializedState

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound="LLMClientSerializedState")
ConfigT = TypeVar("ConfigT", bound="BaseLLMClientConfig")

# Error message constants
_NO_MESSAGES_ERROR = "No messages provided"
_NO_PROMPTS_ERROR = "No prompts provided"
_FAILED_RESPONSE_ERROR_TEMPLATE = "Failed to get response from {}"
_FAILED_EMBEDDING_ERROR_TEMPLATE = "Failed to get embeddings from {}"
_LOGPROBS_NOT_SUPPORTED_ERROR_TEMPLATE = "Logprobs not supported by {}"


class BaseLLMClientConfig(ABC, BaseModel):
    """Base configuration class for LLM clients.

    Attributes:
        api_key (str): API key for authenticating requests to the LLM service.
        supports_structured_output (bool): Indicates whether the client supports structured output.
        default_model (str): The default language model to use when none is specified.
        default_embedding_model (str): The default embedding model for generating vector representations.
        supports_logprobs (bool): Indicates whether the client supports log probabilities.
    """

    api_key: str
    supports_structured_output: StrictBool
    default_model: str
    default_embedding_model: str
    supports_logprobs: StrictBool

    @abstractmethod
    def _prevent_instantiation(self) -> None:
        pass

    @abstractmethod
    def serialize(self) -> LLMClientSerializedState:
        """Serialize config to typed state object (excluding API key for security).

        Returns:
            LLMClientSerializedState: Configuration as typed state object without sensitive data.
        """
        pass

    @classmethod
    @abstractmethod
    def deserialize(
        cls, state: LLMClientSerializedState, api_key: str, **kwargs
    ) -> "BaseLLMClientConfig":
        """Deserialize config from typed state object and API key.

        Args:
            state: Serialized configuration state object.
            api_key: API key to use for the config.
            **kwargs: Additional configuration parameters that may be needed.

        Returns:
            BaseLLMClientConfig: Reconstructed configuration instance.
        """
        pass


class BaseLLMClient(ABC):
    """Abstract base class providing shared functionality for all LLM clients.

    This class contains all the common functionality shared between synchronous
    and asynchronous LLM clients, including validation, XML parsing, response
    construction, and logging setup. Both LLMClient and AsyncLLMClient inherit
    from this base class.

    Attributes:
        config (BaseLLMClientConfig): The configuration settings for the LLM client.
        logger (logging.Logger): Logger instance for logging messages and errors.
    """

    def __init__(self, config: BaseLLMClientConfig):
        """Initialize the base LLM client.

        Args:
            config (BaseLLMClientConfig): The configuration settings for the LLM client.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__module__)

    # ================================
    # Abstract Methods (Must be implemented by subclasses)
    # ================================

    @property
    @abstractmethod
    def enum_value(self) -> LLMClientEnum:
        """Return the unique LLMClientEnum value associated with this client.

        Returns:
            LLMClientEnum: The unique enumeration value associated with this client.
        """
        raise NotImplementedError("Subclasses must implement this method")

    # ================================
    # Shared Validation Methods
    # ================================

    def _validate_messages(self, messages: list[Message]) -> None:
        """Validate that messages list is not empty.

        Args:
            messages: List of messages to validate.

        Raises:
            LLMValidationError: If messages list is empty.
        """
        if len(messages) == 0:
            raise LLMValidationError(_NO_MESSAGES_ERROR, self.enum_value)

    def _validate_prompts(self, text_list: list[str]) -> None:
        """Validate that text prompts list is not empty.

        Args:
            text_list: List of text prompts to validate.

        Raises:
            LLMValidationError: If text list is empty.
        """
        if len(text_list) == 0:
            raise LLMValidationError(_NO_PROMPTS_ERROR, self.enum_value)

    # ================================
    # Shared Response Construction Methods
    # ================================

    def _construct_llm_response(
        self, new_response: str, usage: LLMUsage, messages: list[Message], model: str
    ) -> LLMResponse:
        """Construct an LLMResponse from raw response content.

        Args:
            new_response: Raw response content from the LLM.
            usage: Token usage information.
            messages: Original conversation messages.
            model: Model used for the request.

        Returns:
            LLMResponse: Complete response object with conversation history.
        """
        new_message = Message(role=RoleEnum.ASSISTANT, content=new_response)
        new_message_list = messages + [new_message]

        return LLMResponse(
            provider=self.enum_value,
            model=model,
            messages=new_message_list,
            usage=usage,
        )

    def _construct_llm_response_with_structured_response(
        self, new_response: BaseModel, usage: LLMUsage, messages: list[Message]
    ) -> LLMResponse:
        """Construct an LLMResponse from structured response content.

        Args:
            new_response: Structured response object (Pydantic model).
            usage: Token usage information.
            messages: Original conversation messages.

        Returns:
            LLMResponse: Complete response object with structured content as JSON.
        """
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

    # ================================
    # Shared XML Parsing Methods
    # ================================

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
