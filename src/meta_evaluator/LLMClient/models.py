"""Package meta_evaluator.LLMClient.models contains data models used for interacting with large language model (LLM) clients.

The package contains models for representing messages, responses, and usage statistics of LLM clients.

The package contains the following models:
*   Message: Represents a message in a conversation with an LLM client.
*   LLMResponse: Represents a response from an LLM client.
*   LLMUsage: Represents the usage statistics of an LLM client.

The models are used to provide a unified interface for interacting with LLM clients from different providers, such as OpenAI, Anthropic, and Gemini. The models are also used to provide type safety and to enable auto-completion when interacting with LLM clients in Python.
"""

from enum import Enum
import time
from typing import Any, Literal, Optional, Union
import uuid
from pydantic import BaseModel, model_validator

_RESPONSE_ID_UUID_LENGTH = 12


class RoleEnum(str, Enum):
    """Enum representing different roles in a conversation.

    This enum defines the roles supported in a conversation with an LLM client.

    The role enum is used when sending a prompt to the LLM client to specify the role of the message sender.

    Attributes:
        SYSTEM: Represents the system role, responsible for managing the conversation flow and providing context.
        USER: Represents the user role, typically the human participant interacting with the LLM.
        ASSISTANT: Represents the assistant role, which is the LLM itself providing responses to the user's queries.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Class representing a message in a conversation.

    This class represents a message sent by one of the participants in a conversation
    with an LLM client. It contains the message content and the role of the sender.

    The class is used to represent messages in the conversation history that is passed
    to the provider-specific LLM client implementations.

    Attributes:
        role: Enum value from RoleEnum indicating the role of the message sender.
        content: The text content of the message.
    """

    role: RoleEnum
    content: str

    def __str__(self):
        """Returns a string representation of the message.

        The string representation is a formatted string with the role and content of the message.
        The format is: "{role}: {content}".

        Returns:
            str: The string representation of the message.
        """
        return f"{self.role.value}: {self.content}"


class LLMClientEnum(Enum):
    """Enum class representing the supported LLM clients.

    This enum class is used to identify the provider-specific LLM client that is
    used to interact with the LLM service. The enum values are used in the
    LLMClient implementations as a unique identifier for the client.

    The enum values are:

    - OPENAI: OpenAI LLM client
    - AZURE_OPENAI: Azure OpenAI LLM client
    - GEMINI: Gemini LLM client
    - ANTHROPIC: Anthropic LLM client

    The values are used in the LLMClient implementations to identify the client
    that is used to interact with the LLM service.
    The enum values are also used in logging and other contexts to identify the
    client that is being used.
    """

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class LLMUsage(BaseModel):
    """Represents the usage statistics of a single interaction with an LLM client.

    The LLMUsage class is used to capture the usage statistics of a single interaction
    with an LLM client. The usage statistics include the number of prompt tokens,
    number of completion tokens, and the total number of tokens used in the interaction.

    The usage statistics are used by the LLMClient implementations to track the usage
    of the LLM clients and to provide detailed information about the usage to the
    user.

    Attributes:
        prompt_tokens: The number of prompt tokens used in the interaction.
        completion_tokens: The number of completion tokens used in the interaction.
        total_tokens: The total number of tokens used in the interaction.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMResponse(BaseModel):
    """Represents a response from a Large Language Model (LLM) client interaction.

    The `LLMResponse` class models the entire response from an LLM client after a prompt
    is sent. It encapsulates details such as the unique identifier for the response,
    the provider of the LLM service, the model used, the conversation messages, and the
    usage statistics.

    Attributes:
        id (str): A unique identifier for the response. If not provided, it is auto-generated
            based on the current timestamp, provider, model, and a random number.
        provider (LLMClientEnum): The provider of the LLM service (e.g., OpenAI, Azure).
        model (str): The specific model used for generating the response.
        messages (list[Message]): A list of messages representing the conversation, including
            both the prompt messages and the response from the LLM.
        usage (LLMUsage): An object containing the usage statistics for the interaction, such as
            the number of tokens used.

    Properties:
        latest_response (Message): Retrieves the most recent message from the conversation.
        content (str): Retrieves the content of the most recent message.

    Methods:
        model_post_init(__context: Any): Initializes the `id` attribute if it is not provided,
            using the current time, provider, model, and a random number.
    """

    id: str = ""
    provider: LLMClientEnum
    model: str
    messages: list[Message]
    usage: LLMUsage

    @model_validator(mode="after")
    def _validate_messages_not_empty(self) -> "LLMResponse":
        if not self.messages:
            raise ValueError("LLMResponse must have at least one message")
        return self

    @property
    def latest_response(self) -> Message:
        """Returns the most recent message from the conversation.

        This property returns the latest message from the conversation, which is the
        last element in the list of messages. If the list is empty, returns None.

        Returns:
            Message: The most recent message from the conversation.
        """
        return self.messages[-1]

    @property
    def content(self) -> str:
        """Returns the content of the most recent message from the conversation.

        This property returns the content of the latest message from the conversation,
        which is the last element in the list of messages. If the list is empty, returns
        an empty string.

        Returns:
            str: The content of the most recent message from the conversation.
        """
        return self.latest_response.content

    def model_post_init(self, __context: Any) -> None:
        """Initializes the `id` attribute if it is not provided.

        This method is a post-init hook for Pydantic models. It is called after the
        model has been initialized with the provided data. If the `id` attribute is
        not provided, it is initialized with a unique identifier based on the current
        timestamp, provider, model, and a random number.

        The `id` attribute is used to identify the response in logging and other
        contexts.
        """
        if not self.id:
            timestamp = int(time.time())
            self.id = f"{self.provider.value}_{self.model}_{timestamp}_{uuid.uuid4().hex[:_RESPONSE_ID_UUID_LENGTH]}"


class TagConfig(BaseModel):
    """Configuration for parsing and validating XML tags from raw text.

    This class defines the parsing rules for a specific XML tag, including value
    validation, cardinality constraints, and error handling behavior when multiple
    instances are found but only one is expected.

    The parser will extract all instances of the specified XML tag from the input
    text and apply the configured validation and cardinality rules to determine
    the final result and any validation errors.

    Attributes:
        name: The XML tag name to search for (e.g., "user_id", "status").
            Case-sensitive matching against opening and closing tags.
        allowed_values: List of valid string values for this tag's content.
            If None, any string content is accepted (freeform text).
            If provided, tag content must exactly match one of these values
            or it will be considered invalid.
        cardinality: Expected number of tag instances in the input.
            "one": Expect exactly one instance of this tag.
            "many": Accept more than one instance of this tag.
        multiple_handling: Behavior when cardinality="one" but multiple tags found.
            Only applies when cardinality="one" and multiple valid tags are present.
            "error": Generate validation error, return None for this tag.
            "allow_both": Accept multiple values, return as list despite cardinality="one".
            "error_if_different": Accept if all values identical, error if different.

    Examples:
        Single required field with validation:
            >>> config = TagConfig(
            ...     name="status",
            ...     allowed_values=["active", "inactive", "pending"],
            ...     cardinality="one",
            ...     multiple_handling="error"
            ... )

        Multiple tags with restricted values:
            >>> config = TagConfig(
            ...     name="tags",
            ...     allowed_values=["red", "blue", "green"],
            ...     cardinality="many"
            ... )

        Single freeform text field that allows duplicates if identical:
            >>> config = TagConfig(
            ...     name="description",
            ...     allowed_values=None,
            ...     cardinality="one",
            ...     multiple_handling="error_if_different"
            ... )

    Note:
        The multiple_handling attribute is ignored when cardinality="many" since
        multiple instances are expected in that case.
    """

    name: str
    allowed_values: Optional[list[str]] = None  # None = freeform
    cardinality: Literal["one", "many"] = "many"
    # Only matters when cardinality="one" but found multiple
    multiple_handling: Literal["error", "allow_both", "error_if_different"] = "error"


class ErrorType(str, Enum):
    """Error types for XML tag parsing.

    Attributes:
        INVALID_VALUE: Tag value is invalid (e.g. not in allowed_values).
    """

    INVALID_VALUE = "invalid_value"
    CARDINALITY_MISMATCH = "cardinality_mismatch"
    MULTIPLE_VALUES_CONFLICT = "multiple_values_conflict"
    TAG_NOT_FOUND = "tag_not_found"


class ParseError(BaseModel):
    """Structured parsing error with context."""

    error_type: ErrorType
    tag_name: str
    message: str
    found_values: Optional[list[str]] = None
    expected_values: Optional[list[str]] = None

    def __str__(self) -> str:
        """Returns the message of the parse error."""
        return self.message


class ParseResult(BaseModel):
    """Result of XML tag parsing with structured error reporting.

    Provides both the successfully parsed data and detailed error information
    for any validation or cardinality issues encountered during parsing.

    Attributes:
        data: Successfully parsed tag values. Keys are tag names, values are
            either single strings (cardinality="one") or lists (cardinality="many").
        errors: List of structured parsing errors with context about what failed.
        success: True if no errors occurred during parsing.
        partial_success: True if some data was parsed despite errors.
    """

    data: dict[str, Union[str, list[str]]]
    errors: list[ParseError] = []

    @property
    def success(self) -> bool:
        """True if parsing completed without any errors."""
        return len(self.errors) == 0

    @property
    def partial_success(self) -> bool:
        """True if some data was parsed, even if errors occurred.

        Returns:
            bool: True if there is any parsed data, even if errors occurred.
        """
        return len(self.data) > 0

    def get_errors_by_tag(self, tag_name: str) -> list[ParseError]:
        """Get all errors for a specific tag.

        Returns:
            list[ParseError]: List of errors that occurred when parsing the specified tag.
        """
        return [e for e in self.errors if e.tag_name == tag_name]

    def get_errors_by_type(self, error_type: ErrorType) -> list[ParseError]:
        """Get all errors of a specific type.

        Returns:
            list[ParseError]: List of errors with the specified type.
        """
        return [e for e in self.errors if e.error_type == error_type]
