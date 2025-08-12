"""Enumerations for the llm_client domain."""

from enum import Enum


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


class LLMClientEnum(Enum):
    """Enum class representing the supported synchronous LLM clients.

    This enum class is used to identify the provider-specific LLM client that is
    used to interact with the LLM service synchronously. The enum values are used in the
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
    TEST = "test"


class AsyncLLMClientEnum(Enum):
    """Enum class representing the supported asynchronous LLM clients.

    This enum class is used to identify the provider-specific async LLM client that is
    used to interact with the LLM service asynchronously.

    The enum values are:

    - OPENAI: Async OpenAI LLM client
    - AZURE_OPENAI: Async Azure OpenAI LLM client
    - ANTHROPIC: Async Anthropic LLM client

    The values are used in the AsyncLLMClient implementations to identify the client
    that is used to interact with the LLM service.
    The enum values are also used in logging and other contexts to identify the
    client that is being used.
    """

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class ErrorType(str, Enum):
    """Error types for XML tag parsing.

    Attributes:
        INVALID_VALUE: Tag value is invalid (e.g. not in allowed_values).
    """

    INVALID_VALUE = "invalid_value"
    CARDINALITY_MISMATCH = "cardinality_mismatch"
    MULTIPLE_VALUES_CONFLICT = "multiple_values_conflict"
    TAG_NOT_FOUND = "tag_not_found"
