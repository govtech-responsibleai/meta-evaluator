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


class ErrorType(str, Enum):
    """Error types for XML tag parsing.

    Attributes:
        INVALID_VALUE: Tag value is invalid (e.g. not in allowed_values).
    """

    INVALID_VALUE = "invalid_value"
    CARDINALITY_MISMATCH = "cardinality_mismatch"
    MULTIPLE_VALUES_CONFLICT = "multiple_values_conflict"
    TAG_NOT_FOUND = "tag_not_found"
