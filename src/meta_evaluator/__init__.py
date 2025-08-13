"""This is a placeholder for the MetaEvaluator class.

Note: This class is currently under development
"""

# Temporarily disable beartype to resolve import issues

# Apply beartype after all imports are complete to avoid path resolution conflicts
from beartype.claw import beartype_this_package

from .llm_client import (
    AsyncLLMClientEnum,
    ErrorType,
    LLMClient,
    LLMClientConfig,
    LLMClientEnum,
    RoleEnum,
)
from .llm_client.exceptions import LLMAPIError, LLMClientError, LLMValidationError
from .llm_client.models import LLMResponse, LLMUsage, Message
from .meta_evaluator import MetaEvaluator

beartype_this_package()

__all__ = [
    "LLMClient",
    "LLMClientConfig",
    "LLMClientEnum",
    "AsyncLLMClientEnum",
    "RoleEnum",
    "ErrorType",
    "LLMAPIError",
    "LLMClientError",
    "LLMValidationError",
    "LLMUsage",
    "Message",
    "LLMResponse",
    "MetaEvaluator",
]
