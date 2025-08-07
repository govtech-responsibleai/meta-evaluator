"""This is a placeholder for the MetaEvaluator class.

Note: This class is currently under development
"""

from beartype.claw import beartype_this_package

beartype_this_package()

from .llm_client import LLMClient, LLMClientConfig  # noqa: E402
from .llm_client.exceptions import LLMAPIError, LLMValidationError, LLMClientError  # noqa: E402
from .llm_client.models import LLMUsage, Message, LLMResponse  # noqa: E402
from .llm_client.enums import LLMClientEnum, RoleEnum  # noqa: E402
from .meta_evaluator import MetaEvaluator  # noqa: E402

__all__ = [
    "LLMClientEnum",
    "LLMAPIError",
    "LLMValidationError",
    "LLMClientError",
    "LLMUsage",
    "Message",
    "LLMResponse",
    "LLMClientConfig",
    "LLMClient",
    "RoleEnum",
    "MetaEvaluator",
]
