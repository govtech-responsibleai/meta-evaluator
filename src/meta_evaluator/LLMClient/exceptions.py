"""All Exceptions for LLMClient."""

from .models import LLMClientEnum


class LLMClientError(Exception):
    """Base exception for LLM client errors."""

    def __init__(
        self, message: str, provider: LLMClientEnum, original_error: Exception
    ):
        """Initialize the LLMClientError exception.

        Args:
            message (str): The message to be displayed.
            provider (LLMClientEnum): The provider of the LLM client that raised the error.
            original_error (Exception): The original error that was raised.
        """
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error
