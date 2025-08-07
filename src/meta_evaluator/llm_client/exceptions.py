"""All Exceptions for LLMClient."""

from abc import ABC

from .models import LLMClientEnum


class LLMClientError(Exception, ABC):
    """Base exception for LLM client errors."""

    def __init__(self, message: str, provider: LLMClientEnum):
        """Initialize the LLMClientError exception.

        Args:
            message (str): The message to be displayed.
            provider (LLMClientEnum): The provider of the LLM client that raised the error.
        """
        self.message = message
        self.provider = provider
        super().__init__(self.message)

    def __str__(self) -> str:
        """Returns a string representation of the error.

        The string representation is formatted as:
        [{provider.value}] {message}

        Returns:
            str: The formatted string representation of the error.
        """
        return f"[{self.provider.value}] {super().__str__()}"


class LLMAPIError(LLMClientError):
    """Wraps provider API errors in LLMClient."""

    def __init__(
        self, message: str, provider: LLMClientEnum, original_error: Exception
    ):
        """Initialize the LLMAPIError exception.

        Args:
            message (str): The message to be displayed.
            provider (LLMClientEnum): The provider of the LLM client that raised the error.
            original_error (Exception): The original error that was raised.
        """
        super().__init__(message=message, provider=provider)
        self.original_error = original_error

    def __str__(self) -> str:
        """Returns a string representation of the error.

        The string representation is in the following format:
        [{provider.value}] {message} | Original: {original_error}

        Returns:
            str: A string representation of the error.
        """
        return f"[{self.provider.value}] {super(LLMClientError, self).__str__()} | Original: {self.original_error}"


class LLMValidationError(LLMClientError):
    """Client-side validation errors."""

    def __init__(self, message: str, provider: LLMClientEnum):
        """Initialize the LLMValidationError exception.

        Args:
            message (str): The message to be displayed.
            provider (LLMClientEnum): The provider of the LLM client that raised the error.
        """
        super().__init__(message=message, provider=provider)
