"""File for all Judge related exceptions."""


class JudgeError(Exception):
    """Base class for Judge exceptions."""

    pass


class LLMAPIError(JudgeError):
    """Wraps provider API errors in LLMClient."""

    def __init__(self, message: str, llm_client: str, original_error: Exception):
        """Initialize the LLMAPIError exception.

        Args:
            message (str): The message to be displayed.
            llm_client (str): The LLM client that was used.
            original_error (Exception): The original error that was raised.
        """
        super().__init__(message)
        self.llm_client = llm_client
        self.original_error = original_error

    def __str__(self) -> str:
        """Returns a string representation of the error.

        The string representation is in the following format:
        [{llm_client}] {message} | Original: {original_error}

        Returns:
            str: A string representation of the error.
        """
        return f"[{self.llm_client}] {super(JudgeError, self).__str__()} | Original: {self.original_error}"


class UnsupportedFormatMethodError(JudgeError):
    """Raised when the specified answering method is not supported for the given model/provider.

    This exception is raised when a user tries to use an answering method (like 'structured')
    that is not supported by their model/provider combination. It provides guidance on
    alternative methods to try.

    Attributes:
        method: The answering method that was attempted.
        model: The model/provider combination that doesn't support the method.
        supported_methods: List of alternative methods the user can try.
        message: A descriptive error message with suggestions.

    Example:
        >>> # If 'structured' doesn't work with a particular model
        UnsupportedFormatMethodError: The 'structured' method is not supported for model 'anthropic/claude-3-haiku'. Please try one of these alternatives: ['instructor', 'xml']
    """

    def __init__(
        self,
        method: str,
        model: str,
        suggested_methods: list[str],
    ):
        """Initialize the UnsupportedFormatMethodError.

        Args:
            method: The answering method that was attempted.
            model: The model/provider combination that doesn't support the method.
            suggested_methods: List of alternative methods the user can try.
        """
        self.method = method
        self.model = model
        self.suggested_methods = suggested_methods
        message = (
            f"The '{method}' method is not supported for model '{model}'. "
            f"Please try one of these alternatives: {suggested_methods}"
        )
        super().__init__(message)
