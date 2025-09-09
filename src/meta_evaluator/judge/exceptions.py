"""File for all Judge related exceptions."""

from typing import Optional

from pydantic import Field


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


class MissingTemplateVariablesError(JudgeError):
    """Raised when the prompt template is missing required variable placeholders.

    This exception is raised when a user's prompt.md file doesn't contain the required
    template variables (in curly brackets) for the columns defined in their EvalTask.
    The prompt must include placeholders like {column_name} for all prompt_columns
    and response_columns defined in the evaluation task.

    Attributes:
        missing_variables: List of variable names that are missing from the template.
        prompt_columns: List of prompt columns from the EvalTask.
        response_columns: List of response columns from the EvalTask.
        message: A descriptive error message with examples.

    Example:
        >>> # If prompt.md is missing {system} and {response} placeholders
        MissingTemplateVariablesError: Your prompt template is missing required variables: ['system', 'response'].
        Please add these placeholders to your prompt.md file. Example: "Evaluate this system: {system}. The response was: {response}"
    """

    def __init__(
        self,
        missing_variables: list[str],
        prompt_columns: Optional[list[str]] = Field(default=None),
        response_columns: Optional[list[str]] = Field(..., min_length=1),
    ):
        """Initialize the MissingTemplateVariablesError.

        Args:
            missing_variables: List of variable names missing from the template.
            prompt_columns: List of prompt columns from the EvalTask.
            response_columns: List of response columns from the EvalTask.
        """
        self.missing_variables = missing_variables
        self.prompt_columns = prompt_columns or []
        self.response_columns = response_columns or []

        # Create example with the missing variables
        example_parts = []
        if self.prompt_columns:
            example_parts.extend(["{" + col + "}" for col in self.prompt_columns])
        if self.response_columns:
            example_parts.extend(["{" + col + "}" for col in self.response_columns])

        example_text = "Evaluate this content: " + " and ".join(example_parts)

        message = (
            f"Your prompt template is missing required variables: {missing_variables}. "
            f"Please add these placeholders to your prompt.md file. "
            f'Example: "{example_text}"'
        )
        super().__init__(message)
