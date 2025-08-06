"""Custom exceptions for the results module."""

from abc import ABC


class ResultsError(Exception, ABC):
    """Base exception for all results-related errors."""

    def __init__(self, message: str):
        """Initialize the exception with a message.

        Args:
            message: The error message.
        """
        self.message = message
        super().__init__(self.message)


class EmptyResultsError(ResultsError):
    """Error raised when results data is empty."""

    def __init__(self, message: str = "Results data is empty"):
        """Initialize the exception.

        Args:
            message: The error message.
        """
        self.message = message
        super().__init__(self.message)


class ResultsValidationError(ResultsError):
    """Error raised when results validation fails."""

    def __init__(self, validation_issue: str):
        """Initialize the exception.

        Args:
            validation_issue: Description of the validation issue.
        """
        super().__init__(f"Results validation failed: {validation_issue}")


class TaskNotFoundError(ResultsError):
    """Error raised when a task is not found in the task schema."""

    def __init__(self, task_name: str):
        """Initialize the exception.

        Args:
            task_name: The name of the task that was not found.
        """
        super().__init__(f"Task '{task_name}' not found in task schema")


class ResultsDataFormatError(ResultsError):
    """Error raised when an unsupported results data format is encountered."""

    def __init__(self, format_name: str, data_filename: str):
        """Initialize the exception.

        Args:
            format_name: The unsupported format name.
            data_filename: The name of the data file that was attempted to be saved.
        """
        super().__init__(
            f"Unsupported results data format '{format_name}' for '{data_filename}'"
        )


class BuilderInitializationError(ResultsError):
    """Error raised when initializing the results builder fails."""

    def __init__(self, message: str):
        """Initialize the exception.

        Args:
            message: The error message.
        """
        super().__init__(message)


class ResultsStateParsingError(ResultsError):
    """Error raised when parsing results state file fails."""

    def __init__(self, details: str, original_error: Exception):
        """Initialize the exception.

        Args:
            details: Description of what failed to parse.
            original_error: The original parsing error.
        """
        self.original_error = original_error
        super().__init__(f"Failed to parse results state: {details}")


class InvalidFileError(ResultsError):
    """Error raised when a file is invalid."""

    def __init__(self, message: str, original_error: Exception):
        """Initialize the exception.

        Args:
            message: The error message.
            original_error: The original error.
        """
        self.original_error = original_error
        super().__init__(f"Invalid file: {message}")


class IncompleteResultsError(ResultsError):
    """Error raised when evaluation results are incomplete."""

    def __init__(self, message: str):
        """Initialize the exception.

        Args:
            message: The error message.
        """
        super().__init__(f"Incomplete results: {message}")


class MismatchedTasksError(ResultsError):
    """Error raised when tasks are mismatched."""

    def __init__(self, task_names: list[str], message: str):
        """Initialize the exception.

        Args:
            task_names: The list of task names that are mismatched.
            message: The error message.
        """
        super().__init__(f"Mismatched tasks: {task_names}. {message}")
