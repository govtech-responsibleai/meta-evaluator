"""Custom exceptions for the eval_task module."""

from abc import ABC


class EvalTaskError(Exception, ABC):
    """Base exception for all evaluation task-related errors."""

    def __init__(self, message: str):
        """Initialize the exception with a message.

        Args:
            message: The error message.
        """
        self.message = message
        super().__init__(self.message)


class TaskSchemaError(EvalTaskError):
    """Exception raised when task_schemas has errors."""

    def __init__(self, message: str = "Error with task_schemas"):
        """Initialize the exception.

        Args:
            message: Custom error message. Defaults to standard message.
        """
        super().__init__(message)
