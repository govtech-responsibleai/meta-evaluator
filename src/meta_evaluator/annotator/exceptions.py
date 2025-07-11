"""Custom exceptions for the annotator module."""

from typing import Optional


class AnnotationError(Exception):
    """Base exception for all annotation-related errors."""

    def __init__(self, message: str):
        """Initialize the exception with a message.

        Args:
            message: The error message.
        """
        self.message = message
        super().__init__(self.message)


class AnnotatorInitializationError(AnnotationError):
    """Exception raised when the annotator app faces initialization errors."""

    def __init__(self, part: str):
        """Initialize the exception."""
        super().__init__(f"Initialization error: No {part} initialized")


class NameValidationError(AnnotationError):
    """Exception raised when annotator name validation fails."""

    def __init__(self):
        """Initialize the exception."""
        super().__init__("Missing annotator name")


class AnnotationValidationError(AnnotationError):
    """Exception raised when annotation validation fails."""

    def __init__(self, field: str, error: Exception):
        """Initialize the exception."""
        super().__init__(f"Error processing annotation for {field}: {error}")


class SaveError(AnnotationError):
    """Exception raised when saving annotations fails."""

    def __init__(
        self,
        message: str,
        filepath: Optional[str] = None,
    ):
        """Initialize the exception."""
        super().__init__(f"{message}: {filepath}")


class PortOccupiedError(AnnotationError):
    """Exception raised when the specified port is already in use."""

    def __init__(self, port: str | int):
        """Initialize the exception."""
        super().__init__(
            f"Port {port} is already in use. Please specify a different port.",
        )
