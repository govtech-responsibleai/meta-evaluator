"""Custom exceptions for the annotator module."""

from typing import Optional, Any, Dict


class AnnotationError(Exception):
    """Base exception for all annotation-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception with a message and details.

        Args:
            message: The error message.
            details: Additional details about the error.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AnnotatorInitializationError(AnnotationError):
    """Exception raised when the annotator app faces initialization errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception with a message and details.

        Args:
            message: The error message.
            details: Additional details about the error.
        """
        super().__init__(message, details)


class AnnotationValidationError(AnnotationError):
    """Exception raised when annotation validation fails."""

    def __init__(
        self, message: str, field: Optional[str] = None, value: Optional[Any] = None
    ):
        """Initialize the exception with a message and the field that failed validation.

        Args:
            message: The error message.
            field: The field that failed validation.
            value: The value that failed validation.
        """
        details = {"field": field, "value": value}
        super().__init__(message, details)


class SaveError(AnnotationError):
    """Exception raised when saving annotations fails."""

    def __init__(
        self,
        message: str,
        filepath: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        """Initialise the exception with a message and the path to the file that failed to save.

        Args:
            message: The error message.
            filepath: The path to the file that failed to save.
            original_error: The original error that occurred.
        """
        details = {
            "filepath": filepath,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(message, details)
