"""Custom exceptions for the scores module."""

from abc import ABC


class ScoringError(Exception, ABC):
    """Base exception for all scoring-related errors."""

    def __init__(self, message: str):
        """Initialize the exception with a message.

        Args:
            message: The error message.
        """
        self.message = message
        super().__init__(self.message)


class AltTestInvalidScoringFunctionError(ScoringError):
    """Exception raised for Alt-Test invalid scoring function."""

    def __init__(self, message: str):
        """Initialize the exception.

        Args:
            message: The error message.
        """
        super().__init__(message)


class AltTestInsufficientAnnotationsError(ScoringError):
    """Exception raised for Alt-Test insufficient annotations."""

    def __init__(self, message: str):
        """Initialize the exception.

        Args:
            message: The error message.
        """
        super().__init__(message)


class InvalidAggregationModeError(ScoringError):
    """Exception raised for invalid aggregation mode configuration."""

    def __init__(self, message: str):
        """Initialize the exception.

        Args:
            message: The error message.
        """
        super().__init__(message)


class MultiLabelScoringError(ScoringError):
    """Exception raised for invalid multi-label scoring configuration.

    Covers an unsupported ``average`` for a multi-label ClassificationScorer,
    a foreign slot value found during positional binarize, and Cohen's kappa
    applied to a multi-label (vector) task.
    """

    def __init__(self, message: str):
        """Initialize the exception.

        Args:
            message: The error message.
        """
        super().__init__(message)
