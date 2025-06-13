"""Custom exceptions for MetaEvaluator."""

from abc import ABC


class MetaEvaluatorException(Exception, ABC):
    """Base exception class for MetaEvaluator errors."""

    def __init__(self, message: str):
        """Initialize the exception with a message.

        Args:
            message: The error message.
        """
        self.message = message
        super().__init__(self.message)


class MissingConfigurationException(MetaEvaluatorException):
    """Exception raised when required configuration is missing."""

    def __init__(self, parameter_name: str):
        """Initialize with missing parameter name.

        Args:
            parameter_name: The name of the missing configuration parameter.
        """
        super().__init__(f"Missing required configuration: {parameter_name}")


class ClientAlreadyExistsException(MetaEvaluatorException):
    """Exception raised when trying to add a client that already exists."""

    def __init__(self, client_type: str):
        """Initialize with client type.

        Args:
            client_type: The type of client that already exists.
        """
        super().__init__(
            f"Client of type {client_type} already exists. Use override_existing=True to replace it."
        )


class ClientNotFoundException(MetaEvaluatorException):
    """Exception raised when trying to get a client that doesn't exist."""

    def __init__(self, client_type: str):
        """Initialize with client type.

        Args:
            client_type: The type of client that was not found.
        """
        super().__init__(f"Client of type {client_type} not found in registry.")


class DataAlreadyExistsException(MetaEvaluatorException):
    """Exception raised when trying to add data that already exists."""

    def __init__(self):
        """Initialize the exception."""
        super().__init__("Data already exists. Use overwrite=True to replace it.")


class DataFilenameExtensionMismatchException(MetaEvaluatorException):
    """Exception raised when data filename extension doesn't match data format."""

    def __init__(self, filename: str, expected_extension: str, data_format: str):
        """Initialize with filename and expected extension.

        Args:
            filename: The provided data filename.
            expected_extension: The expected file extension.
            data_format: The data format that was specified.
        """
        super().__init__(
            f"Data filename '{filename}' must have extension '.{expected_extension}' "
            f"to match data_format '{data_format}'"
        )
