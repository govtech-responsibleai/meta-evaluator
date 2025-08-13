"""Custom exceptions for MetaEvaluator organized by category."""

from abc import ABC


# Base exception class
class MetaEvaluatorError(Exception, ABC):
    """Base exception class for MetaEvaluator errors."""

    def __init__(self, message: str):
        """Initialize the exception with a message.

        Args:
            message: The error message.
        """
        self.message = message
        super().__init__(self.message)


# Configuration-related errors
class MetaEvaluatorConfigurationError(MetaEvaluatorError):
    """Base class for configuration-related errors."""

    pass


class MissingConfigurationError(MetaEvaluatorConfigurationError):
    """Error raised when required configuration is missing."""

    def __init__(self, parameter_name: str):
        """Initialize with missing parameter name.

        Args:
            parameter_name: The name of the missing configuration parameter.
        """
        super().__init__(f"Missing required configuration: {parameter_name}")


class DataFilenameExtensionMismatchError(MetaEvaluatorConfigurationError):
    """Error raised when data filename extension doesn't match data format."""

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


class InvalidYAMLStructureError(MetaEvaluatorConfigurationError):
    """Error raised when YAML structure is invalid."""

    def __init__(self, details: str):
        """Initialize with validation details.

        Args:
            details: Details about what's wrong with the YAML structure.
        """
        super().__init__(f"Invalid YAML structure: {details}")


class PromptFileNotFoundError(MetaEvaluatorConfigurationError):
    """Error raised when a prompt file referenced in YAML cannot be found."""

    def __init__(self, file_path: str):
        """Initialize with file path.

        Args:
            file_path: The path to the prompt file that was not found.
        """
        super().__init__(f"Prompt file not found: {file_path}")


class DataAlreadyExistsError(MetaEvaluatorConfigurationError):
    """Error raised when trying to add data that already exists."""

    def __init__(self):
        """Initialize the exception."""
        super().__init__("Data already exists. Use overwrite=True to replace it.")


class EvalTaskAlreadyExistsError(MetaEvaluatorConfigurationError):
    """Error raised when trying to add evaluation task that already exists."""

    def __init__(self):
        """Initialize the exception."""
        super().__init__(
            "Evaluation task already exists. Use overwrite=True to replace it."
        )


class EvalDataNotFoundError(MetaEvaluatorConfigurationError):
    """Error raised when evaluation data is not found."""

    def __init__(self, message: str):
        """Initialize with message.

        Args:
            message: The error message.
        """
        super().__init__(f"Evaluation data not found. {message}")


class EvalTaskNotFoundError(MetaEvaluatorConfigurationError):
    """Error raised when evaluation task is not found."""

    def __init__(self, message: str):
        """Initialize with message.

        Args:
            message: The error message.
        """
        super().__init__(f"Evaluation task not found. {message}")


class DataFormatError(MetaEvaluatorConfigurationError):
    """Error raised when data format is invalid or incompatible."""

    def __init__(self, message: str):
        """Initialize with error message.

        Args:
            message: The error message describing the data format issue.
        """
        super().__init__(f"Data format error: {message}")


class InvalidFileError(MetaEvaluatorConfigurationError):
    """Error raised when a file is invalid or corrupted."""

    def __init__(self, file_path: str, reason: str = ""):
        """Initialize with file path and optional reason.

        Args:
            file_path: The path to the invalid file.
            reason: Optional reason why the file is invalid.
        """
        message = f"Invalid file: {file_path}"
        if reason:
            message += f" - {reason}"
        super().__init__(message)


class InsufficientDataError(MetaEvaluatorConfigurationError):
    """Error raised when there is insufficient data for an operation."""

    def __init__(self, message: str):
        """Initialize with error message.

        Args:
            message: The error message describing the insufficient data issue.
        """
        super().__init__(f"Insufficient data: {message}")


class IncompatibleTaskError(MetaEvaluatorConfigurationError):
    """Error raised when a task is incompatible with the current configuration."""

    def __init__(self, message: str):
        """Initialize with error message.

        Args:
            message: The error message describing the incompatibility.
        """
        super().__init__(f"Incompatible task: {message}")


# Client-related errors
class MetaEvaluatorClientError(MetaEvaluatorError):
    """Base class for client-related errors."""

    pass


class ClientAlreadyExistsError(MetaEvaluatorClientError):
    """Error raised when trying to add a client that already exists."""

    def __init__(self, client_type: str):
        """Initialize with client type.

        Args:
            client_type: The type of client that already exists.
        """
        super().__init__(
            f"Client of type {client_type} already exists. Use override_existing=True to replace it."
        )


class ClientNotFoundError(MetaEvaluatorClientError):
    """Error raised when trying to get a client that doesn't exist."""

    def __init__(self, client_type: str, client_method: str = ""):
        """Initialize with client type.

        Args:
            client_type: The type of client that was not found.
            client_method: Optional method name suggestion for configuring the client.
        """
        message = f"Client of type {client_type} not found in registry."
        if client_method:
            message += f" Use {client_method} to configure it."
        super().__init__(message)


class AsyncClientAlreadyExistsError(MetaEvaluatorClientError):
    """Error raised when trying to add an async client that already exists."""

    def __init__(self, client_type: str):
        """Initialize with async client type.

        Args:
            client_type: The type of async client that already exists.
        """
        super().__init__(
            f"Async client of type {client_type} already exists. Use override_existing=True to replace it."
        )


class AsyncClientNotFoundError(MetaEvaluatorClientError):
    """Error raised when trying to get an async client that doesn't exist."""

    def __init__(self, client_type: str, client_method: str = ""):
        """Initialize with client type.

        Args:
            client_type: The type of client that was not found.
            client_method: Optional method name suggestion for configuring the client.
        """
        message = f"Async client of type {client_type} not found in registry."
        if client_method:
            message += f" Use {client_method} to configure it."
        super().__init__(message)


# Judge-related errors
class MetaEvaluatorJudgeError(MetaEvaluatorError):
    """Base class for judge-related errors."""

    pass


class JudgeAlreadyExistsError(MetaEvaluatorJudgeError):
    """Error raised when trying to add a judge that already exists."""

    def __init__(self, judge_id: str):
        """Initialize with judge ID.

        Args:
            judge_id: The ID of the judge that already exists.
        """
        super().__init__(
            f"Judge with ID '{judge_id}' already exists. Use override_existing=True to replace it."
        )


class JudgeNotFoundError(MetaEvaluatorJudgeError):
    """Error raised when trying to get a judge that doesn't exist."""

    def __init__(self, judge_id: str):
        """Initialize with judge ID.

        Args:
            judge_id: The ID of the judge that was not found.
        """
        super().__init__(f"Judge with ID '{judge_id}' not found in registry.")


class JudgeExecutionError(MetaEvaluatorJudgeError):
    """Error raised when judge execution fails."""

    def __init__(self, judge_id: str, error: str):
        """Initialize with judge ID and error details.

        Args:
            judge_id: The ID of the judge that failed.
            error: Details about the execution error.
        """
        super().__init__(f"Judge '{judge_id}' execution failed: {error}")


class ResultsSaveError(MetaEvaluatorJudgeError):
    """Error raised when saving judge results fails."""

    def __init__(self, judge_id: str, run_id: str, error: str):
        """Initialize with judge ID, run ID, and error details.

        Args:
            judge_id: The ID of the judge whose results failed to save.
            run_id: The run ID of the evaluation.
            error: Details about the save error.
        """
        super().__init__(
            f"Failed to save results for judge '{judge_id}' in run '{run_id}': {error}"
        )


# Scoring-related errors
class MetaEvaluatorScoringError(MetaEvaluatorError):
    """Base class for scoring-related errors."""

    pass


class ScoringConfigError(MetaEvaluatorScoringError):
    """Error raised when scoring configuration is invalid."""

    def __init__(self, message: str):
        """Initialize with error message.

        Args:
            message: The error message describing the configuration issue.
        """
        super().__init__(f"Scoring configuration error: {message}")
