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


class EvalTaskAlreadyExistsException(MetaEvaluatorException):
    """Exception raised when trying to add evaluation task that already exists."""

    def __init__(self):
        """Initialize the exception."""
        super().__init__(
            "Evaluation task already exists. Use overwrite=True to replace it."
        )


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


class JudgeAlreadyExistsException(MetaEvaluatorException):
    """Exception raised when trying to add a judge that already exists."""

    def __init__(self, judge_id: str):
        """Initialize with judge ID.

        Args:
            judge_id: The ID of the judge that already exists.
        """
        super().__init__(
            f"Judge with ID '{judge_id}' already exists. Use override_existing=True to replace it."
        )


class JudgeNotFoundException(MetaEvaluatorException):
    """Exception raised when trying to get a judge that doesn't exist."""

    def __init__(self, judge_id: str):
        """Initialize with judge ID.

        Args:
            judge_id: The ID of the judge that was not found.
        """
        super().__init__(f"Judge with ID '{judge_id}' not found in registry.")


class InvalidYAMLStructureException(MetaEvaluatorException):
    """Exception raised when YAML structure is invalid."""

    def __init__(self, details: str):
        """Initialize with validation details.

        Args:
            details: Details about what's wrong with the YAML structure.
        """
        super().__init__(f"Invalid YAML structure: {details}")


class PromptFileNotFoundException(MetaEvaluatorException):
    """Exception raised when a prompt file referenced in YAML cannot be found."""

    def __init__(self, file_path: str):
        """Initialize with file path.

        Args:
            file_path: The path to the prompt file that was not found.
        """
        super().__init__(f"Prompt file not found: {file_path}")


class EvalTaskNotSetException(MetaEvaluatorException):
    """Exception raised when trying to run judges without an evaluation task."""

    def __init__(self):
        """Initialize the exception."""
        super().__init__("eval_task must be set before running judges")


class EvalDataNotSetException(MetaEvaluatorException):
    """Exception raised when trying to run judges without evaluation data."""

    def __init__(self):
        """Initialize the exception."""
        super().__init__("data must be set before running judges")


class NoJudgesAvailableException(MetaEvaluatorException):
    """Exception raised when no judges are available to run."""

    def __init__(self):
        """Initialize the exception."""
        super().__init__("No judges available to run")


class LLMClientNotConfiguredException(MetaEvaluatorException):
    """Exception raised when required LLM client is not configured."""

    def __init__(self, judge_id: str, required_client: str):
        """Initialize with judge ID and required client type.

        Args:
            judge_id: The ID of the judge requiring the client.
            required_client: The type of LLM client that is required.
        """
        super().__init__(
            f"No LLM client configured for judge '{judge_id}' "
            f"(requires {required_client})"
        )


class JudgeExecutionException(MetaEvaluatorException):
    """Exception raised when judge execution fails."""

    def __init__(self, judge_id: str, error: str):
        """Initialize with judge ID and error details.

        Args:
            judge_id: The ID of the judge that failed.
            error: Details about the execution error.
        """
        super().__init__(f"Judge '{judge_id}' execution failed: {error}")


class ResultsSaveException(MetaEvaluatorException):
    """Exception raised when saving judge results fails."""

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
