"""File for all Judge related exceptions."""


class JudgeError(Exception):
    """Base class for Judge exceptions."""

    pass


class IncorrectClientError(JudgeError):
    """Exception raised when the LLMClient is incorrect."""

    def __init__(self, expected_client, actual_client):
        """Initializes the IncorrectClientError exception with the expected and actual LLMClient enums.

        Args:
            expected_client: The expected LLMClientEnum or AsyncLLMClientEnum.
            actual_client: The actual LLMClientEnum or AsyncLLMClientEnum.
        """
        expected_type = type(expected_client).__name__
        actual_type = type(actual_client).__name__
        expected_value = expected_client.value
        actual_value = actual_client.value

        super().__init__(
            f"Incorrect LLMClient. Expected: {expected_type}.{expected_value}, Actual: {actual_type}.{actual_value}"
        )
