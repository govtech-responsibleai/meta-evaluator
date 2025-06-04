"""File for all Judge related exceptions."""


class JudgeError(Exception):
    """Base class for Judge exceptions."""

    pass


class IncorrectClientError(JudgeError):
    """Exception raised when the LLMClient is incorrect."""

    def __init__(self, expected_client: str, actual_client: str):
        """Initializes the IncorrectClientError exception with the expected and actual LLMClient strings.

        Args:
            expected_client (str): The expected LLMClient string.
            actual_client (str): The actual LLMClient string.
        """
        super().__init__(
            f"Incorrect LLMClient. Expected: {expected_client}, Actual: {actual_client}"
        )
