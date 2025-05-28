"""File for testing the LLMClient module."""

import pytest
import logging
from unittest.mock import MagicMock
from pydantic import ValidationError

from meta_evaluator.LLMClient import LLMClientConfig, LLMClient
from meta_evaluator.LLMClient.LLM_client import (
    _FAILED_RESPONSE_ERROR_TEMPLATE,
    _NO_MESSAGES_ERROR,
)
from meta_evaluator.LLMClient.models import (
    Message,
    LLMClientEnum,
    LLMResponse,
    LLMUsage,
    RoleEnum,
)
from meta_evaluator.LLMClient.exceptions import (
    LLMAPIError,
    LLMValidationError,
)


class LLMClientConfigConcreteTest(LLMClientConfig):
    """A concrete configuration class for testing LLMClient."""

    api_key: str = "test-api-key"
    supports_structured_output: bool = False
    default_model: str = "test-default-model"
    default_embedding_model: str = "test-default-embedding-model"

    def _prevent_instantiation(self) -> None:
        pass


class ConcreteTestLLMClient(LLMClient):
    """A concrete LLMClient subclass for testing."""

    @property
    def enum_value(self) -> LLMClientEnum:
        """Return the unique LLMClientEnum value for the test client."""
        return LLMClientEnum.OPENAI

    def _prompt(
        self, model: str, messages: list[Message], get_logprobs: bool
    ) -> tuple[str, LLMUsage]:
        """Abstract method implementation for testing.

        This method is intended to be mocked in actual tests.

        Args:
            model: The model to use.
            messages: The list of messages.
            get_logprobs: Whether to get logprobs.

        Returns:
            A tuple containing a mock response and LLMUsage.
        """
        raise NotImplementedError(
            "ConcreteTestLLMClient._prompt should be mocked for tests"
        )


class TestLLMClientConfig:
    """Test the LLMClientConfig abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that LLMClientConfig cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            LLMClientConfig(
                api_key="test-api-key",
                supports_structured_output=True,
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )  # type: ignore

    def test_concrete_subclass_can_instantiate(self):
        """Test that a concrete subclass can be instantiated with valid data."""

        class ConcreteConfig(LLMClientConfig):
            def _prevent_instantiation(self) -> None:
                pass

        config = ConcreteConfig(
            api_key="test-api-key",
            supports_structured_output=True,
            default_model="gpt-4",
            default_embedding_model="text-embedding-ada-002",
            supports_logprobs=False,
        )

        assert config.api_key == "test-api-key"
        assert config.supports_structured_output is True
        assert config.default_model == "gpt-4"
        assert config.default_embedding_model == "text-embedding-ada-002"

    def test_validation_works_for_missing_fields(self):
        """Test that Pydantic validation is properly configured."""

        class ConcreteConfig(LLMClientConfig):
            def _prevent_instantiation(self) -> None:
                pass

        with pytest.raises(ValidationError):
            ConcreteConfig()  # type: ignore

        with pytest.raises(ValidationError):
            ConcreteConfig(api_key="test-key")  # type: ignore


class TestLLMClient:
    """Test suite for the abstract LLMClient class."""

    @pytest.fixture
    def valid_config(self) -> LLMClientConfigConcreteTest:
        """Provides a valid LLMClientConfigConcreteTest instance.

        The instance is configured with valid values for all fields. This fixture is
        used to provide a valid config to the LLMClient class under test.

        Returns:
            LLMClientConfigConcreteTest: A valid instance of
                LLMClientConfigConcreteTest.
        """
        return LLMClientConfigConcreteTest(
            api_key="test-api-key",
            supports_structured_output=True,
            default_model="gpt-4",
            default_embedding_model="text-embedding-ada-002",
            supports_logprobs=False,
        )

    @pytest.fixture
    def concrete_client(
        self, valid_config: LLMClientConfigConcreteTest
    ) -> ConcreteTestLLMClient:
        """Provides a ConcreteTestLLMClient instance with valid config.

        Args:
            valid_config (LLMClientConfigConcreteTest): A valid configuration instance for the LLM client.

        Returns:
            ConcreteTestLLMClient: An instance of ConcreteTestLLMClient initialized with the provided config.
        """
        return ConcreteTestLLMClient(valid_config)

    @pytest.fixture
    def mock_logger(self, mocker) -> MagicMock:
        """Mocks the logger instance used by LLMClient.

        This fixture patches the logging.getLogger method to return a mock logger
        instance. This allows us to verify the logging calls made by the LLMClient
        class.

        Returns:
            MagicMock: A mock logger instance.
        """
        return mocker.patch("logging.getLogger").return_value

    @pytest.fixture
    def mock_raw_response(self) -> tuple[str, LLMUsage]:
        """Provides a mock raw response and usage statistics for testing LLMClient implementations.

        Returns:
            tuple[str, LLMUsage]: A tuple containing the mock raw response and usage statistics.
        """
        return (
            "Test response content from mock",
            LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    @pytest.fixture
    def valid_messages(self) -> list[Message]:
        """Provides a list of valid Message objects.

        The list contains one message with role USER and content "Hello, LLM!".

        Returns:
            list[Message]: A list of valid Message objects.
        """
        return [Message(role=RoleEnum.USER, content="Hello, LLM!")]

    def test_initialization_with_valid_config(
        self, valid_config: LLMClientConfigConcreteTest
    ):
        """Test case 1: Verify initialization succeeds with a valid config.

        Args:
            valid_config: A pytest fixture providing a valid config.
        """
        client = ConcreteTestLLMClient(valid_config)
        assert isinstance(client, ConcreteTestLLMClient)
        assert client.config is valid_config
        assert isinstance(client.logger, logging.Logger)

    # Test Case 2 removed because BearType takes care of testing issues
    # TODO: Rename test case numbers

    def test_prompt_happy_path_default_model(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_raw_response: tuple[str, LLMUsage],
        mocker,
    ):
        """Test case 3: Verify prompt uses default model and logs correctly on success.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mock_raw_response: A pytest fixture providing a mock raw response tuple.
            mocker: The pytest-mock mocker fixture.
        """
        mock__prompt = mocker.patch.object(
            concrete_client, "_prompt", return_value=mock_raw_response
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        response = concrete_client.prompt(messages=valid_messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Test response content from mock"
        assert response.usage.total_tokens == 30
        assert response.provider == concrete_client.enum_value
        assert response.model == concrete_client.config.default_model
        assert len(response.messages) == 2  # Original message + assistant response
        assert response.messages[0].role == RoleEnum.USER
        assert response.messages[1].role == RoleEnum.ASSISTANT
        assert response.messages[1].content == "Test response content from mock"

        mock__prompt.assert_called_once_with(
            model=concrete_client.config.default_model,
            messages=valid_messages,
            get_logprobs=False,
        )

        mock_logger.info.assert_any_call(
            f"Using model: {concrete_client.config.default_model}"
        )
        mock_logger.info.assert_any_call(f"Input Payload: {valid_messages}")
        mock_logger.info.assert_any_call(f"Latest response: {response.latest_response}")
        mock_logger.info.assert_any_call(f"Output usage: {response.usage}")
        assert mock_logger.info.call_count == 4

    def test_prompt_happy_path_explicit_model(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_raw_response: tuple[str, LLMUsage],
        mocker,
    ):
        """Test case 4: Verify prompt uses explicit model and logs correctly on success.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mock_raw_response: A pytest fixture providing a mock raw response tuple.
            mocker: The pytest-mock mocker fixture.
        """
        explicit_model = "my-specific-model"

        mock__prompt = mocker.patch.object(
            concrete_client, "_prompt", return_value=mock_raw_response
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        response = concrete_client.prompt(messages=valid_messages, model=explicit_model)

        assert isinstance(response, LLMResponse)
        assert response.content == "Test response content from mock"
        assert response.usage.total_tokens == 30
        assert response.provider == concrete_client.enum_value
        assert (
            response.model == concrete_client.config.default_model
        )  # Note: this uses default_model, not the explicit model
        assert len(response.messages) == 2

        mock__prompt.assert_called_once_with(
            model=explicit_model,
            messages=valid_messages,
            get_logprobs=False,
        )

        mock_logger.info.assert_any_call(f"Using model: {explicit_model}")
        mock_logger.info.assert_any_call(f"Input Payload: {valid_messages}")
        mock_logger.info.assert_any_call(f"Latest response: {response.latest_response}")
        mock_logger.info.assert_any_call(f"Output usage: {response.usage}")
        assert mock_logger.info.call_count == 4

    def test_prompt_handles_empty_messages_list(
        self, concrete_client: ConcreteTestLLMClient, mocker
    ):
        """Test case 5: Verify prompt raises LLMValidationError for empty messages list.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            mocker: The pytest-mock mocker fixture.
        """
        mock__prompt = mocker.patch.object(concrete_client, "_prompt")

        with pytest.raises(LLMValidationError) as excinfo:
            concrete_client.prompt(messages=[])

        assert _NO_MESSAGES_ERROR in str(excinfo.value)
        assert excinfo.value.provider == concrete_client.enum_value
        mock__prompt.assert_not_called()

    def test_prompt_handles_exception_from__prompt(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mocker,
    ):
        """Test case 6: Verify prompt wraps exceptions from _prompt in LLMAPIError.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mocker: The pytest-mock mocker fixture.
        """
        original_exception = RuntimeError("Simulated API connection failure")

        mock__prompt = mocker.patch.object(
            concrete_client, "_prompt", side_effect=original_exception
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        with pytest.raises(LLMAPIError) as excinfo:
            concrete_client.prompt(messages=valid_messages)

        assert _FAILED_RESPONSE_ERROR_TEMPLATE.format(
            concrete_client.enum_value
        ) in str(excinfo.value)
        assert excinfo.value.provider == concrete_client.enum_value
        assert excinfo.value.original_error is original_exception

        mock__prompt.assert_called_once_with(
            model=concrete_client.config.default_model,
            messages=valid_messages,
            get_logprobs=False,
        )

        mock_logger.info.assert_any_call(
            f"Using model: {concrete_client.config.default_model}"
        )
        mock_logger.info.assert_any_call(f"Input Payload: {valid_messages}")
        # Should not log output since exception occurred
        assert mock_logger.info.call_count == 2

    def test_logging_content_verification(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_raw_response: tuple[str, LLMUsage],
        mocker,
    ):
        """Test case 7: Verify the exact content logged by the prompt method.

        This test explicitly checks the arguments passed to the logger's info method.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mock_raw_response: A pytest fixture providing a mock raw response tuple.
            mocker: The pytest-mock mocker fixture.
        """
        mocker.patch.object(concrete_client, "_prompt", return_value=mock_raw_response)
        mock_logger = mocker.patch.object(concrete_client, "logger")

        response = concrete_client.prompt(messages=valid_messages)

        expected_log_calls = [
            mocker.call(f"Using model: {concrete_client.config.default_model}"),
            mocker.call(f"Input Payload: {valid_messages}"),
            mocker.call(f"Latest response: {response.latest_response}"),
            mocker.call(f"Output usage: {response.usage}"),
        ]

        mock_logger.info.assert_has_calls(expected_log_calls)
        assert mock_logger.info.call_count == len(expected_log_calls)

    def test_construct_llm_response(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
    ):
        """Test case 8: Verify _construct_llm_response builds response correctly.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
        """
        raw_response = "Test assistant response"
        usage = LLMUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15)

        # Make a copy since the method modifies the original list
        messages_copy = valid_messages.copy()

        response = concrete_client._construct_llm_response(
            raw_response, usage, messages_copy
        )

        assert isinstance(response, LLMResponse)
        assert response.provider == concrete_client.enum_value
        assert response.model == concrete_client.config.default_model
        assert response.usage == usage
        assert len(response.messages) == 2  # Original + assistant message
        assert response.messages[0] == valid_messages[0]  # Original message unchanged
        assert response.messages[1].role == RoleEnum.ASSISTANT
        assert response.messages[1].content == raw_response
        assert response.content == raw_response
        assert response.latest_response.content == raw_response

    def test_validate_messages(self, concrete_client: ConcreteTestLLMClient):
        """Test case 9: Verify _validate_messages works correctly.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
        """
        # Should not raise for non-empty messages
        valid_messages = [Message(role=RoleEnum.USER, content="Hello")]
        concrete_client._validate_messages(valid_messages)  # Should not raise

        # Should raise for empty messages
        with pytest.raises(LLMValidationError) as excinfo:
            concrete_client._validate_messages([])

        assert _NO_MESSAGES_ERROR in str(excinfo.value)
        assert excinfo.value.provider == concrete_client.enum_value
