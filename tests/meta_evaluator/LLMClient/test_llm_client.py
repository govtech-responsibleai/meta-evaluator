"""File for testing the LLMClient module."""

import pytest
import logging
from unittest.mock import MagicMock
from pydantic import ValidationError

from meta_evaluator.LLMClient import LLMClientConfig, LLMClient
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

    def _prompt(self, model: str, messages: list[Message]) -> LLMResponse:
        """Abstract method implementation for testing.

        This method is intended to be mocked in actual tests.

        Args:
            model: The model to use.
            messages: The list of messages.

        Returns:
            A mock LLMResponse.
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
        )

        assert config.api_key == "test-api-key"
        assert config.supports_structured_output is True
        assert config.default_model == "gpt-4"
        assert config.default_embedding_model == "text-embedding-ada-002"

    def test_validation_works_for_missing_fields(self):
        """Test that Pydantic validation is properly configured."""

        class ConcreteConfig(LLMClientConfig):
            pass

            def _prevent_instantiation(self) -> None:
                pass

        with pytest.raises(ValidationError):
            ConcreteConfig()  # type: ignore

        with pytest.raises(ValidationError):
            ConcreteConfig(api_key="test-key")  # type: ignore

    def test_validation_works_for_wrong_types(self):
        """Test that field types are validated correctly."""

        class ConcreteConfig(LLMClientConfig):
            pass

            def _prevent_instantiation(self) -> None:
                pass

        with pytest.raises(ValidationError):  # type: ignore
            ConcreteConfig(
                api_key=123,  # type: ignore
                supports_structured_output=True,
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

        with pytest.raises(ValidationError):  # type: ignore
            ConcreteConfig(
                api_key="test-key",
                supports_structured_output="yes",  # type: ignore
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )


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
    def mock_llm_response(self) -> LLMResponse:
        """Provides a mock LLMResponse object for testing LLMClient implementations.

        The mock object is initialized with a valid provider, model, messages, and
        usage statistics.

        Returns:
            LLMResponse: A mock LLMResponse object.
        """
        return LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="test-model-returned-in-response",
            messages=[
                Message(role=RoleEnum.USER, content="Test input message"),
                Message(
                    role=RoleEnum.ASSISTANT, content="Test response content from mock"
                ),
            ],
            usage=LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
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

    def test_initialization_with_invalid_config_type(self):
        """Test case 2: Verify TypeError is raised with invalid config type."""
        invalid_config = {
            "api_key": "fake",
            "default_model": "gpt-4",
            "supports_structured_output": False,
            "default_embedding_model": "embed-model",
        }
        with pytest.raises(
            TypeError, match="config must be an instance of LLMClientConfig"
        ):
            ConcreteTestLLMClient(invalid_config)  # type: ignore

    def test_prompt_happy_path_default_model(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_llm_response: LLMResponse,
        mocker,
    ):
        """Test case 3: Verify prompt uses default model and logs correctly on success.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mock_llm_response: A pytest fixture providing a mock LLMResponse.
            mock_logger: A pytest fixture providing a mock logger.
            mocker: The pytest-mock mocker fixture.
        """
        mock__prompt = mocker.patch.object(
            concrete_client, "_prompt", return_value=mock_llm_response
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        response = concrete_client.prompt(messages=valid_messages)

        assert response is mock_llm_response
        mock__prompt.assert_called_once_with(
            concrete_client.config.default_model, valid_messages
        )

        mock_logger.info.assert_any_call(
            f"Using model: {concrete_client.config.default_model}"
        )
        mock_logger.info.assert_any_call(f"Input Payload: {valid_messages}")
        mock_logger.info.assert_any_call(
            f"Latest response: {mock_llm_response.latest_response}"
        )
        mock_logger.info.assert_any_call(f"Output usage: {mock_llm_response.usage}")
        assert mock_logger.info.call_count == 4

    def test_prompt_happy_path_explicit_model(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_llm_response: LLMResponse,
        mocker,
    ):
        """Test case 4: Verify prompt uses explicit model and logs correctly on success.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mock_llm_response: A pytest fixture providing a mock LLMResponse.
            mock_logger: A pytest fixture providing a mock logger.
            mocker: The pytest-mock mocker fixture.
        """
        explicit_model = "my-specific-model"

        mock__prompt = mocker.patch.object(
            concrete_client, "_prompt", return_value=mock_llm_response
        )

        mock_logger = mocker.patch.object(concrete_client, "logger")

        response = concrete_client.prompt(messages=valid_messages, model=explicit_model)

        assert response is mock_llm_response
        mock__prompt.assert_called_once_with(explicit_model, valid_messages)

        mock_logger.info.assert_any_call(f"Using model: {explicit_model}")
        mock_logger.info.assert_any_call(f"Input Payload: {valid_messages}")
        mock_logger.info.assert_any_call(
            f"Latest response: {mock_llm_response.latest_response}"
        )
        mock_logger.info.assert_any_call(f"Output usage: {mock_llm_response.usage}")
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

        assert "No messages provided" in str(excinfo.value)
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
            mock_logger: A pytest fixture providing a mock logger.
            mocker: The pytest-mock mocker fixture.
        """
        original_exception = RuntimeError("Simulated API connection failure")

        mock__prompt = mocker.patch.object(
            concrete_client, "_prompt", side_effect=original_exception
        )

        mock_logger = mocker.patch.object(concrete_client, "logger")

        with pytest.raises(LLMAPIError) as excinfo:
            concrete_client.prompt(messages=valid_messages)

        assert f"Failed to get response from {concrete_client.enum_value}" in str(
            excinfo.value
        )
        assert excinfo.value.provider == concrete_client.enum_value

        assert excinfo.value.original_error is original_exception
        mock__prompt.assert_called_once_with(
            concrete_client.config.default_model, valid_messages
        )

        mock_logger.info.assert_any_call(
            f"Using model: {concrete_client.config.default_model}"
        )
        mock_logger.info.assert_any_call(f"Input Payload: {valid_messages}")

    def test_logging_content_verification(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_llm_response: LLMResponse,
        mocker,
    ):
        """Test case 7: Verify the exact content logged by the prompt method.

        This test explicitly checks the arguments passed to the logger's info method.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mock_llm_response: A pytest fixture providing a mock LLMResponse.
            mock_logger: A pytest fixture providing a mock logger.
            mocker: The pytest-mock mocker fixture.
        """
        mocker.patch.object(concrete_client, "_prompt", return_value=mock_llm_response)

        mock_logger = mocker.patch.object(concrete_client, "logger")

        concrete_client.prompt(messages=valid_messages)

        expected_log_calls = [
            mocker.call(f"Using model: {concrete_client.config.default_model}"),
            mocker.call(f"Input Payload: {valid_messages}"),
            mocker.call(f"Latest response: {mock_llm_response.latest_response}"),
            mocker.call(f"Output usage: {mock_llm_response.usage}"),
        ]

        mock_logger.info.assert_has_calls(expected_log_calls)
        assert mock_logger.info.call_count == len(expected_log_calls)
