"""File for testing the LLMClient module."""

import logging
from typing import TypeVar

import pytest
from pydantic import BaseModel, ValidationError

from meta_evaluator.llm_client import LLMClientConfig
from meta_evaluator.llm_client.client import (
    _FAILED_RESPONSE_ERROR_TEMPLATE,
    _NO_MESSAGES_ERROR,
)
from meta_evaluator.llm_client.exceptions import (
    LLMAPIError,
    LLMValidationError,
)
from meta_evaluator.llm_client.models import (
    LLMResponse,
    LLMUsage,
    Message,
    RoleEnum,
)
from meta_evaluator.llm_client.serialization import (
    LLMClientSerializedState,
    MockLLMClientSerializedState,
)

# Import test classes from conftest for type hints
from .conftest import ConcreteTestLLMClient, LLMClientConfigConcreteTest

T = TypeVar("T", bound=BaseModel)


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

            def serialize(self) -> MockLLMClientSerializedState:
                return MockLLMClientSerializedState(
                    default_model=self.default_model,
                    default_embedding_model=self.default_embedding_model,
                    supports_structured_output=self.supports_structured_output,
                    supports_logprobs=self.supports_logprobs,
                    supports_instructor=False,
                )

            @classmethod
            def deserialize(
                cls,
                state: LLMClientSerializedState,
                api_key: str,
                **kwargs,
            ) -> "ConcreteConfig":
                if not isinstance(state, MockLLMClientSerializedState):
                    raise TypeError(
                        f"Expected MockLLMClientSerializedState, got {type(state).__name__}"
                    )

                return cls(
                    api_key=api_key,
                    supports_structured_output=True,
                    default_model="test",
                    default_embedding_model="test",
                    supports_logprobs=False,
                )

        config = ConcreteConfig(  # type: ignore
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

            def serialize(self) -> MockLLMClientSerializedState:
                return MockLLMClientSerializedState(
                    default_model=self.default_model,
                    default_embedding_model=self.default_embedding_model,
                    supports_structured_output=self.supports_structured_output,
                    supports_logprobs=self.supports_logprobs,
                    supports_instructor=False,
                )

            @classmethod
            def deserialize(
                cls,
                state: LLMClientSerializedState,
                api_key: str,
                **kwargs,
            ) -> "ConcreteConfig":
                if not isinstance(state, MockLLMClientSerializedState):
                    raise TypeError(
                        f"Expected MockLLMClientSerializedState, got {type(state).__name__}"
                    )

                return cls(
                    api_key=api_key,
                    supports_structured_output=True,
                    default_model="test",
                    default_embedding_model="test",
                    supports_logprobs=False,
                )

        with pytest.raises(ValidationError):
            ConcreteConfig()  # type: ignore

        with pytest.raises(ValidationError):
            ConcreteConfig(api_key="test-key")  # type: ignore


class TestLLMClient:
    """Test suite for the abstract LLMClient class."""

    def test_initialization_with_valid_config(
        self, structured_output_config: LLMClientConfigConcreteTest
    ):
        """Verify initialization succeeds with a valid config.

        Args:
            structured_output_config: A pytest fixture providing a valid config.
        """
        client = ConcreteTestLLMClient(structured_output_config)
        assert isinstance(client, ConcreteTestLLMClient)
        assert client.config is structured_output_config
        assert isinstance(client.logger, logging.Logger)

    # Test Case 2 removed because BearType takes care of testing issues
    # TODO: Rename test case numbers

    def test_prompt_happy_path_default_model(
        self,
        structured_output_client: ConcreteTestLLMClient,
        simple_user_message: list[Message],
        mock_raw_response: tuple[str, LLMUsage],
        mocker,
    ):
        """Verify prompt uses default model and logs correctly on success.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            simple_user_message: A pytest fixture providing valid messages.
            mock_raw_response: A pytest fixture providing a mock raw response tuple.
            mocker: The pytest-mock mocker fixture.
        """
        mock__prompt = mocker.patch.object(
            structured_output_client, "_prompt", return_value=mock_raw_response
        )
        mock_logger = mocker.patch.object(structured_output_client, "logger")

        response = structured_output_client.prompt(messages=simple_user_message)

        assert isinstance(response, LLMResponse)
        assert response.content == "Test response content from mock"
        assert response.usage.total_tokens == 40  # prompt_tokens + completion_tokens
        assert response.provider == structured_output_client.enum_value
        assert response.model == structured_output_client.config.default_model
        assert len(response.messages) == 2  # Original message + assistant response
        assert response.messages[0].role == RoleEnum.USER
        assert response.messages[1].role == RoleEnum.ASSISTANT
        assert response.messages[1].content == "Test response content from mock"

        mock__prompt.assert_called_once_with(
            model=structured_output_client.config.default_model,
            messages=simple_user_message,
            get_logprobs=False,
        )

        mock_logger.info.assert_any_call(
            f"Using model: {structured_output_client.config.default_model}"
        )
        mock_logger.info.assert_any_call(f"Input Payload: {simple_user_message}")
        mock_logger.info.assert_any_call(f"Latest response: {response.latest_response}")
        mock_logger.info.assert_any_call(f"Output usage: {response.usage}")
        assert mock_logger.info.call_count == 4

    def test_prompt_happy_path_explicit_model(
        self,
        structured_output_client: ConcreteTestLLMClient,
        simple_user_message: list[Message],
        mock_raw_response: tuple[str, LLMUsage],
        mocker,
    ):
        """Verify prompt uses explicit model and logs correctly on success.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            simple_user_message: A pytest fixture providing valid messages.
            mock_raw_response: A pytest fixture providing a mock raw response tuple.
            mocker: The pytest-mock mocker fixture.
        """
        explicit_model = "my-specific-model"

        mock__prompt = mocker.patch.object(
            structured_output_client, "_prompt", return_value=mock_raw_response
        )
        mock_logger = mocker.patch.object(structured_output_client, "logger")

        response = structured_output_client.prompt(
            messages=simple_user_message, model=explicit_model
        )

        assert isinstance(response, LLMResponse)
        assert response.content == "Test response content from mock"
        assert response.usage.total_tokens == 40
        assert response.provider == structured_output_client.enum_value
        assert response.model == explicit_model
        assert len(response.messages) == 2

        mock__prompt.assert_called_once_with(
            model=explicit_model,
            messages=simple_user_message,
            get_logprobs=False,
        )

        mock_logger.info.assert_any_call(f"Using model: {explicit_model}")
        mock_logger.info.assert_any_call(f"Input Payload: {simple_user_message}")
        mock_logger.info.assert_any_call(f"Latest response: {response.latest_response}")
        mock_logger.info.assert_any_call(f"Output usage: {response.usage}")
        assert mock_logger.info.call_count == 4

    def test_prompt_handles_empty_messages_list(
        self, structured_output_client: ConcreteTestLLMClient, mocker
    ):
        """Verify prompt raises LLMValidationError for empty messages list.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            mocker: The pytest-mock mocker fixture.
        """
        mock__prompt = mocker.patch.object(structured_output_client, "_prompt")

        with pytest.raises(LLMValidationError) as excinfo:
            structured_output_client.prompt(messages=[])

        assert _NO_MESSAGES_ERROR in str(excinfo.value)
        assert excinfo.value.provider == structured_output_client.enum_value
        mock__prompt.assert_not_called()

    def test_prompt_handles_exception_from__prompt(
        self,
        structured_output_client: ConcreteTestLLMClient,
        simple_user_message: list[Message],
        mocker,
    ):
        """Verify prompt wraps exceptions from _prompt in LLMAPIError.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            simple_user_message: A pytest fixture providing valid messages.
            mocker: The pytest-mock mocker fixture.
        """
        original_exception = RuntimeError("Simulated API connection failure")

        mock__prompt = mocker.patch.object(
            structured_output_client, "_prompt", side_effect=original_exception
        )
        mock_logger = mocker.patch.object(structured_output_client, "logger")

        with pytest.raises(LLMAPIError) as excinfo:
            structured_output_client.prompt(messages=simple_user_message)

        assert _FAILED_RESPONSE_ERROR_TEMPLATE.format(
            structured_output_client.enum_value
        ) in str(excinfo.value)
        assert excinfo.value.provider == structured_output_client.enum_value
        assert excinfo.value.original_error is original_exception

        mock__prompt.assert_called_once_with(
            model=structured_output_client.config.default_model,
            messages=simple_user_message,
            get_logprobs=False,
        )

        mock_logger.info.assert_any_call(
            f"Using model: {structured_output_client.config.default_model}"
        )
        mock_logger.info.assert_any_call(f"Input Payload: {simple_user_message}")
        # Should not log output since exception occurred
        assert mock_logger.info.call_count == 2

    def test_logging_content_verification(
        self,
        structured_output_client: ConcreteTestLLMClient,
        simple_user_message: list[Message],
        mock_raw_response: tuple[str, LLMUsage],
        mocker,
    ):
        """Verify the exact content logged by the prompt method.

        This test explicitly checks the arguments passed to the logger's info method.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            simple_user_message: A pytest fixture providing valid messages.
            mock_raw_response: A pytest fixture providing a mock raw response tuple.
            mocker: The pytest-mock mocker fixture.
        """
        mocker.patch.object(
            structured_output_client, "_prompt", return_value=mock_raw_response
        )
        mock_logger = mocker.patch.object(structured_output_client, "logger")

        response = structured_output_client.prompt(messages=simple_user_message)

        expected_log_calls = [
            mocker.call(
                f"Using model: {structured_output_client.config.default_model}"
            ),
            mocker.call(f"Input Payload: {simple_user_message}"),
            mocker.call(f"Latest response: {response.latest_response}"),
            mocker.call(f"Output usage: {response.usage}"),
        ]

        mock_logger.info.assert_has_calls(expected_log_calls)
        assert mock_logger.info.call_count == len(expected_log_calls)

    def test_construct_llm_response(
        self,
        structured_output_client: ConcreteTestLLMClient,
        simple_user_message: list[Message],
    ):
        """Verify _construct_llm_response builds response correctly.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            simple_user_message: A pytest fixture providing valid messages.
        """
        raw_response = "Test assistant response"
        usage = LLMUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15)

        # Make a copy since the method modifies the original list
        messages_copy = simple_user_message.copy()

        response = structured_output_client._construct_llm_response(
            raw_response,
            usage,
            messages_copy,
            structured_output_client.config.default_model,
        )

        assert isinstance(response, LLMResponse)
        assert response.provider == structured_output_client.enum_value
        assert response.model == structured_output_client.config.default_model
        assert response.usage == usage
        assert len(response.messages) == 2  # Original + assistant message
        assert (
            response.messages[0] == simple_user_message[0]
        )  # Original message unchanged
        assert response.messages[1].role == RoleEnum.ASSISTANT
        assert response.messages[1].content == raw_response
        assert response.content == raw_response
        assert response.latest_response.content == raw_response

    def test_validate_messages(self, structured_output_client: ConcreteTestLLMClient):
        """Verify _validate_messages works correctly.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
        """
        # Should not raise for non-empty messages
        valid_messages = [Message(role=RoleEnum.USER, content="Hello")]
        structured_output_client._validate_messages(valid_messages)  # Should not raise

        # Should raise for empty messages
        with pytest.raises(LLMValidationError) as excinfo:
            structured_output_client._validate_messages([])

        assert _NO_MESSAGES_ERROR in str(excinfo.value)
        assert excinfo.value.provider == structured_output_client.enum_value
