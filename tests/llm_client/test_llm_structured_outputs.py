"""File for testing the structured output functionality in LLMClient module."""

import pytest
from pydantic import BaseModel

from meta_evaluator.llm_client.exceptions import (
    LLMAPIError,
    LLMValidationError,
)
from meta_evaluator.llm_client.LLM_client import (
    _FAILED_RESPONSE_ERROR_TEMPLATE,
    _LOGPROBS_NOT_SUPPORTED_ERROR_TEMPLATE,
    _NO_MESSAGES_ERROR,
)
from meta_evaluator.llm_client.models import (
    LLMResponse,
    LLMUsage,
    Message,
    RoleEnum,
)

# Import test classes from conftest for type hints
from .conftest import (
    ConcreteTestLLMClient,
    ExampleResponseModel,
)

# Classes imported from conftest above


class TestStructuredOutput:
    """Test suite for structured output functionality."""

    # ===== _prompt_with_structured_response Tests (Abstract Method) =====

    def test_prompt_with_structured_response_abstract_method_raises_not_implemented(
        self,
        structured_output_client: ConcreteTestLLMClient,
        structured_output_messages: list[Message],
    ):
        """Test that the abstract _prompt_with_structured_response method raises NotImplementedError.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            structured_output_messages: A pytest fixture providing valid messages.
        """
        with pytest.raises(NotImplementedError) as excinfo:
            structured_output_client._prompt_with_structured_response(
                messages=structured_output_messages,
                response_model=ExampleResponseModel,
                model="test-model",
            )

        assert "should be mocked for tests" in str(excinfo.value)

    # ===== _construct_llm_response_with_structured_response Tests =====

    def test_construct_llm_response_with_structured_response_basic(
        self,
        structured_output_client: ConcreteTestLLMClient,
        simple_user_message: list[Message],
        mock_structured_response: ExampleResponseModel,
        mock_usage: LLMUsage,
    ):
        """Test _construct_llm_response_with_structured_response builds response correctly.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            simple_user_message: A pytest fixture providing valid messages.
            mock_structured_response: A pytest fixture providing a mock structured response.
            mock_usage: A pytest fixture providing mock usage statistics.
        """
        # Make a copy since the method modifies the original list
        messages_copy = simple_user_message.copy()

        response = (
            structured_output_client._construct_llm_response_with_structured_response(
                mock_structured_response, mock_usage, messages_copy
            )
        )

        assert isinstance(response, LLMResponse)
        assert response.provider == structured_output_client.enum_value
        assert response.model == structured_output_client.config.default_model
        assert response.usage == mock_usage
        assert len(response.messages) == 2  # Original + assistant message
        assert (
            response.messages[0] == simple_user_message[0]
        )  # Original message unchanged
        assert response.messages[1].role == RoleEnum.ASSISTANT

        # Verify the content is JSON serialized
        expected_json = mock_structured_response.model_dump_json()
        assert response.messages[1].content == expected_json
        assert response.content == expected_json
        assert response.latest_response.content == expected_json

    def test_construct_llm_response_with_structured_response_json_serialization(
        self,
        structured_output_client: ConcreteTestLLMClient,
        structured_output_messages: list[Message],
        mock_usage: LLMUsage,
    ):
        """Test that structured response is properly JSON serialized.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            structured_output_messages: A pytest fixture providing valid messages.
            mock_usage: A pytest fixture providing mock usage statistics.
        """
        # Create a specific structured response to verify JSON content
        structured_response = ExampleResponseModel(
            task_id="json_test_123",
            status="active",
            confidence=0.87,
            tags=["test", "json"],
        )

        response = (
            structured_output_client._construct_llm_response_with_structured_response(
                structured_response, mock_usage, structured_output_messages.copy()
            )
        )

        # Parse the JSON to verify it's valid and contains expected data
        import json

        parsed_content = json.loads(response.content)

        assert parsed_content["task_id"] == "json_test_123"
        assert parsed_content["status"] == "active"
        assert parsed_content["confidence"] == 0.87
        assert parsed_content["tags"] == ["test", "json"]

    # ===== prompt_with_structured_response Tests =====

    def test_prompt_with_structured_response_happy_path_default_model(
        self,
        structured_output_client: ConcreteTestLLMClient,
        structured_output_messages: list[Message],
        mock_structured_response: ExampleResponseModel,
        mock_usage: LLMUsage,
        mocker,
    ):
        """Test prompt_with_structured_response uses default model and logs correctly on success.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            structured_output_messages: A pytest fixture providing valid messages.
            mock_structured_response: A pytest fixture providing a mock structured response.
            mock_usage: A pytest fixture providing mock usage statistics.
            mocker: The pytest-mock mocker fixture.
        """
        mock_prompt_structured = mocker.patch.object(
            structured_output_client,
            "_prompt_with_structured_response",
            return_value=(mock_structured_response, mock_usage),
        )
        mock_logger = mocker.patch.object(structured_output_client, "logger")

        structured_response, llm_response = (
            structured_output_client.prompt_with_structured_response(
                messages=structured_output_messages, response_model=ExampleResponseModel
            )
        )

        # Verify returned structured response
        assert isinstance(structured_response, ExampleResponseModel)
        assert structured_response.task_id == "task_123"
        assert structured_response.status == "completed"
        assert structured_response.confidence == 0.95
        assert structured_response.tags == ["important", "urgent"]

        # Verify LLM response
        assert isinstance(llm_response, LLMResponse)
        assert llm_response.usage == mock_usage
        assert llm_response.provider == structured_output_client.enum_value
        assert llm_response.model == structured_output_client.config.default_model
        assert len(llm_response.messages) == 2
        assert llm_response.messages[0].role == RoleEnum.USER
        assert llm_response.messages[1].role == RoleEnum.ASSISTANT

        # Verify content is JSON serialized
        expected_json = mock_structured_response.model_dump_json()
        assert llm_response.content == expected_json

        # Verify method calls
        mock_prompt_structured.assert_called_once_with(
            messages=structured_output_messages,
            response_model=ExampleResponseModel,
            model=structured_output_client.config.default_model,
        )

        # Verify logging
        mock_logger.info.assert_any_call(
            f"Using model: {structured_output_client.config.default_model}"
        )
        mock_logger.info.assert_any_call(f"Input Payload: {structured_output_messages}")
        mock_logger.info.assert_any_call(
            f"Latest response: {llm_response.latest_response}"
        )
        mock_logger.info.assert_any_call(f"Output usage: {llm_response.usage}")
        assert mock_logger.info.call_count == 4

    def test_prompt_with_structured_response_happy_path_explicit_model(
        self,
        structured_output_client: ConcreteTestLLMClient,
        structured_output_messages: list[Message],
        mock_structured_response: ExampleResponseModel,
        mock_usage: LLMUsage,
        mocker,
    ):
        """Test prompt_with_structured_response uses explicit model and logs correctly on success.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            structured_output_messages: A pytest fixture providing valid messages.
            mock_structured_response: A pytest fixture providing a mock structured response.
            mock_usage: A pytest fixture providing mock usage statistics.
            mocker: The pytest-mock mocker fixture.
        """
        explicit_model = "my-specific-structured-model"

        mock_prompt_structured = mocker.patch.object(
            structured_output_client,
            "_prompt_with_structured_response",
            return_value=(mock_structured_response, mock_usage),
        )
        mock_logger = mocker.patch.object(structured_output_client, "logger")

        structured_response, llm_response = (
            structured_output_client.prompt_with_structured_response(
                messages=structured_output_messages,
                response_model=ExampleResponseModel,
                model=explicit_model,
            )
        )

        # Verify returns
        assert isinstance(structured_response, ExampleResponseModel)
        assert isinstance(llm_response, LLMResponse)
        assert llm_response.usage == mock_usage
        assert (
            llm_response.model == structured_output_client.config.default_model
        )  # Uses default_model in response

        # Verify method calls with explicit model
        mock_prompt_structured.assert_called_once_with(
            messages=structured_output_messages,
            response_model=ExampleResponseModel,
            model=explicit_model,
        )

        # Verify logging shows explicit model
        mock_logger.info.assert_any_call(f"Using model: {explicit_model}")
        mock_logger.info.assert_any_call(f"Input Payload: {structured_output_messages}")
        mock_logger.info.assert_any_call(
            f"Latest response: {llm_response.latest_response}"
        )
        mock_logger.info.assert_any_call(f"Output usage: {llm_response.usage}")
        assert mock_logger.info.call_count == 4

    def test_prompt_with_structured_response_handles_empty_messages_list(
        self, structured_output_client: ConcreteTestLLMClient, mocker
    ):
        """Test prompt_with_structured_response raises LLMValidationError for empty messages list.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            mocker: The pytest-mock mocker fixture.
        """
        mock_prompt_structured = mocker.patch.object(
            structured_output_client, "_prompt_with_structured_response"
        )

        with pytest.raises(LLMValidationError) as excinfo:
            structured_output_client.prompt_with_structured_response(
                messages=[], response_model=ExampleResponseModel
            )

        assert _NO_MESSAGES_ERROR in str(excinfo.value)
        assert excinfo.value.provider == structured_output_client.enum_value
        mock_prompt_structured.assert_not_called()

    def test_prompt_with_structured_response_handles_exception_from_prompt_structured(
        self,
        structured_output_client: ConcreteTestLLMClient,
        structured_output_messages: list[Message],
        mocker,
    ):
        """Test prompt_with_structured_response wraps exceptions from _prompt_with_structured_response in LLMAPIError.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            structured_output_messages: A pytest fixture providing valid messages.
            mocker: The pytest-mock mocker fixture.
        """
        original_exception = RuntimeError("Simulated structured API connection failure")

        mock_prompt_structured = mocker.patch.object(
            structured_output_client,
            "_prompt_with_structured_response",
            side_effect=original_exception,
        )
        mock_logger = mocker.patch.object(structured_output_client, "logger")

        with pytest.raises(LLMAPIError) as excinfo:
            structured_output_client.prompt_with_structured_response(
                messages=structured_output_messages, response_model=ExampleResponseModel
            )

        assert _FAILED_RESPONSE_ERROR_TEMPLATE.format(
            structured_output_client.enum_value
        ) in str(excinfo.value)
        assert excinfo.value.provider == structured_output_client.enum_value
        assert excinfo.value.original_error is original_exception

        mock_prompt_structured.assert_called_once_with(
            messages=structured_output_messages,
            response_model=ExampleResponseModel,
            model=structured_output_client.config.default_model,
        )

        # Should log input but not output since exception occurred
        mock_logger.info.assert_any_call(
            f"Using model: {structured_output_client.config.default_model}"
        )
        mock_logger.info.assert_any_call(f"Input Payload: {structured_output_messages}")
        assert mock_logger.info.call_count == 2

    def test_prompt_with_structured_response_logging_content_verification(
        self,
        structured_output_client: ConcreteTestLLMClient,
        structured_output_messages: list[Message],
        mock_structured_response: ExampleResponseModel,
        mock_usage: LLMUsage,
        mocker,
    ):
        """Test the exact content logged by prompt_with_structured_response method.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            structured_output_messages: A pytest fixture providing valid messages.
            mock_structured_response: A pytest fixture providing a mock structured response.
            mock_usage: A pytest fixture providing mock usage statistics.
            mocker: The pytest-mock mocker fixture.
        """
        mocker.patch.object(
            structured_output_client,
            "_prompt_with_structured_response",
            return_value=(mock_structured_response, mock_usage),
        )
        mock_logger = mocker.patch.object(structured_output_client, "logger")

        structured_response, llm_response = (
            structured_output_client.prompt_with_structured_response(
                messages=structured_output_messages, response_model=ExampleResponseModel
            )
        )

        expected_log_calls = [
            mocker.call(
                f"Using model: {structured_output_client.config.default_model}"
            ),
            mocker.call(f"Input Payload: {structured_output_messages}"),
            mocker.call(f"Latest response: {llm_response.latest_response}"),
            mocker.call(f"Output usage: {llm_response.usage}"),
        ]

        mock_logger.info.assert_has_calls(expected_log_calls)
        assert mock_logger.info.call_count == len(expected_log_calls)

    def test_prompt_with_structured_response_with_different_response_models(
        self,
        structured_output_client: ConcreteTestLLMClient,
        structured_output_messages: list[Message],
        mock_usage: LLMUsage,
        mocker,
    ):
        """Test prompt_with_structured_response works with different Pydantic models.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            structured_output_messages: A pytest fixture providing valid messages.
            mock_usage: A pytest fixture providing mock usage statistics.
            mocker: The pytest-mock mocker fixture.
        """

        # Define a different response model
        class DifferentResponseModel(BaseModel):
            name: str
            age: int
            active: bool

        different_response = DifferentResponseModel(
            name="Test User", age=30, active=True
        )

        mock_prompt_structured = mocker.patch.object(
            structured_output_client,
            "_prompt_with_structured_response",
            return_value=(different_response, mock_usage),
        )

        structured_response, llm_response = (
            structured_output_client.prompt_with_structured_response(
                messages=structured_output_messages,
                response_model=DifferentResponseModel,
            )
        )

        # Verify the different model is used correctly
        assert isinstance(structured_response, DifferentResponseModel)
        assert structured_response.name == "Test User"
        assert structured_response.age == 30
        assert structured_response.active is True

        # Verify JSON content
        import json

        parsed_content = json.loads(llm_response.content)
        assert parsed_content["name"] == "Test User"
        assert parsed_content["age"] == 30
        assert parsed_content["active"] is True

        mock_prompt_structured.assert_called_once_with(
            messages=structured_output_messages,
            response_model=DifferentResponseModel,
            model=structured_output_client.config.default_model,
        )

    def test_prompt_with_structured_response_return_tuple_structure(
        self,
        structured_output_client: ConcreteTestLLMClient,
        structured_output_messages: list[Message],
        mock_structured_response: ExampleResponseModel,
        mock_usage: LLMUsage,
        mocker,
    ):
        """Test that prompt_with_structured_response returns the correct tuple structure.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            structured_output_messages: A pytest fixture providing valid messages.
            mock_structured_response: A pytest fixture providing a mock structured response.
            mock_usage: A pytest fixture providing mock usage statistics.
            mocker: The pytest-mock mocker fixture.
        """
        mocker.patch.object(
            structured_output_client,
            "_prompt_with_structured_response",
            return_value=(mock_structured_response, mock_usage),
        )

        result = structured_output_client.prompt_with_structured_response(
            messages=structured_output_messages, response_model=ExampleResponseModel
        )

        # Verify it returns a tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        structured_response, llm_response = result

        # Verify first element is the structured response
        assert isinstance(structured_response, ExampleResponseModel)
        assert structured_response is mock_structured_response

        # Verify second element is LLMResponse
        assert isinstance(llm_response, LLMResponse)
        assert llm_response.usage == mock_usage

    # ===== Integration Tests with Regular prompt() Method =====

    def test_regular_prompt_method_still_works_with_new_signature(
        self,
        structured_output_client: ConcreteTestLLMClient,
        structured_output_messages: list[Message],
        mock_usage: LLMUsage,
        mocker,
    ):
        """Test that regular prompt method still works with the updated _prompt signature.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            structured_output_messages: A pytest fixture providing valid messages.
            mock_usage: A pytest fixture providing mock usage statistics.
            mocker: The pytest-mock mocker fixture.
        """
        mock_raw_response = "Regular prompt response"

        mock_prompt = mocker.patch.object(
            structured_output_client,
            "_prompt",
            return_value=(mock_raw_response, mock_usage),
        )

        response = structured_output_client.prompt(messages=structured_output_messages)

        # Verify the _prompt method is called with get_logprobs=False by default
        mock_prompt.assert_called_once_with(
            model=structured_output_client.config.default_model,
            messages=structured_output_messages,
            get_logprobs=False,
        )

        assert response.content == mock_raw_response
        assert response.usage == mock_usage

    def test_regular_prompt_method_with_get_logprobs_not_supported(
        self,
        structured_output_client: ConcreteTestLLMClient,
        structured_output_messages: list[Message],
        mocker,
    ):
        """Test prompt method raises validation error when logprobs requested but not supported.

        Args:
            structured_output_client: A pytest fixture providing a concrete client instance.
            structured_output_messages: A pytest fixture providing valid messages.
            mocker: The pytest-mock mocker fixture.
        """
        mock_prompt = mocker.patch.object(structured_output_client, "_prompt")

        with pytest.raises(LLMValidationError) as excinfo:
            structured_output_client.prompt(
                messages=structured_output_messages, get_logprobs=True
            )

        assert _LOGPROBS_NOT_SUPPORTED_ERROR_TEMPLATE.format(
            structured_output_client.enum_value
        ) in str(excinfo.value)
        mock_prompt.assert_not_called()

    def test_regular_prompt_method_with_get_logprobs_supported(
        self,
        logprobs_client: ConcreteTestLLMClient,
        structured_output_messages: list[Message],
        mock_usage: LLMUsage,
        mocker,
    ):
        """Test prompt method works when logprobs requested and supported.

        Args:
            logprobs_client: A pytest fixture providing a client with logprobs support.
            structured_output_messages: A pytest fixture providing valid messages.
            mock_usage: A pytest fixture providing mock usage statistics.
            mocker: The pytest-mock mocker fixture.
        """
        mock_raw_response = "Response with logprobs"

        mock_prompt = mocker.patch.object(
            logprobs_client, "_prompt", return_value=(mock_raw_response, mock_usage)
        )

        response = logprobs_client.prompt(
            messages=structured_output_messages, get_logprobs=True
        )

        # Verify the _prompt method is called with get_logprobs=True
        mock_prompt.assert_called_once_with(
            model=logprobs_client.config.default_model,
            messages=structured_output_messages,
            get_logprobs=True,
        )

        assert response.content == mock_raw_response
        assert response.usage == mock_usage
