"""File for testing the structured output functionality in LLMClient module."""

import pytest
from pydantic import BaseModel

from meta_evaluator.llm_client import LLMClientConfig, LLMClient
from meta_evaluator.llm_client.LLM_client import (
    _FAILED_RESPONSE_ERROR_TEMPLATE,
    _NO_MESSAGES_ERROR,
    _LOGPROBS_NOT_SUPPORTED_ERROR_TEMPLATE,
)
from meta_evaluator.llm_client.models import (
    Message,
    LLMClientEnum,
    LLMResponse,
    LLMUsage,
    RoleEnum,
)
from meta_evaluator.llm_client.exceptions import (
    LLMAPIError,
    LLMValidationError,
)


class ExampleResponseModel(BaseModel):
    """A test Pydantic model for structured output testing."""

    task_id: str
    status: str
    confidence: float
    tags: list[str]


class LLMClientConfigConcreteTest(LLMClientConfig):
    """A concrete configuration class for testing LLMClient."""

    api_key: str = "test-api-key"
    supports_structured_output: bool = True
    default_model: str = "test-default-model"
    default_embedding_model: str = "test-default-embedding-model"
    supports_logprobs: bool = False

    def _prevent_instantiation(self) -> None:
        pass


class ConcreteTestLLMClient(LLMClient):
    """A concrete LLMClient subclass for testing structured output."""

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

    def _get_embedding(self, text_list: list[str], model: str) -> list[list[float]]:
        """Abstract method implementation for testing.

        This method is intended to be mocked in actual tests.

        Args:
            text_list: List of text prompts to generate embeddings for.
            model: The embedding model to use for this request.

        Returns:
            List of embedding vectors, one for each input prompt.
        """
        raise NotImplementedError(
            "ConcreteTestLLMClient._get_embedding should be mocked for tests"
        )

    def _prompt_with_structured_response(
        self, messages: list[Message], response_model: type[BaseModel], model: str
    ) -> tuple[BaseModel, LLMUsage]:
        """Abstract method implementation for testing.

        This method is intended to be mocked in actual tests.

        Args:
            messages: The list of messages.
            response_model: The Pydantic model for the response.
            model: The model to use.

        Returns:
            A tuple containing the structured response and LLMUsage.
        """
        raise NotImplementedError(
            "ConcreteTestLLMClient._prompt_with_structured_response should be mocked for tests"
        )


class TestStructuredOutput:
    """Test suite for structured output functionality."""

    @pytest.fixture
    def valid_config(self) -> LLMClientConfigConcreteTest:
        """Provides a valid LLMClientConfigConcreteTest instance.

        Returns:
            LLMClientConfigConcreteTest: A valid instance with structured output enabled.
        """
        return LLMClientConfigConcreteTest(
            api_key="test-api-key",
            supports_structured_output=True,
            default_model="gpt-4",
            default_embedding_model="text-embedding-ada-002",
            supports_logprobs=False,
        )

    @pytest.fixture
    def logprobs_config(self) -> LLMClientConfigConcreteTest:
        """Provides a valid config with logprobs support enabled.

        Returns:
            LLMClientConfigConcreteTest: A valid instance with logprobs support.
        """
        return LLMClientConfigConcreteTest(
            api_key="test-api-key",
            supports_structured_output=True,
            default_model="gpt-4",
            default_embedding_model="text-embedding-ada-002",
            supports_logprobs=True,
        )

    @pytest.fixture
    def concrete_client(
        self, valid_config: LLMClientConfigConcreteTest
    ) -> ConcreteTestLLMClient:
        """Provides a ConcreteTestLLMClient instance with valid config.

        Args:
            valid_config: A valid configuration instance for the LLM client.

        Returns:
            ConcreteTestLLMClient: An instance initialized with the provided config.
        """
        return ConcreteTestLLMClient(valid_config)

    @pytest.fixture
    def logprobs_client(
        self, logprobs_config: LLMClientConfigConcreteTest
    ) -> ConcreteTestLLMClient:
        """Provides a ConcreteTestLLMClient instance with logprobs support.

        Args:
            logprobs_config: A valid configuration instance with logprobs support.

        Returns:
            ConcreteTestLLMClient: An instance initialized with logprobs support.
        """
        return ConcreteTestLLMClient(logprobs_config)

    @pytest.fixture
    def mock_structured_response(self) -> ExampleResponseModel:
        """Provides a mock structured response for testing.

        Returns:
            ExampleResponseModel: A mock structured response instance.
        """
        return ExampleResponseModel(
            task_id="task_123",
            status="completed",
            confidence=0.95,
            tags=["important", "urgent"],
        )

    @pytest.fixture
    def mock_usage(self) -> LLMUsage:
        """Provides mock usage statistics.

        Returns:
            LLMUsage: A mock usage statistics instance.
        """
        return LLMUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40)

    @pytest.fixture
    def valid_messages(self) -> list[Message]:
        """Provides a list of valid Message objects.

        Returns:
            list[Message]: A list of valid Message objects.
        """
        return [Message(role=RoleEnum.USER, content="Generate structured response")]

    # ===== _prompt_with_structured_response Tests (Abstract Method) =====

    def test_prompt_with_structured_response_abstract_method_raises_not_implemented(
        self, concrete_client: ConcreteTestLLMClient, valid_messages: list[Message]
    ):
        """Test that the abstract _prompt_with_structured_response method raises NotImplementedError.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
        """
        with pytest.raises(NotImplementedError) as excinfo:
            concrete_client._prompt_with_structured_response(
                messages=valid_messages,
                response_model=ExampleResponseModel,
                model="test-model",
            )

        assert "should be mocked for tests" in str(excinfo.value)

    # ===== _construct_llm_response_with_structured_response Tests =====

    def test_construct_llm_response_with_structured_response_basic(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_structured_response: ExampleResponseModel,
        mock_usage: LLMUsage,
    ):
        """Test _construct_llm_response_with_structured_response builds response correctly.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mock_structured_response: A pytest fixture providing a mock structured response.
            mock_usage: A pytest fixture providing mock usage statistics.
        """
        # Make a copy since the method modifies the original list
        messages_copy = valid_messages.copy()

        response = concrete_client._construct_llm_response_with_structured_response(
            mock_structured_response, mock_usage, messages_copy
        )

        assert isinstance(response, LLMResponse)
        assert response.provider == concrete_client.enum_value
        assert response.model == concrete_client.config.default_model
        assert response.usage == mock_usage
        assert len(response.messages) == 2  # Original + assistant message
        assert response.messages[0] == valid_messages[0]  # Original message unchanged
        assert response.messages[1].role == RoleEnum.ASSISTANT

        # Verify the content is JSON serialized
        expected_json = mock_structured_response.model_dump_json()
        assert response.messages[1].content == expected_json
        assert response.content == expected_json
        assert response.latest_response.content == expected_json

    def test_construct_llm_response_with_structured_response_json_serialization(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_usage: LLMUsage,
    ):
        """Test that structured response is properly JSON serialized.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mock_usage: A pytest fixture providing mock usage statistics.
        """
        # Create a specific structured response to verify JSON content
        structured_response = ExampleResponseModel(
            task_id="json_test_123",
            status="active",
            confidence=0.87,
            tags=["test", "json"],
        )

        response = concrete_client._construct_llm_response_with_structured_response(
            structured_response, mock_usage, valid_messages.copy()
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
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_structured_response: ExampleResponseModel,
        mock_usage: LLMUsage,
        mocker,
    ):
        """Test prompt_with_structured_response uses default model and logs correctly on success.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mock_structured_response: A pytest fixture providing a mock structured response.
            mock_usage: A pytest fixture providing mock usage statistics.
            mocker: The pytest-mock mocker fixture.
        """
        mock_prompt_structured = mocker.patch.object(
            concrete_client,
            "_prompt_with_structured_response",
            return_value=(mock_structured_response, mock_usage),
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        structured_response, llm_response = (
            concrete_client.prompt_with_structured_response(
                messages=valid_messages, response_model=ExampleResponseModel
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
        assert llm_response.provider == concrete_client.enum_value
        assert llm_response.model == concrete_client.config.default_model
        assert len(llm_response.messages) == 2
        assert llm_response.messages[0].role == RoleEnum.USER
        assert llm_response.messages[1].role == RoleEnum.ASSISTANT

        # Verify content is JSON serialized
        expected_json = mock_structured_response.model_dump_json()
        assert llm_response.content == expected_json

        # Verify method calls
        mock_prompt_structured.assert_called_once_with(
            messages=valid_messages,
            response_model=ExampleResponseModel,
            model=concrete_client.config.default_model,
        )

        # Verify logging
        mock_logger.info.assert_any_call(
            f"Using model: {concrete_client.config.default_model}"
        )
        mock_logger.info.assert_any_call(f"Input Payload: {valid_messages}")
        mock_logger.info.assert_any_call(
            f"Latest response: {llm_response.latest_response}"
        )
        mock_logger.info.assert_any_call(f"Output usage: {llm_response.usage}")
        assert mock_logger.info.call_count == 4

    def test_prompt_with_structured_response_happy_path_explicit_model(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_structured_response: ExampleResponseModel,
        mock_usage: LLMUsage,
        mocker,
    ):
        """Test prompt_with_structured_response uses explicit model and logs correctly on success.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mock_structured_response: A pytest fixture providing a mock structured response.
            mock_usage: A pytest fixture providing mock usage statistics.
            mocker: The pytest-mock mocker fixture.
        """
        explicit_model = "my-specific-structured-model"

        mock_prompt_structured = mocker.patch.object(
            concrete_client,
            "_prompt_with_structured_response",
            return_value=(mock_structured_response, mock_usage),
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        structured_response, llm_response = (
            concrete_client.prompt_with_structured_response(
                messages=valid_messages,
                response_model=ExampleResponseModel,
                model=explicit_model,
            )
        )

        # Verify returns
        assert isinstance(structured_response, ExampleResponseModel)
        assert isinstance(llm_response, LLMResponse)
        assert llm_response.usage == mock_usage
        assert (
            llm_response.model == concrete_client.config.default_model
        )  # Uses default_model in response

        # Verify method calls with explicit model
        mock_prompt_structured.assert_called_once_with(
            messages=valid_messages,
            response_model=ExampleResponseModel,
            model=explicit_model,
        )

        # Verify logging shows explicit model
        mock_logger.info.assert_any_call(f"Using model: {explicit_model}")
        mock_logger.info.assert_any_call(f"Input Payload: {valid_messages}")
        mock_logger.info.assert_any_call(
            f"Latest response: {llm_response.latest_response}"
        )
        mock_logger.info.assert_any_call(f"Output usage: {llm_response.usage}")
        assert mock_logger.info.call_count == 4

    def test_prompt_with_structured_response_handles_empty_messages_list(
        self, concrete_client: ConcreteTestLLMClient, mocker
    ):
        """Test prompt_with_structured_response raises LLMValidationError for empty messages list.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            mocker: The pytest-mock mocker fixture.
        """
        mock_prompt_structured = mocker.patch.object(
            concrete_client, "_prompt_with_structured_response"
        )

        with pytest.raises(LLMValidationError) as excinfo:
            concrete_client.prompt_with_structured_response(
                messages=[], response_model=ExampleResponseModel
            )

        assert _NO_MESSAGES_ERROR in str(excinfo.value)
        assert excinfo.value.provider == concrete_client.enum_value
        mock_prompt_structured.assert_not_called()

    def test_prompt_with_structured_response_handles_exception_from_prompt_structured(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mocker,
    ):
        """Test prompt_with_structured_response wraps exceptions from _prompt_with_structured_response in LLMAPIError.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mocker: The pytest-mock mocker fixture.
        """
        original_exception = RuntimeError("Simulated structured API connection failure")

        mock_prompt_structured = mocker.patch.object(
            concrete_client,
            "_prompt_with_structured_response",
            side_effect=original_exception,
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        with pytest.raises(LLMAPIError) as excinfo:
            concrete_client.prompt_with_structured_response(
                messages=valid_messages, response_model=ExampleResponseModel
            )

        assert _FAILED_RESPONSE_ERROR_TEMPLATE.format(
            concrete_client.enum_value
        ) in str(excinfo.value)
        assert excinfo.value.provider == concrete_client.enum_value
        assert excinfo.value.original_error is original_exception

        mock_prompt_structured.assert_called_once_with(
            messages=valid_messages,
            response_model=ExampleResponseModel,
            model=concrete_client.config.default_model,
        )

        # Should log input but not output since exception occurred
        mock_logger.info.assert_any_call(
            f"Using model: {concrete_client.config.default_model}"
        )
        mock_logger.info.assert_any_call(f"Input Payload: {valid_messages}")
        assert mock_logger.info.call_count == 2

    def test_prompt_with_structured_response_logging_content_verification(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_structured_response: ExampleResponseModel,
        mock_usage: LLMUsage,
        mocker,
    ):
        """Test the exact content logged by prompt_with_structured_response method.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mock_structured_response: A pytest fixture providing a mock structured response.
            mock_usage: A pytest fixture providing mock usage statistics.
            mocker: The pytest-mock mocker fixture.
        """
        mocker.patch.object(
            concrete_client,
            "_prompt_with_structured_response",
            return_value=(mock_structured_response, mock_usage),
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        structured_response, llm_response = (
            concrete_client.prompt_with_structured_response(
                messages=valid_messages, response_model=ExampleResponseModel
            )
        )

        expected_log_calls = [
            mocker.call(f"Using model: {concrete_client.config.default_model}"),
            mocker.call(f"Input Payload: {valid_messages}"),
            mocker.call(f"Latest response: {llm_response.latest_response}"),
            mocker.call(f"Output usage: {llm_response.usage}"),
        ]

        mock_logger.info.assert_has_calls(expected_log_calls)
        assert mock_logger.info.call_count == len(expected_log_calls)

    def test_prompt_with_structured_response_with_different_response_models(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_usage: LLMUsage,
        mocker,
    ):
        """Test prompt_with_structured_response works with different Pydantic models.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
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
            concrete_client,
            "_prompt_with_structured_response",
            return_value=(different_response, mock_usage),
        )

        structured_response, llm_response = (
            concrete_client.prompt_with_structured_response(
                messages=valid_messages, response_model=DifferentResponseModel
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
            messages=valid_messages,
            response_model=DifferentResponseModel,
            model=concrete_client.config.default_model,
        )

    def test_prompt_with_structured_response_return_tuple_structure(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_structured_response: ExampleResponseModel,
        mock_usage: LLMUsage,
        mocker,
    ):
        """Test that prompt_with_structured_response returns the correct tuple structure.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mock_structured_response: A pytest fixture providing a mock structured response.
            mock_usage: A pytest fixture providing mock usage statistics.
            mocker: The pytest-mock mocker fixture.
        """
        mocker.patch.object(
            concrete_client,
            "_prompt_with_structured_response",
            return_value=(mock_structured_response, mock_usage),
        )

        result = concrete_client.prompt_with_structured_response(
            messages=valid_messages, response_model=ExampleResponseModel
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
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_usage: LLMUsage,
        mocker,
    ):
        """Test that regular prompt method still works with the updated _prompt signature.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mock_usage: A pytest fixture providing mock usage statistics.
            mocker: The pytest-mock mocker fixture.
        """
        mock_raw_response = "Regular prompt response"

        mock_prompt = mocker.patch.object(
            concrete_client, "_prompt", return_value=(mock_raw_response, mock_usage)
        )

        response = concrete_client.prompt(messages=valid_messages)

        # Verify the _prompt method is called with get_logprobs=False by default
        mock_prompt.assert_called_once_with(
            model=concrete_client.config.default_model,
            messages=valid_messages,
            get_logprobs=False,
        )

        assert response.content == mock_raw_response
        assert response.usage == mock_usage

    def test_regular_prompt_method_with_get_logprobs_not_supported(
        self,
        concrete_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mocker,
    ):
        """Test prompt method raises validation error when logprobs requested but not supported.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            valid_messages: A pytest fixture providing valid messages.
            mocker: The pytest-mock mocker fixture.
        """
        mock_prompt = mocker.patch.object(concrete_client, "_prompt")

        with pytest.raises(LLMValidationError) as excinfo:
            concrete_client.prompt(messages=valid_messages, get_logprobs=True)

        assert _LOGPROBS_NOT_SUPPORTED_ERROR_TEMPLATE.format(
            concrete_client.enum_value
        ) in str(excinfo.value)
        mock_prompt.assert_not_called()

    def test_regular_prompt_method_with_get_logprobs_supported(
        self,
        logprobs_client: ConcreteTestLLMClient,
        valid_messages: list[Message],
        mock_usage: LLMUsage,
        mocker,
    ):
        """Test prompt method works when logprobs requested and supported.

        Args:
            logprobs_client: A pytest fixture providing a client with logprobs support.
            valid_messages: A pytest fixture providing valid messages.
            mock_usage: A pytest fixture providing mock usage statistics.
            mocker: The pytest-mock mocker fixture.
        """
        mock_raw_response = "Response with logprobs"

        mock_prompt = mocker.patch.object(
            logprobs_client, "_prompt", return_value=(mock_raw_response, mock_usage)
        )

        response = logprobs_client.prompt(messages=valid_messages, get_logprobs=True)

        # Verify the _prompt method is called with get_logprobs=True
        mock_prompt.assert_called_once_with(
            model=logprobs_client.config.default_model,
            messages=valid_messages,
            get_logprobs=True,
        )

        assert response.content == mock_raw_response
        assert response.usage == mock_usage
