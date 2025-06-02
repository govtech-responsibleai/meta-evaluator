"""Comprehensive tests for embeddings functionality in LLMClient."""

import pytest
from typing import TypeVar

from meta_evaluator.llm_client import LLMClientConfig, LLMClient
from meta_evaluator.llm_client.LLM_client import (
    _FAILED_EMBEDDING_ERROR_TEMPLATE,
    _NO_PROMPTS_ERROR,
)
from meta_evaluator.llm_client.models import (
    Message,
    LLMClientEnum,
    LLMUsage,
)
from meta_evaluator.llm_client.exceptions import (
    LLMAPIError,
    LLMValidationError,
)
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClientConfigConcreteTest(LLMClientConfig):
    """A concrete configuration class for testing LLMClient embeddings."""

    api_key: str = "test-api-key"
    supports_structured_output: bool = False
    default_model: str = "test-default-model"
    default_embedding_model: str = "test-default-embedding-model"
    supports_logprobs: bool = False

    def _prevent_instantiation(self) -> None:
        """Prevent instantiation as required by abstract base class."""
        pass


class ConcreteTestLLMClient(LLMClient):
    """A concrete LLMClient subclass for testing embeddings functionality."""

    @property
    def enum_value(self) -> LLMClientEnum:
        """Return the unique LLMClientEnum value for the test client.

        Returns:
            LLMClientEnum: The test client's enum identifier.
        """
        return LLMClientEnum.OPENAI

    def _prompt(
        self, model: str, messages: list[Message], get_logprobs: bool = False
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
        self, messages: list[Message], response_model: type[T], model: str
    ) -> tuple[T, LLMUsage]:
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


class TestEmbeddings:
    """Comprehensive test suite for embeddings functionality."""

    @pytest.fixture
    def valid_config(self) -> LLMClientConfigConcreteTest:
        """Provides a valid LLMClientConfigConcreteTest instance.

        Returns:
            LLMClientConfigConcreteTest: A valid instance of
                LLMClientConfigConcreteTest with embeddings support.
        """
        return LLMClientConfigConcreteTest(
            api_key="test-api-key",
            supports_structured_output=False,
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
            valid_config: A valid configuration instance for the LLM client.

        Returns:
            ConcreteTestLLMClient: An instance of ConcreteTestLLMClient initialized with the provided config.
        """
        return ConcreteTestLLMClient(valid_config)

    @pytest.fixture
    def single_text_input(self) -> list[str]:
        """Provides a single text input for embedding tests.

        Returns:
            list[str]: A list containing one text prompt for embedding.
        """
        return ["Hello, world!"]

    @pytest.fixture
    def multiple_text_input(self) -> list[str]:
        """Provides multiple text inputs for embedding tests.

        Returns:
            list[str]: A list containing multiple text prompts for embedding.
        """
        return ["Hello, world!", "How are you?", "This is a test."]

    @pytest.fixture
    def mock_single_embedding(self) -> list[list[float]]:
        """Provides a mock embedding response for single text input.

        Returns:
            list[list[float]]: A list containing one mock embedding vector.
        """
        return [[0.1, 0.2, 0.3, 0.4, 0.5]]

    @pytest.fixture
    def mock_multiple_embeddings(self) -> list[list[float]]:
        """Provides mock embedding responses for multiple text inputs.

        Returns:
            list[list[float]]: A list containing multiple mock embedding vectors.
        """
        return [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9, 1.0],
            [1.1, 1.2, 1.3, 1.4, 1.5],
        ]

    # ===== Happy Path Tests =====

    def test_single_text_embedding_with_default_model(
        self,
        concrete_client: ConcreteTestLLMClient,
        single_text_input: list[str],
        mock_single_embedding: list[list[float]],
        mocker,
    ):
        """Test single text embedding using default model with comprehensive logging verification.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            single_text_input: A pytest fixture providing single text input.
            mock_single_embedding: A pytest fixture providing mock embedding response.
            mocker: The pytest-mock mocker fixture.
        """
        mock_get_embedding = mocker.patch.object(
            concrete_client, "_get_embedding", return_value=mock_single_embedding
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        result = concrete_client.get_embedding(text_list=single_text_input)

        # Verify return value structure and content
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert all(isinstance(val, float) for val in result[0])
        assert result == mock_single_embedding

        # Verify method call delegation
        mock_get_embedding.assert_called_once_with(
            text_list=single_text_input,
            model=concrete_client.config.default_embedding_model,
        )

        # Verify logging
        mock_logger.info.assert_any_call(
            f"Using embedding model: {concrete_client.config.default_embedding_model}"
        )
        mock_logger.info.assert_any_call(
            f"Processing {len(single_text_input)} prompts for embeddings"
        )
        mock_logger.info.assert_any_call(
            f"Successfully generated {len(result)} embeddings"
        )
        assert mock_logger.info.call_count == 3

    def test_multiple_texts_embedding_with_default_model(
        self,
        concrete_client: ConcreteTestLLMClient,
        multiple_text_input: list[str],
        mock_multiple_embeddings: list[list[float]],
        mocker,
    ):
        """Test multiple texts embedding using default model with batch processing verification.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            multiple_text_input: A pytest fixture providing multiple text inputs.
            mock_multiple_embeddings: A pytest fixture providing mock embedding responses.
            mocker: The pytest-mock mocker fixture.
        """
        mock_get_embedding = mocker.patch.object(
            concrete_client, "_get_embedding", return_value=mock_multiple_embeddings
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        result = concrete_client.get_embedding(text_list=multiple_text_input)

        # Verify return value structure and content
        assert isinstance(result, list)
        assert len(result) == len(multiple_text_input)
        assert all(isinstance(embedding, list) for embedding in result)
        assert all(
            all(isinstance(val, float) for val in embedding) for embedding in result
        )
        assert result == mock_multiple_embeddings

        # Verify method call delegation
        mock_get_embedding.assert_called_once_with(
            text_list=multiple_text_input,
            model=concrete_client.config.default_embedding_model,
        )

        # Verify batch processing logging
        mock_logger.info.assert_any_call(
            f"Using embedding model: {concrete_client.config.default_embedding_model}"
        )
        mock_logger.info.assert_any_call(
            f"Processing {len(multiple_text_input)} prompts for embeddings"
        )
        mock_logger.info.assert_any_call(
            f"Successfully generated {len(result)} embeddings"
        )
        assert mock_logger.info.call_count == 3

    def test_single_text_embedding_with_explicit_model(
        self,
        concrete_client: ConcreteTestLLMClient,
        single_text_input: list[str],
        mock_single_embedding: list[list[float]],
        mocker,
    ):
        """Test single text embedding using explicit model parameter.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            single_text_input: A pytest fixture providing single text input.
            mock_single_embedding: A pytest fixture providing mock embedding response.
            mocker: The pytest-mock mocker fixture.
        """
        explicit_model = "text-embedding-3-large"

        mock_get_embedding = mocker.patch.object(
            concrete_client, "_get_embedding", return_value=mock_single_embedding
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        result = concrete_client.get_embedding(
            text_list=single_text_input, model=explicit_model
        )

        # Verify return value
        assert result == mock_single_embedding

        # Verify explicit model is used
        mock_get_embedding.assert_called_once_with(
            text_list=single_text_input, model=explicit_model
        )

        # Verify logging shows explicit model
        mock_logger.info.assert_any_call(f"Using embedding model: {explicit_model}")
        mock_logger.info.assert_any_call(
            f"Processing {len(single_text_input)} prompts for embeddings"
        )
        mock_logger.info.assert_any_call(
            f"Successfully generated {len(result)} embeddings"
        )
        assert mock_logger.info.call_count == 3

    def test_multiple_texts_embedding_with_explicit_model(
        self,
        concrete_client: ConcreteTestLLMClient,
        multiple_text_input: list[str],
        mock_multiple_embeddings: list[list[float]],
        mocker,
    ):
        """Test multiple texts embedding using explicit model parameter.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            multiple_text_input: A pytest fixture providing multiple text inputs.
            mock_multiple_embeddings: A pytest fixture providing mock embedding responses.
            mocker: The pytest-mock mocker fixture.
        """
        explicit_model = "text-embedding-3-small"

        mock_get_embedding = mocker.patch.object(
            concrete_client, "_get_embedding", return_value=mock_multiple_embeddings
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        result = concrete_client.get_embedding(
            text_list=multiple_text_input, model=explicit_model
        )

        # Verify return value
        assert result == mock_multiple_embeddings
        assert len(result) == len(multiple_text_input)

        # Verify explicit model is used for batch processing
        mock_get_embedding.assert_called_once_with(
            text_list=multiple_text_input, model=explicit_model
        )

        # Verify logging shows explicit model and batch size
        mock_logger.info.assert_any_call(f"Using embedding model: {explicit_model}")
        mock_logger.info.assert_any_call(
            f"Processing {len(multiple_text_input)} prompts for embeddings"
        )
        mock_logger.info.assert_any_call(
            f"Successfully generated {len(result)} embeddings"
        )
        assert mock_logger.info.call_count == 3

    # ===== Error Path Tests =====

    def test_empty_text_list_validation_error(
        self, concrete_client: ConcreteTestLLMClient, mocker
    ):
        """Test that empty text list raises LLMValidationError with proper error message.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            mocker: The pytest-mock mocker fixture.
        """
        mock_get_embedding = mocker.patch.object(concrete_client, "_get_embedding")

        with pytest.raises(LLMValidationError) as excinfo:
            concrete_client.get_embedding(text_list=[])

        # Verify exception details
        assert _NO_PROMPTS_ERROR in str(excinfo.value)
        assert excinfo.value.provider == concrete_client.enum_value

        # Verify _get_embedding was never called
        mock_get_embedding.assert_not_called()

    def test_api_failure_handling_with_runtime_error(
        self,
        concrete_client: ConcreteTestLLMClient,
        single_text_input: list[str],
        mocker,
    ):
        """Test API failure handling when _get_embedding raises RuntimeError.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            single_text_input: A pytest fixture providing single text input.
            mocker: The pytest-mock mocker fixture.
        """
        original_exception = RuntimeError("Simulated API connection failure")

        mock_get_embedding = mocker.patch.object(
            concrete_client, "_get_embedding", side_effect=original_exception
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        with pytest.raises(LLMAPIError) as excinfo:
            concrete_client.get_embedding(text_list=single_text_input)

        # Verify exception wrapping
        assert _FAILED_EMBEDDING_ERROR_TEMPLATE.format(
            concrete_client.enum_value
        ) in str(excinfo.value)
        assert excinfo.value.provider == concrete_client.enum_value
        assert excinfo.value.original_error is original_exception

        # Verify method was called before failing
        mock_get_embedding.assert_called_once_with(
            text_list=single_text_input,
            model=concrete_client.config.default_embedding_model,
        )

        # Verify partial logging (input logged, success not logged)
        mock_logger.info.assert_any_call(
            f"Using embedding model: {concrete_client.config.default_embedding_model}"
        )
        mock_logger.info.assert_any_call(
            f"Processing {len(single_text_input)} prompts for embeddings"
        )
        # Should not log success since exception occurred
        assert mock_logger.info.call_count == 2

    def test_not_implemented_error_handling(
        self,
        concrete_client: ConcreteTestLLMClient,
        single_text_input: list[str],
        mocker,
    ):
        """Test NotImplementedError handling for clients that don't support embeddings.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            single_text_input: A pytest fixture providing single text input.
            mocker: The pytest-mock mocker fixture.
        """
        original_exception = NotImplementedError(
            "This client does not support embeddings"
        )

        mock_get_embedding = mocker.patch.object(
            concrete_client, "_get_embedding", side_effect=original_exception
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        with pytest.raises(LLMAPIError) as excinfo:
            concrete_client.get_embedding(text_list=single_text_input)

        # Verify NotImplementedError is wrapped as LLMAPIError
        assert _FAILED_EMBEDDING_ERROR_TEMPLATE.format(
            concrete_client.enum_value
        ) in str(excinfo.value)
        assert excinfo.value.provider == concrete_client.enum_value
        assert excinfo.value.original_error is original_exception
        assert isinstance(excinfo.value.original_error, NotImplementedError)

        # Verify method was called before failing
        mock_get_embedding.assert_called_once_with(
            text_list=single_text_input,
            model=concrete_client.config.default_embedding_model,
        )

        # Verify partial logging (input logged, success not logged)
        mock_logger.info.assert_any_call(
            f"Using embedding model: {concrete_client.config.default_embedding_model}"
        )
        mock_logger.info.assert_any_call(
            f"Processing {len(single_text_input)} prompts for embeddings"
        )
        assert mock_logger.info.call_count == 2

    def test_logging_content_verification(
        self,
        concrete_client: ConcreteTestLLMClient,
        multiple_text_input: list[str],
        mock_multiple_embeddings: list[list[float]],
        mocker,
    ):
        """Test the exact content logged by the get_embedding method.

        This test explicitly checks the arguments passed to the logger's info method.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
            multiple_text_input: A pytest fixture providing multiple text inputs.
            mock_multiple_embeddings: A pytest fixture providing mock embedding responses.
            mocker: The pytest-mock mocker fixture.
        """
        mocker.patch.object(
            concrete_client, "_get_embedding", return_value=mock_multiple_embeddings
        )
        mock_logger = mocker.patch.object(concrete_client, "logger")

        result = concrete_client.get_embedding(text_list=multiple_text_input)

        expected_log_calls = [
            mocker.call(
                f"Using embedding model: {concrete_client.config.default_embedding_model}"
            ),
            mocker.call(
                f"Processing {len(multiple_text_input)} prompts for embeddings"
            ),
            mocker.call(f"Successfully generated {len(result)} embeddings"),
        ]

        mock_logger.info.assert_has_calls(expected_log_calls)
        assert mock_logger.info.call_count == len(expected_log_calls)

    def test_validate_prompts_method(self, concrete_client: ConcreteTestLLMClient):
        """Test _validate_prompts works correctly for embeddings validation.

        Args:
            concrete_client: A pytest fixture providing a concrete client instance.
        """
        # Should not raise for non-empty prompts
        valid_prompts = ["Hello world", "How are you?"]
        concrete_client._validate_prompts(valid_prompts)  # Should not raise

        # Should raise for empty prompts
        with pytest.raises(LLMValidationError) as excinfo:
            concrete_client._validate_prompts([])

        assert _NO_PROMPTS_ERROR in str(excinfo.value)
        assert excinfo.value.provider == concrete_client.enum_value
