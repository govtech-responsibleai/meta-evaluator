"""Comprehensive tests for embeddings functionality in LLMClient."""

from typing import TypeVar

import pytest
from pydantic import BaseModel

from meta_evaluator.llm_client.exceptions import (
    LLMAPIError,
    LLMValidationError,
)
from meta_evaluator.llm_client.LLM_client import (
    _FAILED_EMBEDDING_ERROR_TEMPLATE,
    _NO_PROMPTS_ERROR,
)

# Import test classes from conftest for type hints
from .conftest import ConcreteTestLLMClient

T = TypeVar("T", bound=BaseModel)


class TestEmbeddings:
    """Comprehensive test suite for embeddings functionality."""

    # ===== Happy Path Tests =====

    def test_single_text_embedding_with_default_model(
        self,
        basic_llm_client: ConcreteTestLLMClient,
        single_text_input: list[str],
        mock_single_embedding: list[list[float]],
        mocker,
    ):
        """Test single text embedding using default model with comprehensive logging verification.

        Args:
            basic_llm_client: A pytest fixture providing a concrete client instance.
            single_text_input: A pytest fixture providing single text input.
            mock_single_embedding: A pytest fixture providing mock embedding response.
            mocker: The pytest-mock mocker fixture.
        """
        mock_get_embedding = mocker.patch.object(
            basic_llm_client, "_get_embedding", return_value=mock_single_embedding
        )
        mock_logger = mocker.patch.object(basic_llm_client, "logger")

        result = basic_llm_client.get_embedding(text_list=single_text_input)

        # Verify return value structure and content
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert all(isinstance(val, float) for val in result[0])
        assert result == mock_single_embedding

        # Verify method call delegation
        mock_get_embedding.assert_called_once_with(
            text_list=single_text_input,
            model=basic_llm_client.config.default_embedding_model,
        )

        # Verify logging
        mock_logger.info.assert_any_call(
            f"Using embedding model: {basic_llm_client.config.default_embedding_model}"
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
        basic_llm_client: ConcreteTestLLMClient,
        multiple_text_inputs: list[str],
        mock_multiple_embeddings: list[list[float]],
        mocker,
    ):
        """Test multiple texts embedding using default model with batch processing verification.

        Args:
            basic_llm_client: A pytest fixture providing a concrete client instance.
            multiple_text_inputs: A pytest fixture providing multiple text inputs.
            mock_multiple_embeddings: A pytest fixture providing mock embedding responses.
            mocker: The pytest-mock mocker fixture.
        """
        mock_get_embedding = mocker.patch.object(
            basic_llm_client, "_get_embedding", return_value=mock_multiple_embeddings
        )
        mock_logger = mocker.patch.object(basic_llm_client, "logger")

        result = basic_llm_client.get_embedding(text_list=multiple_text_inputs)

        # Verify return value structure and content
        assert isinstance(result, list)
        assert len(result) == len(multiple_text_inputs)
        assert all(isinstance(embedding, list) for embedding in result)
        assert all(
            all(isinstance(val, float) for val in embedding) for embedding in result
        )
        assert result == mock_multiple_embeddings

        # Verify method call delegation
        mock_get_embedding.assert_called_once_with(
            text_list=multiple_text_inputs,
            model=basic_llm_client.config.default_embedding_model,
        )

        # Verify batch processing logging
        mock_logger.info.assert_any_call(
            f"Using embedding model: {basic_llm_client.config.default_embedding_model}"
        )
        mock_logger.info.assert_any_call(
            f"Processing {len(multiple_text_inputs)} prompts for embeddings"
        )
        mock_logger.info.assert_any_call(
            f"Successfully generated {len(result)} embeddings"
        )
        assert mock_logger.info.call_count == 3

    def test_single_text_embedding_with_explicit_model(
        self,
        basic_llm_client: ConcreteTestLLMClient,
        single_text_input: list[str],
        mock_single_embedding: list[list[float]],
        mocker,
    ):
        """Test single text embedding using explicit model parameter.

        Args:
            basic_llm_client: A pytest fixture providing a concrete client instance.
            single_text_input: A pytest fixture providing single text input.
            mock_single_embedding: A pytest fixture providing mock embedding response.
            mocker: The pytest-mock mocker fixture.
        """
        explicit_model = "text-embedding-3-large"

        mock_get_embedding = mocker.patch.object(
            basic_llm_client, "_get_embedding", return_value=mock_single_embedding
        )
        mock_logger = mocker.patch.object(basic_llm_client, "logger")

        result = basic_llm_client.get_embedding(
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
        basic_llm_client: ConcreteTestLLMClient,
        multiple_text_inputs: list[str],
        mock_multiple_embeddings: list[list[float]],
        mocker,
    ):
        """Test multiple texts embedding using explicit model parameter.

        Args:
            basic_llm_client: A pytest fixture providing a concrete client instance.
            multiple_text_inputs: A pytest fixture providing multiple text inputs.
            mock_multiple_embeddings: A pytest fixture providing mock embedding responses.
            mocker: The pytest-mock mocker fixture.
        """
        explicit_model = "text-embedding-3-small"

        mock_get_embedding = mocker.patch.object(
            basic_llm_client, "_get_embedding", return_value=mock_multiple_embeddings
        )
        mock_logger = mocker.patch.object(basic_llm_client, "logger")

        result = basic_llm_client.get_embedding(
            text_list=multiple_text_inputs, model=explicit_model
        )

        # Verify return value
        assert result == mock_multiple_embeddings
        assert len(result) == len(multiple_text_inputs)

        # Verify explicit model is used for batch processing
        mock_get_embedding.assert_called_once_with(
            text_list=multiple_text_inputs, model=explicit_model
        )

        # Verify logging shows explicit model and batch size
        mock_logger.info.assert_any_call(f"Using embedding model: {explicit_model}")
        mock_logger.info.assert_any_call(
            f"Processing {len(multiple_text_inputs)} prompts for embeddings"
        )
        mock_logger.info.assert_any_call(
            f"Successfully generated {len(result)} embeddings"
        )
        assert mock_logger.info.call_count == 3

    # ===== Error Path Tests =====

    def test_empty_text_list_validation_error(
        self, basic_llm_client: ConcreteTestLLMClient, mocker
    ):
        """Test that empty text list raises LLMValidationError with proper error message.

        Args:
            basic_llm_client: A pytest fixture providing a concrete client instance.
            mocker: The pytest-mock mocker fixture.
        """
        mock_get_embedding = mocker.patch.object(basic_llm_client, "_get_embedding")

        with pytest.raises(LLMValidationError) as excinfo:
            basic_llm_client.get_embedding(text_list=[])

        # Verify exception details
        assert _NO_PROMPTS_ERROR in str(excinfo.value)
        assert excinfo.value.provider == basic_llm_client.enum_value

        # Verify _get_embedding was never called
        mock_get_embedding.assert_not_called()

    def test_api_failure_handling_with_runtime_error(
        self,
        basic_llm_client: ConcreteTestLLMClient,
        single_text_input: list[str],
        mocker,
    ):
        """Test API failure handling when _get_embedding raises RuntimeError.

        Args:
            basic_llm_client: A pytest fixture providing a concrete client instance.
            single_text_input: A pytest fixture providing single text input.
            mocker: The pytest-mock mocker fixture.
        """
        original_exception = RuntimeError("Simulated API connection failure")

        mock_get_embedding = mocker.patch.object(
            basic_llm_client, "_get_embedding", side_effect=original_exception
        )
        mock_logger = mocker.patch.object(basic_llm_client, "logger")

        with pytest.raises(LLMAPIError) as excinfo:
            basic_llm_client.get_embedding(text_list=single_text_input)

        # Verify exception wrapping
        assert _FAILED_EMBEDDING_ERROR_TEMPLATE.format(
            basic_llm_client.enum_value
        ) in str(excinfo.value)
        assert excinfo.value.provider == basic_llm_client.enum_value
        assert excinfo.value.original_error is original_exception

        # Verify method was called before failing
        mock_get_embedding.assert_called_once_with(
            text_list=single_text_input,
            model=basic_llm_client.config.default_embedding_model,
        )

        # Verify partial logging (input logged, success not logged)
        mock_logger.info.assert_any_call(
            f"Using embedding model: {basic_llm_client.config.default_embedding_model}"
        )
        mock_logger.info.assert_any_call(
            f"Processing {len(single_text_input)} prompts for embeddings"
        )
        # Should not log success since exception occurred
        assert mock_logger.info.call_count == 2

    def test_not_implemented_error_handling(
        self,
        basic_llm_client: ConcreteTestLLMClient,
        single_text_input: list[str],
        mocker,
    ):
        """Test NotImplementedError handling for clients that don't support embeddings.

        Args:
            basic_llm_client: A pytest fixture providing a concrete client instance.
            single_text_input: A pytest fixture providing single text input.
            mocker: The pytest-mock mocker fixture.
        """
        original_exception = NotImplementedError(
            "This client does not support embeddings"
        )

        mock_get_embedding = mocker.patch.object(
            basic_llm_client, "_get_embedding", side_effect=original_exception
        )
        mock_logger = mocker.patch.object(basic_llm_client, "logger")

        with pytest.raises(LLMAPIError) as excinfo:
            basic_llm_client.get_embedding(text_list=single_text_input)

        # Verify NotImplementedError is wrapped as LLMAPIError
        assert _FAILED_EMBEDDING_ERROR_TEMPLATE.format(
            basic_llm_client.enum_value
        ) in str(excinfo.value)
        assert excinfo.value.provider == basic_llm_client.enum_value
        assert excinfo.value.original_error is original_exception
        assert isinstance(excinfo.value.original_error, NotImplementedError)

        # Verify method was called before failing
        mock_get_embedding.assert_called_once_with(
            text_list=single_text_input,
            model=basic_llm_client.config.default_embedding_model,
        )

        # Verify partial logging (input logged, success not logged)
        mock_logger.info.assert_any_call(
            f"Using embedding model: {basic_llm_client.config.default_embedding_model}"
        )
        mock_logger.info.assert_any_call(
            f"Processing {len(single_text_input)} prompts for embeddings"
        )
        assert mock_logger.info.call_count == 2

    def test_logging_content_verification(
        self,
        basic_llm_client: ConcreteTestLLMClient,
        multiple_text_inputs: list[str],
        mock_multiple_embeddings: list[list[float]],
        mocker,
    ):
        """Test the exact content logged by the get_embedding method.

        This test explicitly checks the arguments passed to the logger's info method.

        Args:
            basic_llm_client: A pytest fixture providing a concrete client instance.
            multiple_text_inputs: A pytest fixture providing multiple text inputs.
            mock_multiple_embeddings: A pytest fixture providing mock embedding responses.
            mocker: The pytest-mock mocker fixture.
        """
        mocker.patch.object(
            basic_llm_client, "_get_embedding", return_value=mock_multiple_embeddings
        )
        mock_logger = mocker.patch.object(basic_llm_client, "logger")

        result = basic_llm_client.get_embedding(text_list=multiple_text_inputs)

        expected_log_calls = [
            mocker.call(
                f"Using embedding model: {basic_llm_client.config.default_embedding_model}"
            ),
            mocker.call(
                f"Processing {len(multiple_text_inputs)} prompts for embeddings"
            ),
            mocker.call(f"Successfully generated {len(result)} embeddings"),
        ]

        mock_logger.info.assert_has_calls(expected_log_calls)
        assert mock_logger.info.call_count == len(expected_log_calls)

    def test_validate_prompts_method(self, basic_llm_client: ConcreteTestLLMClient):
        """Test _validate_prompts works correctly for embeddings validation.

        Args:
            basic_llm_client: A pytest fixture providing a concrete client instance.
        """
        # Should not raise for non-empty prompts
        valid_prompts = ["Hello world", "How are you?"]
        basic_llm_client._validate_prompts(valid_prompts)  # Should not raise

        # Should raise for empty prompts
        with pytest.raises(LLMValidationError) as excinfo:
            basic_llm_client._validate_prompts([])

        assert _NO_PROMPTS_ERROR in str(excinfo.value)
        assert excinfo.value.provider == basic_llm_client.enum_value
