"""Test suite for OpenAI LLM client implementation.

This module contains comprehensive tests for both the OpenAIConfig
and OpenAIClient classes, covering configuration validation,
client initialization, API integration, and error handling scenarios.
All tests use mocked responses to avoid external API dependencies.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from meta_evaluator.llm_client.enums import LLMClientEnum
from meta_evaluator.llm_client.exceptions import LLMAPIError
from meta_evaluator.llm_client.models import Message
from meta_evaluator.llm_client.openai_client import (
    OpenAIClient,
    OpenAIConfig,
)

from .conftest import ExampleResponseModel


class TestOpenAIConfig:
    """Test suite for the OpenAIConfig class.

    This class tests the OpenAI configuration implementation,
    including default values, serialization, and integration
    with the base LLMClientConfig class.
    """

    # Note: openai_config_data and openai_config fixtures are provided by conftest.py

    def test_happy_path_instantiation(self, openai_config_data: dict[str, Any]) -> None:
        """Test successful creation with all required fields provided.

        Verifies that an OpenAIConfig instance can be created successfully
        when all required fields are provided with valid values.

        Args:
            openai_config_data: A dictionary containing valid configuration data.
        """
        config = OpenAIConfig(**openai_config_data)

        assert config.api_key == "test-api-key"
        assert config.default_model == "gpt-4o-2024-11-20"
        assert config.default_embedding_model == "text-embedding-3-large"
        assert config.supports_structured_output is True
        assert config.supports_logprobs is True

    def test_default_boolean_values(self, openai_config_data: dict[str, Any]) -> None:
        """Test that OpenAI-specific capabilities have correct default values.

        Verifies that the supports_structured_output and supports_logprobs
        fields are set to their expected default values for OpenAI.

        Args:
            openai_config_data: A dictionary containing valid configuration data.
        """
        config = OpenAIConfig(**openai_config_data)

        assert config.supports_structured_output is True
        assert config.supports_logprobs is True

    def test_prevent_instantiation_method_exists(
        self, openai_config: OpenAIConfig
    ) -> None:
        """Test that _prevent_instantiation method can be called without error.

        Verifies that the _prevent_instantiation method required by the abstract
        base class is implemented and can be called without raising exceptions.

        Args:
            openai_config: A valid OpenAIConfig instance.
        """
        # Should not raise an exception
        openai_config._prevent_instantiation()

    def test_missing_api_key_raises_validation_error(self) -> None:
        """Test that missing api_key field raises ValidationError.

        Verifies that attempting to create an OpenAIConfig instance
        without providing the required api_key field results in a ValidationError.
        """
        with pytest.raises(ValidationError):
            OpenAIConfig(  # type: ignore
                default_model="gpt-4o-2024-11-20",
                default_embedding_model="text-embedding-3-large",
            )

    def test_missing_default_model_raises_validation_error(self) -> None:
        """Test that missing default_model field raises ValidationError.

        Verifies that attempting to create an OpenAIConfig instance
        without providing the required default_model field results in a ValidationError.
        """
        with pytest.raises(ValidationError):
            OpenAIConfig(  # type: ignore
                api_key="test-key",
                default_embedding_model="text-embedding-3-large",
            )

    def test_serialization_excludes_api_key(self, openai_config: OpenAIConfig) -> None:
        """Test that serialization properly excludes the API key.

        Verifies that when an OpenAIConfig instance is serialized,
        the API key is properly excluded for security reasons.

        Args:
            openai_config: A valid OpenAIConfig instance.
        """
        serialized = openai_config.serialize()

        # Verify that the serialized data contains expected fields
        assert serialized.default_model == "gpt-4o-2024-11-20"
        assert serialized.default_embedding_model == "text-embedding-3-large"
        assert serialized.supports_structured_output is True
        assert serialized.supports_logprobs is True

        # Verify that API key is not in the serialized data
        assert not hasattr(serialized, "api_key")

    def test_deserialization_with_api_key(self, openai_config: OpenAIConfig) -> None:
        """Test that deserialization works correctly with API key.

        Verifies that an OpenAIConfig instance can be properly reconstructed
        from serialized state when an API key is provided.

        Args:
            openai_config: A valid OpenAIConfig instance.
        """
        serialized = openai_config.serialize()

        # Deserialize with new API key
        new_api_key = "new-test-api-key"
        deserialized = OpenAIConfig.deserialize(serialized, new_api_key)

        assert deserialized.api_key == new_api_key
        assert deserialized.default_model == "gpt-4o-2024-11-20"
        assert deserialized.default_embedding_model == "text-embedding-3-large"
        assert deserialized.supports_structured_output is True
        assert deserialized.supports_logprobs is True


class TestOpenAIClient:
    """Test suite for the OpenAIClient class.

    This class tests the OpenAI client implementation, including
    initialization, message conversion, API integration, and error handling.
    All tests use mocked responses to avoid external API dependencies.
    """

    # Note: openai_config and openai_client fixtures are provided by conftest.py

    @patch("meta_evaluator.llm_client.openai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.openai_client.OpenAI")
    def test_client_initialization(
        self,
        mock_openai: Mock,
        mock_instructor: Mock,
        openai_config: OpenAIConfig,
    ) -> None:
        """Test that OpenAIClient initializes correctly with proper configuration.

        Verifies that the client properly initializes the OpenAI client
        and instructor client with the provided configuration.

        Args:
            mock_openai: Mock for the OpenAI client class.
            mock_instructor: Mock for the instructor.from_openai function.
            openai_config: A valid OpenAI configuration.
        """
        client = OpenAIClient(openai_config)

        # Verify that OpenAI client was initialized with correct API key
        mock_openai.assert_called_once_with(api_key="test-api-key")

        # Verify that instructor client was initialized
        mock_instructor.assert_called_once_with(mock_openai.return_value)

        # Verify client properties
        assert client.config == openai_config
        assert client.enum_value == LLMClientEnum.OPENAI

    def test_convert_messages_to_openai_format(
        self, openai_config: OpenAIConfig, sample_messages: list[Message]
    ) -> None:
        """Test message conversion to OpenAI format.

        Verifies that internal Message objects are correctly converted
        to OpenAI's ChatCompletionMessageParam format.

        Args:
            openai_config: A valid OpenAI configuration.
            sample_messages: Sample messages for testing.
        """
        with (
            patch("meta_evaluator.llm_client.openai_client.OpenAI"),
            patch("meta_evaluator.llm_client.openai_client.instructor.from_openai"),
        ):
            client = OpenAIClient(openai_config)
            openai_messages = client._convert_messages_to_openai_format(sample_messages)

        assert len(openai_messages) == 3

        # Check system message
        assert openai_messages[0]["role"] == "system"
        assert openai_messages[0]["content"] == "You are a helpful assistant."

        # Check user message
        assert openai_messages[1]["role"] == "user"
        assert openai_messages[1]["content"] == "Hello, how are you?"

        # Check assistant message
        assert openai_messages[2]["role"] == "assistant"
        assert openai_messages[2]["content"] == "I'm doing well, thank you!"  # type: ignore

    @patch("meta_evaluator.llm_client.openai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.openai_client.OpenAI")
    def test_prompt_without_logprobs(
        self,
        mock_openai_class: Mock,
        mock_instructor: Mock,
        openai_config: OpenAIConfig,
        simple_user_message: list[Message],
        mock_openai_response: Mock,
    ) -> None:
        """Test _prompt method without logprobs.

        Verifies that the _prompt method correctly calls the OpenAI API
        without logprobs and returns the expected response format.

        Args:
            mock_openai_class: Mock for the OpenAI client class.
            mock_instructor: Mock for the instructor function.
            openai_config: A valid OpenAI configuration.
            simple_user_message: A simple user message for testing.
            mock_openai_response: Mock OpenAI API response.
        """
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response

        client = OpenAIClient(openai_config)
        content, usage = client._prompt(
            model="gpt-4o-2024-11-20", messages=simple_user_message, get_logprobs=False
        )

        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-2024-11-20"  # Uses hardcoded model
        assert len(call_kwargs["messages"]) == 1
        assert "logprobs" not in call_kwargs

        # Verify response
        assert content == "Hello! I'm an AI assistant."
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 8
        assert usage.total_tokens == 18

    @patch("meta_evaluator.llm_client.openai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.openai_client.OpenAI")
    def test_prompt_with_logprobs(
        self,
        mock_openai_class: Mock,
        mock_instructor: Mock,
        openai_config: OpenAIConfig,
        simple_user_message: list[Message],
        mock_openai_response: Mock,
    ) -> None:
        """Test _prompt method with logprobs enabled.

        Verifies that the _prompt method correctly calls the OpenAI API
        with logprobs enabled when requested.

        Args:
            mock_openai_class: Mock for the OpenAI client class.
            mock_instructor: Mock for the instructor function.
            openai_config: A valid OpenAI configuration.
            simple_user_message: A simple user message for testing.
            mock_openai_response: Mock OpenAI API response.
        """
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response

        client = OpenAIClient(openai_config)
        content, usage = client._prompt(
            model="gpt-4o-2024-11-20", messages=simple_user_message, get_logprobs=True
        )

        # Verify API call includes logprobs
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["top_logprobs"] == 5

        # Verify response
        assert content == "Hello! I'm an AI assistant."
        assert usage.total_tokens == 18

    @patch("meta_evaluator.llm_client.openai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.openai_client.OpenAI")
    def test_prompt_with_empty_content_raises_error(
        self,
        mock_openai_class: Mock,
        mock_instructor: Mock,
        openai_config: OpenAIConfig,
        simple_user_message: list[Message],
    ) -> None:
        """Test that _prompt raises LLMAPIError when response content is empty.

        Verifies that the method properly handles cases where the OpenAI API
        returns a response with empty or None content.

        Args:
            mock_openai_class: Mock for the OpenAI client class.
            mock_instructor: Mock for the instructor function.
            openai_config: A valid OpenAI configuration.
            simple_user_message: A simple user message for testing.
        """
        # Setup mock with empty content
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 0
        mock_response.usage.total_tokens = 10
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(openai_config)

        with pytest.raises(LLMAPIError) as exc_info:
            client._prompt(
                model="gpt-4o-2024-11-20",
                messages=simple_user_message,
                get_logprobs=False,
            )

        # Verify the error details
        assert "Expected non-empty content" in str(exc_info.value)
        assert exc_info.value.provider == LLMClientEnum.OPENAI
        assert isinstance(exc_info.value.original_error, ValueError)

    @patch("meta_evaluator.llm_client.openai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.openai_client.OpenAI")
    def test_prompt_with_missing_usage_raises_error(
        self,
        mock_openai_class: Mock,
        mock_instructor: Mock,
        openai_config: OpenAIConfig,
        simple_user_message: list[Message],
    ) -> None:
        """Test that _prompt raises LLMAPIError when usage data is missing.

        Verifies that the method properly handles cases where the OpenAI API
        returns a response without usage information.

        Args:
            mock_openai_class: Mock for the OpenAI client class.
            mock_instructor: Mock for the instructor function.
            openai_config: A valid OpenAI configuration.
            simple_user_message: A simple user message for testing.
        """
        # Setup mock with missing usage
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(openai_config)

        with pytest.raises(LLMAPIError) as exc_info:
            client._prompt(
                model="gpt-4o-2024-11-20",
                messages=simple_user_message,
                get_logprobs=False,
            )

        # Verify the error details
        assert "Expected usage data" in str(exc_info.value)
        assert exc_info.value.provider == LLMClientEnum.OPENAI
        assert isinstance(exc_info.value.original_error, ValueError)

    @patch("meta_evaluator.llm_client.openai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.openai_client.OpenAI")
    def test_get_embedding(
        self,
        mock_openai_class: Mock,
        mock_instructor: Mock,
        openai_config: OpenAIConfig,
        mock_embedding_response: Mock,
    ) -> None:
        """Test _get_embedding method.

        Verifies that the _get_embedding method correctly calls the OpenAI
        embeddings API and returns the expected format.

        Args:
            mock_openai_class: Mock for the OpenAI client class.
            mock_instructor: Mock for the instructor function.
            openai_config: A valid OpenAI configuration.
            mock_embedding_response: Mock embedding response.
        """
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.embeddings.create.return_value = mock_embedding_response

        client = OpenAIClient(openai_config)
        embeddings = client._get_embedding(
            text_list=["Hello world"], model="text-embedding-3-large"
        )

        # Verify API call
        mock_client.embeddings.create.assert_called_once_with(
            input=["Hello world"], model="text-embedding-3-large"
        )

        # Verify response
        assert embeddings == [[0.1, 0.2, 0.3, 0.4, 0.5]]

    @patch("meta_evaluator.llm_client.openai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.openai_client.OpenAI")
    def test_prompt_with_structured_response(
        self,
        mock_openai_class: Mock,
        mock_instructor: Mock,
        openai_config: OpenAIConfig,
        structured_output_messages: list[Message],
        mock_structured_response: Mock,
    ) -> None:
        """Test _prompt_with_structured_response method.

        Verifies that the structured response method correctly uses the
        instructor client to get structured outputs.

        Args:
            mock_openai_class: Mock for the OpenAI client class.
            mock_instructor: Mock for the instructor function.
            openai_config: A valid OpenAI configuration.
            structured_output_messages: Messages for structured output testing.
            mock_structured_response: Mock structured response.
        """
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_instructor_client = Mock()
        mock_instructor.return_value = mock_instructor_client

        mock_completion = Mock()
        mock_completion.usage.prompt_tokens = 15
        mock_completion.usage.completion_tokens = 25
        mock_completion.usage.total_tokens = 40

        mock_instructor_client.chat.completions.create_with_completion.return_value = (
            mock_structured_response,
            mock_completion,
        )

        client = OpenAIClient(openai_config)

        response, usage = client._prompt_with_structured_response(
            messages=structured_output_messages,
            response_model=ExampleResponseModel,
            model="gpt-4o-2024-11-20",
        )

        # Verify instructor client was used
        mock_instructor_client.chat.completions.create_with_completion.assert_called_once()
        call_kwargs = mock_instructor_client.chat.completions.create_with_completion.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-2024-11-20"  # Uses hardcoded model
        assert call_kwargs["response_model"] == ExampleResponseModel

        # Verify response
        assert response == mock_structured_response
        assert usage.prompt_tokens == 15
        assert usage.completion_tokens == 25
        assert usage.total_tokens == 40
