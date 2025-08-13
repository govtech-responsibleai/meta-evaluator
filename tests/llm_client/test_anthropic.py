"""Test suite for Anthropic LLM client implementation.

This module contains comprehensive tests for both the AnthropicConfig
and AnthropicClient classes, covering configuration validation,
client initialization, API integration, and error handling scenarios.
All tests use mocked responses to avoid external API dependencies.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from meta_evaluator.llm_client.anthropic_client import AnthropicClient, AnthropicConfig
from meta_evaluator.llm_client.enums import LLMClientEnum
from meta_evaluator.llm_client.exceptions import LLMAPIError
from meta_evaluator.llm_client.models import Message, RoleEnum
from meta_evaluator.llm_client.serialization import AnthropicSerializedState

from .conftest import ExampleResponseModel


class TestAnthropicConfig:
    """Test suite for the AnthropicConfig class.

    This class tests the Anthropic configuration implementation,
    including default values, serialization, and integration
    with the base LLMClientConfig class.
    """

    def test_happy_path_instantiation(
        self, anthropic_config_data: dict[str, Any]
    ) -> None:
        """Test successful creation with all required fields provided.

        Verifies that an AnthropicConfig instance can be created successfully
        when all required fields are provided with valid values.

        Args:
            anthropic_config_data: A dictionary containing valid configuration data.
        """
        config = AnthropicConfig(**anthropic_config_data)

        assert config.api_key == "test-anthropic-key"
        assert config.default_model == "claude-3-5-sonnet-20241022"
        assert config.supports_structured_output is True
        assert config.supports_logprobs is False  # Anthropic doesn't support logprobs

    def test_serialization_excludes_api_key(
        self, anthropic_config: AnthropicConfig
    ) -> None:
        """Test that serialization properly excludes the API key.

        Verifies that the serialize method excludes sensitive information
        like API keys from the serialized state, while preserving all
        other configuration parameters.

        Args:
            anthropic_config: A configured AnthropicConfig instance.
        """
        serialized = anthropic_config.serialize()

        assert isinstance(serialized, AnthropicSerializedState)
        serialized_dict = serialized.model_dump()

        assert "api_key" not in serialized_dict
        assert serialized_dict["default_model"] == anthropic_config.default_model
        assert (
            serialized_dict["supports_structured_output"]
            == anthropic_config.supports_structured_output
        )
        assert (
            serialized_dict["supports_logprobs"] == anthropic_config.supports_logprobs
        )
        assert serialized_dict["client_type"] == LLMClientEnum.ANTHROPIC

    def test_deserialization_reconstructs_config(
        self, anthropic_config: AnthropicConfig
    ) -> None:
        """Test that deserialization correctly reconstructs the configuration.

        Verifies that a configuration can be serialized and then deserialized
        back to an equivalent configuration instance, ensuring data persistence
        and consistency across serialization boundaries.

        Args:
            anthropic_config: A configured AnthropicConfig instance.
        """
        serialized = anthropic_config.serialize()
        new_api_key = "new-test-api-key"

        reconstructed = AnthropicConfig.deserialize(serialized, new_api_key)

        assert reconstructed.api_key == new_api_key
        assert reconstructed.default_model == anthropic_config.default_model
        assert (
            reconstructed.supports_structured_output
            == anthropic_config.supports_structured_output
        )
        assert reconstructed.supports_logprobs == anthropic_config.supports_logprobs

    def test_missing_api_key_raises_validation_error(
        self, anthropic_config_data: dict[str, Any]
    ) -> None:
        """Test that missing API key raises ValidationError.

        Verifies that attempting to create a configuration without providing
        the required API key parameter results in a proper validation error.

        Args:
            anthropic_config_data: Configuration data dictionary.
        """
        del anthropic_config_data["api_key"]

        with pytest.raises(ValidationError) as exc_info:
            AnthropicConfig(**anthropic_config_data)

        assert "api_key" in str(exc_info.value)


class TestAnthropicClient:
    """Test suite for the AnthropicClient class.

    This class tests the Anthropic client implementation, including
    message handling, API integration, error scenarios, and response
    processing. All tests use mocked API responses.
    """

    def test_initialization_with_valid_config(
        self, anthropic_config: AnthropicConfig
    ) -> None:
        """Test that client initializes correctly with valid configuration.

        Verifies that an AnthropicClient can be created successfully
        when provided with a valid configuration object.

        Args:
            anthropic_config: A valid AnthropicConfig instance.
        """
        with (
            patch("meta_evaluator.llm_client.anthropic_client.Anthropic"),
            patch(
                "meta_evaluator.llm_client.anthropic_client.instructor.from_anthropic"
            ),
        ):
            client = AnthropicClient(anthropic_config)

            assert client.config == anthropic_config
            assert client.enum_value == LLMClientEnum.ANTHROPIC

    def test_prompt_success_with_mocked_response(
        self,
        anthropic_client: AnthropicClient,
        simple_user_message: list[Message],
        mock_anthropic_response: Mock,
    ) -> None:
        """Test successful prompt with mocked Anthropic response.

        Verifies that the client can successfully send a prompt and
        process the response when the Anthropic API returns a valid response.

        Args:
            anthropic_client: An AnthropicClient instance.
            simple_user_message: A list containing a simple user message.
            mock_anthropic_response: A mocked Anthropic API response.
        """
        with patch.object(
            anthropic_client._anthropic_client.messages, "create"
        ) as mock_create:
            mock_create.return_value = mock_anthropic_response

            response = anthropic_client.prompt(simple_user_message)

            assert response.content == "Hello! I'm Claude, an AI assistant."
            assert response.usage.prompt_tokens == 10
            assert response.usage.completion_tokens == 15
            assert response.usage.total_tokens == 25
            assert len(response.messages) == 2  # Original + assistant response

    def test_prompt_with_system_message(
        self,
        anthropic_client: AnthropicClient,
        mock_anthropic_response: Mock,
    ) -> None:
        """Test prompt handling with system message.

        Verifies that system messages are properly handled and passed
        to the Anthropic API in the correct format.

        Args:
            anthropic_client: An AnthropicClient instance.
            mock_anthropic_response: A mocked Anthropic API response.
        """
        messages = [
            Message(role=RoleEnum.SYSTEM, content="You are a helpful assistant."),
            Message(role=RoleEnum.USER, content="Hello!"),
        ]

        with patch.object(
            anthropic_client._anthropic_client.messages, "create"
        ) as mock_create:
            mock_create.return_value = mock_anthropic_response

            anthropic_client.prompt(messages)

            mock_create.assert_called_once()
            call_args = mock_create.call_args[1]
            assert call_args["system"] == "You are a helpful assistant."
            assert len(call_args["messages"]) == 1
            assert call_args["messages"][0]["role"] == "user"
            assert call_args["messages"][0]["content"] == "Hello!"

    def test_prompt_with_multiple_system_messages_raises_error(
        self,
        anthropic_client: AnthropicClient,
    ) -> None:
        """Test that multiple system messages raise an error.

        Verifies that providing multiple system messages raises an
        appropriate error since Anthropic only supports one system message.

        Args:
            anthropic_client: An AnthropicClient instance.
        """
        messages = [
            Message(role=RoleEnum.SYSTEM, content="System message 1"),
            Message(role=RoleEnum.SYSTEM, content="System message 2"),
            Message(role=RoleEnum.USER, content="Hello!"),
        ]

        with pytest.raises(LLMAPIError) as exc_info:
            anthropic_client.prompt(messages)

        assert "Multiple system messages are not supported" in str(exc_info.value)

    def test_get_embedding_raises_error(
        self,
        anthropic_client: AnthropicClient,
        single_text_input: list[str],
    ) -> None:
        """Test that get_embedding raises error for Anthropic.

        Verifies that attempting to get embeddings raises an appropriate
        error since Anthropic doesn't provide embedding models.

        Args:
            anthropic_client: An AnthropicClient instance.
            single_text_input: A list with a single text input.
        """
        with pytest.raises(LLMAPIError) as exc_info:
            anthropic_client.get_embedding(single_text_input)

        assert "No embedding model specified" in str(exc_info.value)

    def test_prompt_with_structured_response(
        self,
        anthropic_client: AnthropicClient,
        simple_user_message: list[Message],
        mock_structured_response: ExampleResponseModel,
    ) -> None:
        """Test structured response functionality.

        Verifies that the client can handle structured output requests
        using instructor integration.

        Args:
            anthropic_client: An AnthropicClient instance.
            simple_user_message: A list containing a simple user message.
            mock_structured_response: A mock structured response object.
        """
        with patch.object(
            anthropic_client._instructor_client.messages, "create"
        ) as mock_create:
            mock_create.return_value = mock_structured_response

            structured_response, llm_response = (
                anthropic_client.prompt_with_structured_response(
                    simple_user_message, ExampleResponseModel
                )
            )

            assert isinstance(structured_response, ExampleResponseModel)
            assert structured_response == mock_structured_response
            mock_create.assert_called_once()

    def test_prompt_api_error_handling(
        self,
        anthropic_client: AnthropicClient,
        simple_user_message: list[Message],
    ) -> None:
        """Test API error handling during prompt.

        Verifies that API errors are properly caught and wrapped
        in LLMAPIError exceptions with appropriate context.

        Args:
            anthropic_client: An AnthropicClient instance.
            simple_user_message: A list containing a simple user message.
        """
        with patch.object(
            anthropic_client._anthropic_client.messages, "create"
        ) as mock_create:
            mock_create.side_effect = Exception("API Error")

            with pytest.raises(LLMAPIError) as exc_info:
                anthropic_client.prompt(simple_user_message)

            assert "Failed to get response from Anthropic" in str(exc_info.value)
            assert exc_info.value.provider == LLMClientEnum.ANTHROPIC

    def test_convert_message_unsupported_role(
        self, anthropic_client: AnthropicClient
    ) -> None:
        """Test conversion of unsupported message role.

        Verifies that attempting to convert a message with an unsupported
        role raises an appropriate error.

        Args:
            anthropic_client: An AnthropicClient instance.
        """
        system_message = Message(role=RoleEnum.SYSTEM, content="System prompt")

        with pytest.raises(LLMAPIError) as exc_info:
            anthropic_client._convert_message_to_anthropic_format(system_message)

        assert "Unsupported role" in str(exc_info.value)
        assert "SYSTEM" in str(exc_info.value)

    def test_empty_response_content_handling(
        self,
        anthropic_client: AnthropicClient,
        simple_user_message: list[Message],
    ) -> None:
        """Test handling of empty response content.

        Verifies that the client gracefully handles responses with
        empty or missing content.

        Args:
            anthropic_client: An AnthropicClient instance.
            simple_user_message: A list containing a simple user message.
        """
        mock_response = Mock()
        mock_response.content = []  # Empty content
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 0

        with patch.object(
            anthropic_client._anthropic_client.messages, "create"
        ) as mock_create:
            mock_create.return_value = mock_response

            response = anthropic_client.prompt(simple_user_message)

            assert response.content == ""
            assert response.usage.completion_tokens == 0
