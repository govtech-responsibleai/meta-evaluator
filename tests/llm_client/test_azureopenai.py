"""Test suite for Azure OpenAI LLM client implementation.

This module contains comprehensive tests for both the AzureOpenAIConfig
and AzureOpenAIClient classes, covering configuration validation,
client initialization, API integration, and error handling scenarios.
All tests use mocked responses to avoid external API dependencies.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from meta_evaluator.llm_client.azureopenai_client import (
    AzureOpenAIClient,
    AzureOpenAIConfig,
)
from meta_evaluator.llm_client.enums import LLMClientEnum, RoleEnum
from meta_evaluator.llm_client.exceptions import LLMAPIError
from meta_evaluator.llm_client.models import LLMUsage, Message


class TestAzureOpenAIConfig:
    """Test suite for the AzureOpenAIConfig class.

    This class tests the Azure OpenAI configuration implementation,
    including required field validation, default values, and integration
    with the base LLMClientConfig class.
    """

    def test_happy_path_instantiation(self, azure_config_data: dict[str, Any]) -> None:
        """Test successful creation with all required fields provided.

        Verifies that an AzureOpenAIConfig instance can be created successfully
        when all required fields are provided with valid values.

        Args:
            azure_config_data: A dictionary containing valid configuration data.
        """
        config = AzureOpenAIConfig(**azure_config_data)

        assert config.api_key == "test-api-key"
        assert config.endpoint == "https://test.openai.azure.com/"
        assert config.api_version == "2023-12-01-preview"
        assert config.default_model == "gpt-4"
        assert config.default_embedding_model == "text-embedding-ada-002"
        assert config.supports_structured_output is True
        assert config.supports_logprobs is True

    def test_default_boolean_values(self, azure_config_data: dict[str, Any]) -> None:
        """Test that Azure-specific capabilities have correct default values.

        Verifies that the supports_structured_output and supports_logprobs
        fields are set to their expected default values for Azure OpenAI.

        Args:
            azure_config_data: A dictionary containing valid configuration data.
        """
        config = AzureOpenAIConfig(**azure_config_data)

        assert config.supports_structured_output is True
        assert config.supports_logprobs is True

    def test_prevent_instantiation_method_exists(
        self, azure_config: AzureOpenAIConfig
    ) -> None:
        """Test that _prevent_instantiation method can be called without error.

        Verifies that the _prevent_instantiation method required by the abstract
        base class is implemented and can be called without raising exceptions.

        Args:
            azure_config: A valid AzureOpenAIConfig instance.

        """
        # Should not raise an exception
        azure_config._prevent_instantiation()

    def test_missing_endpoint_raises_validation_error(self) -> None:
        """Test that missing endpoint field raises ValidationError.

        Verifies that attempting to create an AzureOpenAIConfig instance
        without providing the required endpoint field results in a ValidationError.
        """
        with pytest.raises(ValidationError):
            AzureOpenAIConfig(  # type: ignore
                api_key="test-key",
                api_version="2023-12-01-preview",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_missing_api_version_raises_validation_error(self) -> None:
        """Test that missing api_version field raises ValidationError.

        Verifies that attempting to create an AzureOpenAIConfig instance
        without providing the required api_version field results in a ValidationError.
        """
        with pytest.raises(ValidationError):
            AzureOpenAIConfig(  # type: ignore
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_missing_api_key_raises_validation_error(self) -> None:
        """Test that missing api_key field raises ValidationError.

        Verifies that attempting to create an AzureOpenAIConfig instance
        without providing the required api_key field (inherited from base class)
        results in a ValidationError.
        """
        with pytest.raises(ValidationError):
            AzureOpenAIConfig(  # type: ignore
                endpoint="https://test.openai.azure.com/",
                api_version="2023-12-01-preview",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_missing_default_model_raises_validation_error(self) -> None:
        """Test that missing default_model field raises ValidationError.

        Verifies that attempting to create an AzureOpenAIConfig instance
        without providing the required default_model field (inherited from base class)
        results in a ValidationError.

        """
        with pytest.raises(ValidationError):
            AzureOpenAIConfig(  # type: ignore
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
                api_version="2023-12-01-preview",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_missing_default_embedding_model_raises_validation_error(self) -> None:
        """Test that missing default_embedding_model field raises ValidationError.

        Verifies that attempting to create an AzureOpenAIConfig instance
        without providing the required default_embedding_model field
        (inherited from base class) results in a ValidationError.
        """
        with pytest.raises(ValidationError):
            AzureOpenAIConfig(  # type: ignore
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
                api_version="2023-12-01-preview",
                default_model="gpt-4",
            )

    def test_empty_strings_are_allowed(self) -> None:
        """Test that empty strings are accepted for all string fields.

        Verifies that the configuration class accepts empty strings for all
        string fields, following the design decision to allow empty values
        and let validation occur at API call time.
        """
        config = AzureOpenAIConfig(
            api_key="",
            endpoint="",
            api_version="",
            default_model="",
            default_embedding_model="",
        )

        assert config.api_key == ""
        assert config.endpoint == ""
        assert config.api_version == ""
        assert config.default_model == ""
        assert config.default_embedding_model == ""

    def test_whitespace_strings_are_preserved(self) -> None:
        """Test that whitespace-only strings are preserved exactly as provided.

        Verifies that strings containing only whitespace characters are
        stored without modification, maintaining the principle that the
        configuration class acts as a simple data holder.
        """
        config = AzureOpenAIConfig(
            api_key="   ",
            endpoint="  \t  ",
            api_version=" \n ",
            default_model="    ",
            default_embedding_model=" ",
        )

        assert config.api_key == "   "
        assert config.endpoint == "  \t  "
        assert config.api_version == " \n "
        assert config.default_model == "    "
        assert config.default_embedding_model == " "

    def test_model_dump_serialization(self, azure_config: AzureOpenAIConfig) -> None:
        """Test that configuration can be serialized to a dictionary.

        Verifies that the Pydantic model_dump method correctly serializes
        the configuration instance to a dictionary containing all field values.

        Args:
            azure_config: A valid AzureOpenAIConfig instance.

        """
        config_dict = azure_config.model_dump()

        assert config_dict["api_key"] == "test-api-key"
        assert config_dict["endpoint"] == "https://test.openai.azure.com/"
        assert config_dict["api_version"] == "2023-12-01-preview"
        assert config_dict["default_model"] == "gpt-4"
        assert config_dict["default_embedding_model"] == "text-embedding-ada-002"
        assert config_dict["supports_structured_output"] is True
        assert config_dict["supports_logprobs"] is True

    def test_reconstruction_from_dict(self, azure_config: AzureOpenAIConfig) -> None:
        """Test that configuration can be reconstructed from a dictionary.

        Verifies that an AzureOpenAIConfig instance can be created from
        a dictionary produced by model_dump, ensuring round-trip serialization
        works correctly.

        Args:
            azure_config: A valid AzureOpenAIConfig instance.

        """
        config_dict = azure_config.model_dump()
        reconstructed = AzureOpenAIConfig(**config_dict)

        assert reconstructed.api_key == azure_config.api_key
        assert reconstructed.endpoint == azure_config.endpoint
        assert reconstructed.api_version == azure_config.api_version
        assert reconstructed.default_model == azure_config.default_model
        assert (
            reconstructed.default_embedding_model
            == azure_config.default_embedding_model
        )
        assert (
            reconstructed.supports_structured_output
            == azure_config.supports_structured_output
        )
        assert reconstructed.supports_logprobs == azure_config.supports_logprobs

    def test_boolean_fields_can_be_overridden(self) -> None:
        """Test that default boolean values can be explicitly overridden.

        Verifies that the supports_structured_output and supports_logprobs
        fields can be set to non-default values when explicitly provided
        during instantiation.

        """
        config = AzureOpenAIConfig(
            api_key="test-key",
            endpoint="https://test.openai.azure.com/",
            api_version="2023-12-01-preview",
            default_model="gpt-4",
            default_embedding_model="text-embedding-ada-002",
            supports_structured_output=False,
            supports_logprobs=False,
        )

        assert config.supports_structured_output is False
        assert config.supports_logprobs is False

    def test_inheritance_from_base_class(self, azure_config: AzureOpenAIConfig) -> None:
        """Test that AzureOpenAIConfig properly inherits from LLMClientConfig.

        Verifies that the AzureOpenAIConfig class correctly inherits all
        functionality from the LLMClientConfig base class.

        Args:
            azure_config: A valid AzureOpenAIConfig instance.

        """
        from meta_evaluator.llm_client import LLMClientConfig

        assert isinstance(azure_config, LLMClientConfig)

        # Test that base class fields are accessible
        assert hasattr(azure_config, "api_key")
        assert hasattr(azure_config, "supports_structured_output")
        assert hasattr(azure_config, "default_model")
        assert hasattr(azure_config, "default_embedding_model")

        # Test that Azure-specific fields are accessible
        assert hasattr(azure_config, "endpoint")
        assert hasattr(azure_config, "api_version")


class TestAzureOpenAIClient:
    """Test suite for the AzureOpenAIClient class.

    This class tests the Azure OpenAI client implementation, including
    initialization, message conversion, API integration, and error handling.
    All tests use mocked responses to avoid external API dependencies.
    """

    @pytest.fixture
    def azure_config(self) -> AzureOpenAIConfig:
        """Provide a valid AzureOpenAIConfig instance for client testing.

        Returns:
            AzureOpenAIConfig: A valid configuration instance for creating test clients.
        """
        return AzureOpenAIConfig(
            api_key="test-api-key",
            endpoint="https://test.openai.azure.com/",
            api_version="2023-12-01-preview",
            default_model="gpt-4",
            default_embedding_model="text-embedding-ada-002",
        )

    @pytest.fixture
    def sample_messages(self) -> list[Message]:
        """Provide sample messages for testing conversation flows.

        Returns:
            list[Message]: A list of sample messages representing a conversation.
        """
        return [
            Message(role=RoleEnum.SYSTEM, content="You are a helpful assistant."),
            Message(role=RoleEnum.USER, content="Hello, how are you?"),
            Message(role=RoleEnum.ASSISTANT, content="I'm doing well, thank you!"),
        ]

    @pytest.fixture
    def mock_azure_response(self) -> Mock:
        """Provide a mock Azure OpenAI API response for testing.

        Returns:
            Mock: A mock response object with typical Azure OpenAI structure.
        """
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello! I'm an AI assistant."
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 8
        mock_response.usage.total_tokens = 18
        return mock_response

    @pytest.fixture
    def mock_embedding_response(self) -> Mock:
        """Provide a mock Azure OpenAI embedding response for testing.

        Returns:
            Mock: A mock embedding response with typical structure.
        """
        mock_response = Mock()
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_response.data = [mock_embedding]
        return mock_response

    @patch("meta_evaluator.llm_client.azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.azureopenai_client.AzureOpenAI")
    def test_client_initialization(
        self,
        mock_azure_openai: Mock,
        mock_instructor: Mock,
        azure_config: AzureOpenAIConfig,
    ) -> None:
        """Test that AzureOpenAIClient initializes correctly with proper configuration.

        Verifies that the client properly initializes the Azure OpenAI client
        and instructor client with the correct parameters from the configuration.

        Args:
            mock_azure_openai: Mock for the AzureOpenAI client constructor.
            mock_instructor: Mock for the instructor.from_openai function.
            azure_config: A valid configuration instance.

        """
        client = AzureOpenAIClient(azure_config)

        # Verify Azure OpenAI client was initialized with correct parameters
        mock_azure_openai.assert_called_once_with(
            api_key="test-api-key",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2023-12-01-preview",
        )

        # Verify instructor client was initialized
        mock_instructor.assert_called_once_with(mock_azure_openai.return_value)

        # Verify configuration is stored
        assert client.config == azure_config
        assert isinstance(client.config, AzureOpenAIConfig)

    def test_enum_value_property(self, azure_config: AzureOpenAIConfig) -> None:
        """Test that enum_value property returns correct LLMClientEnum value.

        Verifies that the client correctly identifies itself as Azure OpenAI
        through the enum_value property.

        Args:
            azure_config: A valid configuration instance.

        """
        with (
            patch("meta_evaluator.llm_client.azureopenai_client.AzureOpenAI"),
            patch(
                "meta_evaluator.llm_client.azureopenai_client.instructor.from_openai"
            ),
        ):
            client = AzureOpenAIClient(azure_config)
            assert client.enum_value == LLMClientEnum.AZURE_OPENAI

    def test_convert_messages_to_openai_format(
        self, azure_config: AzureOpenAIConfig, sample_messages: list[Message]
    ) -> None:
        """Test message conversion to OpenAI format.

        Verifies that internal Message objects are correctly converted to
        the format expected by the OpenAI API.

        Args:
            azure_config: A valid configuration instance.
            sample_messages: Sample messages for testing conversion.

        """
        with (
            patch("meta_evaluator.llm_client.azureopenai_client.AzureOpenAI"),
            patch(
                "meta_evaluator.llm_client.azureopenai_client.instructor.from_openai"
            ),
        ):
            client = AzureOpenAIClient(azure_config)

            converted = client._convert_messages_to_openai_format(sample_messages)

            assert len(converted) == 3
            assert converted[0]["role"] == "system"
            assert converted[0]["content"] == "You are a helpful assistant."
            assert converted[1]["role"] == "user"
            assert converted[1]["content"] == "Hello, how are you?"
            assert converted[2]["role"] == "assistant"
            assert converted[2]["content"] == "I'm doing well, thank you!"  # type: ignore

    def test_convert_empty_messages_list(self, azure_config: AzureOpenAIConfig) -> None:
        """Test message conversion with empty messages list.

        Verifies that empty message lists are handled correctly during conversion.

        Args:
            azure_config: A valid configuration instance.

        """
        with (
            patch("meta_evaluator.llm_client.azureopenai_client.AzureOpenAI"),
            patch(
                "meta_evaluator.llm_client.azureopenai_client.instructor.from_openai"
            ),
        ):
            client = AzureOpenAIClient(azure_config)

            converted = client._convert_messages_to_openai_format([])
            assert converted == []

    def test_convert_single_message_each_role(
        self, azure_config: AzureOpenAIConfig
    ) -> None:
        """Test message conversion for each individual role type.

        Verifies that each role type (USER, SYSTEM, ASSISTANT) is converted
        correctly when appearing as a single message.

        Args:
            azure_config: A valid configuration instance.

        """
        with (
            patch("meta_evaluator.llm_client.azureopenai_client.AzureOpenAI"),
            patch(
                "meta_evaluator.llm_client.azureopenai_client.instructor.from_openai"
            ),
        ):
            client = AzureOpenAIClient(azure_config)

            # Test USER role
            user_message = [Message(role=RoleEnum.USER, content="Test user message")]
            converted = client._convert_messages_to_openai_format(user_message)
            assert converted[0]["role"] == "user"
            assert converted[0]["content"] == "Test user message"

            # Test SYSTEM role
            system_message = [
                Message(role=RoleEnum.SYSTEM, content="Test system message")
            ]
            converted = client._convert_messages_to_openai_format(system_message)
            assert converted[0]["role"] == "system"
            assert converted[0]["content"] == "Test system message"

            # Test ASSISTANT role
            assistant_message = [
                Message(role=RoleEnum.ASSISTANT, content="Test assistant message")
            ]
            converted = client._convert_messages_to_openai_format(assistant_message)
            assert converted[0]["role"] == "assistant"
            assert converted[0]["content"] == "Test assistant message"  # type: ignore

    @patch("meta_evaluator.llm_client.azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.azureopenai_client.AzureOpenAI")
    def test_prompt_without_logprobs(
        self,
        mock_azure_openai: Mock,
        mock_instructor: Mock,
        azure_config: AzureOpenAIConfig,
        sample_messages: list[Message],
        mock_azure_response: Mock,
    ) -> None:
        """Test _prompt method without logprobs parameter.

        Verifies that the _prompt method correctly calls the Azure OpenAI API
        without logprobs and properly parses the response.

        Args:
            mock_azure_openai: Mock for the AzureOpenAI client constructor.
            mock_instructor: Mock for the instructor.from_openai function.
            azure_config: A valid configuration instance.
            sample_messages: Sample messages for testing.
            mock_azure_response: Mock response from Azure OpenAI API.

        """
        mock_client = mock_azure_openai.return_value
        mock_client.chat.completions.create.return_value = mock_azure_response

        client = AzureOpenAIClient(azure_config)
        content, usage = client._prompt("gpt-4", sample_messages, get_logprobs=False)

        # Verify API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-4"
        assert len(call_args[1]["messages"]) == 3
        assert "logprobs" not in call_args[1]

        # Verify response parsing
        assert content == "Hello! I'm an AI assistant."
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 8
        assert usage.total_tokens == 18

    @patch("meta_evaluator.llm_client.azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.azureopenai_client.AzureOpenAI")
    def test_prompt_with_logprobs(
        self,
        mock_azure_openai: Mock,
        mock_instructor: Mock,
        azure_config: AzureOpenAIConfig,
        sample_messages: list[Message],
        mock_azure_response: Mock,
    ) -> None:
        """Test _prompt method with logprobs parameter enabled.

        Verifies that the _prompt method correctly includes logprobs parameters
        when requested and the configuration supports it.

        Args:
            mock_azure_openai: Mock for the AzureOpenAI client constructor.
            mock_instructor: Mock for the instructor.from_openai function.
            azure_config: A valid configuration instance.
            sample_messages: Sample messages for testing.
            mock_azure_response: Mock response from Azure OpenAI API.

        """
        mock_client = mock_azure_openai.return_value
        mock_client.chat.completions.create.return_value = mock_azure_response

        client = AzureOpenAIClient(azure_config)
        content, usage = client._prompt("gpt-4", sample_messages, get_logprobs=True)

        # Verify API was called with logprobs
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["logprobs"] is True
        assert call_args[1]["top_logprobs"] == 5

        # Verify response parsing still works
        assert content == "Hello! I'm an AI assistant."
        assert isinstance(usage, LLMUsage)

    @patch("meta_evaluator.llm_client.azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.azureopenai_client.AzureOpenAI")
    def test_prompt_with_empty_content_raises_error(
        self,
        mock_azure_openai: Mock,
        mock_instructor: Mock,
        azure_config: AzureOpenAIConfig,
        sample_messages: list[Message],
    ) -> None:
        """Test _prompt method raises ValueError for empty content response.

        Verifies that the client properly handles API responses with empty
        or None content by raising an appropriate error.

        Args:
            mock_azure_openai: Mock for the AzureOpenAI client constructor.
            mock_instructor: Mock for the instructor.from_openai function.
            azure_config: A valid configuration instance.
            sample_messages: Sample messages for testing.

        """
        mock_client = mock_azure_openai.return_value
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response

        client = AzureOpenAIClient(azure_config)

        with pytest.raises(LLMAPIError, match="Expected non-empty content"):
            client._prompt("gpt-4", sample_messages, get_logprobs=False)

    @patch("meta_evaluator.llm_client.azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.azureopenai_client.AzureOpenAI")
    def test_prompt_with_missing_usage_raises_error(
        self,
        mock_azure_openai: Mock,
        mock_instructor: Mock,
        azure_config: AzureOpenAIConfig,
        sample_messages: list[Message],
    ) -> None:
        """Test _prompt method raises ValueError for missing usage data.

        Verifies that the client properly handles API responses with missing
        usage information by raising an appropriate error.

        Args:
            mock_azure_openai: Mock for the AzureOpenAI client constructor.
            mock_instructor: Mock for the instructor.from_openai function.
            azure_config: A valid configuration instance.
            sample_messages: Sample messages for testing.

        """
        mock_client = mock_azure_openai.return_value
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Valid content"
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        client = AzureOpenAIClient(azure_config)

        with pytest.raises(LLMAPIError, match="Expected usage data"):
            client._prompt("gpt-4", sample_messages, get_logprobs=False)

    @patch("meta_evaluator.llm_client.azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.azureopenai_client.AzureOpenAI")
    def test_prompt_with_structured_response_success(
        self,
        mock_azure_openai: Mock,
        mock_instructor: Mock,
        azure_config: AzureOpenAIConfig,
        sample_messages: list[Message],
    ) -> None:
        """Test _prompt_with_structured_response with successful API calls."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            test_field: str

        # Mock instructor response - now returns tuple (response, completion)
        mock_instructor_client = mock_instructor.return_value
        test_response = TestModel(test_field="test_value")

        # Mock the completion object
        mock_completion = Mock()
        mock_completion.usage.prompt_tokens = 15
        mock_completion.usage.completion_tokens = 10
        mock_completion.usage.total_tokens = 25

        # Set up the mock to return the tuple
        mock_instructor_client.chat.completions.create_with_completion.return_value = (
            test_response,
            mock_completion,
        )

        client = AzureOpenAIClient(azure_config)
        response, usage = client._prompt_with_structured_response(
            sample_messages, TestModel, "gpt-4"
        )

        # Verify instructor was called with the new method
        mock_instructor_client.chat.completions.create_with_completion.assert_called_once()

        # Remove the old assertion for the regular client call since we don't make that call anymore

        # Verify response and usage
        assert response.test_field == "test_value"
        assert usage.prompt_tokens == 15
        assert usage.completion_tokens == 10
        assert usage.total_tokens == 25

    @patch("meta_evaluator.llm_client.azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.azureopenai_client.AzureOpenAI")
    def test_get_embedding_success(
        self,
        mock_azure_openai: Mock,
        mock_instructor: Mock,
        azure_config: AzureOpenAIConfig,
        mock_embedding_response: Mock,
    ) -> None:
        """Test _get_embedding method with successful API response.

        Verifies that embedding generation works correctly with proper
        API calls and response parsing.

        Args:
            mock_azure_openai: Mock for the AzureOpenAI client constructor.
            mock_instructor: Mock for the instructor.from_openai function.
            azure_config: A valid configuration instance.
            mock_embedding_response: Mock embedding response from Azure OpenAI API.

        """
        mock_client = mock_azure_openai.return_value
        mock_client.embeddings.create.return_value = mock_embedding_response

        client = AzureOpenAIClient(azure_config)
        embeddings = client._get_embedding(["Hello world"], "text-embedding-ada-002")

        # Verify API was called correctly
        mock_client.embeddings.create.assert_called_once_with(
            input=["Hello world"], model="text-embedding-ada-002"
        )

        # Verify response parsing
        assert embeddings == [[0.1, 0.2, 0.3, 0.4, 0.5]]

    @patch("meta_evaluator.llm_client.azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.azureopenai_client.AzureOpenAI")
    def test_get_embedding_multiple_texts(
        self,
        mock_azure_openai: Mock,
        mock_instructor: Mock,
        azure_config: AzureOpenAIConfig,
    ) -> None:
        """Test _get_embedding method with multiple text inputs.

        Verifies that the embedding method correctly handles multiple text
        inputs and returns the appropriate number of embeddings.

        Args:
            mock_azure_openai: Mock for the AzureOpenAI client constructor.
            mock_instructor: Mock for the instructor.from_openai function.
            azure_config: A valid configuration instance.

        """
        mock_client = mock_azure_openai.return_value

        # Mock response with multiple embeddings
        mock_response = Mock()
        mock_embedding1 = Mock()
        mock_embedding1.embedding = [0.1, 0.2, 0.3]
        mock_embedding2 = Mock()
        mock_embedding2.embedding = [0.4, 0.5, 0.6]
        mock_response.data = [mock_embedding1, mock_embedding2]
        mock_client.embeddings.create.return_value = mock_response

        client = AzureOpenAIClient(azure_config)
        embeddings = client._get_embedding(
            ["First text", "Second text"], "text-embedding-ada-002"
        )

        # Verify correct number of embeddings returned
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

    @patch("meta_evaluator.llm_client.azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.azureopenai_client.AzureOpenAI")
    def test_api_error_handling(
        self,
        mock_azure_openai: Mock,
        mock_instructor: Mock,
        azure_config: AzureOpenAIConfig,
        sample_messages: list[Message],
    ) -> None:
        """Test that API errors are properly bubbled up for base class handling.

        Verifies that exceptions from the Azure OpenAI API are allowed to
        propagate so the base class can wrap them in LLMAPIError.

        Args:
            mock_azure_openai: Mock for the AzureOpenAI client constructor.
            mock_instructor: Mock for the instructor.from_openai function.
            azure_config: A valid configuration instance.
            sample_messages: Sample messages for testing.

        """
        mock_client = mock_azure_openai.return_value
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        client = AzureOpenAIClient(azure_config)

        with pytest.raises(Exception, match="API Error"):
            client._prompt("gpt-4", sample_messages, get_logprobs=False)

    @patch("meta_evaluator.llm_client.azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.azureopenai_client.AzureOpenAI")
    def test_embedding_api_error_handling(
        self,
        mock_azure_openai: Mock,
        mock_instructor: Mock,
        azure_config: AzureOpenAIConfig,
    ) -> None:
        """Test that embedding API errors are properly bubbled up.

        Verifies that exceptions from the Azure OpenAI embedding API are
        allowed to propagate for proper error handling by the base class.

        Args:
            mock_azure_openai: Mock for the AzureOpenAI client constructor.
            mock_instructor: Mock for the instructor.from_openai function.
            azure_config: A valid configuration instance.

        """
        mock_client = mock_azure_openai.return_value
        mock_client.embeddings.create.side_effect = Exception("Embedding API Error")

        client = AzureOpenAIClient(azure_config)

        with pytest.raises(Exception, match="Embedding API Error"):
            client._get_embedding(["Test text"], "text-embedding-ada-002")
