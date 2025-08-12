"""Tests for AsyncAzureOpenAIClient and AsyncAzureOpenAIConfig."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel, ValidationError

from meta_evaluator.llm_client.async_azureopenai_client import (
    AsyncAzureOpenAIClient,
    AsyncAzureOpenAIConfig,
)
from meta_evaluator.llm_client.enums import AsyncLLMClientEnum
from meta_evaluator.llm_client.models import Message, RoleEnum, TagConfig
from meta_evaluator.llm_client.serialization import AzureOpenAISerializedState


class TestModel(BaseModel):
    """Test model for structured response validation."""

    test_field: str


class TestAsyncAzureOpenAIConfig:
    """Test AsyncAzureOpenAIConfig validation and functionality."""

    def test_happy_path_instantiation(self, azure_config_data):
        """Test that AsyncAzureOpenAIConfig can be instantiated with valid data."""
        config = AsyncAzureOpenAIConfig(**azure_config_data)
        assert config.api_key == "test-api-key"
        assert config.endpoint == "https://test.openai.azure.com/"
        assert config.api_version == "2023-12-01-preview"
        assert config.default_model == "gpt-4"
        assert config.default_embedding_model == "text-embedding-ada-002"
        assert config.supports_structured_output is True
        assert config.supports_logprobs is True

    def test_serialization_excludes_api_key(self, async_azure_config):
        """Test that serialization properly excludes the API key."""
        serialized = async_azure_config.serialize()

        assert isinstance(serialized, AzureOpenAISerializedState)
        serialized_dict = serialized.model_dump()

        assert "api_key" not in serialized_dict
        assert serialized_dict["endpoint"] == async_azure_config.endpoint
        assert serialized_dict["api_version"] == async_azure_config.api_version
        assert serialized_dict["default_model"] == async_azure_config.default_model
        assert (
            serialized_dict["default_embedding_model"]
            == async_azure_config.default_embedding_model
        )
        assert (
            serialized_dict["supports_structured_output"]
            == async_azure_config.supports_structured_output
        )
        assert (
            serialized_dict["supports_logprobs"] == async_azure_config.supports_logprobs
        )
        assert serialized_dict["supports_instructor"] is True

    def test_deserialization_with_api_key(self):
        """Test that deserialization works with API key."""
        state = AzureOpenAISerializedState(
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            default_model="gpt-4",
            default_embedding_model="text-embedding-ada-002",
            supports_structured_output=True,
            supports_logprobs=False,
            supports_instructor=True,
        )

        config = AsyncAzureOpenAIConfig.deserialize(state, api_key="new-api-key")

        assert config.api_key == "new-api-key"
        assert config.endpoint == "https://test.openai.azure.com"
        assert config.api_version == "2024-02-15-preview"
        assert config.default_model == "gpt-4"
        assert config.default_embedding_model == "text-embedding-ada-002"
        assert config.supports_structured_output is True
        assert config.supports_logprobs is False

    def test_validation_requires_api_key(self, azure_config_data):
        """Test that api_key is required during instantiation."""
        config_data = azure_config_data.copy()
        del config_data["api_key"]

        with pytest.raises(ValidationError):
            AsyncAzureOpenAIConfig(**config_data)


class TestAsyncAzureOpenAIClient:
    """Test AsyncAzureOpenAIClient functionality."""

    def test_client_initialization(self, async_azure_client):
        """Test that AsyncAzureOpenAIClient initializes correctly."""
        assert isinstance(async_azure_client, AsyncAzureOpenAIClient)
        assert async_azure_client.enum_value == AsyncLLMClientEnum.AZURE_OPENAI
        assert isinstance(async_azure_client.config, AsyncAzureOpenAIConfig)

    @patch("meta_evaluator.llm_client.async_azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.async_azureopenai_client.AsyncAzureOpenAI")
    @pytest.mark.asyncio
    async def test_prompt_with_structured_response(
        self,
        mock_async_azure_openai,
        mock_instructor,
        async_azure_config,
        simple_user_message,
    ):
        """Test async prompt with structured response."""
        # Mock instructor response - returns tuple (response, completion)
        mock_instructor_client = mock_instructor.return_value
        test_response = TestModel(test_field="test_value")

        # Mock the completion object
        mock_completion = Mock()
        mock_completion.usage.prompt_tokens = 15
        mock_completion.usage.completion_tokens = 10
        mock_completion.usage.total_tokens = 25

        # Set up the async mock to return the tuple
        mock_instructor_client.chat.completions.create_with_completion = AsyncMock(
            return_value=(test_response, mock_completion)
        )

        # Create client and test the private method
        client = AsyncAzureOpenAIClient(async_azure_config)
        (
            structured_response,
            llm_response,
        ) = await client._prompt_with_structured_response(
            messages=simple_user_message, response_model=TestModel, model="gpt-4"
        )

        # Verify instructor was called
        mock_instructor_client.chat.completions.create_with_completion.assert_called_once()

        # Verify response and usage
        assert structured_response.test_field == "test_value"
        assert llm_response.prompt_tokens == 15
        assert llm_response.completion_tokens == 10
        assert llm_response.total_tokens == 25

    @patch("meta_evaluator.llm_client.async_azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.async_azureopenai_client.AsyncAzureOpenAI")
    @pytest.mark.asyncio
    async def test_prompt_with_xml_tags(
        self,
        mock_async_azure_openai,
        mock_instructor,
        async_azure_config,
        simple_user_message,
    ):
        """Test async prompt with XML tags."""
        # Mock response with XML content
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "<sentiment>positive</sentiment>"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 20

        mock_client = mock_async_azure_openai.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        tag_configs = [
            TagConfig(
                name="sentiment",
                allowed_values=["positive", "negative", "neutral"],
                cardinality="one",
            )
        ]

        client = AsyncAzureOpenAIClient(async_azure_config)
        parse_result, llm_response = await client.prompt_with_xml_tags(
            messages=simple_user_message, tag_configs=tag_configs
        )

        assert parse_result.data["sentiment"] == "positive"
        assert llm_response.usage.total_tokens == 20

    @patch("meta_evaluator.llm_client.async_azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.async_azureopenai_client.AsyncAzureOpenAI")
    @pytest.mark.asyncio
    async def test_get_embedding_batch(
        self,
        mock_async_azure_openai,
        mock_instructor,
        async_azure_config,
        mock_embedding_response,
        batch_text_lists,
    ):
        """Test async batch embedding processing."""
        mock_client = mock_async_azure_openai.return_value
        mock_client.embeddings.create = AsyncMock(return_value=mock_embedding_response)

        client = AsyncAzureOpenAIClient(async_azure_config)
        embeddings = await client.get_embedding_batch(batch_text_lists)

        assert len(embeddings) == len(batch_text_lists)
        # Should be called once for each text list in the batch
        assert mock_client.embeddings.create.call_count == len(batch_text_lists)

    # Batch processing tests
    @patch("meta_evaluator.llm_client.async_azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.async_azureopenai_client.AsyncAzureOpenAI")
    @pytest.mark.asyncio
    async def test_prompt_batch(
        self,
        mock_async_azure_openai,
        mock_instructor,
        async_azure_config,
        batch_message_lists,
        mock_azure_response,
    ):
        """Test async batch prompt processing."""
        mock_client = mock_async_azure_openai.return_value
        mock_client.chat.completions.create = AsyncMock(
            return_value=mock_azure_response
        )

        client = AsyncAzureOpenAIClient(async_azure_config)
        responses = await client.prompt_batch(batch_message_lists)

        assert len(responses) == len(batch_message_lists)
        assert all(
            response.content == "Hello! I'm an AI assistant." for response in responses
        )
        # Should be called once for each message list in the batch
        assert mock_client.chat.completions.create.call_count == len(
            batch_message_lists
        )

    @patch("meta_evaluator.llm_client.async_azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.async_azureopenai_client.AsyncAzureOpenAI")
    @pytest.mark.asyncio
    async def test_prompt_with_structured_response_batch(
        self,
        mock_async_azure_openai,
        mock_instructor,
        async_azure_config,
        batch_structured_items,
    ):
        """Test async batch structured response processing."""
        # Mock instructor response - returns tuple (response, completion)
        mock_instructor_client = mock_instructor.return_value
        test_response = TestModel(test_field="test_value")

        # Mock the completion object
        mock_completion = Mock()
        mock_completion.usage.prompt_tokens = 15
        mock_completion.usage.completion_tokens = 10
        mock_completion.usage.total_tokens = 25

        # Set up the async mock to return the tuple
        mock_instructor_client.chat.completions.create_with_completion = AsyncMock(
            return_value=(test_response, mock_completion)
        )

        client = AsyncAzureOpenAIClient(async_azure_config)
        results = await client.prompt_with_structured_response_batch(
            batch_structured_items
        )

        assert len(results) == len(batch_structured_items)
        for structured_response, llm_response in results:
            assert structured_response.test_field == "test_value"
            assert llm_response.usage.prompt_tokens == 15
        # Should be called once for each item in the batch
        assert (
            mock_instructor_client.chat.completions.create_with_completion.call_count
            == len(batch_structured_items)
        )

    @patch("meta_evaluator.llm_client.async_azureopenai_client.instructor.from_openai")
    @patch("meta_evaluator.llm_client.async_azureopenai_client.AsyncAzureOpenAI")
    @pytest.mark.asyncio
    async def test_prompt_with_xml_tags_batch(
        self, mock_async_azure_openai, mock_instructor, async_azure_config
    ):
        """Test async batch XML tag processing."""
        # Mock response with XML content
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "<sentiment>positive</sentiment>"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 20

        mock_client = mock_async_azure_openai.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        tag_configs = [
            TagConfig(
                name="sentiment",
                allowed_values=["positive", "negative", "neutral"],
                cardinality="one",
            )
        ]

        batch_items = [
            ([Message(role=RoleEnum.USER, content="Great!")], tag_configs),
            ([Message(role=RoleEnum.USER, content="Terrible!")], tag_configs),
        ]

        client = AsyncAzureOpenAIClient(async_azure_config)
        results = await client.prompt_with_xml_tags_batch(batch_items)

        assert len(results) == len(batch_items)
        for parse_result, llm_response in results:
            assert parse_result.data["sentiment"] == "positive"
        # Should be called once for each item in the batch
        assert mock_client.chat.completions.create.call_count == len(batch_items)
