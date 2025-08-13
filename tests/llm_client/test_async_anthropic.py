"""Tests for AsyncAnthropicClient and AsyncAnthropicConfig."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from meta_evaluator.llm_client.async_anthropic_client import (
    AsyncAnthropicClient,
    AsyncAnthropicConfig,
)
from meta_evaluator.llm_client.enums import AsyncLLMClientEnum
from meta_evaluator.llm_client.exceptions import LLMAPIError
from meta_evaluator.llm_client.models import Message, RoleEnum, TagConfig
from meta_evaluator.llm_client.serialization import AnthropicSerializedState

from .conftest import ExampleResponseModel


class TestModel(BaseModel):
    """Test model for structured response validation."""

    test_field: str


class TestAsyncAnthropicConfig:
    """Test AsyncAnthropicConfig validation and functionality."""

    def test_happy_path_instantiation(self, anthropic_config_data):
        """Test that AsyncAnthropicConfig can be instantiated with valid data."""
        config = AsyncAnthropicConfig(**anthropic_config_data)
        assert config.api_key == "test-anthropic-key"
        assert config.default_model == "claude-3-5-sonnet-20241022"
        assert config.supports_structured_output is True
        assert config.supports_logprobs is False  # Anthropic doesn't support logprobs

    def test_serialization_excludes_api_key(self, async_anthropic_config):
        """Test that serialization properly excludes the API key."""
        serialized = async_anthropic_config.serialize()

        assert isinstance(serialized, AnthropicSerializedState)
        serialized_dict = serialized.model_dump()

        assert "api_key" not in serialized_dict
        assert serialized_dict["default_model"] == async_anthropic_config.default_model
        assert (
            serialized_dict["supports_structured_output"]
            == async_anthropic_config.supports_structured_output
        )
        assert (
            serialized_dict["supports_logprobs"]
            == async_anthropic_config.supports_logprobs
        )

    def test_deserialization_reconstructs_config(self, async_anthropic_config):
        """Test deserialization with API key injection."""
        serialized = async_anthropic_config.serialize()
        new_api_key = "new-test-key"

        reconstructed = AsyncAnthropicConfig.deserialize(serialized, new_api_key)

        assert reconstructed.api_key == new_api_key
        assert reconstructed.default_model == async_anthropic_config.default_model


class TestAsyncAnthropicClient:
    """Test AsyncAnthropicClient functionality."""

    def test_client_initialization(self, async_anthropic_config):
        """Test that the client initializes properly."""
        with (
            patch("meta_evaluator.llm_client.async_anthropic_client.AsyncAnthropic"),
            patch(
                "meta_evaluator.llm_client.async_anthropic_client.instructor.from_anthropic"
            ),
        ):
            client = AsyncAnthropicClient(async_anthropic_config)
            assert client.config == async_anthropic_config
            assert client.enum_value == AsyncLLMClientEnum.ANTHROPIC

    @pytest.mark.asyncio
    async def test_prompt_success_with_mocked_response(
        self, async_anthropic_client, simple_user_message, mock_anthropic_response
    ):
        """Test successful async prompt execution."""
        with patch.object(
            async_anthropic_client._anthropic_client.messages,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_anthropic_response

            response = await async_anthropic_client.prompt(simple_user_message)

            assert response.content == "Hello! I'm Claude, an AI assistant."
            assert response.usage.prompt_tokens == 10
            assert response.usage.completion_tokens == 15
            assert response.usage.total_tokens == 25
            assert len(response.messages) == 2  # Original + assistant response

    @pytest.mark.asyncio
    async def test_prompt_with_system_message(
        self, async_anthropic_client, mock_anthropic_response
    ):
        """Test async prompt with system message handling."""
        messages = [
            Message(role=RoleEnum.SYSTEM, content="You are a helpful assistant."),
            Message(role=RoleEnum.USER, content="Hello!"),
        ]

        with patch.object(
            async_anthropic_client._anthropic_client.messages,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_anthropic_response

            await async_anthropic_client.prompt(messages)

            mock_create.assert_called_once()
            call_args = mock_create.call_args[1]
            assert call_args["system"] == "You are a helpful assistant."
            assert len(call_args["messages"]) == 1
            assert call_args["messages"][0]["role"] == "user"
            assert call_args["messages"][0]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_prompt_with_multiple_system_messages_raises_error(
        self, async_anthropic_client
    ):
        """Test that multiple system messages raise an error."""
        messages = [
            Message(role=RoleEnum.SYSTEM, content="System message 1"),
            Message(role=RoleEnum.SYSTEM, content="System message 2"),
            Message(role=RoleEnum.USER, content="Hello!"),
        ]

        with pytest.raises(LLMAPIError) as exc_info:
            await async_anthropic_client.prompt(messages)

        assert "Multiple system messages are not supported" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_embedding_raises_error(
        self, async_anthropic_client, single_text_input
    ):
        """Test that get_embedding raises error for Anthropic."""
        with pytest.raises(LLMAPIError) as exc_info:
            await async_anthropic_client.get_embedding(single_text_input)

        assert "No embedding model specified" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_prompt_with_structured_response(
        self, async_anthropic_client, simple_user_message, mock_structured_response
    ):
        """Test async structured response functionality."""
        with patch.object(
            async_anthropic_client._instructor_client.messages,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_structured_response

            (
                structured_response,
                llm_response,
            ) = await async_anthropic_client.prompt_with_structured_response(
                simple_user_message, ExampleResponseModel
            )

            assert isinstance(structured_response, ExampleResponseModel)
            assert structured_response == mock_structured_response
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_prompt_api_error_handling(
        self, async_anthropic_client, simple_user_message
    ):
        """Test API error handling during async prompt."""
        with patch.object(
            async_anthropic_client._anthropic_client.messages,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.side_effect = Exception("API Error")

            with pytest.raises(LLMAPIError) as exc_info:
                await async_anthropic_client.prompt(simple_user_message)

            assert "Failed to get response from Anthropic" in str(exc_info.value)
            assert exc_info.value.provider == AsyncLLMClientEnum.ANTHROPIC

    def test_convert_message_unsupported_role(self, async_anthropic_client):
        """Test conversion of unsupported message role."""
        system_message = Message(role=RoleEnum.SYSTEM, content="System prompt")

        with pytest.raises(LLMAPIError) as exc_info:
            async_anthropic_client._convert_message_to_anthropic_format(system_message)

        assert "Unsupported role" in str(exc_info.value)
        assert "SYSTEM" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_response_content_handling(
        self, async_anthropic_client, simple_user_message
    ):
        """Test handling of empty response content."""
        mock_response = Mock()
        mock_response.content = []  # Empty content
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 0

        with patch.object(
            async_anthropic_client._anthropic_client.messages,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            response = await async_anthropic_client.prompt(simple_user_message)

            assert response.content == ""
            assert response.usage.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_prompt_with_xml_tags(
        self, async_anthropic_client, simple_user_message
    ):
        """Test async XML tag parsing functionality."""
        tag_configs = [
            TagConfig(
                name="sentiment",
                allowed_values=["positive", "negative", "neutral"],
                cardinality="one",
            )
        ]

        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "<sentiment>positive</sentiment>"
        mock_response.content = [mock_content]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 15

        with patch.object(
            async_anthropic_client._anthropic_client.messages,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            (
                parse_result,
                llm_response,
            ) = await async_anthropic_client.prompt_with_xml_tags(
                simple_user_message, tag_configs
            )

            assert parse_result.success
            assert parse_result.data["sentiment"] == "positive"
            assert llm_response.content == "<sentiment>positive</sentiment>"

    @pytest.mark.asyncio
    async def test_batch_prompt_processing(
        self, async_anthropic_client, batch_message_lists
    ):
        """Test batch processing of multiple prompts."""
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Batch response"
        mock_response.content = [mock_content]
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 10

        with patch.object(
            async_anthropic_client._anthropic_client.messages,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            responses = await async_anthropic_client.prompt_batch(
                batch_message_lists, batch_size=2, max_concurrency=2
            )

            assert len(responses) == 3
            for response in responses:
                assert response.content == "Batch response"
                assert response.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_batch_structured_response_processing(
        self, async_anthropic_client, batch_structured_items
    ):
        """Test batch processing of structured responses."""
        mock_structured_response = ExampleResponseModel(
            task_id="batch_task",
            status="completed",
            confidence=0.9,
            tags=["batch", "test"],
        )

        with patch.object(
            async_anthropic_client._instructor_client.messages,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_structured_response

            results = (
                await async_anthropic_client.prompt_with_structured_response_batch(
                    batch_structured_items, batch_size=2, max_concurrency=2
                )
            )

            assert len(results) == 3
            for structured_response, llm_response in results:
                assert isinstance(structured_response, ExampleResponseModel)
                assert structured_response.task_id == "batch_task"

    @pytest.mark.asyncio
    async def test_batch_xml_tags_processing(
        self, async_anthropic_client, simple_user_message
    ):
        """Test batch processing of XML tag parsing."""
        tag_configs = [
            TagConfig(
                name="category", allowed_values=["tech", "general"], cardinality="one"
            )
        ]

        items = [
            ([simple_user_message[0]], tag_configs),
            ([simple_user_message[0]], tag_configs),
            ([simple_user_message[0]], tag_configs),
        ]

        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "<category>tech</category>"
        mock_response.content = [mock_content]
        mock_response.usage.input_tokens = 8
        mock_response.usage.output_tokens = 12

        with patch.object(
            async_anthropic_client._anthropic_client.messages,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            results = await async_anthropic_client.prompt_with_xml_tags_batch(
                items, batch_size=2, max_concurrency=2
            )

            assert len(results) == 3
            for parse_result, llm_response in results:
                assert parse_result.success
                assert parse_result.data["category"] == "tech"
