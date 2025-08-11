"""File for testing the AsyncLLMClient module."""

import asyncio
import logging
from typing import TypeVar

import pytest
from pydantic import BaseModel, ValidationError

from meta_evaluator.llm_client.async_client import AsyncLLMClientConfig
from meta_evaluator.llm_client.exceptions import LLMAPIError
from meta_evaluator.llm_client.models import (
    LLMResponse,
    LLMUsage,
    Message,
    ParseResult,
    RoleEnum,
    TagConfig,
)
from meta_evaluator.llm_client.serialization import (
    LLMClientSerializedState,
    MockLLMClientSerializedState,
)

# Import test classes from conftest for type hints
from .conftest import (
    AsyncLLMClientConfigConcreteTest,
    ConcreteTestAsyncLLMClient,
    ExampleResponseModel,
)

T = TypeVar("T", bound=BaseModel)


class TestAsyncLLMClientConfig:
    """Test the AsyncLLMClientConfig abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that AsyncLLMClientConfig cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AsyncLLMClientConfig(
                api_key="test-api-key",
                supports_structured_output=True,
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )  # type: ignore

    def test_concrete_subclass_can_instantiate(self):
        """Test that a concrete subclass can be instantiated with valid data."""

        class ConcreteAsyncConfig(AsyncLLMClientConfig):
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
            ) -> "ConcreteAsyncConfig":
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

        config = ConcreteAsyncConfig(  # type: ignore
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

        class ConcreteAsyncConfig(AsyncLLMClientConfig):
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
            ) -> "ConcreteAsyncConfig":
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
            ConcreteAsyncConfig()  # type: ignore

        with pytest.raises(ValidationError):
            ConcreteAsyncConfig(api_key="test-key")  # type: ignore


class TestAsyncLLMClient:
    """Test suite for the abstract AsyncLLMClient class."""

    def test_initialization_with_valid_config(
        self, async_structured_output_config: AsyncLLMClientConfigConcreteTest
    ):
        """Verify initialization succeeds with a valid config.

        Args:
            async_structured_output_config: A pytest fixture providing a valid async config.
        """
        client = ConcreteTestAsyncLLMClient(async_structured_output_config)
        assert isinstance(client, ConcreteTestAsyncLLMClient)
        assert client.config is async_structured_output_config
        assert isinstance(client.logger, logging.Logger)

    @pytest.mark.asyncio
    async def test_prompt_happy_path_default_model(
        self,
        async_structured_output_client: ConcreteTestAsyncLLMClient,
        simple_user_message: list[Message],
        mock_raw_response: tuple[str, LLMUsage],
        mocker,
    ):
        """Verify async prompt uses default model and logs correctly on success.

        Args:
            async_structured_output_client: A pytest fixture providing a concrete async client instance.
            simple_user_message: A pytest fixture providing valid messages.
            mock_raw_response: A pytest fixture providing a mock raw response tuple.
            mocker: The pytest-mock mocker fixture.
        """
        mock__prompt = mocker.patch.object(
            async_structured_output_client, "_prompt", return_value=mock_raw_response
        )
        mock_logger = mocker.patch.object(async_structured_output_client, "logger")

        response = await async_structured_output_client.prompt(
            messages=simple_user_message
        )

        assert isinstance(response, LLMResponse)
        assert response.content == "Test response content from mock"
        assert response.usage.total_tokens == 40  # prompt_tokens + completion_tokens
        assert response.provider == async_structured_output_client.enum_value
        assert response.model == async_structured_output_client.config.default_model
        assert len(response.messages) == 2  # Original message + assistant response
        assert response.messages[0].role == RoleEnum.USER
        assert response.messages[1].role == RoleEnum.ASSISTANT
        assert response.messages[1].content == "Test response content from mock"

        mock__prompt.assert_called_once_with(
            model=async_structured_output_client.config.default_model,
            messages=simple_user_message,
            get_logprobs=False,
        )

        mock_logger.info.assert_any_call(
            f"Using model: {async_structured_output_client.config.default_model}"
        )
        mock_logger.info.assert_any_call(f"Input Payload: {simple_user_message}")
        mock_logger.info.assert_any_call(f"Latest response: {response.latest_response}")
        mock_logger.info.assert_any_call(f"Output usage: {response.usage}")
        assert mock_logger.info.call_count == 4

    @pytest.mark.asyncio
    async def test_prompt_with_structured_response(
        self,
        async_structured_output_client: ConcreteTestAsyncLLMClient,
        simple_user_message: list[Message],
        mock_structured_response,
        mock_raw_response: tuple[str, LLMUsage],
        mocker,
    ):
        """Verify async prompt with structured response works correctly.

        Args:
            async_structured_output_client: A pytest fixture providing a concrete async client instance.
            simple_user_message: A pytest fixture providing valid messages.
            mock_structured_response: A pytest fixture providing a mock structured response.
            mock_raw_response: A pytest fixture providing a mock raw response tuple.
            mocker: The pytest-mock mocker fixture.
        """
        mock__prompt_with_structured_response = mocker.patch.object(
            async_structured_output_client,
            "_prompt_with_structured_response",
            return_value=(mock_structured_response, mock_raw_response[1]),
        )
        mock_logger = mocker.patch.object(async_structured_output_client, "logger")

        (
            structured_response,
            llm_response,
        ) = await async_structured_output_client.prompt_with_structured_response(
            messages=simple_user_message, response_model=ExampleResponseModel
        )

        assert structured_response == mock_structured_response
        assert isinstance(llm_response, LLMResponse)
        assert llm_response.provider == async_structured_output_client.enum_value

        mock__prompt_with_structured_response.assert_called_once_with(
            messages=simple_user_message,
            response_model=ExampleResponseModel,
            model=async_structured_output_client.config.default_model,
        )

        mock_logger.info.assert_any_call(
            f"Using model: {async_structured_output_client.config.default_model}"
        )
        mock_logger.info.assert_any_call(f"Input Payload: {simple_user_message}")

    @pytest.mark.asyncio
    async def test_prompt_with_xml_tags(
        self,
        async_structured_output_client: ConcreteTestAsyncLLMClient,
        simple_user_message: list[Message],
        mock_raw_response: tuple[str, LLMUsage],
        mocker,
    ):
        """Verify async prompt with XML tags works correctly.

        Args:
            async_structured_output_client: A pytest fixture providing a concrete async client instance.
            simple_user_message: A pytest fixture providing valid messages.
            mock_raw_response: A pytest fixture providing a mock raw response tuple.
            mocker: The pytest-mock mocker fixture.
        """
        mock__prompt = mocker.patch.object(
            async_structured_output_client, "_prompt", return_value=mock_raw_response
        )
        mock_logger = mocker.patch.object(async_structured_output_client, "logger")

        tag_configs = [
            TagConfig(
                name="sentiment",
                allowed_values=["positive", "negative", "neutral"],
                cardinality="one",
            )
        ]

        (
            parse_result,
            llm_response,
        ) = await async_structured_output_client.prompt_with_xml_tags(
            messages=simple_user_message, tag_configs=tag_configs
        )

        assert isinstance(parse_result, ParseResult)
        assert isinstance(llm_response, LLMResponse)
        assert llm_response.provider == async_structured_output_client.enum_value

        mock__prompt.assert_called_once_with(
            model=async_structured_output_client.config.default_model,
            messages=simple_user_message,
            get_logprobs=False,
        )

        mock_logger.info.assert_any_call(
            f"Using model: {async_structured_output_client.config.default_model}"
        )
        mock_logger.info.assert_any_call(f"Input Payload: {simple_user_message}")

    @pytest.mark.asyncio
    async def test_get_embedding_batch(
        self,
        async_structured_output_client: ConcreteTestAsyncLLMClient,
        batch_text_lists: list[list[str]],
        mock_multiple_embeddings,
        mocker,
    ):
        """Verify async batch embedding processing works correctly.

        Args:
            async_structured_output_client: A pytest fixture providing a concrete async client instance.
            batch_text_lists: A pytest fixture providing multiple text lists for batch processing.
            mock_multiple_embeddings: A pytest fixture providing mock embeddings.
            mocker: The pytest-mock mocker fixture.
        """
        # Mock the _get_embedding method to return different embeddings for each call
        mock__get_embedding = mocker.patch.object(
            async_structured_output_client,
            "_get_embedding",
            return_value=mock_multiple_embeddings[:2],
        )
        mock_logger = mocker.patch.object(async_structured_output_client, "logger")

        embeddings = await async_structured_output_client.get_embedding_batch(
            batch_text_lists
        )

        assert len(embeddings) == len(batch_text_lists)
        assert all(isinstance(embedding_list, list) for embedding_list in embeddings)

        # Should be called once for each text list in the batch
        assert mock__get_embedding.call_count == len(batch_text_lists)

        # Verify logging happened
        assert mock_logger.info.call_count >= len(
            batch_text_lists
        )  # At least one log per batch item

    # ===== _batch_process TESTS =====

    @pytest.mark.asyncio
    async def test_batch_process_basic_functionality(
        self, async_structured_output_client: ConcreteTestAsyncLLMClient
    ):
        """Test basic _batch_process functionality with simple async method."""

        # Create a simple async method to test
        async def mock_async_method(item):
            await asyncio.sleep(0.001)  # Simulate async work
            return f"processed_{item}"

        items = ["item1", "item2", "item3"]
        results = await async_structured_output_client._batch_process(
            items, mock_async_method, batch_size=2, max_concurrency=2
        )

        assert len(results) == len(items)
        assert results == ["processed_item1", "processed_item2", "processed_item3"]

    @pytest.mark.asyncio
    async def test_batch_process_concurrency_control(
        self, async_structured_output_client: ConcreteTestAsyncLLMClient
    ):
        """Test that max_concurrency parameter properly limits concurrent operations."""
        max_concurrency = 2
        concurrent_count = 0
        max_concurrent_reached = 0

        async def mock_async_method(item):
            nonlocal concurrent_count, max_concurrent_reached
            concurrent_count += 1
            max_concurrent_reached = max(max_concurrent_reached, concurrent_count)

            # Simulate some work
            await asyncio.sleep(0.05)

            concurrent_count -= 1
            return f"processed_{item}"

        items = ["item1", "item2", "item3", "item4", "item5"]
        results = await async_structured_output_client._batch_process(
            items, mock_async_method, batch_size=10, max_concurrency=max_concurrency
        )

        assert len(results) == len(items)
        # Verify concurrency was limited
        assert max_concurrent_reached <= max_concurrency

    @pytest.mark.asyncio
    async def test_batch_process_batch_size_control(
        self, async_structured_output_client: ConcreteTestAsyncLLMClient
    ):
        """Test that batch_size parameter controls how many items are processed per batch."""
        batch_size = 2
        batch_stats = {"items_processed": 0}

        async def mock_async_method(item):
            # Track items processed
            batch_stats["items_processed"] += 1
            await asyncio.sleep(0.001)
            return f"processed_{item}"

        items = ["item1", "item2", "item3", "item4", "item5"]
        results = await async_structured_output_client._batch_process(
            items, mock_async_method, batch_size=batch_size, max_concurrency=5
        )

        assert len(results) == len(items)
        assert batch_stats["items_processed"] == len(items)
        # All items should be processed
        assert results == [
            "processed_item1",
            "processed_item2",
            "processed_item3",
            "processed_item4",
            "processed_item5",
        ]

    @pytest.mark.asyncio
    async def test_batch_process_error_handling_single_failure(
        self, async_structured_output_client: ConcreteTestAsyncLLMClient
    ):
        """Test _batch_process handles individual item failures without stopping the batch."""

        async def mock_async_method(item):
            await asyncio.sleep(0.001)
            if item == "fail_item":
                raise ValueError("Simulated failure")
            return f"processed_{item}"

        items = ["item1", "fail_item", "item3"]
        results = await async_structured_output_client._batch_process(
            items, mock_async_method, batch_size=10, max_concurrency=5
        )

        assert len(results) == len(items)
        assert results[0] == "processed_item1"
        assert isinstance(results[1], ValueError)
        assert str(results[1]) == "Simulated failure"
        assert results[2] == "processed_item3"

    @pytest.mark.asyncio
    async def test_batch_process_error_handling_multiple_failures(
        self, async_structured_output_client: ConcreteTestAsyncLLMClient
    ):
        """Test _batch_process handles multiple failures in a batch."""

        async def mock_async_method(item):
            await asyncio.sleep(0.001)
            if "fail" in item:
                if item == "fail1":
                    raise LLMAPIError(
                        "API Error",
                        async_structured_output_client.enum_value,
                        Exception("API failed"),
                    )
                else:
                    raise RuntimeError(f"Runtime error for {item}")
            return f"processed_{item}"

        items = ["item1", "fail1", "item3", "fail2", "item5"]
        results = await async_structured_output_client._batch_process(
            items, mock_async_method, batch_size=3, max_concurrency=2
        )

        assert len(results) == len(items)
        assert results[0] == "processed_item1"
        assert isinstance(results[1], LLMAPIError)
        assert results[2] == "processed_item3"
        assert isinstance(results[3], RuntimeError)
        assert results[4] == "processed_item5"

    @pytest.mark.asyncio
    async def test_batch_process_empty_list(
        self, async_structured_output_client: ConcreteTestAsyncLLMClient
    ):
        """Test _batch_process handles empty input list."""

        async def mock_async_method(item):
            return f"processed_{item}"

        items = []
        results = await async_structured_output_client._batch_process(
            items, mock_async_method, batch_size=10, max_concurrency=5
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_batch_process_batch_size_larger_than_items(
        self, async_structured_output_client: ConcreteTestAsyncLLMClient
    ):
        """Test _batch_process when batch_size is larger than number of items."""

        async def mock_async_method(item):
            await asyncio.sleep(0.001)
            return f"processed_{item}"

        items = ["item1", "item2"]
        results = await async_structured_output_client._batch_process(
            items, mock_async_method, batch_size=10, max_concurrency=5
        )

        assert len(results) == len(items)
        assert results == ["processed_item1", "processed_item2"]
