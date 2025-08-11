"""Integration tests for AsyncOpenAIClient with real API calls."""

import pytest
from pydantic import BaseModel

from meta_evaluator.llm_client.models import Message, RoleEnum


class SimpleResponse(BaseModel):
    """Simple response model for testing structured output."""

    sentiment: str
    confidence: float


@pytest.mark.integration
class TestAsyncOpenAIIntegration:
    """Integration tests for AsyncOpenAIClient using real API calls."""

    @pytest.mark.asyncio
    async def test_chat_completion_integration(self, async_openai_client_integration):
        """Test basic async chat completion with real API."""
        messages = [
            Message(
                role=RoleEnum.SYSTEM,
                content="You are a helpful assistant that responds briefly.",
            ),
            Message(role=RoleEnum.USER, content="Say hello!"),
        ]

        response = await async_openai_client_integration.prompt(messages)

        assert response.content is not None
        assert len(response.content) > 0
        assert response.usage.total_tokens > 0
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert len(response.messages) == 3  # system + user + assistant

    @pytest.mark.asyncio
    async def test_structured_output_integration(self, async_openai_client_integration):
        """Test async structured output with real API."""
        messages = [
            Message(
                role=RoleEnum.SYSTEM,
                content="Analyze the sentiment of the given text. Respond with sentiment (positive/negative/neutral) and confidence (0.0-1.0).",
            ),
            Message(role=RoleEnum.USER, content="I love this product! It's amazing!"),
        ]

        (
            structured_response,
            llm_response,
        ) = await async_openai_client_integration.prompt_with_structured_response(
            messages=messages, response_model=SimpleResponse
        )

        assert isinstance(structured_response, SimpleResponse)
        assert structured_response.sentiment in ["positive", "negative", "neutral"]
        assert 0.0 <= structured_response.confidence <= 1.0
        assert llm_response.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_embedding_integration(self, async_openai_client_integration):
        """Test async embedding generation with real API."""
        texts = ["Hello world", "This is a test"]

        embeddings = await async_openai_client_integration.get_embedding(texts)

        assert len(embeddings) == 2
        assert all(len(embedding) > 0 for embedding in embeddings)
        assert all(
            isinstance(val, float) for embedding in embeddings for val in embedding
        )
