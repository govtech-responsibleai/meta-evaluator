"""Integration test suite for Azure OpenAI LLM client implementation.

This module contains integration tests that actually call the Azure OpenAI API.
These tests require valid Azure OpenAI credentials in environment variables:
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_VERSION

Tests will be skipped if these environment variables are not set.
"""

import os
import pytest
from dotenv import load_dotenv

from meta_evaluator.llm_client.azureopenai_client import (
    AzureOpenAIConfig,
    AzureOpenAIClient,
)
from meta_evaluator.llm_client.models import Message, RoleEnum

load_dotenv()

# Check if required environment variables are set
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION")

AZURE_CREDENTIALS_AVAILABLE = all(
    [
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_VERSION,
    ]
)


@pytest.mark.skipif(
    not AZURE_CREDENTIALS_AVAILABLE,
    reason="Azure OpenAI credentials not available in environment variables",
)
class TestAzureOpenAIClientIntegration:
    """Integration test suite for the AzureOpenAIClient class.

    This class tests the Azure OpenAI client implementation with real API calls.
    All tests require valid Azure OpenAI credentials in environment variables.
    """

    @pytest.fixture
    def azure_config(self) -> AzureOpenAIConfig:
        """Provide a valid AzureOpenAIConfig instance from environment variables.

        Returns:
            AzureOpenAIConfig: A valid configuration instance for creating test clients.
        """
        assert AZURE_OPENAI_API_KEY is not None
        assert AZURE_OPENAI_ENDPOINT is not None
        assert AZURE_OPENAI_VERSION is not None
        return AzureOpenAIConfig(
            api_key=AZURE_OPENAI_API_KEY,
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_VERSION,
            default_model="gpt-4o-2024-11-20",
            default_embedding_model="text-embedding-3-large",
        )

    @pytest.fixture
    def azure_client(self, azure_config: AzureOpenAIConfig) -> AzureOpenAIClient:
        """Provide a valid AzureOpenAIClient instance for integration testing.

        Args:
            azure_config: A valid configuration instance.

        Returns:
            AzureOpenAIClient: A valid client instance for integration testing.
        """
        return AzureOpenAIClient(azure_config)

    def test_chat_completion_integration(self, azure_client: AzureOpenAIClient) -> None:
        """Test actual chat completion with Azure OpenAI API.

        Verifies that the client can successfully make a real API call
        to Azure OpenAI and receive a valid response.
        """
        messages = [
            Message(role=RoleEnum.USER, content="Say hello in exactly one word.")
        ]

        content, usage = azure_client._prompt(
            model="gpt-4o-2024-11-20", messages=messages, get_logprobs=False
        )

        # Verify we got a response
        assert content is not None
        assert len(content.strip()) > 0

        # Verify usage data
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_embedding_integration(self, azure_client: AzureOpenAIClient) -> None:
        """Test actual embedding generation with Azure OpenAI API.

        Verifies that the client can successfully generate embeddings
        using the Azure OpenAI embedding API.
        """
        texts = ["Hello world", "This is a test"]

        embeddings = azure_client._get_embedding(
            text_list=texts, model="text-embedding-3-large"
        )

        # Verify we got embeddings for both texts
        assert len(embeddings) == 2

        # Verify each embedding is a list of floats
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, (int, float)) for x in embedding)

        # Verify embeddings are different for different texts
        assert embeddings[0] != embeddings[1]
