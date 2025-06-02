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
from pydantic import BaseModel

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


class PersonInfo(BaseModel):
    """Test model for structured output."""

    name: str
    age: int
    occupation: str


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

        # Log usage and response information
        print("\n=== Chat Completion Test ===")
        print(f"Response content: {content}")
        print(f"Usage - Prompt tokens: {usage.prompt_tokens}")
        print(f"Usage - Completion tokens: {usage.completion_tokens}")
        print(f"Usage - Total tokens: {usage.total_tokens}")

        # Verify we got a response
        assert content is not None
        assert len(content.strip()) > 0

        # Verify usage data
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_structured_output_integration(
        self, azure_client: AzureOpenAIClient
    ) -> None:
        """Test structured output with Azure OpenAI API using instructor.

        Verifies that the client can successfully extract structured data
        using the instructor library with Azure OpenAI.
        """
        messages = [
            Message(
                role=RoleEnum.USER,
                content="Extract information: John Doe is a 30-year-old software engineer.",
            )
        ]

        structured_response, usage = azure_client._prompt_with_structured_response(
            messages=messages, response_model=PersonInfo, model="gpt-4o-2024-11-20"
        )

        # Log usage and response information
        print("\n=== Structured Output Test ===")
        print(f"Structured response: {structured_response}")
        print(f"Response type: {type(structured_response)}")
        print(f"Name: {structured_response.name}")
        print(f"Age: {structured_response.age}")
        print(f"Occupation: {structured_response.occupation}")
        print(f"Usage - Prompt tokens: {usage.prompt_tokens}")
        print(f"Usage - Completion tokens: {usage.completion_tokens}")
        print(f"Usage - Total tokens: {usage.total_tokens}")

        # Verify we got a structured response
        assert isinstance(structured_response, PersonInfo)
        assert structured_response.name is not None
        assert structured_response.age is not None
        assert structured_response.occupation is not None

        # Verify the extracted data matches expectations
        assert "john" in structured_response.name.lower()
        assert structured_response.age == 30
        assert "engineer" in structured_response.occupation.lower()

        # Verify usage data
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_high_level_prompt_integration(
        self, azure_client: AzureOpenAIClient
    ) -> None:
        """Test the high-level prompt method that returns LLMResponse.

        Verifies that the public prompt method works correctly and returns
        both the response content and comprehensive usage information.
        """
        messages = [
            Message(role=RoleEnum.USER, content="Write a haiku about programming.")
        ]

        response = azure_client.prompt(messages, model="gpt-4o-2024-11-20")

        # Log usage and response information
        print("\n=== High-Level Prompt Test ===")
        print(f"Response content: {response.content}")
        print(f"Latest response: {response.latest_response}")
        print(f"Provider: {response.provider}")
        print(f"Model: {response.model}")
        print(f"Message count: {len(response.messages)}")
        print(f"Usage - Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Usage - Completion tokens: {response.usage.completion_tokens}")
        print(f"Usage - Total tokens: {response.usage.total_tokens}")

        # Verify response structure
        assert response.content is not None
        assert len(response.content.strip()) > 0
        assert len(response.messages) == 2  # Original user message + assistant response
        assert response.messages[0].role == RoleEnum.USER
        assert response.messages[1].role == RoleEnum.ASSISTANT

        # Verify usage data
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0
        assert (
            response.usage.total_tokens
            == response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_high_level_structured_output_integration(
        self, azure_client: AzureOpenAIClient
    ) -> None:
        """Test the high-level structured output method.

        Verifies that the public prompt_with_structured_response method works correctly
        and returns both structured data and comprehensive response information.
        """
        messages = [
            Message(
                role=RoleEnum.USER,
                content="Extract: Sarah Smith is a 25-year-old data scientist working at Tech Corp.",
            )
        ]

        structured_response, llm_response = (
            azure_client.prompt_with_structured_response(
                messages=messages, response_model=PersonInfo, model="gpt-4o-2024-11-20"
            )
        )

        # Log usage and response information
        print("\n=== High-Level Structured Output Test ===")
        print(f"Structured response: {structured_response}")
        print(f"Name: {structured_response.name}")
        print(f"Age: {structured_response.age}")
        print(f"Occupation: {structured_response.occupation}")
        print(f"LLM Response content: {llm_response.content}")
        print(f"Provider: {llm_response.provider}")
        print(f"Model: {llm_response.model}")
        print(f"Usage - Prompt tokens: {llm_response.usage.prompt_tokens}")
        print(f"Usage - Completion tokens: {llm_response.usage.completion_tokens}")
        print(f"Usage - Total tokens: {llm_response.usage.total_tokens}")

        # Verify structured response
        assert isinstance(structured_response, PersonInfo)
        assert "sarah" in structured_response.name.lower()
        assert structured_response.age == 25
        assert "scientist" in structured_response.occupation.lower()

        # Verify LLM response structure
        assert llm_response.content is not None
        assert len(llm_response.messages) == 2
        assert llm_response.usage.total_tokens > 0

    def test_embedding_integration(self, azure_client: AzureOpenAIClient) -> None:
        """Test actual embedding generation with Azure OpenAI API.

        Verifies that the client can successfully generate embeddings
        using the Azure OpenAI embedding API.
        """
        texts = ["Hello world", "This is a test"]

        embeddings = azure_client._get_embedding(
            text_list=texts, model="text-embedding-3-large"
        )

        # Log embedding information
        print("\n=== Embedding Test ===")
        print(f"Input texts: {texts}")
        print(f"Number of embeddings generated: {len(embeddings)}")
        print(f"Embedding dimensions: {len(embeddings[0])}")
        print(f"First embedding (first 5 values): {embeddings[0][:5]}")
        print(f"Second embedding (first 5 values): {embeddings[1][:5]}")

        # Verify we got embeddings for both texts
        assert len(embeddings) == 2

        # Verify each embedding is a list of floats
        for i, embedding in enumerate(embeddings):
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, (int, float)) for x in embedding)
            print(f"Embedding {i + 1} length: {len(embedding)}")

        # Verify embeddings are different for different texts
        assert embeddings[0] != embeddings[1]

    def test_high_level_embedding_integration(
        self, azure_client: AzureOpenAIClient
    ) -> None:
        """Test the high-level embedding method.

        Verifies that the public get_embedding method works correctly.
        """
        texts = [
            "Machine learning is fascinating",
            "Python is a great programming language",
        ]

        embeddings = azure_client.get_embedding(texts, model="text-embedding-3-large")

        # Log embedding information
        print("\n=== High-Level Embedding Test ===")
        print(f"Input texts: {texts}")
        print(f"Number of embeddings: {len(embeddings)}")
        print(f"Embedding dimensions: {len(embeddings[0])}")

        # Verify embeddings
        assert len(embeddings) == 2
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, (int, float)) for x in embedding)

        # Verify embeddings are different
        assert embeddings[0] != embeddings[1]
