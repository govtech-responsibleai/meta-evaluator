"""Fixtures for MetaEvaluator tests.

This conftest provides fixtures for MetaEvaluator testing.
"""

import pytest
from unittest.mock import MagicMock

from meta_evaluator.llm_client.openai_client import OpenAIClient
from meta_evaluator.llm_client.azureopenai_client import AzureOpenAIClient
from meta_evaluator.llm_client.serialization import (
    OpenAISerializedState,
    AzureOpenAISerializedState,
)


@pytest.fixture
def mock_openai_client():
    """Create a properly mocked OpenAI client for testing.

    Returns:
        MagicMock: A mock OpenAI client with configured attributes.
    """
    mock_client = MagicMock(spec=OpenAIClient)
    mock_config = MagicMock()
    mock_config.default_model = "gpt-4"
    mock_config.default_embedding_model = "text-embedding-3-large"
    mock_config.supports_structured_output = True
    mock_config.supports_logprobs = True

    # Mock the serialize method to return a proper OpenAISerializedState
    serialized_state = OpenAISerializedState(
        default_model=mock_config.default_model,
        default_embedding_model=mock_config.default_embedding_model,
        supports_structured_output=mock_config.supports_structured_output,
        supports_logprobs=mock_config.supports_logprobs,
        supports_instructor=True,
    )
    mock_config.serialize.return_value = serialized_state

    mock_client.config = mock_config
    return mock_client


@pytest.fixture
def mock_azure_openai_client():
    """Create a properly mocked Azure OpenAI client for testing.

    Returns:
        MagicMock: A mock Azure OpenAI client with configured attributes.
    """
    mock_client = MagicMock(spec=AzureOpenAIClient)
    mock_config = MagicMock()
    mock_config.endpoint = "https://test.openai.azure.com"
    mock_config.api_version = "2024-02-15-preview"
    mock_config.default_model = "gpt-4"
    mock_config.default_embedding_model = "text-embedding-ada-002"
    mock_config.supports_structured_output = True
    mock_config.supports_logprobs = True

    # Mock the serialize method to return a proper AzureOpenAISerializedState
    serialized_state = AzureOpenAISerializedState(
        endpoint=mock_config.endpoint,
        api_version=mock_config.api_version,
        default_model=mock_config.default_model,
        default_embedding_model=mock_config.default_embedding_model,
        supports_structured_output=mock_config.supports_structured_output,
        supports_logprobs=mock_config.supports_logprobs,
        supports_instructor=True,
    )
    mock_config.serialize.return_value = serialized_state

    mock_client.config = mock_config
    return mock_client


def create_mock_openai_client(**config_overrides):
    """Helper function to create a customized mock OpenAI client.

    Args:
        **config_overrides: Override default configuration values.

    Returns:
        MagicMock: A mock OpenAI client with custom configuration.
    """
    mock_client = MagicMock(spec=OpenAIClient)
    mock_config = MagicMock()
    mock_config.default_model = config_overrides.get("default_model", "gpt-4")
    mock_config.default_embedding_model = config_overrides.get(
        "default_embedding_model", "text-embedding-3-large"
    )
    mock_config.supports_structured_output = config_overrides.get(
        "supports_structured_output", True
    )
    mock_config.supports_logprobs = config_overrides.get("supports_logprobs", True)

    # Mock the serialize method to return a proper OpenAISerializedState
    serialized_state = OpenAISerializedState(
        default_model=mock_config.default_model,
        default_embedding_model=mock_config.default_embedding_model,
        supports_structured_output=mock_config.supports_structured_output,
        supports_logprobs=mock_config.supports_logprobs,
        supports_instructor=True,
    )
    mock_config.serialize.return_value = serialized_state

    mock_client.config = mock_config
    return mock_client


def create_mock_azure_openai_client(**config_overrides):
    """Helper function to create a customized mock Azure OpenAI client.

    Args:
        **config_overrides: Override default configuration values.

    Returns:
        MagicMock: A mock Azure OpenAI client with custom configuration.
    """
    mock_client = MagicMock(spec=AzureOpenAIClient)
    mock_config = MagicMock()
    mock_config.endpoint = config_overrides.get(
        "endpoint", "https://test.openai.azure.com"
    )
    mock_config.api_version = config_overrides.get("api_version", "2024-02-15-preview")
    mock_config.default_model = config_overrides.get("default_model", "gpt-4")
    mock_config.default_embedding_model = config_overrides.get(
        "default_embedding_model", "text-embedding-ada-002"
    )
    mock_config.supports_structured_output = config_overrides.get(
        "supports_structured_output", True
    )
    mock_config.supports_logprobs = config_overrides.get("supports_logprobs", True)

    # Mock the serialize method to return a proper AzureOpenAISerializedState
    serialized_state = AzureOpenAISerializedState(
        endpoint=mock_config.endpoint,
        api_version=mock_config.api_version,
        default_model=mock_config.default_model,
        default_embedding_model=mock_config.default_embedding_model,
        supports_structured_output=mock_config.supports_structured_output,
        supports_logprobs=mock_config.supports_logprobs,
        supports_instructor=True,
    )
    mock_config.serialize.return_value = serialized_state

    mock_client.config = mock_config
    return mock_client
