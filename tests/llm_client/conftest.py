"""LLM Client test fixtures.

This conftest provides LLM client-specific fixtures used across multiple test modules.
Common fixtures are defined in the main tests/conftest.py file.
"""

import os
import warnings
from typing import Any, Optional, TypeVar
from unittest.mock import MagicMock, Mock

import pytest
from pydantic import BaseModel

from meta_evaluator.llm_client import LLMClient, LLMClientConfig
from meta_evaluator.llm_client.anthropic_client import (
    AnthropicClient,
    AnthropicConfig,
)
from meta_evaluator.llm_client.async_anthropic_client import (
    AsyncAnthropicClient,
    AsyncAnthropicConfig,
)
from meta_evaluator.llm_client.async_azureopenai_client import (
    AsyncAzureOpenAIClient,
    AsyncAzureOpenAIConfig,
)
from meta_evaluator.llm_client.async_client import AsyncLLMClient, AsyncLLMClientConfig
from meta_evaluator.llm_client.async_openai_client import (
    AsyncOpenAIClient,
    AsyncOpenAIConfig,
)
from meta_evaluator.llm_client.azureopenai_client import (
    AzureOpenAIClient,
    AzureOpenAIConfig,
)
from meta_evaluator.llm_client.enums import AsyncLLMClientEnum, LLMClientEnum
from meta_evaluator.llm_client.models import LLMUsage, Message, RoleEnum
from meta_evaluator.llm_client.openai_client import (
    OpenAIClient,
    OpenAIConfig,
)
from meta_evaluator.llm_client.serialization import (
    LLMClientSerializedState,
    MockLLMClientSerializedState,
)

T = TypeVar("T", bound=BaseModel)

# ==== TEST CLASSES ====


class LLMClientConfigConcreteTest(LLMClientConfig):
    """A concrete configuration class for testing LLMClient."""

    api_key: str = "test-api-key"
    supports_structured_output: bool = False
    default_model: str = "test-default-model"
    default_embedding_model: Optional[str] = "test-default-embedding-model"

    def _prevent_instantiation(self) -> None:
        pass

    def serialize(self) -> MockLLMClientSerializedState:
        """Serialize test config to typed state object (excluding API key for security).

        Returns:
            MockLLMClientSerializedState: Configuration as typed state object without sensitive data.
        """
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
    ) -> "LLMClientConfigConcreteTest":
        """Deserialize test config from typed state object and API key.

        Args:
            state: Serialized configuration state object.
            api_key: API key to use for the config.
            **kwargs: Additional configuration parameters.

        Returns:
            LLMClientConfigConcreteTest: Reconstructed configuration instance.

        Raises:
            TypeError: If state is not a MockLLMClientSerializedState instance.
        """
        if not isinstance(state, MockLLMClientSerializedState):
            raise TypeError(
                f"Expected MockLLMClientSerializedState, got {type(state).__name__}"
            )

        return cls(
            api_key=api_key,
            default_model=state.default_model,
            default_embedding_model=state.default_embedding_model,
            supports_structured_output=state.supports_structured_output,
            supports_logprobs=state.supports_logprobs,
        )


class ConcreteTestLLMClient(LLMClient):
    """A concrete LLMClient subclass for testing."""

    @property
    def enum_value(self) -> LLMClientEnum:
        """Return the unique LLMClientEnum value for the test client."""
        return LLMClientEnum.OPENAI

    def _prompt(
        self, model: str, messages: list[Message], get_logprobs: bool
    ) -> tuple[str, LLMUsage]:
        """Abstract method implementation for testing.

        This method is intended to be mocked in actual tests.

        Args:
            model: The model to use.
            messages: The list of messages.
            get_logprobs: Whether to get logprobs.

        Returns:
            A tuple containing a mock response and LLMUsage.
        """
        raise NotImplementedError(
            "ConcreteTestLLMClient._prompt should be mocked for tests"
        )

    def _get_embedding(self, text_list: list[str], model: str) -> list[list[float]]:
        """Abstract method implementation for testing.

        This method is intended to be mocked in actual tests.

        Args:
            text_list: List of text prompts to generate embeddings for.
            model: The embedding model to use for this request.

        Returns:
            List of embedding vectors, one for each input prompt.
        """
        raise NotImplementedError(
            "ConcreteTestLLMClient._get_embedding should be mocked for tests"
        )

    def _prompt_with_structured_response(
        self, messages: list[Message], response_model: type[T], model: str
    ) -> tuple[T, LLMUsage]:
        raise NotImplementedError(
            "ConcreteTestLLMClient._prompt_with_structured_response should be mocked for tests"
        )


class AsyncLLMClientConfigConcreteTest(AsyncLLMClientConfig):
    """A concrete async configuration class for testing AsyncLLMClient."""

    api_key: str = "test-api-key"
    supports_structured_output: bool = False
    default_model: str = "test-default-model"
    default_embedding_model: Optional[str] = "test-default-embedding-model"

    def _prevent_instantiation(self) -> None:
        pass

    def serialize(self) -> MockLLMClientSerializedState:
        """Serialize test config to typed state object (excluding API key for security).

        Returns:
            MockLLMClientSerializedState: Configuration as typed state object without sensitive data.
        """
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
    ) -> "AsyncLLMClientConfigConcreteTest":
        """Deserialize test config from typed state object and API key.

        Args:
            state: Serialized configuration state object.
            api_key: API key to use for the config.
            **kwargs: Additional configuration parameters.

        Returns:
            AsyncLLMClientConfigConcreteTest: Reconstructed configuration instance.

        Raises:
            TypeError: If state is not a MockLLMClientSerializedState instance.
        """
        if not isinstance(state, MockLLMClientSerializedState):
            raise TypeError(
                f"Expected MockLLMClientSerializedState, got {type(state).__name__}"
            )

        return cls(
            api_key=api_key,
            default_model=state.default_model,
            default_embedding_model=state.default_embedding_model,
            supports_structured_output=state.supports_structured_output,
            supports_logprobs=state.supports_logprobs,
        )


class ConcreteTestAsyncLLMClient(AsyncLLMClient):
    """A concrete AsyncLLMClient subclass for testing."""

    @property
    def enum_value(self) -> AsyncLLMClientEnum:
        """Return the unique AsyncLLMClientEnum value for the test client."""
        return AsyncLLMClientEnum.OPENAI

    async def _prompt(
        self, model: str, messages: list[Message], get_logprobs: bool
    ) -> tuple[str, LLMUsage]:
        """Abstract method implementation for testing.

        This method is intended to be mocked in actual tests.

        Args:
            model: The model to use.
            messages: The list of messages.
            get_logprobs: Whether to get logprobs.

        Returns:
            A tuple containing a mock response and LLMUsage.
        """
        raise NotImplementedError(
            "ConcreteTestAsyncLLMClient._prompt should be mocked for tests"
        )

    async def _get_embedding(
        self, text_list: list[str], model: str
    ) -> list[list[float]]:
        """Abstract method implementation for testing.

        This method is intended to be mocked in actual tests.

        Args:
            text_list: List of text prompts to generate embeddings for.
            model: The embedding model to use for this request.

        Returns:
            List of embedding vectors, one for each input prompt.
        """
        raise NotImplementedError(
            "ConcreteTestAsyncLLMClient._get_embedding should be mocked for tests"
        )

    async def _prompt_with_structured_response(
        self, messages: list[Message], response_model: type[T], model: str
    ) -> tuple[T, LLMUsage]:
        raise NotImplementedError(
            "ConcreteTestAsyncLLMClient._prompt_with_structured_response should be mocked for tests"
        )


class ExampleResponseModel(BaseModel):
    """Example Pydantic model for structured output testing."""

    task_id: str
    status: str
    confidence: float
    tags: list[str]


# ==== BASIC CONFIG AND CLIENT FIXTURES ====


@pytest.fixture
def basic_llm_config() -> LLMClientConfigConcreteTest:
    """Provides a basic LLM client configuration for testing.

    Returns:
        LLMClientConfigConcreteTest: A basic configuration instance.
    """
    return LLMClientConfigConcreteTest(
        api_key="test-api-key",
        supports_structured_output=False,
        default_model="gpt-4",
        default_embedding_model="text-embedding-ada-002",
        supports_logprobs=False,
    )


@pytest.fixture
def structured_output_config() -> LLMClientConfigConcreteTest:
    """Provides a configuration with structured output support.

    Returns:
        LLMClientConfigConcreteTest: A configuration with structured output enabled.
    """
    return LLMClientConfigConcreteTest(
        api_key="test-api-key",
        supports_structured_output=True,
        default_model="gpt-4",
        default_embedding_model="text-embedding-ada-002",
        supports_logprobs=False,
    )


@pytest.fixture
def logprobs_config() -> LLMClientConfigConcreteTest:
    """Provides a configuration with logprobs support.

    Returns:
        LLMClientConfigConcreteTest: A configuration with logprobs enabled.
    """
    return LLMClientConfigConcreteTest(
        api_key="test-api-key",
        supports_structured_output=True,
        default_model="gpt-4",
        default_embedding_model="text-embedding-ada-002",
        supports_logprobs=True,
    )


@pytest.fixture
def basic_llm_client(
    basic_llm_config: LLMClientConfigConcreteTest,
) -> ConcreteTestLLMClient:
    """Provides a basic LLM client instance.

    Args:
        basic_llm_config: A basic configuration instance.

    Returns:
        ConcreteTestLLMClient: A client instance for testing.
    """
    return ConcreteTestLLMClient(basic_llm_config)


@pytest.fixture
def structured_output_client(
    structured_output_config: LLMClientConfigConcreteTest,
) -> ConcreteTestLLMClient:
    """Provides an LLM client with structured output support.

    Args:
        structured_output_config: A configuration with structured output enabled.

    Returns:
        ConcreteTestLLMClient: A client instance with structured output support.
    """
    return ConcreteTestLLMClient(structured_output_config)


@pytest.fixture
def logprobs_client(
    logprobs_config: LLMClientConfigConcreteTest,
) -> ConcreteTestLLMClient:
    """Provides an LLM client with logprobs support.

    Args:
        logprobs_config: A configuration with logprobs enabled.

    Returns:
        ConcreteTestLLMClient: A client instance with logprobs support.
    """
    return ConcreteTestLLMClient(logprobs_config)


# ==== ASYNC CLIENT FIXTURES ====


@pytest.fixture
def async_basic_llm_config() -> AsyncLLMClientConfigConcreteTest:
    """Provides a basic async LLM client configuration for testing.

    Returns:
        AsyncLLMClientConfigConcreteTest: A basic async configuration instance.
    """
    return AsyncLLMClientConfigConcreteTest(
        api_key="test-api-key",
        supports_structured_output=False,
        default_model="gpt-4",
        default_embedding_model="text-embedding-ada-002",
        supports_logprobs=False,
    )


@pytest.fixture
def async_structured_output_config() -> AsyncLLMClientConfigConcreteTest:
    """Provides an async configuration with structured output support.

    Returns:
        AsyncLLMClientConfigConcreteTest: An async configuration with structured output enabled.
    """
    return AsyncLLMClientConfigConcreteTest(
        api_key="test-api-key",
        supports_structured_output=True,
        default_model="gpt-4",
        default_embedding_model="text-embedding-ada-002",
        supports_logprobs=False,
    )


@pytest.fixture
def async_logprobs_config() -> AsyncLLMClientConfigConcreteTest:
    """Provides an async configuration with logprobs support.

    Returns:
        AsyncLLMClientConfigConcreteTest: An async configuration with logprobs enabled.
    """
    return AsyncLLMClientConfigConcreteTest(
        api_key="test-api-key",
        supports_structured_output=True,
        default_model="gpt-4",
        default_embedding_model="text-embedding-ada-002",
        supports_logprobs=True,
    )


@pytest.fixture
def async_basic_llm_client(
    async_basic_llm_config: AsyncLLMClientConfigConcreteTest,
) -> ConcreteTestAsyncLLMClient:
    """Provides a basic async LLM client instance.

    Args:
        async_basic_llm_config: A basic async configuration instance.

    Returns:
        ConcreteTestAsyncLLMClient: An async client instance for testing.
    """
    return ConcreteTestAsyncLLMClient(async_basic_llm_config)


@pytest.fixture
def async_structured_output_client(
    async_structured_output_config: AsyncLLMClientConfigConcreteTest,
) -> ConcreteTestAsyncLLMClient:
    """Provides an async LLM client with structured output support.

    Args:
        async_structured_output_config: An async configuration with structured output enabled.

    Returns:
        ConcreteTestAsyncLLMClient: An async client instance with structured output support.
    """
    return ConcreteTestAsyncLLMClient(async_structured_output_config)


@pytest.fixture
def async_logprobs_client(
    async_logprobs_config: AsyncLLMClientConfigConcreteTest,
) -> ConcreteTestAsyncLLMClient:
    """Provides an async LLM client with logprobs support.

    Args:
        async_logprobs_config: An async configuration with logprobs enabled.

    Returns:
        ConcreteTestAsyncLLMClient: An async client instance with logprobs support.
    """
    return ConcreteTestAsyncLLMClient(async_logprobs_config)


# ==== AZURE OPENAI FIXTURES ====


@pytest.fixture
def azure_config_data() -> dict[str, Any]:
    """Provides valid Azure OpenAI configuration data.

    Returns:
        dict[str, Any]: Configuration data for Azure OpenAI.
    """
    return {
        "api_key": "test-api-key",
        "endpoint": "https://test.openai.azure.com/",
        "api_version": "2023-12-01-preview",
        "default_model": "gpt-4",
        "default_embedding_model": "text-embedding-ada-002",
    }


@pytest.fixture
def azure_config(azure_config_data: dict[str, Any]) -> AzureOpenAIConfig:
    """Provides a valid Azure OpenAI configuration.

    Args:
        azure_config_data: Configuration data dictionary.

    Returns:
        AzureOpenAIConfig: A valid Azure OpenAI configuration instance.
    """
    return AzureOpenAIConfig(**azure_config_data)


@pytest.fixture
def azure_client(azure_config: AzureOpenAIConfig) -> AzureOpenAIClient:
    """Provides an Azure OpenAI client instance.

    Args:
        azure_config: A valid Azure OpenAI configuration.

    Returns:
        AzureOpenAIClient: An Azure OpenAI client instance.
    """
    return AzureOpenAIClient(azure_config)


@pytest.fixture
def async_azure_config(azure_config_data: dict[str, Any]) -> AsyncAzureOpenAIConfig:
    """Provides a valid async Azure OpenAI configuration.

    Args:
        azure_config_data: Configuration data dictionary.

    Returns:
        AsyncAzureOpenAIConfig: A valid async Azure OpenAI configuration instance.
    """
    return AsyncAzureOpenAIConfig(**azure_config_data)


@pytest.fixture
def async_azure_client(
    async_azure_config: AsyncAzureOpenAIConfig,
) -> AsyncAzureOpenAIClient:
    """Provides an async Azure OpenAI client instance.

    Args:
        async_azure_config: A valid async Azure OpenAI configuration.

    Returns:
        AsyncAzureOpenAIClient: An async Azure OpenAI client instance.
    """
    return AsyncAzureOpenAIClient(async_azure_config)


# ==== OPENAI FIXTURES ====


@pytest.fixture
def openai_config_data() -> dict[str, Any]:
    """Provides valid OpenAI configuration data for unit tests.

    Returns:
        dict[str, Any]: Configuration data for OpenAI unit tests.
    """
    return {
        "api_key": "test-api-key",
        "default_model": "gpt-4o-2024-11-20",
        "default_embedding_model": "text-embedding-3-large",
    }


@pytest.fixture
def openai_config(openai_config_data: dict[str, Any]) -> OpenAIConfig:
    """Provides a valid OpenAI configuration for unit tests.

    Args:
        openai_config_data: Configuration data dictionary.

    Returns:
        OpenAIConfig: A valid OpenAI configuration instance for unit tests.
    """
    return OpenAIConfig(**openai_config_data)


@pytest.fixture
def openai_client(openai_config: OpenAIConfig) -> OpenAIClient:
    """Provides an OpenAI client for unit tests.

    Args:
        openai_config: A valid OpenAI configuration.

    Returns:
        OpenAIClient: OpenAI client instance for unit tests.
    """
    return OpenAIClient(openai_config)


@pytest.fixture
def async_openai_config(openai_config_data: dict[str, Any]) -> AsyncOpenAIConfig:
    """Provides a valid async OpenAI configuration for unit tests.

    Args:
        openai_config_data: Configuration data dictionary.

    Returns:
        AsyncOpenAIConfig: A valid async OpenAI configuration instance for unit tests.
    """
    return AsyncOpenAIConfig(**openai_config_data)


@pytest.fixture
def async_openai_client(async_openai_config: AsyncOpenAIConfig) -> AsyncOpenAIClient:
    """Provides an async OpenAI client for unit tests.

    Args:
        async_openai_config: A valid async OpenAI configuration.

    Returns:
        AsyncOpenAIClient: Async OpenAI client instance for unit tests.
    """
    return AsyncOpenAIClient(async_openai_config)


# ==== AZURE OPENAI INTEGRATION FIXTURES ====


@pytest.fixture
def azure_config_integration() -> AzureOpenAIConfig:
    """Provides Azure OpenAI configuration from environment variables for integration tests.

    Returns:
        AzureOpenAIConfig: Azure OpenAI configuration if credentials available.
    """
    # Check if required environment variables are set
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_version = os.getenv("AZURE_OPENAI_VERSION")

    azure_credentials_available = all(
        [
            azure_endpoint,
            azure_api_key,
            azure_version,
        ]
    )

    if not azure_credentials_available:
        warnings.warn(
            "Azure OpenAI credentials not available in environment variables. "
            "Integration tests will be skipped.",
            UserWarning,
            stacklevel=2,
        )
        pytest.skip("Azure OpenAI credentials not available in environment variables")

    assert azure_endpoint is not None
    assert azure_api_key is not None
    assert azure_version is not None
    return AzureOpenAIConfig(
        api_key=azure_api_key,
        endpoint=azure_endpoint,
        api_version=azure_version,
        default_model="gpt-4o-2024-11-20",
        default_embedding_model="text-embedding-3-large",
    )


@pytest.fixture
def azure_client_integration(
    azure_config_integration: AzureOpenAIConfig,
) -> AzureOpenAIClient:
    """Provides an Azure OpenAI client for integration testing.

    Args:
        azure_config_integration: Azure OpenAI configuration from environment variables.

    Returns:
        AzureOpenAIClient: Azure OpenAI client instance for integration testing.
    """
    return AzureOpenAIClient(azure_config_integration)


@pytest.fixture
def async_azure_config_integration() -> AsyncAzureOpenAIConfig:
    """Provides async Azure OpenAI configuration from environment variables for integration tests.

    Returns:
        AsyncAzureOpenAIConfig: Async Azure OpenAI configuration if credentials available.
    """
    # Check if required environment variables are set
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_version = os.getenv("AZURE_OPENAI_VERSION")

    azure_credentials_available = all(
        [
            azure_endpoint,
            azure_api_key,
            azure_version,
        ]
    )

    if not azure_credentials_available:
        warnings.warn(
            "Azure OpenAI credentials not available in environment variables. "
            "Integration tests will be skipped.",
            UserWarning,
            stacklevel=2,
        )
        pytest.skip("Azure OpenAI credentials not available in environment variables")

    assert azure_endpoint is not None
    assert azure_api_key is not None
    assert azure_version is not None
    return AsyncAzureOpenAIConfig(
        api_key=azure_api_key,
        endpoint=azure_endpoint,
        api_version=azure_version,
        default_model="gpt-4o-2024-11-20",
        default_embedding_model="text-embedding-3-large",
    )


@pytest.fixture
def async_azure_client_integration(
    async_azure_config_integration: AsyncAzureOpenAIConfig,
) -> AsyncAzureOpenAIClient:
    """Provides an async Azure OpenAI client for integration testing.

    Args:
        async_azure_config_integration: Async Azure OpenAI configuration from environment variables.

    Returns:
        AsyncAzureOpenAIClient: Async Azure OpenAI client instance for integration testing.
    """
    return AsyncAzureOpenAIClient(async_azure_config_integration)


# ==== OPENAI INTEGRATION FIXTURES ====


@pytest.fixture
def openai_config_integration() -> OpenAIConfig:
    """Provides OpenAI configuration from environment variables for integration tests.

    Returns:
        OpenAIConfig: OpenAI configuration if API key available.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        warnings.warn(
            "OpenAI API key not available in environment variables. "
            "Integration tests will be skipped.",
            UserWarning,
            stacklevel=2,
        )
        pytest.skip("OpenAI API key not available in environment variables")

    return OpenAIConfig(
        api_key=api_key,
        default_model="gpt-4o-2024-11-20",
        default_embedding_model="text-embedding-3-large",
    )


@pytest.fixture
def openai_client_integration(openai_config_integration: OpenAIConfig) -> OpenAIClient:
    """Provides an OpenAI client for integration testing.

    Args:
        openai_config_integration: OpenAI configuration from environment variables.

    Returns:
        OpenAI: OpenAI client instance for integration testing.
    """
    return OpenAIClient(openai_config_integration)


@pytest.fixture
def async_openai_config_integration() -> AsyncOpenAIConfig:
    """Provides async OpenAI configuration from environment variables for integration tests.

    Returns:
        AsyncOpenAIConfig: Async OpenAI configuration if API key available.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        warnings.warn(
            "OpenAI API key not available in environment variables. "
            "Integration tests will be skipped.",
            UserWarning,
            stacklevel=2,
        )
        pytest.skip("OpenAI API key not available in environment variables")

    return AsyncOpenAIConfig(
        api_key=api_key,
        default_model="gpt-4o-2024-11-20",
        default_embedding_model="text-embedding-3-large",
    )


@pytest.fixture
def async_openai_client_integration(
    async_openai_config_integration: AsyncOpenAIConfig,
) -> AsyncOpenAIClient:
    """Provides an async OpenAI client for integration testing.

    Args:
        async_openai_config_integration: Async OpenAI configuration from environment variables.

    Returns:
        AsyncOpenAIClient: Async OpenAI client instance for integration testing.
    """
    return AsyncOpenAIClient(async_openai_config_integration)


# ==== MESSAGE FIXTURES ====


@pytest.fixture
def simple_user_message() -> list[Message]:
    """Provides a simple user message for testing.

    Returns:
        list[Message]: A list with one user message.
    """
    return [Message(role=RoleEnum.USER, content="Hello, LLM!")]


@pytest.fixture
def sample_messages() -> list[Message]:
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
def structured_output_messages() -> list[Message]:
    """Provides messages for structured output testing.

    Returns:
        list[Message]: Messages requesting structured response.
    """
    return [Message(role=RoleEnum.USER, content="Generate structured response")]


# ==== MOCK RESPONSE FIXTURES ====


@pytest.fixture
def mock_usage() -> LLMUsage:
    """Provides mock LLM usage statistics.

    Returns:
        LLMUsage: Mock usage statistics.
    """
    return LLMUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40)


@pytest.fixture
def mock_raw_response(mock_usage: LLMUsage) -> tuple[str, LLMUsage]:
    """Provides a mock raw response for testing.

    Args:
        mock_usage: Mock usage statistics.

    Returns:
        tuple[str, LLMUsage]: Mock response content and usage.
    """
    return ("Test response content from mock", mock_usage)


@pytest.fixture
def mock_structured_response() -> ExampleResponseModel:
    """Provides a mock structured response.

    Returns:
        ExampleResponseModel: A mock structured response instance.
    """
    return ExampleResponseModel(
        task_id="task_123",
        status="completed",
        confidence=0.95,
        tags=["important", "urgent"],
    )


@pytest.fixture
def mock_openai_response() -> Mock:
    """Provide a mock OpenAI API response for testing.

    Returns:
        Mock: A mock response object with typical OpenAI structure.
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Hello! I'm an AI assistant."
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 8
    mock_response.usage.total_tokens = 18
    return mock_response


@pytest.fixture
def mock_azure_response() -> Mock:
    """Provides a mock Azure OpenAI API response.

    Returns:
        Mock: A mock Azure OpenAI response object.
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Hello! I'm an AI assistant."
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 8
    mock_response.usage.total_tokens = 18
    return mock_response


@pytest.fixture
def mock_embedding_response() -> Mock:
    """Provides a mock embedding response.

    Returns:
        Mock: A mock embedding response object.
    """
    mock_response = Mock()
    mock_embedding = Mock()
    mock_embedding.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_response.data = [mock_embedding]
    return mock_response


# ==== EMBEDDING FIXTURES ====


@pytest.fixture
def single_text_input() -> list[str]:
    """Provides a single text input for embedding tests.

    Returns:
        list[str]: A list with one text string.
    """
    return ["Hello, world!"]


@pytest.fixture
def multiple_text_inputs() -> list[str]:
    """Provides multiple text inputs for embedding tests.

    Returns:
        list[str]: A list of multiple text strings.
    """
    return ["Hello, world!", "How are you?", "This is a test."]


@pytest.fixture
def mock_single_embedding() -> list[list[float]]:
    """Provides a mock embedding for single text input.

    Returns:
        list[list[float]]: A list with one embedding vector.
    """
    return [[0.1, 0.2, 0.3, 0.4, 0.5]]


@pytest.fixture
def mock_multiple_embeddings() -> list[list[float]]:
    """Provides mock embeddings for multiple text inputs.

    Returns:
        list[list[float]]: A list of multiple embedding vectors.
    """
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0],
        [1.1, 1.2, 1.3, 1.4, 1.5],
    ]


# ==== MOCK HELPERS ====


@pytest.fixture
def mock_logger(mocker) -> MagicMock:
    """Provides a mock logger instance.

    Args:
        mocker: pytest-mock fixture.

    Returns:
        MagicMock: A mock logger instance.
    """
    return mocker.patch("logging.getLogger").return_value


# ==== BATCH PROCESSING FIXTURES ====


@pytest.fixture
def batch_message_lists() -> list[list[Message]]:
    """Provides multiple message lists for batch processing tests.

    Returns:
        list[list[Message]]: List of message conversations for batch testing.
    """
    return [
        [Message(role=RoleEnum.USER, content="Hello, world!")],
        [Message(role=RoleEnum.USER, content="How are you?")],
        [Message(role=RoleEnum.USER, content="What's the weather like?")],
    ]


@pytest.fixture
def batch_text_lists() -> list[list[str]]:
    """Provides multiple text lists for batch embedding tests.

    Returns:
        list[list[str]]: List of text lists for batch embedding testing.
    """
    return [
        ["Hello world", "Goodbye world"],
        ["Good morning", "Good night"],
        ["How are you?", "Fine, thanks"],
    ]


@pytest.fixture
def batch_structured_items() -> list[tuple[list[Message], type[ExampleResponseModel]]]:
    """Provides batch items for structured response testing.

    Returns:
        list[tuple[list[Message], type[ExampleResponseModel]]]: Batch items for testing.
    """
    return [
        ([Message(role=RoleEnum.USER, content="Task 1")], ExampleResponseModel),
        ([Message(role=RoleEnum.USER, content="Task 2")], ExampleResponseModel),
        ([Message(role=RoleEnum.USER, content="Task 3")], ExampleResponseModel),
    ]


# ==== ANTHROPIC FIXTURES ====


@pytest.fixture
def anthropic_config_data() -> dict[str, Any]:
    """Provides valid Anthropic configuration data for unit tests.

    Returns:
        dict[str, Any]: Configuration data for Anthropic unit tests.
    """
    return {
        "api_key": "test-anthropic-key",
        "default_model": "claude-3-5-sonnet-20241022",
    }


@pytest.fixture
def anthropic_config(anthropic_config_data: dict[str, Any]) -> AnthropicConfig:
    """Provides a valid Anthropic configuration for unit tests.

    Args:
        anthropic_config_data: Configuration data dictionary.

    Returns:
        AnthropicConfig: A valid Anthropic configuration instance for unit tests.
    """
    return AnthropicConfig(**anthropic_config_data)


@pytest.fixture
def anthropic_client(anthropic_config: AnthropicConfig) -> AnthropicClient:
    """Provides an Anthropic client for unit tests.

    Args:
        anthropic_config: A valid Anthropic configuration.

    Returns:
        AnthropicClient: Anthropic client instance for unit tests.
    """
    return AnthropicClient(anthropic_config)


@pytest.fixture
def async_anthropic_config(
    anthropic_config_data: dict[str, Any],
) -> AsyncAnthropicConfig:
    """Provides a valid async Anthropic configuration for unit tests.

    Args:
        anthropic_config_data: Configuration data dictionary.

    Returns:
        AsyncAnthropicConfig: A valid async Anthropic configuration instance for unit tests.
    """
    return AsyncAnthropicConfig(**anthropic_config_data)


@pytest.fixture
def async_anthropic_client(
    async_anthropic_config: AsyncAnthropicConfig,
) -> AsyncAnthropicClient:
    """Provides an async Anthropic client for unit tests.

    Args:
        async_anthropic_config: A valid async Anthropic configuration.

    Returns:
        AsyncAnthropicClient: Async Anthropic client instance for unit tests.
    """
    return AsyncAnthropicClient(async_anthropic_config)


@pytest.fixture
def mock_anthropic_response() -> Mock:
    """Provides a mock Anthropic API response for testing.

    Returns:
        Mock: A mock response object with typical Anthropic structure.
    """
    mock_response = Mock()
    mock_content = Mock()
    mock_content.text = "Hello! I'm Claude, an AI assistant."
    mock_response.content = [mock_content]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 15
    return mock_response


# ==== ANTHROPIC INTEGRATION FIXTURES ====


@pytest.fixture
def anthropic_config_integration() -> AnthropicConfig:
    """Provides Anthropic configuration from environment variables for integration tests.

    Returns:
        AnthropicConfig: Anthropic configuration if API key available.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        warnings.warn(
            "Anthropic API key not available in environment variables. "
            "Integration tests will be skipped.",
            UserWarning,
            stacklevel=2,
        )
        pytest.skip("Anthropic API key not available in environment variables")

    return AnthropicConfig(
        api_key=api_key,
        default_model="claude-3-5-sonnet-20241022",
    )


@pytest.fixture
def anthropic_client_integration(
    anthropic_config_integration: AnthropicConfig,
) -> AnthropicClient:
    """Provides an Anthropic client for integration testing.

    Args:
        anthropic_config_integration: Anthropic configuration from environment variables.

    Returns:
        AnthropicClient: Anthropic client instance for integration testing.
    """
    return AnthropicClient(anthropic_config_integration)


@pytest.fixture
def async_anthropic_config_integration() -> AsyncAnthropicConfig:
    """Provides async Anthropic configuration from environment variables for integration tests.

    Returns:
        AsyncAnthropicConfig: Async Anthropic configuration if API key available.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        warnings.warn(
            "Anthropic API key not available in environment variables. "
            "Integration tests will be skipped.",
            UserWarning,
            stacklevel=2,
        )
        pytest.skip("Anthropic API key not available in environment variables")

    return AsyncAnthropicConfig(
        api_key=api_key,
        default_model="claude-3-5-sonnet-20241022",
    )


@pytest.fixture
def async_anthropic_client_integration(
    async_anthropic_config_integration: AsyncAnthropicConfig,
) -> AsyncAnthropicClient:
    """Provides an async Anthropic client for integration testing.

    Args:
        async_anthropic_config_integration: Async Anthropic configuration from environment variables.

    Returns:
        AsyncAnthropicClient: Async Anthropic client instance for integration testing.
    """
    return AsyncAnthropicClient(async_anthropic_config_integration)
