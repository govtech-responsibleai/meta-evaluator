"""Client management functionality for MetaEvaluator."""

import logging
import os
from typing import Optional

from ..llm_client.azureopenai_client import AzureOpenAIClient, AzureOpenAIConfig
from ..llm_client.client import LLMClient
from ..llm_client.enums import LLMClientEnum
from ..llm_client.openai_client import OpenAIClient, OpenAIConfig
from .exceptions import (
    ClientAlreadyExistsError,
    ClientNotFoundError,
    MissingConfigurationError,
)

# Environment variable constants
_OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
_OPENAI_DEFAULT_MODEL_ENV_VAR = "OPENAI_DEFAULT_MODEL"
_OPENAI_DEFAULT_EMBEDDING_MODEL_ENV_VAR = "OPENAI_DEFAULT_EMBEDDING_MODEL"

_AZURE_OPENAI_API_KEY_ENV_VAR = "AZURE_OPENAI_API_KEY"
_AZURE_OPENAI_ENDPOINT_ENV_VAR = "AZURE_OPENAI_ENDPOINT"
_AZURE_OPENAI_API_VERSION_ENV_VAR = "AZURE_OPENAI_API_VERSION"
_AZURE_OPENAI_DEFAULT_MODEL_ENV_VAR = "AZURE_OPENAI_DEFAULT_MODEL"
_AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL_ENV_VAR = "AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL"


class ClientsMixin:
    """Mixin providing LLM client handling functionality for MetaEvaluator.

    This mixin class handles registration, configuration, and retrieval of various
    LLM clients (OpenAI, Azure OpenAI, etc.). It maintains a client registry that
    maps client types to configured client instances, enabling the MetaEvaluator
    to work with multiple LLM providers seamlessly.

    The mixin automatically handles environment variable fallbacks for API keys
    and configuration parameters, making it easy to configure clients without
    hardcoding sensitive information in code.

    Supported Clients:
        - OpenAI: Standard OpenAI API clients
        - Azure OpenAI: Microsoft Azure-hosted OpenAI models

    Environment Variables:
        OpenAI: OPENAI_API_KEY, OPENAI_DEFAULT_MODEL, OPENAI_DEFAULT_EMBEDDING_MODEL
        Azure: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
               AZURE_OPENAI_DEFAULT_MODEL, AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL

    Attributes:
        client_registry (dict): Maps LLMClientEnum values to configured LLMClient instances.
        logger (logging.Logger): Inherited from MetaEvaluator for consistent logging.

    Examples:
        >>> evaluator = MetaEvaluator()
        >>> # Add OpenAI client with explicit API key
        >>> evaluator.add_openai(api_key="sk-...", default_model="gpt-4")
        >>> # Add Azure client using environment variables
        >>> evaluator.add_azure_openai()  # Uses env vars automatically
        >>> # Retrieve client for judge evaluation
        >>> client = evaluator.get_client(LLMClientEnum.OPENAI)
    """

    # Type hint for logger attribute provided by MetaEvaluator
    logger: logging.Logger

    def __init__(self, *args, **kwargs):
        """Initialize client registry."""
        super().__init__(*args, **kwargs)
        self.client_registry = {}

    def add_openai(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        override_existing: bool = False,
    ):
        """Add an OpenAI client to the registry.

        Args:
            api_key: OpenAI API key. If None, will look for OPENAI_API_KEY in environment.
            default_model: Default model to use. If None, will look for OPENAI_DEFAULT_MODEL in environment.
            default_embedding_model: Default embedding model. If None, will look for OPENAI_DEFAULT_EMBEDDING_MODEL in environment.
            override_existing: Whether to override existing client. Defaults to False.

        Raises:
            MissingConfigurationError: If required parameters are missing from both arguments and environment.
            ClientAlreadyExistsError: If client already exists and override_existing is False.
        """
        self.logger.info("Adding OpenAI client to registry...")

        # Check if client already exists
        if LLMClientEnum.OPENAI in self.client_registry and not override_existing:
            raise ClientAlreadyExistsError("OPENAI")

        # Get configuration values, fallback to environment variables
        final_api_key = api_key or os.getenv(_OPENAI_API_KEY_ENV_VAR)
        final_default_model = default_model or os.getenv(_OPENAI_DEFAULT_MODEL_ENV_VAR)
        final_default_embedding_model = default_embedding_model or os.getenv(
            _OPENAI_DEFAULT_EMBEDDING_MODEL_ENV_VAR
        )

        # Validate required parameters
        if not final_api_key:
            raise MissingConfigurationError(
                f"api_key (or {_OPENAI_API_KEY_ENV_VAR} environment variable)"
            )
        if not final_default_model:
            raise MissingConfigurationError(
                f"default_model (or {_OPENAI_DEFAULT_MODEL_ENV_VAR} environment variable)"
            )
        if not final_default_embedding_model:
            raise MissingConfigurationError(
                f"default_embedding_model (or {_OPENAI_DEFAULT_EMBEDDING_MODEL_ENV_VAR} environment variable)"
            )

        # Create configuration and client
        config = OpenAIConfig(
            api_key=final_api_key,
            default_model=final_default_model,
            default_embedding_model=final_default_embedding_model,
        )
        client = OpenAIClient(config)

        # Add to registry
        self.client_registry[LLMClientEnum.OPENAI] = client
        self.logger.info(
            f"...Successfully added OpenAI client with default model: {final_default_model}"
        )

    def add_azure_openai(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        default_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        override_existing: bool = False,
    ):
        """Add an Azure OpenAI client to the registry.

        Args:
            api_key: Azure OpenAI API key. If None, will look for AZURE_OPENAI_API_KEY in environment.
            endpoint: Azure OpenAI endpoint. If None, will look for AZURE_OPENAI_ENDPOINT in environment.
            api_version: Azure OpenAI API version. If None, will look for AZURE_OPENAI_API_VERSION in environment.
            default_model: Default model to use. If None, will look for AZURE_OPENAI_DEFAULT_MODEL in environment.
            default_embedding_model: Default embedding model. If None, will look for AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL in environment.
            override_existing: Whether to override existing client. Defaults to False.

        Raises:
            MissingConfigurationError: If required parameters are missing from both arguments and environment.
            ClientAlreadyExistsError: If client already exists and override_existing is False.
        """
        self.logger.info("Adding Azure OpenAI client to registry...")

        # Check if client already exists
        if LLMClientEnum.AZURE_OPENAI in self.client_registry and not override_existing:
            raise ClientAlreadyExistsError("AZURE_OPENAI")

        # Get configuration values, fallback to environment variables
        final_api_key = api_key or os.getenv(_AZURE_OPENAI_API_KEY_ENV_VAR)
        final_endpoint = endpoint or os.getenv(_AZURE_OPENAI_ENDPOINT_ENV_VAR)
        final_api_version = api_version or os.getenv(_AZURE_OPENAI_API_VERSION_ENV_VAR)
        final_default_model = default_model or os.getenv(
            _AZURE_OPENAI_DEFAULT_MODEL_ENV_VAR
        )
        final_default_embedding_model = default_embedding_model or os.getenv(
            _AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL_ENV_VAR
        )

        # Validate required parameters
        if not final_api_key:
            raise MissingConfigurationError(
                f"api_key (or {_AZURE_OPENAI_API_KEY_ENV_VAR} environment variable)"
            )
        if not final_endpoint:
            raise MissingConfigurationError(
                f"endpoint (or {_AZURE_OPENAI_ENDPOINT_ENV_VAR} environment variable)"
            )
        if not final_api_version:
            raise MissingConfigurationError(
                f"api_version (or {_AZURE_OPENAI_API_VERSION_ENV_VAR} environment variable)"
            )
        if not final_default_model:
            raise MissingConfigurationError(
                f"default_model (or {_AZURE_OPENAI_DEFAULT_MODEL_ENV_VAR} environment variable)"
            )
        if not final_default_embedding_model:
            raise MissingConfigurationError(
                f"default_embedding_model (or {_AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL_ENV_VAR} environment variable)"
            )

        # Create configuration and client
        config = AzureOpenAIConfig(
            api_key=final_api_key,
            endpoint=final_endpoint,
            api_version=final_api_version,
            default_model=final_default_model,
            default_embedding_model=final_default_embedding_model,
        )
        client = AzureOpenAIClient(config)

        # Add to registry
        self.client_registry[LLMClientEnum.AZURE_OPENAI] = client
        self.logger.info(
            f"...Successfully added Azure OpenAI client with default model: {final_default_model}"
        )

    def get_client(self, client_type: LLMClientEnum) -> LLMClient:
        """Get a client from the registry by type.

        Args:
            client_type: The LLM client enum type to retrieve.

        Returns:
            LLMClient: The requested LLM client instance.

        Raises:
            ClientNotFoundError: If the client type is not found in the registry.
        """
        if client_type not in self.client_registry:
            raise ClientNotFoundError(client_type.value)

        return self.client_registry[client_type]

    def get_client_list(self) -> list[tuple[LLMClientEnum, LLMClient]]:
        """Get a list of client tuples (type, client).

        Returns:
            List of client tuples.
        """
        return list(self.client_registry.items())
