"""Client management functionality for MetaEvaluator."""

import os
from typing import Optional

from ..llm_client.models import LLMClientEnum
from ..llm_client.LLM_client import LLMClient
from ..llm_client.openai_client import OpenAIClient, OpenAIConfig
from ..llm_client.azureopenai_client import AzureOpenAIClient, AzureOpenAIConfig
from .exceptions import (
    MissingConfigurationException,
    ClientAlreadyExistsException,
    ClientNotFoundException,
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
    """Mixin class for MetaEvaluator client management functionality."""

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
            MissingConfigurationException: If required parameters are missing from both arguments and environment.
            ClientAlreadyExistsException: If client already exists and override_existing is False.
        """
        # Check if client already exists
        if LLMClientEnum.OPENAI in self.client_registry and not override_existing:
            raise ClientAlreadyExistsException("OPENAI")

        # Get configuration values, fallback to environment variables
        final_api_key = api_key or os.getenv(_OPENAI_API_KEY_ENV_VAR)
        final_default_model = default_model or os.getenv(_OPENAI_DEFAULT_MODEL_ENV_VAR)
        final_default_embedding_model = default_embedding_model or os.getenv(
            _OPENAI_DEFAULT_EMBEDDING_MODEL_ENV_VAR
        )

        # Validate required parameters
        if not final_api_key:
            raise MissingConfigurationException(
                f"api_key (or {_OPENAI_API_KEY_ENV_VAR} environment variable)"
            )
        if not final_default_model:
            raise MissingConfigurationException(
                f"default_model (or {_OPENAI_DEFAULT_MODEL_ENV_VAR} environment variable)"
            )
        if not final_default_embedding_model:
            raise MissingConfigurationException(
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
            MissingConfigurationException: If required parameters are missing from both arguments and environment.
            ClientAlreadyExistsException: If client already exists and override_existing is False.
        """
        # Check if client already exists
        if LLMClientEnum.AZURE_OPENAI in self.client_registry and not override_existing:
            raise ClientAlreadyExistsException("AZURE_OPENAI")

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
            raise MissingConfigurationException(
                f"api_key (or {_AZURE_OPENAI_API_KEY_ENV_VAR} environment variable)"
            )
        if not final_endpoint:
            raise MissingConfigurationException(
                f"endpoint (or {_AZURE_OPENAI_ENDPOINT_ENV_VAR} environment variable)"
            )
        if not final_api_version:
            raise MissingConfigurationException(
                f"api_version (or {_AZURE_OPENAI_API_VERSION_ENV_VAR} environment variable)"
            )
        if not final_default_model:
            raise MissingConfigurationException(
                f"default_model (or {_AZURE_OPENAI_DEFAULT_MODEL_ENV_VAR} environment variable)"
            )
        if not final_default_embedding_model:
            raise MissingConfigurationException(
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

    def get_client(self, client_type: LLMClientEnum) -> LLMClient:
        """Get a client from the registry by type.

        Args:
            client_type: The LLM client enum type to retrieve.

        Returns:
            LLMClient: The requested LLM client instance.

        Raises:
            ClientNotFoundException: If the client type is not found in the registry.
        """
        if client_type not in self.client_registry:
            raise ClientNotFoundException(client_type.value)

        return self.client_registry[client_type]

    def get_client_list(self) -> list[tuple[LLMClientEnum, LLMClient]]:
        """Get a list of client tuples (type, client).

        Returns:
            List of client tuples.
        """
        return list(self.client_registry.items())
