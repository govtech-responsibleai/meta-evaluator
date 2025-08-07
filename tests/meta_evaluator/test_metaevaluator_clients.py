"""Test suite for the MetaEvaluator client management."""

import json
import pytest
from unittest.mock import MagicMock, patch
from meta_evaluator.meta_evaluator import MetaEvaluator
from meta_evaluator.meta_evaluator.exceptions import (
    MissingConfigurationError,
    ClientAlreadyExistsError,
    ClientNotFoundError,
)
from meta_evaluator.llm_client.models import LLMClientEnum
from meta_evaluator.llm_client.openai_client import OpenAIClient
from meta_evaluator.llm_client.azureopenai_client import AzureOpenAIClient


@pytest.mark.integration
class TestMetaEvaluatorClients:
    """Test suite for MetaEvaluator client management functionality."""

    # === Client-specific Fixtures ===

    @pytest.fixture
    def clean_environment(self, monkeypatch):
        """Provides a clean environment without OpenAI-related environment variables.

        Args:
            monkeypatch: pytest fixture for environment variable manipulation.
        """
        # Remove all OpenAI and Azure OpenAI related environment variables
        env_vars_to_remove = [
            "OPENAI_API_KEY",
            "OPENAI_DEFAULT_MODEL",
            "OPENAI_DEFAULT_EMBEDDING_MODEL",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_DEFAULT_MODEL",
            "AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL",
        ]
        for var in env_vars_to_remove:
            monkeypatch.delenv(var, raising=False)

    @pytest.fixture
    def openai_environment(self, monkeypatch):
        """Provides environment with all OpenAI variables set.

        Args:
            monkeypatch: pytest fixture for environment variable manipulation.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-4")
        monkeypatch.setenv("OPENAI_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-large")

    @pytest.fixture
    def azure_openai_environment(self, monkeypatch):
        """Provides environment with all Azure OpenAI variables set.

        Args:
            monkeypatch: pytest fixture for environment variable manipulation.
        """
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AZURE_OPENAI_DEFAULT_MODEL", "gpt-4")
        monkeypatch.setenv(
            "AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL", "text-embedding-ada-002"
        )

    # === Helper Methods ===

    # === add_openai() Method Tests ===

    def test_add_openai_with_all_parameters(self, meta_evaluator, clean_environment):
        """Test adding OpenAI client with all parameters provided."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            meta_evaluator.add_openai(
                api_key="test-key",
                default_model="gpt-4",
                default_embedding_model="text-embedding-3-large",
            )

            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry
            assert meta_evaluator.client_registry[LLMClientEnum.OPENAI] == mock_client
            mock_client_class.assert_called_once()

    def test_add_openai_from_environment_variables(
        self, meta_evaluator, openai_environment
    ):
        """Test adding OpenAI client using only environment variables."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            meta_evaluator.add_openai()

            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry
            mock_client_class.assert_called_once()

    def test_add_openai_mixed_parameters_and_environment(
        self, meta_evaluator, openai_environment
    ):
        """Test adding OpenAI client with mixed parameters and environment variables."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            # Provide only api_key, others should come from environment
            meta_evaluator.add_openai(api_key="override-key")

            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry
            # Verify the config was created with override key but env vars for models
            call_args = mock_client_class.call_args[0][0]
            assert call_args.api_key == "override-key"
            assert call_args.default_model == "gpt-4"  # From environment
            assert (
                call_args.default_embedding_model == "text-embedding-3-large"
            )  # From environment

    def test_add_openai_parameter_precedence_over_environment(
        self, meta_evaluator, openai_environment
    ):
        """Test that provided parameters take precedence over environment variables."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            meta_evaluator.add_openai(
                api_key="param-key",
                default_model="param-model",
                default_embedding_model="param-embedding",
            )

            call_args = mock_client_class.call_args[0][0]
            assert call_args.api_key == "param-key"
            assert call_args.default_model == "param-model"
            assert call_args.default_embedding_model == "param-embedding"

    def test_add_openai_missing_api_key(self, meta_evaluator, clean_environment):
        """Test MissingConfigurationError when api_key is missing."""
        with pytest.raises(MissingConfigurationError, match="api_key"):
            meta_evaluator.add_openai(
                default_model="gpt-4", default_embedding_model="text-embedding-3-large"
            )

    def test_add_openai_missing_default_model(self, meta_evaluator, clean_environment):
        """Test MissingConfigurationError when default_model is missing."""
        with pytest.raises(MissingConfigurationError, match="default_model"):
            meta_evaluator.add_openai(
                api_key="test-key", default_embedding_model="text-embedding-3-large"
            )

    def test_add_openai_missing_default_embedding_model(
        self, meta_evaluator, clean_environment
    ):
        """Test MissingConfigurationError when default_embedding_model is missing."""
        with pytest.raises(MissingConfigurationError, match="default_embedding_model"):
            meta_evaluator.add_openai(api_key="test-key", default_model="gpt-4")

    def test_add_openai_client_already_exists_no_override(
        self, meta_evaluator, openai_environment
    ):
        """Test ClientAlreadyExistsError when client exists and override_existing is False."""
        with patch("meta_evaluator.meta_evaluator.clients.OpenAIClient"):
            # Add client first time
            meta_evaluator.add_openai()

            # Try to add again without override
            with pytest.raises(
                ClientAlreadyExistsError, match="OPENAI.*already exists"
            ):
                meta_evaluator.add_openai()

    def test_add_openai_client_already_exists_with_override(
        self, meta_evaluator, openai_environment
    ):
        """Test successful override when client exists and override_existing is True."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client1 = MagicMock(spec=OpenAIClient)
            mock_client2 = MagicMock(spec=OpenAIClient)
            mock_client_class.side_effect = [mock_client1, mock_client2]

            # Add client first time
            meta_evaluator.add_openai()
            original_client = meta_evaluator.client_registry[LLMClientEnum.OPENAI]

            # Override with new client
            meta_evaluator.add_openai(override_existing=True)
            new_client = meta_evaluator.client_registry[LLMClientEnum.OPENAI]

            assert original_client != new_client
            assert new_client == mock_client2

    def test_add_openai_empty_string_parameters(
        self, meta_evaluator, clean_environment
    ):
        """Test that empty string parameters are treated as missing."""
        with pytest.raises(MissingConfigurationError, match="api_key"):
            meta_evaluator.add_openai(
                api_key="",  # Empty string should be treated as missing
                default_model="gpt-4",
                default_embedding_model="text-embedding-3-large",
            )

    # === add_azure_openai() Method Tests ===

    def test_add_azure_openai_with_all_parameters(
        self, meta_evaluator, clean_environment
    ):
        """Test adding Azure OpenAI client with all parameters provided."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.AzureOpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=AzureOpenAIClient)
            mock_client_class.return_value = mock_client

            meta_evaluator.add_azure_openai(
                api_key="test-key",
                endpoint="https://test.openai.azure.com",
                api_version="2024-02-15-preview",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

            assert LLMClientEnum.AZURE_OPENAI in meta_evaluator.client_registry
            assert (
                meta_evaluator.client_registry[LLMClientEnum.AZURE_OPENAI]
                == mock_client
            )

    def test_add_azure_openai_from_environment_variables(
        self, meta_evaluator, azure_openai_environment
    ):
        """Test adding Azure OpenAI client using only environment variables."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.AzureOpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=AzureOpenAIClient)
            mock_client_class.return_value = mock_client

            meta_evaluator.add_azure_openai()

            assert LLMClientEnum.AZURE_OPENAI in meta_evaluator.client_registry

    def test_add_azure_openai_missing_api_key(self, meta_evaluator, clean_environment):
        """Test MissingConfigurationError when api_key is missing."""
        with pytest.raises(MissingConfigurationError, match="api_key"):
            meta_evaluator.add_azure_openai(
                endpoint="https://test.openai.azure.com",
                api_version="2024-02-15-preview",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_add_azure_openai_missing_endpoint(self, meta_evaluator, clean_environment):
        """Test MissingConfigurationError when endpoint is missing."""
        with pytest.raises(MissingConfigurationError, match="endpoint"):
            meta_evaluator.add_azure_openai(
                api_key="test-key",
                api_version="2024-02-15-preview",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_add_azure_openai_missing_api_version(
        self, meta_evaluator, clean_environment
    ):
        """Test MissingConfigurationError when api_version is missing."""
        with pytest.raises(MissingConfigurationError, match="api_version"):
            meta_evaluator.add_azure_openai(
                api_key="test-key",
                endpoint="https://test.openai.azure.com",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_add_azure_openai_missing_default_model(
        self, meta_evaluator, clean_environment
    ):
        """Test MissingConfigurationError when default_model is missing."""
        with pytest.raises(MissingConfigurationError, match="default_model"):
            meta_evaluator.add_azure_openai(
                api_key="test-key",
                endpoint="https://test.openai.azure.com",
                api_version="2024-02-15-preview",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_add_azure_openai_missing_default_embedding_model(
        self, meta_evaluator, clean_environment
    ):
        """Test MissingConfigurationError when default_embedding_model is missing."""
        with pytest.raises(MissingConfigurationError, match="default_embedding_model"):
            meta_evaluator.add_azure_openai(
                api_key="test-key",
                endpoint="https://test.openai.azure.com",
                api_version="2024-02-15-preview",
                default_model="gpt-4",
            )

    def test_add_azure_openai_client_already_exists_no_override(
        self, meta_evaluator, azure_openai_environment
    ):
        """Test ClientAlreadyExistsError when Azure client exists and override_existing is False."""
        with patch("meta_evaluator.meta_evaluator.clients.AzureOpenAIClient"):
            # Add client first time
            meta_evaluator.add_azure_openai()

            # Try to add again without override
            with pytest.raises(
                ClientAlreadyExistsError, match="AZURE_OPENAI.*already exists"
            ):
                meta_evaluator.add_azure_openai()

    def test_add_azure_openai_client_already_exists_with_override(
        self, meta_evaluator, azure_openai_environment
    ):
        """Test successful override when Azure client exists and override_existing is True."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.AzureOpenAIClient"
        ) as mock_client_class:
            mock_client1 = MagicMock(spec=AzureOpenAIClient)
            mock_client2 = MagicMock(spec=AzureOpenAIClient)
            mock_client_class.side_effect = [mock_client1, mock_client2]

            # Add client first time
            meta_evaluator.add_azure_openai()
            original_client = meta_evaluator.client_registry[LLMClientEnum.AZURE_OPENAI]

            # Override with new client
            meta_evaluator.add_azure_openai(override_existing=True)
            new_client = meta_evaluator.client_registry[LLMClientEnum.AZURE_OPENAI]

            assert original_client != new_client
            assert new_client == mock_client2

    # === get_client() Method Tests ===

    def test_get_client_existing_openai_client(
        self, meta_evaluator, openai_environment
    ):
        """Test retrieving an existing OpenAI client."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            meta_evaluator.add_openai()
            retrieved_client = meta_evaluator.get_client(LLMClientEnum.OPENAI)

            assert retrieved_client == mock_client
            assert isinstance(retrieved_client, MagicMock)  # Mocked OpenAIClient

    def test_get_client_existing_azure_openai_client(
        self, meta_evaluator, azure_openai_environment
    ):
        """Test retrieving an existing Azure OpenAI client."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.AzureOpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=AzureOpenAIClient)
            mock_client_class.return_value = mock_client

            meta_evaluator.add_azure_openai()
            retrieved_client = meta_evaluator.get_client(LLMClientEnum.AZURE_OPENAI)

            assert retrieved_client == mock_client

    def test_get_client_nonexistent_client(self, meta_evaluator):
        """Test ClientNotFoundError when requesting non-existent client."""
        # (?i) makes the regex case-insensitive to match both "OPENAI" and "openai"
        with pytest.raises(ClientNotFoundError, match="(?i)OPENAI.*not found"):
            meta_evaluator.get_client(LLMClientEnum.OPENAI)

    def test_get_client_from_empty_registry(self, meta_evaluator):
        """Test ClientNotFoundError when registry is empty."""
        assert meta_evaluator.client_registry == {}

        # (?i) makes the regex case-insensitive to match both "GEMINI" and "gemini"
        with pytest.raises(ClientNotFoundError, match="(?i)GEMINI.*not found"):
            meta_evaluator.get_client(LLMClientEnum.GEMINI)

    def test_get_client_different_enum_values(self, meta_evaluator):
        """Test ClientNotFoundError for different enum values."""
        # Test all enum values that don't exist
        for client_enum in [LLMClientEnum.GEMINI, LLMClientEnum.ANTHROPIC]:
            with pytest.raises(
                ClientNotFoundError, match=f"{client_enum.value}.*not found"
            ):
                meta_evaluator.get_client(client_enum)

    # === load_state() Method Tests ===

    def test_load_state_with_openai_client(self, tmp_path):
        """Test loading MetaEvaluator with OpenAI client from JSON."""
        # First save a state with OpenAI client
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            # Create proper mock config with serialize method
            from meta_evaluator.llm_client.openai_client import OpenAIConfig
            from meta_evaluator.llm_client.serialization import OpenAISerializedState

            mock_config = MagicMock(spec=OpenAIConfig)
            mock_config.default_model = "gpt-4"
            mock_config.default_embedding_model = "text-embedding-ada-002"
            mock_config.supports_structured_output = True
            mock_config.supports_logprobs = True

            # Mock the serialize method to return proper state
            mock_state = OpenAISerializedState(
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
                supports_structured_output=True,
                supports_logprobs=True,
                supports_instructor=True,
            )
            mock_config.serialize.return_value = mock_state

            # Create mock client with config
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client.config = mock_config
            mock_client_class.return_value = mock_client

            # Create and save evaluator
            original_evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            original_evaluator.add_openai()

            original_evaluator.save_state("test_state.json", include_data=False)

            # Load from JSON (provide API key for reconstruction)
            loaded_evaluator = MetaEvaluator.load_state(
                project_dir=str(original_evaluator.project_dir),
                state_filename="test_state.json",
                load_data=False,
                openai_api_key="test-api-key",
            )

            # Verify client was reconstructed
            assert LLMClientEnum.OPENAI in loaded_evaluator.client_registry
            assert loaded_evaluator.data is None

    def test_load_state_with_azure_openai_client(self, tmp_path):
        """Test loading MetaEvaluator with Azure OpenAI client from JSON."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.AzureOpenAIClient"
        ) as mock_client_class:
            # Create proper mock config with serialize method
            from meta_evaluator.llm_client.azureopenai_client import AzureOpenAIConfig
            from meta_evaluator.llm_client.serialization import (
                AzureOpenAISerializedState,
            )

            mock_config = MagicMock(spec=AzureOpenAIConfig)
            mock_config.endpoint = "https://test.openai.azure.com"
            mock_config.api_version = "2024-02-15-preview"
            mock_config.default_model = "gpt-4"
            mock_config.default_embedding_model = "text-embedding-ada-002"
            mock_config.supports_structured_output = True
            mock_config.supports_logprobs = True

            # Mock the serialize method to return proper state
            mock_state = AzureOpenAISerializedState(
                endpoint="https://test.openai.azure.com",
                api_version="2024-02-15-preview",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
                supports_structured_output=True,
                supports_logprobs=True,
                supports_instructor=True,
            )
            mock_config.serialize.return_value = mock_state

            # Create mock client with config
            mock_client = MagicMock(spec=AzureOpenAIClient)
            mock_client.config = mock_config
            mock_client_class.return_value = mock_client

            # Create and save evaluator
            original_evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            original_evaluator.add_azure_openai()

            original_evaluator.save_state("test_state.json", include_data=False)

            # Load from JSON (provide API key for reconstruction)
            loaded_evaluator = MetaEvaluator.load_state(
                project_dir=str(original_evaluator.project_dir),
                state_filename="test_state.json",
                load_data=False,
                azure_openai_api_key="test-api-key",
            )

            # Verify client was reconstructed
            assert LLMClientEnum.AZURE_OPENAI in loaded_evaluator.client_registry
            assert loaded_evaluator.data is None

    def test_load_state_missing_api_keys(self, tmp_path, clean_environment):
        """Test MissingConfigurationError when API keys are missing."""
        # Create valid state file with OpenAI client
        state_data = {
            "version": "1.0",
            "client_registry": {
                "openai": {
                    "client_type": "openai",
                    "default_model": "gpt-4",
                    "default_embedding_model": "text-embedding-ada-002",
                    "supports_structured_output": True,
                    "supports_logprobs": True,
                    "supports_instructor": True,
                }
            },
            "data": None,
        }

        state_file = tmp_path / "test.json"
        with open(state_file, "w") as f:
            json.dump(state_data, f)

        with pytest.raises(MissingConfigurationError, match="api_key"):
            MetaEvaluator.load_state(str(tmp_path), "test.json")

    def test_load_state_with_custom_api_keys(self, tmp_path):
        """Test loading with custom API keys provided as parameters."""
        # Create state file with both OpenAI and Azure OpenAI clients
        state_data = {
            "version": "1.0",
            "client_registry": {
                "openai": {
                    "client_type": "openai",
                    "default_model": "gpt-4",
                    "default_embedding_model": "text-embedding-ada-002",
                    "supports_structured_output": True,
                    "supports_logprobs": True,
                    "supports_instructor": True,
                },
                "azure_openai": {
                    "client_type": "azure_openai",
                    "endpoint": "https://test.openai.azure.com",
                    "api_version": "2024-02-15-preview",
                    "default_model": "gpt-4",
                    "default_embedding_model": "text-embedding-ada-002",
                    "supports_structured_output": True,
                    "supports_logprobs": True,
                    "supports_instructor": True,
                },
            },
            "data": None,
        }

        state_file = tmp_path / "test.json"
        with open(state_file, "w") as f:
            json.dump(state_data, f)

        with (
            patch(
                "meta_evaluator.meta_evaluator.clients.OpenAIClient"
            ) as mock_openai_class,
            patch(
                "meta_evaluator.meta_evaluator.clients.AzureOpenAIClient"
            ) as mock_azure_class,
        ):
            mock_openai_client = MagicMock(spec=OpenAIClient)
            mock_azure_client = MagicMock(spec=AzureOpenAIClient)
            mock_openai_class.return_value = mock_openai_client
            mock_azure_class.return_value = mock_azure_client

            # Load with custom API keys
            loaded_evaluator = MetaEvaluator.load_state(
                project_dir=str(tmp_path),
                state_filename="test.json",
                load_data=False,
                openai_api_key="custom-openai-key",
                azure_openai_api_key="custom-azure-key",
            )

            # Verify both clients were reconstructed
            assert LLMClientEnum.OPENAI in loaded_evaluator.client_registry
            assert LLMClientEnum.AZURE_OPENAI in loaded_evaluator.client_registry

    # === serialize_client_registry() Method Tests ===

    def test_serialize_client_registry_empty(self, meta_evaluator):
        """Test _serialize_client_registry with empty registry."""
        serialized = meta_evaluator._serialize_client_registry()

        assert serialized == {}
        assert isinstance(serialized, dict)

    def test_serialize_client_registry_with_openai_client(
        self, meta_evaluator, create_mock_openai_client
    ):
        """Test _serialize_client_registry with OpenAI client present."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = create_mock_openai_client()
            mock_client_class.return_value = mock_client

            meta_evaluator.add_openai(
                api_key="test-key",
                default_model="gpt-4",
                default_embedding_model="text-embedding-3-large",
            )
            serialized = meta_evaluator._serialize_client_registry()

            assert "openai" in serialized
            assert serialized["openai"]["client_type"] == "openai"
            assert serialized["openai"]["default_model"] == "gpt-4"
            assert "api_key" not in str(serialized)

    def test_serialize_client_registry_with_both_clients(
        self, meta_evaluator, create_mock_openai_client, create_mock_azure_openai_client
    ):
        """Test _serialize_client_registry with both OpenAI and Azure clients."""
        with (
            patch(
                "meta_evaluator.meta_evaluator.clients.OpenAIClient"
            ) as mock_openai_class,
            patch(
                "meta_evaluator.meta_evaluator.clients.AzureOpenAIClient"
            ) as mock_azure_class,
        ):
            mock_openai_client = create_mock_openai_client()
            mock_openai_class.return_value = mock_openai_client

            mock_azure_client = create_mock_azure_openai_client()
            mock_azure_class.return_value = mock_azure_client

            meta_evaluator.add_openai(
                api_key="test-key",
                default_model="gpt-4",
                default_embedding_model="text-embedding-3-large",
            )
            meta_evaluator.add_azure_openai(
                api_key="test-azure-key",
                endpoint="https://test.openai.azure.com",
                api_version="2024-02-15-preview",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )
            serialized = meta_evaluator._serialize_client_registry()

            assert len(serialized) == 2
            assert "openai" in serialized
            assert "azure_openai" in serialized
            assert serialized["openai"]["client_type"] == "openai"
            assert serialized["azure_openai"]["client_type"] == "azure_openai"

    def test_serialize_single_client_delegates_to_config(self, meta_evaluator):
        """Test _serialize_single_client delegates to client config serialize method."""
        from meta_evaluator.llm_client.LLM_client import LLMClient, LLMClientConfig
        from meta_evaluator.llm_client.serialization import MockLLMClientSerializedState

        # Create mock config with serialize method
        mock_config = MagicMock(spec=LLMClientConfig)
        mock_state = MockLLMClientSerializedState(
            default_model="test-model",
            default_embedding_model="test-embedding",
            supports_structured_output=True,
            supports_logprobs=False,
            supports_instructor=False,
        )
        mock_config.serialize.return_value = mock_state

        # Create mock client with config
        mock_client = MagicMock(spec=LLMClient)
        mock_client.config = mock_config

        result = meta_evaluator._serialize_single_client(
            LLMClientEnum.TEST, mock_client
        )

        # Verify config.serialize was called and result is properly dumped
        mock_config.serialize.assert_called_once()
        assert result["client_type"] == "test"
        assert result["default_model"] == "test-model"

    # === save_state() Method Tests ===

    def test_save_with_multiple_clients_and_sample_data(
        self,
        meta_evaluator,
        mock_sample_eval_data,
        create_mock_openai_client,
        create_mock_azure_openai_client,
    ):
        """Test saving state with multiple clients and SampleEvalData together."""
        with (
            patch(
                "meta_evaluator.meta_evaluator.clients.OpenAIClient"
            ) as mock_openai_class,
            patch(
                "meta_evaluator.meta_evaluator.clients.AzureOpenAIClient"
            ) as mock_azure_class,
        ):
            # Set up OpenAI client
            mock_openai_client = create_mock_openai_client(supports_logprobs=False)
            mock_openai_class.return_value = mock_openai_client

            # Set up Azure client
            mock_azure_client = create_mock_azure_openai_client(
                supports_structured_output=False
            )
            mock_azure_class.return_value = mock_azure_client

            # Add both clients and sample data
            meta_evaluator.add_openai(
                api_key="test-key",
                default_model="gpt-4",
                default_embedding_model="text-embedding-3-large",
            )
            meta_evaluator.add_azure_openai(
                api_key="test-azure-key",
                endpoint="https://test.openai.azure.com",
                api_version="2024-02-15-preview",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )
            meta_evaluator.add_data(mock_sample_eval_data)

            # Save state
            meta_evaluator.save_state(
                "multi_client_test.json",
                include_data=True,
                data_format="csv",
                data_filename="test_state_data.csv",
            )

            # Verify file exists
            state_file = meta_evaluator.project_dir / "multi_client_test.json"
            assert state_file.exists()

            # Verify contents
            with open(state_file) as f:
                state_data = json.load(f)

            # Verify multiple clients
            assert len(state_data["client_registry"]) == 2
            assert "openai" in state_data["client_registry"]
            assert "azure_openai" in state_data["client_registry"]

            # Verify Azure client specific fields
            azure_config = state_data["client_registry"]["azure_openai"]
            assert azure_config["client_type"] == "azure_openai"
            assert azure_config["endpoint"] == "https://test.openai.azure.com"
            assert azure_config["api_version"] == "2024-02-15-preview"

            # Verify SampleEvalData specific fields
            data_config = state_data["data"]
            assert data_config["type"] == "SampleEvalData"
            assert data_config["sample_name"] == "Test Sample"
            assert data_config["stratification_columns"] == ["topic", "difficulty"]
            assert data_config["sample_percentage"] == 0.3
            assert data_config["seed"] == 42
            assert data_config["sampling_method"] == "stratified_by_columns"
            assert data_config["data_format"] == "csv"

    # === Security Assertion Tests ===

    def test_api_key_security_assertion_would_trigger(self, meta_evaluator):
        """Test that the security assertion would catch API key in serialized data."""
        from meta_evaluator.llm_client.LLM_client import LLMClient, LLMClientConfig

        # Create a mock config that returns a state containing "api_key" (which should never happen)
        mock_config = MagicMock(spec=LLMClientConfig)

        # Create a mock state that will return api_key in model_dump
        mock_state = MagicMock()
        mock_state.model_dump.return_value = {
            "api_key": "secret",
            "client_type": "test",
            "default_model": "test-model",
        }
        mock_config.serialize.return_value = mock_state

        # Create mock client with the config
        mock_client = MagicMock(spec=LLMClient)
        mock_client.config = mock_config

        # Add the client to registry
        meta_evaluator.client_registry[LLMClientEnum.TEST] = mock_client

        # This should trigger the assertion
        with pytest.raises(
            AssertionError, match="API key found in serialized test client"
        ):
            meta_evaluator._serialize_client_registry()

    # === Integration Tests ===

    def test_add_multiple_clients_and_retrieve(
        self, meta_evaluator, openai_environment, azure_openai_environment
    ):
        """Test adding multiple clients and retrieving them."""
        with (
            patch(
                "meta_evaluator.meta_evaluator.clients.OpenAIClient"
            ) as mock_openai_class,
            patch(
                "meta_evaluator.meta_evaluator.clients.AzureOpenAIClient"
            ) as mock_azure_class,
        ):
            mock_openai_client = MagicMock(spec=OpenAIClient)
            mock_azure_client = MagicMock(spec=AzureOpenAIClient)
            mock_openai_class.return_value = mock_openai_client
            mock_azure_class.return_value = mock_azure_client

            # Add both clients
            meta_evaluator.add_openai()
            meta_evaluator.add_azure_openai()

            # Verify both exist in registry
            assert len(meta_evaluator.client_registry) == 2
            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry
            assert LLMClientEnum.AZURE_OPENAI in meta_evaluator.client_registry

            # Retrieve both clients
            openai_client = meta_evaluator.get_client(LLMClientEnum.OPENAI)
            azure_client = meta_evaluator.get_client(LLMClientEnum.AZURE_OPENAI)

            assert openai_client == mock_openai_client
            assert azure_client == mock_azure_client

    def test_override_client_then_retrieve(self, meta_evaluator, openai_environment):
        """Test overriding a client and then retrieving the new one."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client1 = MagicMock(spec=OpenAIClient)
            mock_client2 = MagicMock(spec=OpenAIClient)
            mock_client_class.side_effect = [mock_client1, mock_client2]

            # Add initial client
            meta_evaluator.add_openai()
            initial_client = meta_evaluator.get_client(LLMClientEnum.OPENAI)
            assert initial_client == mock_client1

            # Override client
            meta_evaluator.add_openai(override_existing=True)
            overridden_client = meta_evaluator.get_client(LLMClientEnum.OPENAI)
            assert overridden_client == mock_client2
            assert overridden_client != initial_client

    def test_client_registry_state_management(self, meta_evaluator, openai_environment):
        """Test that client registry state is properly maintained across operations."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            # Initially empty
            assert len(meta_evaluator.client_registry) == 0

            # Add client
            meta_evaluator.add_openai()
            assert len(meta_evaluator.client_registry) == 1
            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry

            # Verify client persists across method calls
            retrieved_client = meta_evaluator.get_client(LLMClientEnum.OPENAI)
            assert retrieved_client == mock_client
            assert len(meta_evaluator.client_registry) == 1

    # === Client + Data + Task Integration Tests ===

    def test_add_client_and_data_together(
        self,
        meta_evaluator,
        sample_eval_data,
        basic_eval_task,
    ):
        """Test adding OpenAI client, data, and evaluation task together, verify all exist independently."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            # Add both client and data
            meta_evaluator.add_openai()
            meta_evaluator.add_data(sample_eval_data)
            meta_evaluator.add_eval_task(basic_eval_task)

            # Verify both exist and are independent
            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry
            assert meta_evaluator.client_registry[LLMClientEnum.OPENAI] == mock_client
            assert meta_evaluator.data == sample_eval_data
            assert meta_evaluator.eval_task == basic_eval_task

    def test_add_data_then_client(
        self, meta_evaluator, sample_eval_data, openai_environment
    ):
        """Test adding data first, then client, verify both preserved."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            # Add data first
            meta_evaluator.add_data(sample_eval_data)
            assert meta_evaluator.data == sample_eval_data
            assert len(meta_evaluator.client_registry) == 0

            # Add client
            meta_evaluator.add_openai()

            # Verify both preserved
            assert meta_evaluator.data == sample_eval_data
            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry
            assert meta_evaluator.client_registry[LLMClientEnum.OPENAI] == mock_client

    def test_add_client_then_data(
        self, meta_evaluator, sample_eval_data, openai_environment
    ):
        """Test adding client first, then data, verify both preserved."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            # Add client first
            meta_evaluator.add_openai()
            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry
            assert meta_evaluator.data is None

            # Add data
            meta_evaluator.add_data(sample_eval_data)

            # Verify both preserved
            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry
            assert meta_evaluator.client_registry[LLMClientEnum.OPENAI] == mock_client
            assert meta_evaluator.data == sample_eval_data

    def test_overwrite_data_preserves_clients(
        self, meta_evaluator, sample_eval_data, another_eval_data, openai_environment
    ):
        """Test that overwriting data doesn't affect client registry."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            # Add client and initial data
            meta_evaluator.add_openai()
            meta_evaluator.add_data(sample_eval_data)

            original_client = meta_evaluator.client_registry[LLMClientEnum.OPENAI]

            # Overwrite data
            meta_evaluator.add_data(another_eval_data, overwrite=True)

            # Verify client preserved, data changed
            assert (
                meta_evaluator.client_registry[LLMClientEnum.OPENAI] == original_client
            )
            assert meta_evaluator.data == another_eval_data
            assert meta_evaluator.data != sample_eval_data

    def test_overwrite_task_preserves_clients(
        self,
        meta_evaluator,
        basic_eval_task,
        another_basic_eval_task,
        openai_environment,
    ):
        """Test that overwriting data doesn't affect client registry."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            # Add client and task
            meta_evaluator.add_openai()
            meta_evaluator.add_eval_task(basic_eval_task)

            original_client = meta_evaluator.client_registry[LLMClientEnum.OPENAI]

            # Overwrite task
            meta_evaluator.add_eval_task(another_basic_eval_task, overwrite=True)

            # Verify client preserved, data changed
            assert (
                meta_evaluator.client_registry[LLMClientEnum.OPENAI] == original_client
            )
            assert meta_evaluator.eval_task == another_basic_eval_task
            assert meta_evaluator.eval_task != basic_eval_task

    # === Client Registry Integrity Tests ===

    def test_add_data_preserves_client_registry(self, meta_evaluator, sample_eval_data):
        """Test that adding data doesn't affect client registry."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            # Add client first
            meta_evaluator.add_openai()
            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry
            original_client = meta_evaluator.client_registry[LLMClientEnum.OPENAI]

            # Add data
            meta_evaluator.add_data(sample_eval_data)

            # Verify client registry is unchanged
            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry
            assert (
                meta_evaluator.client_registry[LLMClientEnum.OPENAI] == original_client
            )
            assert meta_evaluator.data == sample_eval_data

    def test_add_eval_task_preserves_client_registry(
        self, meta_evaluator, basic_eval_task
    ):
        """Test that adding evaluation task doesn't affect client registry."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            # Add client first
            meta_evaluator.add_openai()
            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry
            original_client = meta_evaluator.client_registry[LLMClientEnum.OPENAI]

            # Add evaluation task
            meta_evaluator.add_eval_task(basic_eval_task)

            # Verify client registry is unchanged
            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry
            assert (
                meta_evaluator.client_registry[LLMClientEnum.OPENAI] == original_client
            )
            assert meta_evaluator.eval_task == basic_eval_task

    # === Edge Cases and Error Handling ===

    def test_environment_variables_empty_strings(
        self, meta_evaluator, clean_environment, monkeypatch
    ):
        """Test behavior when environment variables are set to empty strings."""
        # Set environment variables to empty strings
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "")
        monkeypatch.setenv("OPENAI_DEFAULT_EMBEDDING_MODEL", "")

        # Should still raise MissingConfigurationError for api_key
        with pytest.raises(MissingConfigurationError, match="api_key"):
            meta_evaluator.add_openai()

    def test_partial_environment_variables(
        self, meta_evaluator, clean_environment, monkeypatch
    ):
        """Test behavior with partial environment variable configuration."""
        # Set only some environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-4")
        # Missing OPENAI_DEFAULT_EMBEDDING_MODEL

        with pytest.raises(MissingConfigurationError, match="default_embedding_model"):
            meta_evaluator.add_openai()

    def test_type_annotations_and_return_types(
        self, meta_evaluator, openai_environment
    ):
        """Test that return types match annotations."""
        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            meta_evaluator.add_openai()
            retrieved_client = meta_evaluator.get_client(LLMClientEnum.OPENAI)

            # Verify the return type is an LLMClient (or mock thereof)
            assert isinstance(retrieved_client, MagicMock)
            # In a real scenario, this would be: isinstance(retrieved_client, LLMClient)
