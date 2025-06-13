"""Test suite for the MetaEvaluator class with comprehensive path coverage."""

import json
import re
import pytest
from unittest.mock import MagicMock, patch
from meta_evaluator.meta_evaluator.metaevaluator import (
    MetaEvaluator,
    INVALID_JSON_STRUCTURE_MSG,
)
from meta_evaluator.meta_evaluator.exceptions import (
    MissingConfigurationException,
    ClientAlreadyExistsException,
    ClientNotFoundException,
    DataAlreadyExistsException,
    DataFilenameExtensionMismatchException,
)
from meta_evaluator.llm_client.models import LLMClientEnum
from meta_evaluator.llm_client.openai_client import OpenAIClient
from meta_evaluator.llm_client.azureopenai_client import AzureOpenAIClient
from meta_evaluator.llm_client.serialization import (
    OpenAISerializedState,
    AzureOpenAISerializedState,
)
from meta_evaluator.data.EvalData import EvalData, SampleEvalData
from meta_evaluator.data.serialization import DataMetadata


class TestMetaEvaluator:
    """Comprehensive test suite for MetaEvaluator class achieving 100% path coverage."""

    @pytest.fixture
    def meta_evaluator(self) -> MetaEvaluator:
        """Provides a fresh MetaEvaluator instance for testing.

        Returns:
            MetaEvaluator: A new MetaEvaluator instance with empty client registry.
        """
        return MetaEvaluator()

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

    @pytest.fixture
    def sample_eval_data(self):
        """Provides a sample EvalData object for testing.

        Returns:
            EvalData: A mocked EvalData instance for testing purposes.
        """
        import polars as pl

        mock_eval_data = MagicMock(spec=EvalData)
        mock_eval_data.name = "test_dataset"
        mock_eval_data.id_column = "id"

        # Create a mock DataFrame with sample data
        mock_dataframe = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "question": [
                    "What is 2+2?",
                    "What is the capital of France?",
                    "Who wrote Hamlet?",
                ],
                "answer": ["4", "Paris", "Shakespeare"],
            }
        )
        mock_eval_data.data = mock_dataframe

        # Mock the serialise function
        def mock_serialize(data_format, data_filename):
            return DataMetadata(
                name="test_dataset",
                id_column="id",
                data_file=data_filename or f"test_data.{data_format}",
                data_format=data_format,
                type="EvalData",
            )

        mock_eval_data.serialize.side_effect = mock_serialize

        # Mock the write_data function
        mock_dataframe.write_parquet = MagicMock()
        mock_dataframe.write_csv = MagicMock()
        mock_dataframe.to_dict = MagicMock(
            return_value={"id": [1, 2], "text": ["a", "b"]}
        )

        def mock_write_data(filepath, data_format):
            if data_format == "parquet":
                mock_dataframe.write_parquet(filepath)
            elif data_format == "csv":
                mock_dataframe.write_csv(filepath)
            elif data_format == "json":
                with open(filepath, "w") as f:
                    json.dump(mock_dataframe.to_dict(as_series=False), f, indent=2)
            else:
                raise ValueError(f"Unsupported data format: {data_format}")

        mock_eval_data.write_data.side_effect = mock_write_data

        return mock_eval_data

    @pytest.fixture
    def another_eval_data(self):
        """Provides another sample EvalData object for testing.

        Returns:
            EvalData: A different mocked EvalData instance for testing purposes.
        """
        import polars as pl

        mock_eval_data = MagicMock(spec=EvalData)
        mock_eval_data.name = "another_dataset"
        mock_eval_data.id_column = "id"

        # Create a mock DataFrame with different sample data
        mock_dataframe = pl.DataFrame(
            {
                "id": [4, 5, 6],
                "prompt": ["How are you?", "What's the weather?", "Tell me a joke"],
                "response": [
                    "I'm fine",
                    "It's sunny",
                    "Why did the chicken cross the road?",
                ],
            }
        )
        mock_eval_data.data = mock_dataframe

        # Mock the serialise function
        def mock_serialize(data_format, data_filename):
            return DataMetadata(
                name="another_dataset",
                id_column="id",
                data_file=data_filename or f"test_data.{data_format}",
                data_format=data_format,
                type="EvalData",
            )

        mock_eval_data.serialize.side_effect = mock_serialize

        # Mock the write_data function
        mock_dataframe.write_parquet = MagicMock()
        mock_dataframe.write_csv = MagicMock()
        mock_dataframe.to_dict = MagicMock(
            return_value={"id": [1, 2], "text": ["a", "b"]}
        )

        def mock_write_data(filepath, data_format):
            if data_format == "parquet":
                mock_dataframe.write_parquet(filepath)
            elif data_format == "csv":
                mock_dataframe.write_csv(filepath)
            elif data_format == "json":
                with open(filepath, "w") as f:
                    json.dump(mock_dataframe.to_dict(as_series=False), f, indent=2)
            else:
                raise ValueError(f"Unsupported data format: {data_format}")

        mock_eval_data.write_data.side_effect = mock_write_data

        return mock_eval_data

    @pytest.fixture
    def mock_eval_data_with_dataframe(self):
        """Provides mock EvalData with mocked Polars DataFrame for serialization testing.

        Returns:
            EvalData: Mocked EvalData instance with mocked DataFrame methods.
        """
        mock_eval_data = MagicMock(spec=EvalData)
        mock_eval_data.name = "test_dataset"
        mock_eval_data.id_column = "id"

        # Mock the data (polars DataFrame)
        mock_dataframe = MagicMock()
        mock_eval_data.data = mock_dataframe

        # Mock the serialise function
        def mock_serialize(data_format, data_filename):
            return DataMetadata(
                name="test_dataset",
                id_column="id",
                data_file=data_filename or f"test_data.{data_format}",
                data_format=data_format,
                type="EvalData",
            )

        mock_eval_data.serialize.side_effect = mock_serialize

        # Mock the write_data function
        mock_dataframe.write_parquet = MagicMock()
        mock_dataframe.write_csv = MagicMock()
        mock_dataframe.to_dict = MagicMock(
            return_value={"id": [1, 2], "text": ["a", "b"]}
        )

        def mock_write_data(filepath, data_format):
            if data_format == "parquet":
                mock_dataframe.write_parquet(filepath)
            elif data_format == "csv":
                mock_dataframe.write_csv(filepath)
            elif data_format == "json":
                with open(filepath, "w") as f:
                    json.dump(mock_dataframe.to_dict(as_series=False), f, indent=2)
            else:
                raise ValueError(f"Unsupported data format: {data_format}")

        mock_eval_data.write_data.side_effect = mock_write_data

        return mock_eval_data

    @pytest.fixture
    def mock_sample_eval_data(self):
        """Provides mock SampleEvalData with all sampling metadata for testing.

        Returns:
            SampleEvalData: Mocked SampleEvalData instance with complete sampling metadata.
        """
        mock_sample_data = MagicMock(spec=SampleEvalData)
        mock_sample_data.name = "sample_dataset"
        mock_sample_data.id_column = "sample_id"
        mock_sample_data.sample_name = "Test Sample"
        mock_sample_data.stratification_columns = ["topic", "difficulty"]
        mock_sample_data.sample_percentage = 0.3
        mock_sample_data.seed = 42
        mock_sample_data.sampling_method = "stratified_by_columns"

        # Mock the data (polars DataFrame)
        import polars as pl

        # Create an actual DataFrame instead of mocking it for write operations
        actual_dataframe = pl.DataFrame(
            {
                "sample_id": [1, 2],
                "topic": ["math", "science"],
                "difficulty": ["easy", "medium"],
            }
        )
        mock_sample_data.data = actual_dataframe

        # Mock the serialise function
        def mock_serialize(data_format, data_filename):
            return DataMetadata(
                name="sample_dataset",
                id_column="sample_id",
                data_file=data_filename,
                data_format=data_format,
                type="SampleEvalData",
                sample_name="Test Sample",
                stratification_columns=["topic", "difficulty"],
                sample_percentage=0.3,
                seed=42,
                sampling_method="stratified_by_columns",
            )

        mock_sample_data.serialize.side_effect = mock_serialize

        # Mock the write_data function
        def mock_write_data(filepath, data_format):
            if data_format == "parquet":
                actual_dataframe.write_parquet(filepath)
            elif data_format == "csv":
                actual_dataframe.write_csv(filepath)
            elif data_format == "json":
                with open(filepath, "w") as f:
                    json.dump(actual_dataframe.to_dict(as_series=False), f, indent=2)
            else:
                raise ValueError(f"Unsupported data format: {data_format}")

        mock_sample_data.write_data.side_effect = mock_write_data

        return mock_sample_data

    def _create_mock_openai_client(self, **config_overrides):
        """Helper method to create a properly mocked OpenAI client.

        Returns:
            MagicMock: A mock OpenAI client with configured attributes.
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

    def _create_mock_azure_client(self, **config_overrides):
        """Helper method to create a properly mocked Azure OpenAI client.

        Returns:
            MagicMock: A mock Azure OpenAI client with configured attributes.
        """
        mock_client = MagicMock(spec=AzureOpenAIClient)
        mock_config = MagicMock()
        mock_config.endpoint = config_overrides.get(
            "endpoint", "https://test.openai.azure.com"
        )
        mock_config.api_version = config_overrides.get(
            "api_version", "2024-02-15-preview"
        )
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

    # === MetaEvaluator Initialization Tests ===

    def test_initialization_state(self, meta_evaluator):
        """Test that MetaEvaluator initializes with proper initial state."""
        assert meta_evaluator.client_registry == {}
        assert isinstance(meta_evaluator.client_registry, dict)
        assert meta_evaluator.data is None

    # === add_openai() Method Tests ===

    def test_add_openai_with_all_parameters(self, meta_evaluator, clean_environment):
        """Test adding OpenAI client with all parameters provided."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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
        """Test MissingConfigurationException when api_key is missing."""
        with pytest.raises(MissingConfigurationException, match="api_key"):
            meta_evaluator.add_openai(
                default_model="gpt-4", default_embedding_model="text-embedding-3-large"
            )

    def test_add_openai_missing_default_model(self, meta_evaluator, clean_environment):
        """Test MissingConfigurationException when default_model is missing."""
        with pytest.raises(MissingConfigurationException, match="default_model"):
            meta_evaluator.add_openai(
                api_key="test-key", default_embedding_model="text-embedding-3-large"
            )

    def test_add_openai_missing_default_embedding_model(
        self, meta_evaluator, clean_environment
    ):
        """Test MissingConfigurationException when default_embedding_model is missing."""
        with pytest.raises(
            MissingConfigurationException, match="default_embedding_model"
        ):
            meta_evaluator.add_openai(api_key="test-key", default_model="gpt-4")

    def test_add_openai_client_already_exists_no_override(
        self, meta_evaluator, openai_environment
    ):
        """Test ClientAlreadyExistsException when client exists and override_existing is False."""
        with patch("meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"):
            # Add client first time
            meta_evaluator.add_openai()

            # Try to add again without override
            with pytest.raises(
                ClientAlreadyExistsException, match="OPENAI.*already exists"
            ):
                meta_evaluator.add_openai()

    def test_add_openai_client_already_exists_with_override(
        self, meta_evaluator, openai_environment
    ):
        """Test successful override when client exists and override_existing is True."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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
        with pytest.raises(MissingConfigurationException, match="api_key"):
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
            "meta_evaluator.meta_evaluator.metaevaluator.AzureOpenAIClient"
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
            "meta_evaluator.meta_evaluator.metaevaluator.AzureOpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=AzureOpenAIClient)
            mock_client_class.return_value = mock_client

            meta_evaluator.add_azure_openai()

            assert LLMClientEnum.AZURE_OPENAI in meta_evaluator.client_registry

    def test_add_azure_openai_missing_api_key(self, meta_evaluator, clean_environment):
        """Test MissingConfigurationException when api_key is missing."""
        with pytest.raises(MissingConfigurationException, match="api_key"):
            meta_evaluator.add_azure_openai(
                endpoint="https://test.openai.azure.com",
                api_version="2024-02-15-preview",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_add_azure_openai_missing_endpoint(self, meta_evaluator, clean_environment):
        """Test MissingConfigurationException when endpoint is missing."""
        with pytest.raises(MissingConfigurationException, match="endpoint"):
            meta_evaluator.add_azure_openai(
                api_key="test-key",
                api_version="2024-02-15-preview",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_add_azure_openai_missing_api_version(
        self, meta_evaluator, clean_environment
    ):
        """Test MissingConfigurationException when api_version is missing."""
        with pytest.raises(MissingConfigurationException, match="api_version"):
            meta_evaluator.add_azure_openai(
                api_key="test-key",
                endpoint="https://test.openai.azure.com",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_add_azure_openai_missing_default_model(
        self, meta_evaluator, clean_environment
    ):
        """Test MissingConfigurationException when default_model is missing."""
        with pytest.raises(MissingConfigurationException, match="default_model"):
            meta_evaluator.add_azure_openai(
                api_key="test-key",
                endpoint="https://test.openai.azure.com",
                api_version="2024-02-15-preview",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_add_azure_openai_missing_default_embedding_model(
        self, meta_evaluator, clean_environment
    ):
        """Test MissingConfigurationException when default_embedding_model is missing."""
        with pytest.raises(
            MissingConfigurationException, match="default_embedding_model"
        ):
            meta_evaluator.add_azure_openai(
                api_key="test-key",
                endpoint="https://test.openai.azure.com",
                api_version="2024-02-15-preview",
                default_model="gpt-4",
            )

    def test_add_azure_openai_client_already_exists_no_override(
        self, meta_evaluator, azure_openai_environment
    ):
        """Test ClientAlreadyExistsException when Azure client exists and override_existing is False."""
        with patch("meta_evaluator.meta_evaluator.metaevaluator.AzureOpenAIClient"):
            # Add client first time
            meta_evaluator.add_azure_openai()

            # Try to add again without override
            with pytest.raises(
                ClientAlreadyExistsException, match="AZURE_OPENAI.*already exists"
            ):
                meta_evaluator.add_azure_openai()

    def test_add_azure_openai_client_already_exists_with_override(
        self, meta_evaluator, azure_openai_environment
    ):
        """Test successful override when Azure client exists and override_existing is True."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.AzureOpenAIClient"
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
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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
            "meta_evaluator.meta_evaluator.metaevaluator.AzureOpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=AzureOpenAIClient)
            mock_client_class.return_value = mock_client

            meta_evaluator.add_azure_openai()
            retrieved_client = meta_evaluator.get_client(LLMClientEnum.AZURE_OPENAI)

            assert retrieved_client == mock_client

    def test_get_client_nonexistent_client(self, meta_evaluator):
        """Test ClientNotFoundException when requesting non-existent client."""
        # (?i) makes the regex case-insensitive to match both "OPENAI" and "openai"
        with pytest.raises(ClientNotFoundException, match="(?i)OPENAI.*not found"):
            meta_evaluator.get_client(LLMClientEnum.OPENAI)

    def test_get_client_from_empty_registry(self, meta_evaluator):
        """Test ClientNotFoundException when registry is empty."""
        assert meta_evaluator.client_registry == {}

        # (?i) makes the regex case-insensitive to match both "GEMINI" and "gemini"
        with pytest.raises(ClientNotFoundException, match="(?i)GEMINI.*not found"):
            meta_evaluator.get_client(LLMClientEnum.GEMINI)

    def test_get_client_different_enum_values(self, meta_evaluator):
        """Test ClientNotFoundException for different enum values."""
        # Test all enum values that don't exist
        for client_enum in [LLMClientEnum.GEMINI, LLMClientEnum.ANTHROPIC]:
            with pytest.raises(
                ClientNotFoundException, match=f"{client_enum.value}.*not found"
            ):
                meta_evaluator.get_client(client_enum)

    # === add_data() Method Tests ===

    def test_add_data_first_time(self, meta_evaluator, sample_eval_data):
        """Test adding data when no data exists."""
        assert meta_evaluator.data is None

        meta_evaluator.add_data(sample_eval_data)

        assert meta_evaluator.data == sample_eval_data

    def test_add_data_already_exists_no_overwrite(
        self, meta_evaluator, sample_eval_data, another_eval_data
    ):
        """Test DataAlreadyExistsException when data exists and overwrite is False (default)."""
        # Add data first time
        meta_evaluator.add_data(sample_eval_data)
        assert meta_evaluator.data == sample_eval_data

        # Try to add again without overwrite
        with pytest.raises(DataAlreadyExistsException, match="Data already exists"):
            meta_evaluator.add_data(another_eval_data)

        # Verify original data is unchanged
        assert meta_evaluator.data == sample_eval_data

    def test_add_data_already_exists_with_overwrite(
        self, meta_evaluator, sample_eval_data, another_eval_data
    ):
        """Test successful replacement when data exists and overwrite is True."""
        # Add data first time
        meta_evaluator.add_data(sample_eval_data)
        original_data = meta_evaluator.data
        assert original_data == sample_eval_data

        # Overwrite with new data
        meta_evaluator.add_data(another_eval_data, overwrite=True)
        new_data = meta_evaluator.data

        assert new_data == another_eval_data
        assert new_data != original_data

    def test_add_data_preserves_client_registry(
        self, meta_evaluator, sample_eval_data, openai_environment
    ):
        """Test that adding data doesn't affect client registry."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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

    # === Integration Tests ===

    def test_add_multiple_clients_and_retrieve(
        self, meta_evaluator, openai_environment, azure_openai_environment
    ):
        """Test adding multiple clients and retrieving them."""
        with (
            patch(
                "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
            ) as mock_openai_class,
            patch(
                "meta_evaluator.meta_evaluator.metaevaluator.AzureOpenAIClient"
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
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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

    # === Data + Client Integration Tests ===

    def test_add_client_and_data_together(
        self, meta_evaluator, sample_eval_data, openai_environment
    ):
        """Test adding OpenAI client and data together, verify both exist independently."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            # Add both client and data
            meta_evaluator.add_openai()
            meta_evaluator.add_data(sample_eval_data)

            # Verify both exist and are independent
            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry
            assert meta_evaluator.client_registry[LLMClientEnum.OPENAI] == mock_client
            assert meta_evaluator.data == sample_eval_data

    def test_add_data_then_client(
        self, meta_evaluator, sample_eval_data, openai_environment
    ):
        """Test adding data first, then client, verify both preserved."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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

    def test_complete_evaluator_state(
        self,
        meta_evaluator,
        sample_eval_data,
        openai_environment,
        azure_openai_environment,
    ):
        """Test complete object state with both clients and data."""
        with (
            patch(
                "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
            ) as mock_openai_class,
            patch(
                "meta_evaluator.meta_evaluator.metaevaluator.AzureOpenAIClient"
            ) as mock_azure_class,
        ):
            mock_openai_client = MagicMock(spec=OpenAIClient)
            mock_azure_client = MagicMock(spec=AzureOpenAIClient)
            mock_openai_class.return_value = mock_openai_client
            mock_azure_class.return_value = mock_azure_client

            # Add all components
            meta_evaluator.add_openai()
            meta_evaluator.add_azure_openai()
            meta_evaluator.add_data(sample_eval_data)

            # Verify complete state
            assert len(meta_evaluator.client_registry) == 2
            assert LLMClientEnum.OPENAI in meta_evaluator.client_registry
            assert LLMClientEnum.AZURE_OPENAI in meta_evaluator.client_registry
            assert (
                meta_evaluator.client_registry[LLMClientEnum.OPENAI]
                == mock_openai_client
            )
            assert (
                meta_evaluator.client_registry[LLMClientEnum.AZURE_OPENAI]
                == mock_azure_client
            )
            assert meta_evaluator.data == sample_eval_data

    # === Edge Cases and Error Handling ===

    def test_environment_variables_empty_strings(
        self, meta_evaluator, clean_environment, monkeypatch
    ):
        """Test behavior when environment variables are set to empty strings."""
        # Set environment variables to empty strings
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "")
        monkeypatch.setenv("OPENAI_DEFAULT_EMBEDDING_MODEL", "")

        # Should still raise MissingConfigurationException for api_key
        with pytest.raises(MissingConfigurationException, match="api_key"):
            meta_evaluator.add_openai()

    def test_partial_environment_variables(
        self, meta_evaluator, clean_environment, monkeypatch
    ):
        """Test behavior with partial environment variable configuration."""
        # Set only some environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-4")
        # Missing OPENAI_DEFAULT_EMBEDDING_MODEL

        with pytest.raises(
            MissingConfigurationException, match="default_embedding_model"
        ):
            meta_evaluator.add_openai()

    def test_type_annotations_and_return_types(
        self, meta_evaluator, openai_environment
    ):
        """Test that return types match annotations."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            meta_evaluator.add_openai()
            retrieved_client = meta_evaluator.get_client(LLMClientEnum.OPENAI)

            # Verify the return type is an LLMClient (or mock thereof)
            assert isinstance(retrieved_client, MagicMock)
            # In a real scenario, this would be: isinstance(retrieved_client, LLMClient)

    # === save_state() Method Tests ===

    def test_save_state_invalid_file_extension(self, meta_evaluator):
        """Test ValueError when state_file doesn't end with .json."""
        with pytest.raises(ValueError, match="state_file must end with .json"):
            meta_evaluator.save_state("invalid_file.txt")

    def test_save_state_include_data_true_but_no_format(self, meta_evaluator):
        """Test ValueError when include_data=True but data_format is None."""
        with pytest.raises(
            ValueError, match="data_format must be specified when include_data=True"
        ):
            meta_evaluator.save_state("test.json", include_data=True, data_format=None)

    def test_save_state_include_data_false(self, meta_evaluator, tmp_path):
        """Test saving state without data serialization."""
        state_file = tmp_path / "test_state.json"

        # Add a client to have something to serialize
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
        ) as mock_client_class:
            mock_client = self._create_mock_openai_client()
            mock_client_class.return_value = mock_client

            meta_evaluator.add_openai(
                api_key="test-key",
                default_model="gpt-4",
                default_embedding_model="text-embedding-3-large",
            )

        # Save without data
        meta_evaluator.save_state(str(state_file), include_data=False)

        # Verify state file exists and data file doesn't
        assert state_file.exists()
        assert not (tmp_path / "test_state_data.json").exists()
        assert not (tmp_path / "test_state_data.csv").exists()
        assert not (tmp_path / "test_state_data.parquet").exists()

        # Verify state file contents
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["version"] == "1.0"
        assert state_data["data"] is None
        assert "openai" in state_data["client_registry"]

    def test_save_state_with_parquet_format(
        self, meta_evaluator, mock_eval_data_with_dataframe, tmp_path
    ):
        """Test saving state with parquet data format."""
        state_file = tmp_path / "test_state.json"
        data_file = tmp_path / "test_state_data.parquet"

        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        meta_evaluator.save_state(
            str(state_file), include_data=True, data_format="parquet"
        )

        # Verify both files exist
        assert state_file.exists()

        # Verify polars write_parquet was called
        mock_eval_data_with_dataframe.write_data.assert_called_once_with(
            str(data_file), "parquet"
        )

        # Verify state file contents
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["version"] == "1.0"
        assert state_data["data"]["data_format"] == "parquet"
        assert state_data["data"]["data_file"] == "test_state_data.parquet"

    def test_save_state_with_csv_format(
        self, meta_evaluator, mock_eval_data_with_dataframe, tmp_path
    ):
        """Test saving state with CSV data format."""
        state_file = tmp_path / "test_state.json"
        data_file = tmp_path / "test_state_data.csv"

        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        meta_evaluator.save_state(str(state_file), include_data=True, data_format="csv")

        # Verify both files exist
        assert state_file.exists()

        # Verify polars write_csv was called
        mock_eval_data_with_dataframe.write_data.assert_called_once_with(
            str(data_file), "csv"
        )

        # Verify state file contents
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["data"]["data_format"] == "csv"
        assert state_data["data"]["data_file"] == "test_state_data.csv"

    def test_save_state_with_json_format(
        self, meta_evaluator, mock_eval_data_with_dataframe, tmp_path
    ):
        """Test saving state with JSON data format."""
        state_file = tmp_path / "test_state.json"
        data_file = tmp_path / "test_state_data.json"

        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        meta_evaluator.save_state(
            str(state_file), include_data=True, data_format="json"
        )

        # Verify both files exist
        assert state_file.exists()

        # Verify polars to_dict was called for JSON serialization
        mock_eval_data_with_dataframe.write_data.assert_called_once_with(
            str(data_file), "json"
        )

        # Verify data file was created with JSON content
        assert data_file.exists()

    def test_save_state_no_data_present(self, meta_evaluator, tmp_path):
        """Test saving when include_data=True but self.data is None."""
        state_file = tmp_path / "test_state.json"

        # Don't add any data to meta_evaluator
        assert meta_evaluator.data is None

        meta_evaluator.save_state(
            str(state_file), include_data=True, data_format="parquet"
        )

        # Verify state file exists but no data file
        assert state_file.exists()
        assert not (tmp_path / "test_state_data.parquet").exists()

        # Verify state file contents
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["data"] is None

    def test_save_state_creates_directories(self, meta_evaluator, tmp_path):
        """Test that parent directories are created when they don't exist."""
        nested_dir = tmp_path / "deep" / "nested" / "path"
        state_file = nested_dir / "test_state.json"

        # Directory shouldn't exist initially
        assert not nested_dir.exists()

        meta_evaluator.save_state(str(state_file), include_data=False)

        # Directory should be created and file should exist
        assert nested_dir.exists()
        assert state_file.exists()

    def test_save_state_with_custom_data_filename_json(
        self, meta_evaluator, mock_eval_data_with_dataframe, tmp_path
    ):
        """Test saving with custom data filename for JSON format."""
        state_file = tmp_path / "test_state.json"
        custom_data_file = "my_custom_data.json"

        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        meta_evaluator.save_state(
            str(state_file),
            include_data=True,
            data_format="json",
            data_filename=custom_data_file,
        )

        # Verify custom data filename is used
        expected_data_path = tmp_path / custom_data_file
        assert expected_data_path.exists()

        # Verify state file references custom filename
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["version"] == "1.0"
        assert state_data["data"]["data_file"] == custom_data_file
        assert state_data["data"]["data_format"] == "json"

    def test_save_state_with_custom_data_filename_csv(
        self, meta_evaluator, mock_eval_data_with_dataframe, tmp_path
    ):
        """Test saving with custom data filename for CSV format."""
        state_file = tmp_path / "test_state.json"
        custom_data_file = "my_custom_data.csv"

        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        meta_evaluator.save_state(
            str(state_file),
            include_data=True,
            data_format="csv",
            data_filename=custom_data_file,
        )

        # Verify polars write_csv was called with custom path
        expected_data_path = tmp_path / custom_data_file
        mock_eval_data_with_dataframe.write_data.assert_called_once_with(
            str(expected_data_path), "csv"
        )

        # Verify state file references custom filename
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["data"]["data_file"] == custom_data_file
        assert state_data["data"]["data_format"] == "csv"

    def test_save_state_with_custom_data_filename_parquet(
        self, meta_evaluator, mock_eval_data_with_dataframe, tmp_path
    ):
        """Test saving with custom data filename for Parquet format."""
        state_file = tmp_path / "test_state.json"
        custom_data_file = "my_custom_data.parquet"

        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        meta_evaluator.save_state(
            str(state_file),
            include_data=True,
            data_format="parquet",
            data_filename=custom_data_file,
        )

        # Verify polars write_parquet was called with custom path
        expected_data_path = tmp_path / custom_data_file
        mock_eval_data_with_dataframe.write_data.assert_called_once_with(
            str(expected_data_path), "parquet"
        )

        # Verify state file references custom filename
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["data"]["data_file"] == custom_data_file
        assert state_data["data"]["data_format"] == "parquet"

    def test_save_state_data_filename_extension_mismatch_json(self, meta_evaluator):
        """Test DataFilenameExtensionMismatchException for JSON format mismatch."""
        with pytest.raises(
            DataFilenameExtensionMismatchException,
            match="Data filename 'wrong_extension.csv' must have extension '.json' to match data_format 'json'",
        ):
            meta_evaluator.save_state(
                "test.json",
                include_data=True,
                data_format="json",
                data_filename="wrong_extension.csv",
            )

    def test_save_state_data_filename_extension_mismatch_csv(self, meta_evaluator):
        """Test DataFilenameExtensionMismatchException for CSV format mismatch."""
        with pytest.raises(
            DataFilenameExtensionMismatchException,
            match="Data filename 'wrong_extension.json' must have extension '.csv' to match data_format 'csv'",
        ):
            meta_evaluator.save_state(
                "test.json",
                include_data=True,
                data_format="csv",
                data_filename="wrong_extension.json",
            )

    def test_save_state_data_filename_extension_mismatch_parquet(self, meta_evaluator):
        """Test DataFilenameExtensionMismatchException for Parquet format mismatch."""
        with pytest.raises(
            DataFilenameExtensionMismatchException,
            match="Data filename 'wrong_extension.csv' must have extension '.parquet' to match data_format 'parquet'",
        ):
            meta_evaluator.save_state(
                "test.json",
                include_data=True,
                data_format="parquet",
                data_filename="wrong_extension.csv",
            )

    def test_save_state_data_filename_no_validation_when_include_data_false(
        self, meta_evaluator, tmp_path
    ):
        """Test that data_filename extension is not validated when include_data=False."""
        state_file = tmp_path / "test.json"
        # This should not raise an exception even with wrong extension
        # because data_filename is ignored when include_data=False
        meta_evaluator.save_state(
            str(state_file), include_data=False, data_filename="wrong_extension.csv"
        )

    def test_save_state_data_filename_no_validation_when_data_format_none(
        self, meta_evaluator
    ):
        """Test that data_filename extension is not validated when data_format=None."""
        # This should not raise an exception because data_format is None
        # (will raise ValueError for missing data_format instead)
        with pytest.raises(
            ValueError, match="data_format must be specified when include_data=True"
        ):
            meta_evaluator.save_state(
                "test.json",
                include_data=True,
                data_format=None,
                data_filename="any_name.csv",
            )

    def test_save_state_fallback_to_auto_generated_when_data_filename_none(
        self, meta_evaluator, mock_eval_data_with_dataframe, tmp_path
    ):
        """Test that auto-generated filename is used when data_filename=None."""
        state_file = tmp_path / "test_state.json"

        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        meta_evaluator.save_state(
            str(state_file),
            include_data=True,
            data_format="json",
            data_filename=None,  # Explicitly set to None
        )

        # Verify auto-generated filename is used
        auto_generated_file = tmp_path / "test_state_data.json"
        assert auto_generated_file.exists()

        # Verify state file references auto-generated filename
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["data"]["data_file"] == "test_state_data.json"

    # === Private Serialization Method Tests ===

    def test_serialize_dict_include_data_false(self, meta_evaluator):
        """Test _serialize when data should not be included."""
        state = meta_evaluator._serialize(
            include_data=False, data_format=None, data_filename=None
        )

        assert state.version == "1.0"
        assert state.data is None
        assert state.client_registry is not None

    def test_serialize_dict_include_data_true(
        self, meta_evaluator, mock_eval_data_with_dataframe
    ):
        """Test _serialize when data should be included."""
        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        state = meta_evaluator._serialize(
            include_data=True, data_format="parquet", data_filename="test_data.parquet"
        )

        assert state.version == "1.0"
        assert state.data is not None
        assert state.data.data_format == "parquet"
        assert state.data.data_file == "test_data.parquet"

    def test_serialize_client_registry_empty(self, meta_evaluator):
        """Test _serialize_client_registry with empty registry."""
        serialized = meta_evaluator._serialize_client_registry()

        assert serialized == {}
        assert isinstance(serialized, dict)

    def test_serialize_client_registry_with_openai_client(
        self, meta_evaluator, openai_environment
    ):
        """Test _serialize_client_registry with OpenAI client present."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
        ) as mock_client_class:
            mock_client = self._create_mock_openai_client()
            mock_client_class.return_value = mock_client

            meta_evaluator.add_openai()
            serialized = meta_evaluator._serialize_client_registry()

            assert "openai" in serialized
            assert serialized["openai"]["client_type"] == "openai"
            assert serialized["openai"]["default_model"] == "gpt-4"
            assert "api_key" not in str(serialized)

    def test_serialize_client_registry_with_both_clients(
        self, meta_evaluator, openai_environment, azure_openai_environment
    ):
        """Test _serialize_client_registry with both OpenAI and Azure clients."""
        with (
            patch(
                "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
            ) as mock_openai_class,
            patch(
                "meta_evaluator.meta_evaluator.metaevaluator.AzureOpenAIClient"
            ) as mock_azure_class,
        ):
            mock_openai_client = self._create_mock_openai_client()
            mock_openai_class.return_value = mock_openai_client

            mock_azure_client = self._create_mock_azure_client()
            mock_azure_class.return_value = mock_azure_client

            meta_evaluator.add_openai()
            meta_evaluator.add_azure_openai()
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

    def test_save_and_verify_complete_file_contents(
        self,
        meta_evaluator,
        openai_environment,
        mock_eval_data_with_dataframe,
        tmp_path,
    ):
        """Test complete save operation and verify actual file contents match expected structure."""
        state_file = tmp_path / "integration_test.json"
        data_file = tmp_path / "integration_test_data.parquet"

        # Set up complete MetaEvaluator state
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
        ) as mock_client_class:
            mock_client = self._create_mock_openai_client()
            mock_client_class.return_value = mock_client

            meta_evaluator.add_openai()
            meta_evaluator.add_data(mock_eval_data_with_dataframe)

            # Save state
            meta_evaluator.save_state(
                str(state_file), include_data=True, data_format="parquet"
            )

            # Verify files exist
            assert state_file.exists()

            # Verify state file structure
            with open(state_file) as f:
                state_data = json.load(f)

            # Verify top-level structure
            assert state_data["version"] == "1.0"
            assert "client_registry" in state_data
            assert "data" in state_data

            # Verify client registry structure
            assert "openai" in state_data["client_registry"]
            openai_config = state_data["client_registry"]["openai"]
            assert openai_config["client_type"] == "openai"
            assert openai_config["default_model"] == "gpt-4"
            assert openai_config["default_embedding_model"] == "text-embedding-3-large"
            assert openai_config["supports_structured_output"] is True
            assert openai_config["supports_logprobs"] is True
            assert "api_key" not in str(openai_config)

            # Verify data structure
            data_config = state_data["data"]
            assert data_config["name"] == "test_dataset"
            assert data_config["id_column"] == "id"
            assert data_config["data_format"] == "parquet"
            assert data_config["data_file"] == "integration_test_data.parquet"
            assert data_config["type"] == "EvalData"

            # Verify DataFrame method was called
            mock_eval_data_with_dataframe.write_data.assert_called_once_with(
                str(data_file), "parquet"
            )

    def test_save_with_multiple_clients_and_sample_data(
        self,
        meta_evaluator,
        openai_environment,
        azure_openai_environment,
        mock_sample_eval_data,
        tmp_path,
    ):
        """Test saving state with multiple clients and SampleEvalData together."""
        state_file = tmp_path / "multi_client_test.json"

        with (
            patch(
                "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
            ) as mock_openai_class,
            patch(
                "meta_evaluator.meta_evaluator.metaevaluator.AzureOpenAIClient"
            ) as mock_azure_class,
        ):
            # Set up OpenAI client
            mock_openai_client = self._create_mock_openai_client(
                supports_logprobs=False
            )
            mock_openai_class.return_value = mock_openai_client

            # Set up Azure client
            mock_azure_client = self._create_mock_azure_client(
                supports_structured_output=False
            )
            mock_azure_class.return_value = mock_azure_client

            # Add both clients and sample data
            meta_evaluator.add_openai()
            meta_evaluator.add_azure_openai()
            meta_evaluator.add_data(mock_sample_eval_data)

            # Save state
            meta_evaluator.save_state(
                str(state_file), include_data=True, data_format="csv"
            )

            # Verify file exists
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

    def test_save_with_complex_file_paths(self, meta_evaluator, tmp_path):
        """Test saving with complex file paths including spaces and nested directories."""
        # Create nested directory with spaces
        complex_dir = tmp_path / "test dir" / "with spaces" / "and-dashes"
        state_file = complex_dir / "complex file name.json"

        # Directory shouldn't exist initially
        assert not complex_dir.exists()

        meta_evaluator.save_state(str(state_file), include_data=False)

        # Verify directory was created and file exists
        assert complex_dir.exists()
        assert state_file.exists()

        # Verify file is valid JSON
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["version"] == "1.0"

    # === load_state() Method Tests ===

    def test_load_state_with_openai_client(self, tmp_path, openai_environment):
        """Test loading MetaEvaluator with OpenAI client from JSON."""
        # First save a state with OpenAI client
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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
            original_evaluator = MetaEvaluator()
            original_evaluator.add_openai()

            state_file = tmp_path / "test_state.json"
            original_evaluator.save_state(str(state_file), include_data=False)

            # Load from JSON (provide API key for reconstruction)
            loaded_evaluator = MetaEvaluator.load_state(
                str(state_file), load_data=False, openai_api_key="test-api-key"
            )

            # Verify client was reconstructed
            assert LLMClientEnum.OPENAI in loaded_evaluator.client_registry
            assert loaded_evaluator.data is None

    def test_load_state_with_azure_openai_client(
        self, tmp_path, azure_openai_environment
    ):
        """Test loading MetaEvaluator with Azure OpenAI client from JSON."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.AzureOpenAIClient"
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
            original_evaluator = MetaEvaluator()
            original_evaluator.add_azure_openai()

            state_file = tmp_path / "test_state.json"
            original_evaluator.save_state(str(state_file), include_data=False)

            # Load from JSON (provide API key for reconstruction)
            loaded_evaluator = MetaEvaluator.load_state(
                str(state_file), load_data=False, azure_openai_api_key="test-api-key"
            )

            # Verify client was reconstructed
            assert LLMClientEnum.AZURE_OPENAI in loaded_evaluator.client_registry
            assert loaded_evaluator.data is None

    def test_load_state_with_eval_data_json_format(
        self, tmp_path, sample_eval_data, openai_environment
    ):
        """Test loading MetaEvaluator with EvalData in JSON format."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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

            # Create and save evaluator with data
            original_evaluator = MetaEvaluator()
            original_evaluator.add_openai()
            original_evaluator.add_data(sample_eval_data)

            state_file = tmp_path / "test_state.json"
            original_evaluator.save_state(
                str(state_file), include_data=True, data_format="json"
            )

            # Load from JSON
            loaded_evaluator = MetaEvaluator.load_state(
                str(state_file), load_data=True, openai_api_key="test-api-key"
            )

            # Verify data was loaded
            assert loaded_evaluator.data is not None
            assert loaded_evaluator.data.name == sample_eval_data.name
            assert loaded_evaluator.data.id_column == sample_eval_data.id_column
            assert isinstance(loaded_evaluator.data, EvalData)

    def test_load_state_with_sample_eval_data_csv_format(
        self, tmp_path, mock_sample_eval_data, openai_environment
    ):
        """Test loading MetaEvaluator with SampleEvalData in CSV format."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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

            # Create and save evaluator with sample data
            original_evaluator = MetaEvaluator()
            original_evaluator.add_openai()
            original_evaluator.add_data(mock_sample_eval_data)

            state_file = tmp_path / "test_state.json"
            original_evaluator.save_state(
                str(state_file), include_data=True, data_format="csv"
            )

            # Ensure the data file exists
            data_file = tmp_path / "test_state_data.csv"
            assert data_file.exists()

            # Load from JSON
            loaded_evaluator = MetaEvaluator.load_state(
                str(state_file), load_data=True, openai_api_key="test-api-key"
            )

            # Verify sample data was loaded with all metadata
            assert loaded_evaluator.data is not None
            assert isinstance(loaded_evaluator.data, SampleEvalData)
            assert (
                loaded_evaluator.data.sample_name == mock_sample_eval_data.sample_name
            )
            assert (
                loaded_evaluator.data.stratification_columns
                == mock_sample_eval_data.stratification_columns
            )
            assert (
                loaded_evaluator.data.sample_percentage
                == mock_sample_eval_data.sample_percentage
            )
            assert loaded_evaluator.data.seed == mock_sample_eval_data.seed
            assert (
                loaded_evaluator.data.sampling_method
                == mock_sample_eval_data.sampling_method
            )

    def test_load_state_skip_data_loading(
        self, tmp_path, sample_eval_data, openai_environment
    ):
        """Test loading MetaEvaluator while skipping data loading."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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

            # Create and save evaluator with data
            original_evaluator = MetaEvaluator()
            original_evaluator.add_openai()
            original_evaluator.add_data(sample_eval_data)

            state_file = tmp_path / "test_state.json"
            original_evaluator.save_state(
                str(state_file), include_data=True, data_format="json"
            )

            # Load from JSON without data
            loaded_evaluator = MetaEvaluator.load_state(
                str(state_file), load_data=False, openai_api_key="test-api-key"
            )

            # Verify client loaded but data skipped
            assert LLMClientEnum.OPENAI in loaded_evaluator.client_registry
            assert loaded_evaluator.data is None

    def test_load_state_invalid_file_extension(self):
        """Test ValueError when state file doesn't end with .json."""
        with pytest.raises(ValueError, match="state_file must end with .json"):
            MetaEvaluator.load_state("invalid_file.txt")

    def test_load_state_nonexistent_file(self):
        """Test FileNotFoundError when state file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="State file not found"):
            MetaEvaluator.load_state("nonexistent.json")

    def test_load_state_invalid_json(self, tmp_path):
        """Test ValueError when state file contains invalid JSON."""
        state_file = tmp_path / "invalid.json"
        state_file.write_text("{ invalid json }")

        with pytest.raises(
            ValueError,
            match=re.compile(rf"{re.escape(INVALID_JSON_STRUCTURE_MSG)}", re.DOTALL),
        ):
            MetaEvaluator.load_state(str(state_file))

    def test_load_state_missing_required_keys(self, tmp_path):
        """Test ValueError when state file is missing required keys."""
        state_file = tmp_path / "incomplete.json"
        state_file.write_text('{"version": "1.0"}')  # Missing client_registry and data

        with pytest.raises(
            ValueError,
            match=re.compile(
                rf"{re.escape(INVALID_JSON_STRUCTURE_MSG)}.*Field required", re.DOTALL
            ),
        ):
            MetaEvaluator.load_state(str(state_file))

    def test_load_state_missing_api_keys(self, tmp_path, clean_environment):
        """Test MissingConfigurationException when API keys are missing."""
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

        with pytest.raises(MissingConfigurationException, match="api_key"):
            MetaEvaluator.load_state(str(state_file))

    def test_load_state_nonexistent_data_file(self, tmp_path, openai_environment):
        """Test FileNotFoundError when referenced data file doesn't exist."""
        # Create state file that references nonexistent data file
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
            "data": {
                "name": "Test Data",
                "id_column": "message_id",
                "data_file": "nonexistent_data.json",
                "data_format": "json",
                "type": "EvalData",
            },
        }

        state_file = tmp_path / "test.json"
        with open(state_file, "w") as f:
            json.dump(state_data, f)

        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            with pytest.raises(FileNotFoundError, match="Data file not found"):
                MetaEvaluator.load_state(
                    str(state_file), load_data=True, openai_api_key="test-api-key"
                )

    def test_load_state_unsupported_data_format(self, tmp_path, openai_environment):
        """Test ValueError when data format is unsupported."""
        # Create state file with unsupported data format
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
            "data": {
                "name": "Test Data",
                "id_column": "message_id",
                "data_file": "test_data.xml",
                "data_format": "xml",  # Unsupported format
                "type": "EvalData",
            },
        }

        state_file = tmp_path / "test.json"
        with open(state_file, "w") as f:
            json.dump(state_data, f)

        # Create the referenced data file (even though format is unsupported)
        data_file = tmp_path / "test_data.xml"
        data_file.write_text("<data></data>")

        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
        ) as mock_client_class:
            mock_client = MagicMock(spec=OpenAIClient)
            mock_client_class.return_value = mock_client

            with pytest.raises(
                ValueError,
                match=re.compile(
                    rf"{re.escape(INVALID_JSON_STRUCTURE_MSG)}.*Input should be 'json', 'csv' or 'parquet'",
                    re.DOTALL,
                ),
            ):
                MetaEvaluator.load_state(
                    str(state_file), load_data=True, openai_api_key="test-api-key"
                )

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
                "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
            ) as mock_openai_class,
            patch(
                "meta_evaluator.meta_evaluator.metaevaluator.AzureOpenAIClient"
            ) as mock_azure_class,
        ):
            mock_openai_client = MagicMock(spec=OpenAIClient)
            mock_azure_client = MagicMock(spec=AzureOpenAIClient)
            mock_openai_class.return_value = mock_openai_client
            mock_azure_class.return_value = mock_azure_client

            # Load with custom API keys
            loaded_evaluator = MetaEvaluator.load_state(
                str(state_file),
                load_data=False,
                openai_api_key="custom-openai-key",
                azure_openai_api_key="custom-azure-key",
            )

            # Verify both clients were reconstructed
            assert LLMClientEnum.OPENAI in loaded_evaluator.client_registry
            assert LLMClientEnum.AZURE_OPENAI in loaded_evaluator.client_registry
