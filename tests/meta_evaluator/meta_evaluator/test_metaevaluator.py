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
    EvalTaskAlreadyExistsException,
    JudgeAlreadyExistsException,
    JudgeNotFoundException,
    InvalidYAMLStructureException,
    PromptFileNotFoundException,
)
from meta_evaluator.llm_client.models import LLMClientEnum
from meta_evaluator.llm_client.openai_client import OpenAIClient
from meta_evaluator.llm_client.azureopenai_client import AzureOpenAIClient
from meta_evaluator.llm_client.serialization import (
    OpenAISerializedState,
    AzureOpenAISerializedState,
)
from meta_evaluator.data import EvalData, SampleEvalData
from meta_evaluator.data.serialization import DataMetadata
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.common.models import Prompt


class TestMetaEvaluator:
    """Comprehensive test suite for MetaEvaluator class achieving 100% path coverage."""

    @pytest.fixture
    def meta_evaluator(self, tmp_path) -> MetaEvaluator:
        """Provides a fresh MetaEvaluator instance for testing.

        Args:
            tmp_path: Pytest fixture providing temporary directory.

        Returns:
            MetaEvaluator: A new MetaEvaluator instance with temporary project directory.
        """
        return MetaEvaluator(str(tmp_path / "test_project"))

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

        mock_eval_data.serialize_metadata.side_effect = mock_serialize

        # Mock the write_data function
        mock_dataframe.write_parquet = MagicMock()
        mock_dataframe.write_csv = MagicMock()
        mock_dataframe.to_dict = MagicMock(
            return_value={"id": [1, 2], "text": ["a", "b"]}
        )

        def mock_write_data(filepath, data_format):
            match data_format:
                case "parquet":
                    mock_dataframe.write_parquet(filepath)
                case "csv":
                    mock_dataframe.write_csv(filepath)
                case "json":
                    data_dict = mock_dataframe.to_dict(as_series=False)
                    with open(filepath, "w") as f:
                        json.dump(data_dict, f, indent=2)
                case _:
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

        mock_eval_data.serialize_metadata.side_effect = mock_serialize

        # Mock the write_data function
        mock_dataframe.write_parquet = MagicMock()
        mock_dataframe.write_csv = MagicMock()
        mock_dataframe.to_dict = MagicMock(
            return_value={"id": [1, 2], "text": ["a", "b"]}
        )

        def mock_write_data(filepath, data_format):
            match data_format:
                case "parquet":
                    mock_dataframe.write_parquet(filepath)
                case "csv":
                    mock_dataframe.write_csv(filepath)
                case "json":
                    data_dict = mock_dataframe.to_dict(as_series=False)
                    with open(filepath, "w") as f:
                        json.dump(data_dict, f, indent=2)
                case _:
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

        mock_eval_data.serialize_metadata.side_effect = mock_serialize

        # Mock the write_data function
        mock_dataframe.write_parquet = MagicMock()
        mock_dataframe.write_csv = MagicMock()
        mock_dataframe.to_dict = MagicMock(
            return_value={"id": [1, 2], "text": ["a", "b"]}
        )

        def mock_write_data(filepath, data_format):
            match data_format:
                case "parquet":
                    mock_dataframe.write_parquet(filepath)
                case "csv":
                    mock_dataframe.write_csv(filepath)
                case "json":
                    data_dict = mock_dataframe.to_dict(as_series=False)
                    with open(filepath, "w") as f:
                        json.dump(data_dict, f, indent=2)
                case _:
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

        mock_sample_data.serialize_metadata.side_effect = mock_serialize

        # Mock the write_data function
        def mock_write_data(filepath, data_format):
            match data_format:
                case "parquet":
                    actual_dataframe.write_parquet(filepath)
                case "csv":
                    actual_dataframe.write_csv(filepath)
                case "json":
                    data_dict = actual_dataframe.to_dict(as_series=False)
                    with open(filepath, "w") as f:
                        json.dump(data_dict, f, indent=2)
                case _:
                    raise ValueError(f"Unsupported data format: {data_format}")

        mock_sample_data.write_data.side_effect = mock_write_data

        return mock_sample_data

    @pytest.fixture
    def basic_eval_task(self) -> EvalTask:
        """Provides a basic evaluation task for testing.

        Returns:
            EvalTask: A basic evaluation task with sentiment analysis schema.
        """
        return EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="structured",
        )

    @pytest.fixture
    def another_basic_eval_task(self) -> EvalTask:
        """Provides a basic evaluation task for testing.

        Returns:
            EvalTask: A basic evaluation task with sentiment analysis schema.
        """
        return EvalTask(
            task_schemas={"rejection": ["rejected", "accepted"]},
            prompt_columns=["prompt"],
            response_columns=["response"],
            answering_method="structured",
        )

    @pytest.fixture
    def basic_eval_task_no_prompt(self) -> EvalTask:
        """Provides a basic evaluation task for testing.

        Returns:
            EvalTask: A basic evaluation task with sentiment analysis schema.
        """
        return EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=None,
            response_columns=["response"],
            answering_method="structured",
        )

    @pytest.fixture
    def basic_eval_task_empty_prompt(self) -> EvalTask:
        """Provides a basic evaluation task for testing.

        Returns:
            EvalTask: A basic evaluation task with sentiment analysis schema.
        """
        return EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=[],
            response_columns=["response"],
            answering_method="structured",
        )

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
        assert meta_evaluator.eval_task is None

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

    # === add_eval_task() Method Tests ===

    def test_add_eval_task_first_time(self, meta_evaluator, basic_eval_task):
        """Test adding evaluation task when no task exists."""
        assert meta_evaluator.eval_task is None

        meta_evaluator.add_eval_task(basic_eval_task)

        assert meta_evaluator.eval_task == basic_eval_task

    def test_add_eval_task_already_exists_no_overwrite(
        self, meta_evaluator, basic_eval_task, another_basic_eval_task
    ):
        """Test EvalTaskAlreadyExistsException when task exists and overwrite is False (default)."""
        # Add task first time
        meta_evaluator.add_eval_task(basic_eval_task)
        assert meta_evaluator.eval_task == basic_eval_task

        # Try to add again without overwrite
        with pytest.raises(
            EvalTaskAlreadyExistsException, match="Evaluation task already exists"
        ):
            meta_evaluator.add_eval_task(another_basic_eval_task)

        # Verify original task is unchanged
        assert meta_evaluator.eval_task == basic_eval_task

    def test_add_eval_task_already_exists_with_overwrite(
        self, meta_evaluator, basic_eval_task, another_basic_eval_task
    ):
        """Test successful replacement when task exists and overwrite is True."""
        # Add task first time
        meta_evaluator.add_eval_task(basic_eval_task)
        original_task = meta_evaluator.eval_task
        assert original_task == basic_eval_task

        # Overwrite with new task
        meta_evaluator.add_eval_task(another_basic_eval_task, overwrite=True)
        new_task = meta_evaluator.eval_task

        assert new_task == another_basic_eval_task
        assert new_task != original_task

    def test_add_eval_task_preserves_client_registry(
        self, meta_evaluator, basic_eval_task, openai_environment
    ):
        """Test that adding evaluation task doesn't affect client registry."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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

    def test_add_eval_task_no_prompt_columns(
        self, meta_evaluator, basic_eval_task_no_prompt
    ):
        """Test adding evaluation task with no prompt columns."""
        # Add evaluation task
        meta_evaluator.add_eval_task(basic_eval_task_no_prompt)

        # Verify task was added correctly
        assert meta_evaluator.eval_task == basic_eval_task_no_prompt
        assert meta_evaluator.eval_task.prompt_columns is None
        assert meta_evaluator.eval_task.response_columns == ["response"]

    def test_add_eval_task_empty_prompt_columns(
        self, meta_evaluator, basic_eval_task_empty_prompt
    ):
        """Test adding evaluation task with empty prompt columns."""
        # Add evaluation task
        meta_evaluator.add_eval_task(basic_eval_task_empty_prompt)

        # Verify task was added correctly
        assert meta_evaluator.eval_task == basic_eval_task_empty_prompt
        assert meta_evaluator.eval_task.prompt_columns == []
        assert meta_evaluator.eval_task.response_columns == ["response"]

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

    # === Client + Data + Task Integration Tests ===

    def test_add_client_and_data_together(
        self,
        meta_evaluator,
        sample_eval_data,
        basic_eval_task,
        openai_environment,
    ):
        """Test adding OpenAI client, data, and evaluation task together, verify all exist independently."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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

    def test_overwrite_task_preserves_clients(
        self,
        meta_evaluator,
        basic_eval_task,
        another_basic_eval_task,
        openai_environment,
    ):
        """Test that overwriting data doesn't affect client registry."""
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
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

    def test_complete_evaluator_state(
        self,
        meta_evaluator,
        sample_eval_data,
        basic_eval_task,
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
            meta_evaluator.add_eval_task(basic_eval_task)

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
        """Test ValueError when state_filename doesn't end with .json."""
        with pytest.raises(ValueError, match="state_filename must end with .json"):
            meta_evaluator.save_state("invalid_file.txt")

    def test_save_state_include_data_true_but_no_format(self, meta_evaluator):
        """Test ValueError when include_data=True but data_format is None."""
        with pytest.raises(
            ValueError, match="data_format must be specified when include_data=True"
        ):
            meta_evaluator.save_state("test.json", include_data=True, data_format=None)

    def test_save_state_include_data_false(self, meta_evaluator):
        """Test saving state without data serialization."""
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
        meta_evaluator.save_state("test_state.json", include_data=False)

        # Verify state file exists and data file doesn't exist in data directory
        state_file = meta_evaluator.project_dir / "test_state.json"
        data_dir = meta_evaluator.project_dir / "data"

        assert state_file.exists()
        assert not (data_dir / "test_state_data.json").exists()
        assert not (data_dir / "test_state_data.csv").exists()
        assert not (data_dir / "test_state_data.parquet").exists()

        # Verify state file contents
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["version"] == "1.0"
        assert state_data["data"] is None
        assert "openai" in state_data["client_registry"]

    def test_save_state_with_parquet_format(
        self, meta_evaluator, mock_eval_data_with_dataframe
    ):
        """Test saving state with parquet data format."""
        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        meta_evaluator.save_state(
            "test_state.json",
            include_data=True,
            data_format="parquet",
            data_filename="test_state_data.parquet",
        )

        # Verify both files exist
        state_file = meta_evaluator.project_dir / "test_state.json"
        data_file = meta_evaluator.project_dir / "data" / "test_state_data.parquet"
        assert state_file.exists()

        # Verify polars write_parquet was called
        mock_eval_data_with_dataframe.write_data.assert_called_once_with(
            filepath=str(data_file), data_format="parquet"
        )

        # Verify state file contents
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["version"] == "1.0"
        assert state_data["data"]["data_format"] == "parquet"
        assert state_data["data"]["data_file"] == "test_state_data.parquet"

    def test_save_state_with_csv_format(
        self, meta_evaluator, mock_eval_data_with_dataframe
    ):
        """Test saving state with CSV data format."""
        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        meta_evaluator.save_state(
            "test_state.json",
            include_data=True,
            data_format="csv",
            data_filename="test_state_data.csv",
        )

        # Verify both files exist
        state_file = meta_evaluator.project_dir / "test_state.json"
        data_file = meta_evaluator.project_dir / "data" / "test_state_data.csv"
        assert state_file.exists()

        # Verify polars write_csv was called
        mock_eval_data_with_dataframe.write_data.assert_called_once_with(
            filepath=str(data_file), data_format="csv"
        )

        # Verify state file contents
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["data"]["data_format"] == "csv"
        assert state_data["data"]["data_file"] == "test_state_data.csv"

    def test_save_state_with_json_format(
        self, meta_evaluator, mock_eval_data_with_dataframe
    ):
        """Test saving state with JSON data format."""
        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        meta_evaluator.save_state(
            "test_state.json",
            include_data=True,
            data_format="json",
            data_filename="test_state_data.json",
        )

        # Verify both files exist
        state_file = meta_evaluator.project_dir / "test_state.json"
        data_file = meta_evaluator.project_dir / "data" / "test_state_data.json"
        assert state_file.exists()

        # Verify polars to_dict was called for JSON serialization
        mock_eval_data_with_dataframe.write_data.assert_called_once_with(
            filepath=str(data_file), data_format="json"
        )

        # Verify data file was created with JSON content
        assert data_file.exists()

    def test_save_state_no_data_present(self, meta_evaluator):
        """Test saving when include_data=True but self.data is None."""
        # Don't add any data to meta_evaluator
        assert meta_evaluator.data is None

        meta_evaluator.save_state(
            "test_state.json", include_data=True, data_format="csv"
        )

        # Verify state file exists but no data file
        state_file = meta_evaluator.project_dir / "test_state.json"
        data_dir = meta_evaluator.project_dir / "data"
        assert state_file.exists()
        assert not (data_dir / "test_state_data.parquet").exists()

        # Verify state file contents
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["data"] is None

    def test_save_state_include_task_false(self, meta_evaluator):
        """Test saving state without evaluation task serialization."""
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
        meta_evaluator.save_state("test_state.json", include_data=False)

        # Verify state file exists and data file doesn't
        state_file = meta_evaluator.project_dir / "test_state.json"
        data_dir = meta_evaluator.project_dir / "data"
        assert state_file.exists()
        assert not (data_dir / "test_state_data.json").exists()
        assert not (data_dir / "test_state_data.csv").exists()
        assert not (data_dir / "test_state_data.parquet").exists()

        # Verify state file contents
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["version"] == "1.0"
        assert "openai" in state_data["client_registry"]
        assert state_data["eval_task"] is None

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
        self, meta_evaluator, mock_eval_data_with_dataframe
    ):
        """Test saving with custom data filename for JSON format."""
        custom_data_file = "my_custom_data.json"

        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        meta_evaluator.save_state(
            "test_state.json",
            include_data=True,
            data_format="json",
            data_filename=custom_data_file,
        )

        # Verify custom data filename is used
        state_file = meta_evaluator.project_dir / "test_state.json"
        expected_data_path = meta_evaluator.project_dir / "data" / custom_data_file
        assert expected_data_path.exists()

        # Verify state file references custom filename
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["version"] == "1.0"
        assert state_data["data"]["data_file"] == custom_data_file
        assert state_data["data"]["data_format"] == "json"

    def test_save_state_with_custom_data_filename_csv(
        self, meta_evaluator, mock_eval_data_with_dataframe
    ):
        """Test saving with custom data filename for CSV format."""
        custom_data_file = "my_custom_data.csv"

        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        meta_evaluator.save_state(
            "test_state.json",
            include_data=True,
            data_format="csv",
            data_filename=custom_data_file,
        )

        # Verify polars write_csv was called with custom path
        expected_data_path = meta_evaluator.project_dir / "data" / custom_data_file
        mock_eval_data_with_dataframe.write_data.assert_called_once_with(
            filepath=str(expected_data_path), data_format="csv"
        )

        # Verify state file references custom filename
        state_file = meta_evaluator.project_dir / "test_state.json"
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["data"]["data_file"] == custom_data_file
        assert state_data["data"]["data_format"] == "csv"

    def test_save_state_with_custom_data_filename_parquet(
        self, meta_evaluator, mock_eval_data_with_dataframe
    ):
        """Test saving with custom data filename for Parquet format."""
        custom_data_file = "my_custom_data.parquet"

        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        meta_evaluator.save_state(
            "test_state.json",
            include_data=True,
            data_format="parquet",
            data_filename=custom_data_file,
        )

        # Verify polars write_parquet was called with custom path
        expected_data_path = meta_evaluator.project_dir / "data" / custom_data_file
        mock_eval_data_with_dataframe.write_data.assert_called_once_with(
            filepath=str(expected_data_path), data_format="parquet"
        )

        # Verify state file references custom filename
        state_file = meta_evaluator.project_dir / "test_state.json"
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
        self, meta_evaluator
    ):
        """Test that data_filename extension is not validated when include_data=False."""
        # This should not raise an exception even with wrong extension
        # because data_filename is ignored when include_data=False
        meta_evaluator.save_state(
            "test.json", include_data=False, data_filename="wrong_extension.csv"
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
        self, meta_evaluator, mock_eval_data_with_dataframe
    ):
        """Test that auto-generated filename is used when data_filename=None."""
        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        meta_evaluator.save_state(
            "test_state.json",
            include_data=True,
            data_format="json",
            data_filename=None,  # Explicitly set to None
        )

        # Verify auto-generated filename is used
        state_file = meta_evaluator.project_dir / "test_state.json"
        auto_generated_file = (
            meta_evaluator.project_dir / "data" / "test_state_data.json"
        )
        assert auto_generated_file.exists()

        # Verify state file references auto-generated filename
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["data"]["data_file"] == "test_state_data.json"

    def test_save_and_load_eval_task_no_prompt_columns(
        self, meta_evaluator, basic_eval_task_no_prompt, openai_environment
    ):
        """Test saving and loading MetaEvaluator with EvalTask that has no prompt columns."""
        # Add task to evaluator
        meta_evaluator.add_eval_task(basic_eval_task_no_prompt)

        # Save state
        meta_evaluator.save_state(
            "test_state.json",
            include_task=True,
            include_data=True,
            data_format="json",
            data_filename=None,
        )

        # Load state
        loaded_evaluator = MetaEvaluator.load_state(
            project_dir=str(meta_evaluator.project_dir),
            state_filename="test_state.json",
            load_task=True,
            openai_api_key="test-api-key",
        )

        # Verify task was loaded correctly with no prompt columns
        assert loaded_evaluator.eval_task is not None
        assert loaded_evaluator.eval_task.prompt_columns is None
        assert loaded_evaluator.eval_task.response_columns == ["response"]
        assert loaded_evaluator.eval_task.task_schemas == {
            "sentiment": ["positive", "negative", "neutral"]
        }
        assert loaded_evaluator.eval_task.answering_method == "structured"

    def test_save_and_load_eval_task_empty_prompt_columns(
        self, meta_evaluator, basic_eval_task_empty_prompt, tmp_path, openai_environment
    ):
        """Test saving and loading MetaEvaluator with EvalTask that has empty prompt columns."""
        # Add task to evaluator
        meta_evaluator.add_eval_task(basic_eval_task_empty_prompt)

        # Save state
        meta_evaluator.save_state(
            "test_state.json",
            include_task=True,
            include_data=True,
            data_format="json",
            data_filename=None,
        )

        # Load state
        loaded_evaluator = MetaEvaluator.load_state(
            project_dir=str(meta_evaluator.project_dir),
            state_filename="test_state.json",
            load_task=True,
            openai_api_key="test-api-key",
        )

        # Verify task was loaded correctly with empty prompt columns
        assert loaded_evaluator.eval_task is not None
        assert loaded_evaluator.eval_task.prompt_columns == []
        assert loaded_evaluator.eval_task.response_columns == ["response"]
        assert loaded_evaluator.eval_task.task_schemas == {
            "sentiment": ["positive", "negative", "neutral"]
        }
        assert loaded_evaluator.eval_task.answering_method == "structured"

    # === Private Serialization Method Tests ===

    def test_serialize_include_data_false(self, meta_evaluator):
        """Test _serialize when data should not be included."""
        state = meta_evaluator._serialize(
            include_task=True,
            include_data=False,
            data_format=None,
            data_filename=None,
        )

        assert state.version == "1.0"
        assert state.data is None
        assert state.client_registry is not None
        assert state.eval_task is None

    def test_serialize_include_data_true(
        self, meta_evaluator, mock_eval_data_with_dataframe
    ):
        """Test _serialize when data should be included."""
        meta_evaluator.add_data(mock_eval_data_with_dataframe)

        state = meta_evaluator._serialize(
            include_task=True,
            include_data=True,
            data_format="parquet",
            data_filename="test_data.parquet",
        )

        assert state.version == "1.0"
        assert state.data is not None
        assert state.data.data_format == "parquet"
        assert state.data.data_file == "test_data.parquet"
        assert state.eval_task is None

    def test_serialize_include_task_false(self, meta_evaluator, basic_eval_task):
        """Test _serialize when task should not be included."""
        state = meta_evaluator._serialize(
            include_task=False,
            include_data=False,
            data_format=None,
            data_filename=None,
        )

        assert state.version == "1.0"
        assert state.data is None
        assert state.eval_task is None

    def test_serialize_include_task_true(self, meta_evaluator, basic_eval_task):
        """Test _serialize when task should be included."""
        meta_evaluator.add_eval_task(basic_eval_task)

        state = meta_evaluator._serialize(
            include_task=True,
            include_data=False,
            data_format=None,
            data_filename=None,
        )

        assert state.version == "1.0"
        assert state.data is None
        assert state.eval_task is not None

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
        basic_eval_task,
        tmp_path,
    ):
        """Test complete save operation and verify actual file contents match expected structure."""
        # Set up complete MetaEvaluator state
        with patch(
            "meta_evaluator.meta_evaluator.metaevaluator.OpenAIClient"
        ) as mock_client_class:
            mock_client = self._create_mock_openai_client()
            mock_client_class.return_value = mock_client

            meta_evaluator.add_openai()
            meta_evaluator.add_data(mock_eval_data_with_dataframe)
            meta_evaluator.add_eval_task(basic_eval_task)

            # Save state
            meta_evaluator.save_state(
                "integration_test.json", include_data=True, data_format="parquet"
            )

            # Verify files exist
            state_file = meta_evaluator.project_dir / "integration_test.json"
            data_file = (
                meta_evaluator.project_dir / "data" / "integration_test_data.parquet"
            )
            assert state_file.exists()

            # Verify state file structure
            with open(state_file) as f:
                state_data = json.load(f)

            # Verify top-level structure
            assert state_data["version"] == "1.0"
            assert "client_registry" in state_data
            assert "data" in state_data
            assert "eval_task" in state_data

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

            # Verify evaluation task structure
            eval_task_config = state_data["eval_task"]
            assert eval_task_config["task_schemas"] == {
                "sentiment": ["positive", "negative", "neutral"]
            }
            assert eval_task_config["prompt_columns"] == ["text"]
            assert eval_task_config["response_columns"] == ["response"]
            assert eval_task_config["answering_method"] == "structured"

            # Verify DataFrame method was called with data directory path
            mock_eval_data_with_dataframe.write_data.assert_called_once_with(
                filepath=str(data_file), data_format="parquet"
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

    def test_save_with_complex_file_paths(self, meta_evaluator, tmp_path):
        """Test saving with complex file paths including spaces and nested directories."""
        # Test with complex filename path (subdirectories within project_dir)
        complex_filename = "test dir/with spaces/and-dashes/complex file name.json"

        meta_evaluator.save_state(complex_filename, include_data=False)

        # Verify subdirectories were created and file exists within project directory
        state_file = meta_evaluator.project_dir / complex_filename
        assert state_file.exists()
        assert state_file.parent.exists()  # Verify subdirectories were created

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

    def test_load_state_with_eval_data_json_format(
        self, tmp_path, sample_eval_data, openai_environment, basic_eval_task
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

            # Create and save evaluator with data and evaluation task
            original_evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            original_evaluator.add_openai()
            original_evaluator.add_data(sample_eval_data)
            original_evaluator.add_eval_task(basic_eval_task)

            original_evaluator.save_state(
                "test_state.json", include_data=True, data_format="json"
            )

            # Load from JSON
            loaded_evaluator = MetaEvaluator.load_state(
                project_dir=str(original_evaluator.project_dir),
                state_filename="test_state.json",
                load_data=True,
                openai_api_key="test-api-key",
            )

            # Verify data was loaded
            assert loaded_evaluator.data is not None
            assert loaded_evaluator.data.name == sample_eval_data.name
            assert loaded_evaluator.data.id_column == sample_eval_data.id_column
            assert isinstance(loaded_evaluator.data, EvalData)

    def test_load_state_with_sample_eval_data_csv_format(
        self, tmp_path, mock_sample_eval_data, openai_environment, basic_eval_task
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
            original_evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            original_evaluator.add_openai()
            original_evaluator.add_data(mock_sample_eval_data)
            original_evaluator.add_eval_task(basic_eval_task)

            original_evaluator.save_state(
                "test_state.json", include_data=True, data_format="csv"
            )

            # Ensure the data file exists in the data directory
            data_file = original_evaluator.project_dir / "data" / "test_state_data.csv"
            assert data_file.exists()

            # Load from JSON
            loaded_evaluator = MetaEvaluator.load_state(
                project_dir=str(original_evaluator.project_dir),
                state_filename="test_state.json",
                load_data=True,
                openai_api_key="test-api-key",
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

    def test_load_state_with_eval_task(
        self, tmp_path, sample_eval_data, openai_environment, basic_eval_task
    ):
        """Test loading MetaEvaluator with evaluation task from JSON."""
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

            # Create and save evaluator with data and evaluation task
            original_evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            original_evaluator.add_openai()
            original_evaluator.add_data(sample_eval_data)
            original_evaluator.add_eval_task(basic_eval_task)

            original_evaluator.save_state(
                "test_state.json", include_data=True, data_format="json"
            )

            # Load from JSON
            loaded_evaluator = MetaEvaluator.load_state(
                project_dir=str(original_evaluator.project_dir),
                state_filename="test_state.json",
                load_data=True,
                openai_api_key="test-api-key",
            )

            # Verify evaluation task was loaded
            assert loaded_evaluator.eval_task is not None
            assert (
                loaded_evaluator.eval_task.task_schemas == basic_eval_task.task_schemas
            )
            assert (
                loaded_evaluator.eval_task.prompt_columns
                == basic_eval_task.prompt_columns
            )
            assert (
                loaded_evaluator.eval_task.response_columns
                == basic_eval_task.response_columns
            )
            assert (
                loaded_evaluator.eval_task.answering_method
                == basic_eval_task.answering_method
            )

    def test_load_state_skip_data_and_eval_task_loading(
        self, tmp_path, sample_eval_data, openai_environment, basic_eval_task
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

            # Create and save evaluator with data and evaluation task
            original_evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            original_evaluator.add_openai()
            original_evaluator.add_data(sample_eval_data)
            original_evaluator.add_eval_task(basic_eval_task)

            original_evaluator.save_state(
                "test_state.json", include_data=True, data_format="json"
            )

            # Load from JSON without data and evaluation task
            loaded_evaluator = MetaEvaluator.load_state(
                project_dir=str(original_evaluator.project_dir),
                state_filename="test_state.json",
                load_data=False,
                load_task=False,
                openai_api_key="test-api-key",
            )

            # Verify client loaded but data and evaluation task skipped
            assert LLMClientEnum.OPENAI in loaded_evaluator.client_registry
            assert loaded_evaluator.data is None
            assert loaded_evaluator.eval_task is None

    def test_load_state_invalid_file_extension(self, tmp_path):
        """Test ValueError when state filename doesn't end with .json."""
        with pytest.raises(ValueError, match="state_filename must end with .json"):
            MetaEvaluator.load_state(str(tmp_path), "invalid_file.txt")

    def test_load_state_nonexistent_file(self, tmp_path):
        """Test FileNotFoundError when state file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="State file not found"):
            MetaEvaluator.load_state(str(tmp_path), "nonexistent.json")

    def test_load_state_invalid_json(self, tmp_path):
        """Test ValueError when state file contains invalid JSON."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        state_file = project_dir / "invalid.json"
        state_file.write_text("{ invalid json }")

        with pytest.raises(
            ValueError,
            match=re.compile(rf"{re.escape(INVALID_JSON_STRUCTURE_MSG)}", re.DOTALL),
        ):
            MetaEvaluator.load_state(str(project_dir), "invalid.json")

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
            MetaEvaluator.load_state(str(tmp_path), "incomplete.json")

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
            MetaEvaluator.load_state(str(tmp_path), "test.json")

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
                    project_dir=str(tmp_path),
                    state_filename="test.json",
                    load_data=True,
                    openai_api_key="test-api-key",
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
                    project_dir=str(tmp_path),
                    state_filename="test.json",
                    load_data=True,
                    openai_api_key="test-api-key",
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
                project_dir=str(tmp_path),
                state_filename="test.json",
                load_data=False,
                openai_api_key="custom-openai-key",
                azure_openai_api_key="custom-azure-key",
            )

            # Verify both clients were reconstructed
            assert LLMClientEnum.OPENAI in loaded_evaluator.client_registry
            assert LLMClientEnum.AZURE_OPENAI in loaded_evaluator.client_registry

    # ===== JUDGE MANAGEMENT TESTS =====

    @pytest.fixture
    def sample_prompt(self):
        """Provides a sample Prompt object for testing.

        Returns:
            Prompt: A sample prompt for testing.
        """
        return Prompt(
            id="test_prompt",
            prompt="You are a helpful evaluator. Rate the response on accuracy.",
        )

    @pytest.fixture
    def meta_evaluator_with_task(self, meta_evaluator, basic_eval_task):
        """Provides a MetaEvaluator with eval_task set.

        Args:
            meta_evaluator: The MetaEvaluator instance to modify.
            basic_eval_task: The basic evaluation task to add.

        Returns:
            MetaEvaluator: The modified MetaEvaluator instance.
        """
        meta_evaluator.add_eval_task(basic_eval_task)
        return meta_evaluator

    def test_add_judge_success(self, meta_evaluator_with_task, sample_prompt):
        """Test successfully adding a judge."""
        judge_id = "test_judge"

        meta_evaluator_with_task.add_judge(
            judge_id=judge_id,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        # Verify judge was added
        assert judge_id in meta_evaluator_with_task.judge_registry
        judge = meta_evaluator_with_task.judge_registry[judge_id]
        assert judge.id == judge_id
        assert judge.llm_client_enum == LLMClientEnum.OPENAI
        assert judge.model == "gpt-4"
        assert judge.prompt == sample_prompt

    def test_add_judge_without_eval_task(self, meta_evaluator, sample_prompt):
        """Test adding judge fails when no eval_task is set."""
        with pytest.raises(
            ValueError, match="eval_task must be set before adding judges"
        ):
            meta_evaluator.add_judge(
                judge_id="test_judge",
                llm_client_enum=LLMClientEnum.OPENAI,
                model="gpt-4",
                prompt=sample_prompt,
            )

    def test_add_judge_already_exists(self, meta_evaluator_with_task, sample_prompt):
        """Test adding judge with same ID raises exception."""
        judge_id = "test_judge"

        # Add judge first time
        meta_evaluator_with_task.add_judge(
            judge_id=judge_id,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        # Try to add again without override
        with pytest.raises(JudgeAlreadyExistsException):
            meta_evaluator_with_task.add_judge(
                judge_id=judge_id,
                llm_client_enum=LLMClientEnum.OPENAI,
                model="gpt-4",
                prompt=sample_prompt,
            )

    def test_add_judge_override_existing(self, meta_evaluator_with_task, sample_prompt):
        """Test overriding existing judge."""
        judge_id = "test_judge"

        # Add judge first time
        meta_evaluator_with_task.add_judge(
            judge_id=judge_id,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        # Override with different model
        new_prompt = Prompt(id="new_prompt", prompt="New prompt")
        meta_evaluator_with_task.add_judge(
            judge_id=judge_id,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-3.5-turbo",
            prompt=new_prompt,
            override_existing=True,
        )

        # Verify override worked
        judge = meta_evaluator_with_task.judge_registry[judge_id]
        assert judge.model == "gpt-3.5-turbo"
        assert judge.prompt == new_prompt

    def test_get_judge_success(self, meta_evaluator_with_task, sample_prompt):
        """Test successfully retrieving a judge."""
        judge_id = "test_judge"

        meta_evaluator_with_task.add_judge(
            judge_id=judge_id,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        retrieved_judge = meta_evaluator_with_task.get_judge(judge_id)
        assert retrieved_judge.id == judge_id
        assert retrieved_judge.model == "gpt-4"

    def test_get_judge_not_found(self, meta_evaluator_with_task):
        """Test retrieving non-existent judge raises exception."""
        with pytest.raises(JudgeNotFoundException):
            meta_evaluator_with_task.get_judge("non_existent_judge")

    def test_get_judge_list_empty(self, meta_evaluator_with_task):
        """Test getting judge list when empty."""
        judge_list = meta_evaluator_with_task.get_judge_list()
        assert judge_list == []

    def test_get_judge_list_with_judges(self, meta_evaluator_with_task, sample_prompt):
        """Test getting judge list with multiple judges."""
        judge_ids = ["judge1", "judge2", "judge3"]

        for judge_id in judge_ids:
            meta_evaluator_with_task.add_judge(
                judge_id=judge_id,
                llm_client_enum=LLMClientEnum.OPENAI,
                model="gpt-4",
                prompt=sample_prompt,
            )

        judge_list = meta_evaluator_with_task.get_judge_list()
        assert len(judge_list) == 3
        retrieved_ids = [judge_id for judge_id, _ in judge_list]
        assert set(retrieved_ids) == set(judge_ids)

    def test_load_judges_from_yaml_success(self, meta_evaluator_with_task, tmp_path):
        """Test successfully loading judges from YAML file."""
        # Create prompt file
        prompt_file = tmp_path / "test_prompt.md"
        prompt_file.write_text("You are a helpful evaluator.")

        # Create YAML file
        yaml_content = f"""judges:
  - id: test_judge
    llm_client: openai
    model: gpt-4
    prompt_file: {prompt_file}
"""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)

        # Load judges
        meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

        # Verify judge was loaded
        assert "test_judge" in meta_evaluator_with_task.judge_registry
        judge = meta_evaluator_with_task.judge_registry["test_judge"]
        assert judge.id == "test_judge"
        assert judge.llm_client_enum == LLMClientEnum.OPENAI
        assert judge.model == "gpt-4"
        assert judge.prompt.prompt == "You are a helpful evaluator."

    def test_load_judges_from_yaml_relative_path(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading judges with relative prompt paths."""
        # Create prompt file in subdirectory
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        prompt_file = prompt_dir / "test_prompt.md"
        prompt_file.write_text("You are a helpful evaluator.")

        # Create YAML file with relative path
        yaml_content = """judges:
  - id: test_judge
    llm_client: openai
    model: gpt-4
    prompt_file: prompts/test_prompt.md
"""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)

        # Load judges
        meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

        # Verify judge was loaded
        assert "test_judge" in meta_evaluator_with_task.judge_registry
        judge = meta_evaluator_with_task.judge_registry["test_judge"]
        assert judge.prompt.prompt == "You are a helpful evaluator."

    def test_load_judges_from_yaml_without_eval_task(self, meta_evaluator, tmp_path):
        """Test loading judges fails when no eval_task is set."""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text("judges: []")

        with pytest.raises(
            ValueError, match="eval_task must be set before loading judges from YAML"
        ):
            meta_evaluator.load_judges_from_yaml(str(yaml_file))

    def test_load_judges_from_yaml_file_not_found(self, meta_evaluator_with_task):
        """Test loading judges from non-existent YAML file."""
        with pytest.raises(FileNotFoundError):
            meta_evaluator_with_task.load_judges_from_yaml("non_existent.yaml")

    def test_load_judges_from_yaml_invalid_yaml(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading judges from invalid YAML file."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: content: [")

        with pytest.raises(InvalidYAMLStructureException):
            meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

    def test_load_judges_from_yaml_invalid_structure(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading judges from YAML with invalid structure."""
        yaml_content = """judges:
  - id: test_judge
    # missing required fields
"""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(InvalidYAMLStructureException):
            meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

    def test_load_judges_from_yaml_invalid_llm_client(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading judges with invalid LLM client type."""
        prompt_file = tmp_path / "test_prompt.md"
        prompt_file.write_text("Test prompt")

        yaml_content = f"""judges:
  - id: test_judge
    llm_client: invalid_client
    model: gpt-4
    prompt_file: {prompt_file}
"""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="Invalid llm_client"):
            meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

    def test_load_judges_from_yaml_prompt_file_not_found(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading judges with non-existent prompt file."""
        yaml_content = """judges:
  - id: test_judge
    llm_client: openai
    model: gpt-4
    prompt_file: non_existent_prompt.md
"""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(PromptFileNotFoundException):
            meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

    def test_load_judges_from_yaml_multiple_judges(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading multiple judges from YAML."""
        # Create prompt files
        prompt1 = tmp_path / "prompt1.md"
        prompt1.write_text("Prompt 1")
        prompt2 = tmp_path / "prompt2.md"
        prompt2.write_text("Prompt 2")

        yaml_content = f"""judges:
  - id: judge1
    llm_client: openai
    model: gpt-4
    prompt_file: {prompt1}
  - id: judge2
    llm_client: openai
    model: gpt-3.5-turbo
    prompt_file: {prompt2}
"""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)

        meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

        # Verify both judges were loaded
        assert "judge1" in meta_evaluator_with_task.judge_registry
        assert "judge2" in meta_evaluator_with_task.judge_registry

        judge1 = meta_evaluator_with_task.judge_registry["judge1"]
        judge2 = meta_evaluator_with_task.judge_registry["judge2"]

        assert judge1.model == "gpt-4"
        assert judge2.model == "gpt-3.5-turbo"
        assert judge1.prompt.prompt == "Prompt 1"
        assert judge2.prompt.prompt == "Prompt 2"

    def test_load_judges_from_yaml_override_existing(
        self, meta_evaluator_with_task, tmp_path, sample_prompt
    ):
        """Test overriding existing judge when loading from YAML."""
        # Add judge programmatically first
        meta_evaluator_with_task.add_judge(
            judge_id="test_judge",
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        # Create YAML with same judge ID
        prompt_file = tmp_path / "new_prompt.md"
        prompt_file.write_text("New prompt content")

        yaml_content = f"""judges:
  - id: test_judge
    llm_client: openai
    model: gpt-3.5-turbo
    prompt_file: {prompt_file}
"""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)

        # Load with override
        meta_evaluator_with_task.load_judges_from_yaml(
            str(yaml_file), override_existing=True
        )

        # Verify judge was overridden
        judge = meta_evaluator_with_task.judge_registry["test_judge"]
        assert judge.model == "gpt-3.5-turbo"
        assert judge.prompt.prompt == "New prompt content"
