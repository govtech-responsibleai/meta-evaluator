"""Shared fixtures for MetaEvaluator tests."""

import json
import pytest
from unittest.mock import MagicMock
from meta_evaluator.meta_evaluator import MetaEvaluator
from meta_evaluator.data import EvalData, SampleEvalData
from meta_evaluator.data.serialization import DataMetadata
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.common.models import Prompt
from meta_evaluator.llm_client.openai_client import OpenAIClient
from meta_evaluator.llm_client.azureopenai_client import AzureOpenAIClient
from meta_evaluator.llm_client.serialization import (
    OpenAISerializedState,
    AzureOpenAISerializedState,
)


@pytest.fixture
def meta_evaluator(tmp_path) -> MetaEvaluator:
    """Provides a fresh MetaEvaluator instance for testing.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        MetaEvaluator: A new MetaEvaluator instance with temporary project directory.
    """
    return MetaEvaluator(str(tmp_path / "test_project"))


@pytest.fixture
def sample_eval_data():
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
    mock_dataframe.to_dict = MagicMock(return_value={"id": [1, 2], "text": ["a", "b"]})

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
def another_eval_data():
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
    mock_dataframe.to_dict = MagicMock(return_value={"id": [1, 2], "text": ["a", "b"]})

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
def mock_eval_data_with_dataframe():
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
    mock_dataframe.to_dict = MagicMock(return_value={"id": [1, 2], "text": ["a", "b"]})

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
def mock_sample_eval_data():
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
def basic_eval_task() -> EvalTask:
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
def another_basic_eval_task() -> EvalTask:
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
def basic_eval_task_no_prompt() -> EvalTask:
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
def basic_eval_task_empty_prompt() -> EvalTask:
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


@pytest.fixture
def sample_prompt():
    """Provides a sample Prompt object for testing.

    Returns:
        Prompt: A sample prompt for testing.
    """
    return Prompt(
        id="test_prompt",
        prompt="You are a helpful evaluator. Rate the response on accuracy.",
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
