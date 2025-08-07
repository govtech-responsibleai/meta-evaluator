"""Fixtures for data module tests.

This conftest provides fixtures for EvalData, SampleEvalData, DataFrames,
and file-related testing functionality.
"""

import json
from unittest.mock import MagicMock

import polars as pl
import pytest

from meta_evaluator.data import EvalData, SampleEvalData
from meta_evaluator.data.serialization import DataMetadata

# ==== DATAFRAME FIXTURES ====


@pytest.fixture
def valid_dataframe() -> pl.DataFrame:
    """Provides a standard valid DataFrame for testing.

    Returns:
        pl.DataFrame: A DataFrame with sample evaluation data.
    """
    return pl.DataFrame(
        {
            "question": ["What is 2+2?", "What is 3+3?"],
            "answer": ["4", "6"],
            "model_response": ["Four", "Six"],
            "difficulty": ["easy", "easy"],
            "human_rating": [5, 4],
            "extra_col": ["a", "b"],
        }
    )


@pytest.fixture
def minimal_dataframe() -> pl.DataFrame:
    """Provides a minimal valid DataFrame.

    Returns:
        pl.DataFrame: A DataFrame with minimal test data columns.
    """
    return pl.DataFrame(
        {
            "input": ["test1", "test2"],
            "output": ["result1", "result2"],
        }
    )


@pytest.fixture
def valid_dataframe_for_sampling() -> pl.DataFrame:
    """Provides a DataFrame suitable for sampling tests.

    Returns:
        pl.DataFrame: A DataFrame with sample evaluation data for stratification.
    """
    return pl.DataFrame(
        {
            "question": [
                "What is 2+2?",
                "What is 3+3?",
                "What is 4+4?",
                "What is 5+5?",
            ],
            "answer": ["4", "6", "8", "10"],
            "model_response": ["Four", "Six", "Eight", "Ten"],
            "topic": ["math", "math", "math", "math"],
            "difficulty": ["easy", "easy", "medium", "hard"],
            "language": ["en", "en", "en", "fr"],
            "human_rating": [5, 4, 3, 2],
        }
    )


@pytest.fixture
def multi_stratification_dataframe() -> pl.DataFrame:
    """Provides a DataFrame with multiple stratification combinations.

    Returns:
        pl.DataFrame: A DataFrame with varied column value combinations.
    """
    return pl.DataFrame(
        {
            "input": ["q1", "q2", "q3", "q4", "q5", "q6"],
            "output": ["a1", "a2", "a3", "a4", "a5", "a6"],
            "category": ["A", "A", "B", "B", "C", "C"],
            "level": ["1", "2", "1", "2", "1", "2"],
        }
    )


@pytest.fixture
def empty_dataframe() -> pl.DataFrame:
    """Provides an empty DataFrame for error testing.

    Returns:
        pl.DataFrame: An empty DataFrame.
    """
    return pl.DataFrame()


@pytest.fixture
def single_row_partitions_dataframe() -> pl.DataFrame:
    """Provides a DataFrame where each stratification combination has only one row.

    Returns:
        pl.DataFrame: A DataFrame with single-row partitions.
    """
    return pl.DataFrame(
        {
            "input": ["q1", "q2", "q3"],
            "output": ["a1", "a2", "a3"],
            "unique_meta": ["A", "B", "C"],
        }
    )


# ==== EVALDATA FIXTURES ====


@pytest.fixture
def sample_eval_data(valid_dataframe):
    """Provides a sample EvalData object with comprehensive mocking.

    Args:
        valid_dataframe: A valid DataFrame fixture.

    Returns:
        MagicMock: A mocked EvalData instance for testing purposes.
    """
    mock_eval_data = MagicMock(spec=EvalData)
    mock_eval_data.name = "test_dataset"
    mock_eval_data.id_column = "id"
    mock_eval_data.data = valid_dataframe

    # Mock the serialize_metadata function
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
    def mock_write_data(filepath, data_format):
        match data_format:
            case "parquet":
                valid_dataframe.write_parquet(filepath)
            case "csv":
                valid_dataframe.write_csv(filepath)
            case "json":
                data_dict = valid_dataframe.to_dict(as_series=False)
                with open(filepath, "w") as f:
                    json.dump(data_dict, f, indent=2)
            case _:
                raise ValueError(f"Unsupported data format: {data_format}")

    mock_eval_data.write_data.side_effect = mock_write_data
    return mock_eval_data


@pytest.fixture
def mock_sample_eval_data(valid_dataframe_for_sampling):
    """Provides mock SampleEvalData with complete sampling metadata.

    Args:
        valid_dataframe_for_sampling: A DataFrame suitable for sampling.

    Returns:
        MagicMock: Mocked SampleEvalData instance with complete sampling metadata.
    """
    mock_sample_data = MagicMock(spec=SampleEvalData)
    mock_sample_data.name = "sample_dataset"
    mock_sample_data.id_column = "sample_id"
    mock_sample_data.sample_name = "Test Sample"
    mock_sample_data.stratification_columns = ["topic", "difficulty"]
    mock_sample_data.sample_percentage = 0.3
    mock_sample_data.seed = 42
    mock_sample_data.sampling_method = "stratified_by_columns"
    mock_sample_data.data = valid_dataframe_for_sampling

    # Mock the serialize_metadata function
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
                valid_dataframe_for_sampling.write_parquet(filepath)
            case "csv":
                valid_dataframe_for_sampling.write_csv(filepath)
            case "json":
                data_dict = valid_dataframe_for_sampling.to_dict(as_series=False)
                with open(filepath, "w") as f:
                    json.dump(data_dict, f, indent=2)
            case _:
                raise ValueError(f"Unsupported data format: {data_format}")

    mock_sample_data.write_data.side_effect = mock_write_data
    return mock_sample_data


# ==== FILE FIXTURES ====


@pytest.fixture
def valid_csv_file(tmp_path):
    """Create a valid CSV file for testing.

    Returns:
        str: Path to the created CSV file.
    """
    csv_content = """question,answer,model_response,difficulty,rating
What is 2+2?,4,Four,easy,5
What is 3+3?,6,Six,medium,4"""
    csv_file = tmp_path / "valid_data.csv"
    csv_file.write_text(csv_content)
    return str(csv_file)


@pytest.fixture
def valid_json_file(tmp_path):
    """Create a valid JSON file for testing.

    Returns:
        str: Path to the created JSON file.
    """
    json_content = """[
        {
            "question": "What is 2+2?",
            "answer": "4",
            "model_response": "Four",
            "difficulty": "easy",
            "rating": 5
        },
        {
            "question": "What is 3+3?",
            "answer": "6", 
            "model_response": "Six",
            "difficulty": "medium",
            "rating": 4
        }
    ]"""
    json_file = tmp_path / "valid_data.json"
    json_file.write_text(json_content)
    return str(json_file)


@pytest.fixture
def valid_parquet_file(tmp_path, valid_dataframe):  # noqa: D417
    """Create a valid Parquet file for testing.

    Args:
        valid_dataframe: A valid DataFrame to write as Parquet.

    Returns:
        str: Path to the created Parquet file.
    """
    parquet_file = tmp_path / "valid_data.parquet"
    valid_dataframe.write_parquet(str(parquet_file))
    return str(parquet_file)


@pytest.fixture
def minimal_csv_file(tmp_path):
    """Create a minimal CSV file for testing.

    Returns:
        str: Path to the created minimal CSV file.
    """
    csv_content = """input,output
test1,result1
test2,result2"""
    csv_file = tmp_path / "minimal_data.csv"
    csv_file.write_text(csv_content)
    return str(csv_file)


@pytest.fixture
def empty_csv_file(tmp_path):
    """Create an empty CSV file for error testing.

    Returns:
        str: Path to the created empty CSV file.
    """
    csv_file = tmp_path / "empty_data.csv"
    csv_file.write_text("")
    return str(csv_file)


@pytest.fixture
def invalid_csv_file(tmp_path):
    """Create an invalid CSV file for error testing.

    Returns:
        str: Path to the created invalid CSV file.
    """
    csv_content = """invalid;content;here
this,is,malformed"""
    csv_file = tmp_path / "invalid_data.csv"
    csv_file.write_text(csv_content)
    return str(csv_file)


@pytest.fixture
def headers_only_csv_file(tmp_path):
    """Create CSV with headers but no data.

    Returns:
        str: Path to the created CSV file with headers only.
    """
    csv_content = "input,output\n"
    csv_file = tmp_path / "headers_only.csv"
    csv_file.write_text(csv_content)
    return str(csv_file)


@pytest.fixture
def malformed_quotes_csv(tmp_path):
    """Create CSV with malformed quotes that cause parsing errors.

    Returns:
        str: Path to the created CSV file.
    """
    # Improperly escaped quotes cause ComputeError
    csv_content = """input,output
"test test" test,result1
test2,result2"""
    csv_file = tmp_path / "malformed_quotes.csv"
    csv_file.write_text(csv_content)
    return str(csv_file)


@pytest.fixture
def sample_dataframe():
    """Create a sample polars DataFrame for testing.

    Returns:
        pl.DataFrame: A sample polars DataFrame.
    """
    return pl.DataFrame({"input": ["test1", "test2"], "output": ["result1", "result2"]})


@pytest.fixture
def nonexistent_file():
    """Provide a path to a nonexistent file for error testing.

    Returns:
        str: Path to a nonexistent file.
    """
    return "/path/to/nonexistent/file.csv"


@pytest.fixture
def eval_data_with_stratification_columns(valid_dataframe_for_sampling):
    """Provides an EvalData instance for stratification testing.

    Args:
        valid_dataframe_for_sampling: A DataFrame suitable for sampling.

    Returns:
        EvalData: A configured EvalData instance for sampling tests.
    """
    return EvalData(
        name="test_dataset",
        data=valid_dataframe_for_sampling,
    )


@pytest.fixture
def eval_data_minimal(minimal_dataframe):
    """Provides an EvalData instance with minimal data.

    Args:
        minimal_dataframe: A minimal DataFrame fixture.

    Returns:
        EvalData: An EvalData instance with basic columns for testing.
    """
    return EvalData(
        name="minimal_data",
        data=minimal_dataframe,
    )
