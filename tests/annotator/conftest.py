"""Fixtures for Annotator testing."""

import socket
from unittest.mock import Mock

import polars as pl
import pytest

from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.results import HumanAnnotationResultsBuilder


@pytest.fixture
def free_port():
    """Find and return a free port for testing.

    Returns:
        int: An available port number.
    """
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


@pytest.fixture
def temp_annotations_dir(tmp_path):
    """Create a temporary directory for storing annotation results.

    Returns:
        str: Path to a temporary annotations directory.
    """
    return str(tmp_path / "annotations")


@pytest.fixture
def annotator_eval_task() -> EvalTask:
    """Create a test EvalTask for annotator testing.

    Returns:
        EvalTask: A configured evaluation task.
    """
    return EvalTask(
        task_schemas={
            "sentiment": ["positive", "negative", "neutral"],
            "quality": ["high", "medium", "low"],
            "comments": None,
        },
        prompt_columns=["text", "question"],
        response_columns=["response", "answer"],
        answering_method="structured",
    )


@pytest.fixture
def integration_eval_task() -> EvalTask:
    """Create a test EvalTask for integration testing.

    Returns:
        EvalTask: A configured evaluation task.
    """
    return EvalTask(
        task_schemas={
            "sentiment": ["positive", "negative", "neutral"],
            "quality": ["good", "bad"],
        },
        prompt_columns=["text"],
        response_columns=["response"],
        answering_method="structured",
    )


@pytest.fixture
def sample_dataframe():
    """Create a sample Polars DataFrame for testing.

    Returns:
        pl.DataFrame: A DataFrame with sample data.
    """
    return pl.DataFrame(
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


@pytest.fixture
def annotator_eval_data() -> EvalData:
    """Create test EvalData for annotator testing.

    Returns:
        EvalData: A test dataset.
    """
    df = pl.DataFrame(
        {
            "id": ["1", "2"],
            "text": ["I love this!", "I hate this!"],
            "response": ["Great!", "Awful!"],
        }
    )
    return EvalData(
        name="test_data",
        data=df,
        id_column="id",
    )


@pytest.fixture
def integration_eval_data() -> EvalData:
    """Create test EvalData for integration testing.

    Returns:
        EvalData: A test dataset.
    """
    df = pl.DataFrame(
        {
            "id": ["sample_1", "sample_2", "sample_3"],
            "text": [
                "I love this product!",
                "This is terrible!",
                "It's okay, nothing special",
            ],
            "response": [
                "Thank you for your feedback!",
                "We apologize for the inconvenience.",
                "We appreciate your honest review.",
            ],
        }
    )
    return EvalData(
        name="integration_test_data",
        data=df,
        id_column="id",
    )


@pytest.fixture
def mock_eval_data(sample_dataframe):
    """Create a mock EvalData object for testing.

    Returns:
        Mock: A mock EvalData object.
    """
    mock_data = Mock(spec=EvalData)
    mock_data.id_column = "id"
    mock_data.data = sample_dataframe
    return mock_data


@pytest.fixture
def mock_eval_task():
    """Create a mock EvalTask object for testing.

    Returns:
        Mock: A mock EvalTask object.
    """
    mock_task = Mock(spec=EvalTask)
    mock_task.task_schemas = {
        "sentiment": ["positive", "negative", "neutral"],
        "quality": ["high", "medium", "low"],
        "comments": None,
    }
    mock_task.prompt_columns = ["question"]
    mock_task.response_columns = ["answer"]
    mock_task.answering_method = "structured"
    mock_task.annotation_prompt = "Please evaluate the following response:"
    mock_task.get_required_tasks.return_value = ["sentiment", "quality"]
    return mock_task


@pytest.fixture
def sample_task_schemas():
    """Sample task schemas for testing.

    Returns:
        dict: Task schemas mapping.
    """
    return {
        "task1": ["yes", "no", "maybe"],
        "task2": ["good", "bad", "neutral"],
    }


@pytest.fixture
def sample_expected_ids():
    """Sample expected IDs for testing.

    Returns:
        list: A list of sample IDs.
    """
    return ["id1", "id2", "id3"]


@pytest.fixture
def base_human_results_builder() -> HumanAnnotationResultsBuilder:
    """Provide a basic HumanAnnotationResultsBuilder instance.

    Returns:
        HumanAnnotationResultsBuilder: A builder instance.
    """
    return HumanAnnotationResultsBuilder(
        run_id="run_001",
        annotator_id="annotator_1",
        task_schemas={"task1": ["yes", "no"], "task2": ["good", "bad"]},
        expected_ids=["id1", "id2"],
        required_tasks=["task1", "task2"],
        is_sampled_run=False,
    )


@pytest.fixture
def single_task_human_results_builder() -> HumanAnnotationResultsBuilder:
    """Provides a builder with single task and id.

    Returns:
        HumanAnnotationResultsBuilder: A builder with single task.
    """
    return HumanAnnotationResultsBuilder(
        run_id="single_task_run",
        annotator_id="annotator_1",
        task_schemas={"task1": ["yes", "no"]},
        expected_ids=["id1"],
        required_tasks=["task1"],
        is_sampled_run=False,
    )


@pytest.fixture
def mock_results_builder():
    """Mock HumanAnnotationResultsBuilder for testing.

    Returns:
        Mock: A mock results builder.
    """
    mock_builder = Mock(spec=HumanAnnotationResultsBuilder)
    mock_builder.completed_count = 0
    mock_builder._results = {}
    return mock_builder
