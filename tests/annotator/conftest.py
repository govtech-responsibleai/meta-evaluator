"""Fixtures for Annotator testing.

This conftest provides annotator-specific fixtures used across annotator test modules.
Common fixtures are inherited from the main conftest.py.
"""

import socket
from unittest.mock import Mock, patch

import polars as pl
import pytest

from meta_evaluator.annotator.interface.streamlit_session_manager import (
    StreamlitSessionManager,
)
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.results import HumanAnnotationResultsBuilder

# ==== BASIC FIXTURES ====


@pytest.fixture
def free_port():
    """Find and return a free port for testing.

    Returns:
        int: An available port number that can be used for the Streamlit
             app during testing without conflicts.
    """
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


@pytest.fixture
def temp_annotations_dir(tmp_path):
    """Create a temporary directory for storing annotation results.

    Args:
        tmp_path: pytest's temporary path fixture

    Returns:
        str: Path to a temporary annotations directory for test isolation
    """
    return str(tmp_path / "annotations")


# ==== EVAL TASK FIXTURES ====


@pytest.fixture
def annotator_eval_task() -> EvalTask:
    """Create a test EvalTask for annotator testing.

    Returns:
        EvalTask: A configured evaluation task with:
            - Mixed task types (structured sentiment/quality, some free-form)
            - Prompt and response column definitions
            - Structured answering method for radio button interactions
    """
    return EvalTask(
        task_schemas={
            "sentiment": ["positive", "negative", "neutral"],
            "quality": ["high", "medium", "low"],
            "comments": None,  # Free form text
        },
        prompt_columns=["text", "question"],
        response_columns=["response", "answer"],
        answering_method="structured",
    )


@pytest.fixture
def integration_eval_task() -> EvalTask:
    """Create a test EvalTask for integration testing.

    Returns:
        EvalTask: A configured evaluation task for integration tests.
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


# ==== EVAL DATA FIXTURES ====


@pytest.fixture
def sample_dataframe():
    """Create a sample Polars DataFrame for testing annotation workflows.

    Returns:
        pl.DataFrame: A DataFrame with sample question-answer pairs containing:
            - id: Unique identifiers (1, 2, 3)
            - question: Sample questions for annotation
            - answer: Corresponding answers to the questions
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
        EvalData: A test dataset with sample text/response pairs.
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
        EvalData: A test dataset containing:
            - 3 sample text/response pairs for annotation
            - Varied content (positive, negative, neutral sentiment)
            - Properly structured with ID column for tracking
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


# ==== MOCK FIXTURES ====


@pytest.fixture
def mock_eval_data(sample_dataframe):
    """Create a mock EvalData object for testing.

    Args:
        sample_dataframe: Fixture providing sample DataFrame data

    Returns:
        Mock: A mock EvalData object with:
            - id_column: Set to "id" for data identification
            - data: The sample DataFrame for annotation
    """
    mock_data = Mock(spec=EvalData)
    mock_data.id_column = "id"
    mock_data.data = sample_dataframe
    return mock_data


@pytest.fixture
def mock_eval_task():
    """Create a mock EvalTask object for testing annotation workflows.

    Returns:
        Mock: A mock EvalTask object configured with:
            - task_schemas: Mixed task types (structured and free-form)
            - prompt_columns: ["question"] - columns to display as prompts
            - response_columns: ["answer"] - columns to display as responses
            - answering_method: "structured" - annotation method type
            - annotation_prompt: Default prompt text for annotations
    """
    mock_task = Mock(spec=EvalTask)
    mock_task.task_schemas = {
        "sentiment": ["positive", "negative", "neutral"],
        "quality": ["high", "medium", "low"],
        "comments": None,  # Free form text
    }
    mock_task.prompt_columns = ["question"]
    mock_task.response_columns = ["answer"]
    mock_task.answering_method = "structured"
    mock_task.annotation_prompt = "Please evaluate the following response:"
    mock_task.get_required_columns.return_value = ["sentiment", "quality"]
    return mock_task


# ==== SESSION MANAGER FIXTURES ====


class MockSessionState:
    """Mock Streamlit session state that behaves like both dict and object.

    This class simulates Streamlit's session_state object, which can be accessed
    both as a dictionary (with square brackets) and as an object (with dot notation).
    It provides the same interface as the real session_state for testing purposes.

    Features:
        - Dictionary-style access: state['key'] = value
        - Object-style access: state.key = value
        - Proper attribute handling with underscores for internal attributes
        - Support for all standard dict operations (get, contains, delete)
        - Consistent behavior matching Streamlit's actual session_state
    """

    def __init__(self):
        """Initialize state."""
        self._data = {}

    def __contains__(self, key):
        """Check if key exists in the session state.

        Returns:
            bool: True if key exists, False otherwise.
        """
        return key in self._data

    def __getitem__(self, key):
        """Get item using dictionary-style access.

        Returns:
            Any: The value associated with the key.
        """
        return self._data[key]

    def __setitem__(self, key, value):
        """Set item using dictionary-style access."""
        self._data[key] = value

    def __delitem__(self, key):
        """Delete item using dictionary-style access."""
        del self._data[key]

    def __getattr__(self, name):
        """Get attribute using object-style access.

        Returns:
            Any: The attribute value or None if not found.
        """
        return self._data.get(name)

    def __setattr__(self, name, value):
        """Set attribute using object-style access."""
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __delattr__(self, name):
        """Delete attribute using object-style access.

        Raises:
            AttributeError: If the attribute is not found.
        """
        if name.startswith("_"):
            super().__delattr__(name)
        else:
            if name in self._data:
                del self._data[name]
            else:
                raise AttributeError(
                    f"'MockSessionState' object has no attribute '{name}'"
                )

    def get(self, key, default=None):
        """Get item from session state with default if not found.

        Returns:
            Any: The value for key, or default if key not found.
        """
        return self._data.get(key, default)


@pytest.fixture
def mock_streamlit_session_state():
    """Mock streamlit.session_state for tests that need it.

    This fixture patches streamlit.session_state with a MockSessionState
    instance for tests that explicitly request it.
    """
    with patch("streamlit.session_state", MockSessionState()):
        yield


@pytest.fixture
def session_manager():
    """Create a StreamlitSessionManager instance for testing.

    Returns:
        StreamlitSessionManager: A fresh session manager instance that will
                               use the mocked session state from other fixtures
    """
    return StreamlitSessionManager()


# ==== TASK SCHEMA AND ID FIXTURES ====


@pytest.fixture
def sample_task_schemas():
    """Sample task schemas for testing annotation workflows.

    Returns:
        dict: A dictionary mapping task names to their allowed outcomes.
              Contains both structured tasks (with predefined options)
              for testing radio button interactions.
    """
    return {
        "task1": ["yes", "no", "maybe"],
        "task2": ["good", "bad", "neutral"],
    }


@pytest.fixture
def sample_expected_ids():
    """Sample expected IDs for testing annotation workflows.

    Returns:
        list: A list of sample IDs that represent the expected data
              samples to be annotated in the test scenarios.
    """
    return ["id1", "id2", "id3"]


# ==== HUMAN ANNOTATION RESULTS FIXTURES ====


@pytest.fixture
def base_human_results_builder() -> HumanAnnotationResultsBuilder:
    """Provide a basic HumanAnnotationResultsBuilder instance for testing.

    Returns:
        HumanAnnotationResultsBuilder: A builder instance ready to create
                                    annotation result rows for testing.
    """
    return HumanAnnotationResultsBuilder(
        run_id="run_001",
        annotator_id="annotator_1",
        task_schemas={"task1": ["yes", "no"], "task2": ["good", "bad"]},
        expected_ids=["id1", "id2"],
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
        is_sampled_run=False,
    )


@pytest.fixture
def mock_results_builder():
    """Mock HumanAnnotationResultsBuilder for testing.

    Returns:
        Mock: A mock results builder with basic attributes set up
              for testing annotation storage and completion workflows.
    """
    mock_builder = Mock(spec=HumanAnnotationResultsBuilder)
    mock_builder.completed_count = 0
    mock_builder._results = {}
    return mock_builder
