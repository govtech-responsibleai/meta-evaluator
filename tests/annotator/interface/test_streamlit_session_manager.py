"""Tests for StreamlitSessionManager.

This module provides comprehensive unit tests for the StreamlitSessionManager class,
which manages Streamlit session state for the human annotation interface. The session
manager handles user session initialization, navigation between samples, annotation
storage, and results management.

Test Coverage:
- Initialization (session setup, user session creation)
- Navigation (get_previous_outcome, row navigation)
- Results storage (create_success_row, complete_session)
- Utility functions (generate_run_id, generate_annotator_id)
"""

import pytest
import re
from datetime import datetime
from unittest.mock import Mock, patch

from meta_evaluator.annotator.interface.streamlit_session_manager import (
    StreamlitSessionManager,
    generate_run_id,
    generate_annotator_id,
)
from meta_evaluator.results import (
    HumanAnnotationResultsBuilder,
    HumanAnnotationResults,
)
from meta_evaluator.annotator.exceptions import AnnotatorInitializationError


# -------------------------
# Fixtures and Test Helpers
# -------------------------


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


@pytest.fixture(autouse=True)
def mock_streamlit_session_state():
    """Automatically mock streamlit.session_state for all tests in this module.

    This fixture automatically patches streamlit.session_state with a MockSessionState
    instance for every test, ensuring consistent test environment without requiring
    explicit fixture injection.
    """
    with patch("streamlit.session_state", MockSessionState()):
        yield


# Note: session_manager, sample_task_schemas, sample_expected_ids, and
# mock_results_builder fixtures are provided by the shared conftest.py


# -------------------------
# Utility Function Tests
# -------------------------


def test_generate_run_id_format():
    """Test that generate_run_id creates correctly formatted run IDs."""
    run_id = generate_run_id()

    # Check format: annotation_run_YYYYMMDD_HHMMSS_XXXXXXXX
    pattern = r"^annotation_run_\d{8}_\d{6}_[a-f0-9]{8}$"
    assert re.match(pattern, run_id), f"Run ID {run_id} doesn't match expected format"

    # Check that consecutive calls generate different IDs
    run_id2 = generate_run_id()
    assert run_id != run_id2


@pytest.mark.parametrize(
    "input_name,expected",
    [
        ("John Doe", "john_doe"),
        ("John-Doe_123!", "john-doe_123"),
        ("John___Doe", "john_doe"),
        ("_John", "john"),
        ("123John", "user_123john"),
        ("123", "user_123"),
    ],
)
def test_generate_annotator_id(input_name, expected):
    """Test annotator ID generation with various inputs."""
    result = generate_annotator_id(input_name)
    assert result == expected


def test_generate_annotator_id_empty_input():
    """Test annotator ID generation with empty input."""
    with pytest.raises(ValueError, match="Annotator name cannot be empty"):
        generate_annotator_id("")

    with pytest.raises(ValueError, match="Annotator name cannot be empty"):
        generate_annotator_id("   ")


# -------------------------
# Initialization Tests
# -------------------------


def test_session_manager_initialization():
    """Test that StreamlitSessionManager initializes with basic state."""
    manager = StreamlitSessionManager()

    # Check that the session manager properly initializes current_row
    assert manager.current_row == 0


def test_initialize_user_session_sets_up_properly(
    session_manager, sample_task_schemas, sample_expected_ids
):
    """Test that initialize_user_session sets up session state correctly."""
    annotator_name = "Test Annotator"

    session_manager.initialize_user_session(
        annotator_name=annotator_name,
        task_schemas=sample_task_schemas,
        expected_ids=sample_expected_ids,
    )

    # Check that the session manager properly sets up the user session
    assert session_manager.has_user_session
    assert session_manager.annotator_id == "test_annotator"

    # Check that run_id has correct format
    run_id = session_manager.run_id
    pattern = r"^annotation_run_\d{8}_\d{6}_[a-f0-9]{8}$"
    assert re.match(pattern, run_id)

    # Check that results_builder is properly initialized
    assert isinstance(session_manager.results_builder, HumanAnnotationResultsBuilder)


def test_has_user_session_property(session_manager):
    """Test has_user_session property."""
    import streamlit as st

    # Initially no user session
    assert not session_manager.has_user_session

    # Add results_builder to session state
    st.session_state["results_builder"] = Mock()
    assert session_manager.has_user_session


def test_results_builder_property_with_session(
    session_manager, sample_task_schemas, sample_expected_ids
):
    """Test results_builder property when session exists."""
    # Initialize user session properly (this is how results_builder gets created)
    session_manager.initialize_user_session(
        annotator_name="Test User",
        task_schemas=sample_task_schemas,
        expected_ids=sample_expected_ids,
    )

    # Now test that we can access the results_builder
    results_builder = session_manager.results_builder
    assert isinstance(results_builder, HumanAnnotationResultsBuilder)


def test_results_builder_property_without_session(session_manager):
    """Test results_builder property when no session exists."""
    with pytest.raises(AnnotatorInitializationError, match="User Session"):
        session_manager.results_builder


def test_annotated_count_property(session_manager):
    """Test annotated_count property."""
    import streamlit as st

    # No user session
    assert session_manager.annotated_count == 0

    # With user session
    mock_builder = Mock(spec=HumanAnnotationResultsBuilder)
    mock_builder.completed_count = 5
    st.session_state["results_builder"] = mock_builder

    assert session_manager.annotated_count == 5


def test_get_input_key(session_manager):
    """Test get_input_key generates correct keys."""
    session_manager.current_row = 2
    key = session_manager.get_input_key("task1")
    assert key == "outcome_task1_2"


def test_get_input_value(session_manager):
    """Test get_input_value retrieves values from session state."""
    import streamlit as st

    st.session_state["test_key"] = "test_value"

    result = session_manager.get_input_value("test_key")
    assert result == "test_value"

    result = session_manager.get_input_value("nonexistent_key")
    assert result is None


# -------------------------
# Current Row Tests
# -------------------------


def test_current_row_property(session_manager):
    """Test current_row property getter and setter."""
    # Initial value
    assert session_manager.current_row == 0

    # Set new value
    session_manager.current_row = 5
    assert session_manager.current_row == 5


def test_get_previous_outcome_with_existing_result(
    session_manager, sample_task_schemas, sample_expected_ids
):
    """Test get_previous_outcome returns correct previous outcome."""
    # Initialize user session properly
    session_manager.initialize_user_session(
        annotator_name="Test User",
        task_schemas=sample_task_schemas,
        expected_ids=sample_expected_ids,
    )

    # Create a real annotation result first
    session_manager.create_success_row(
        sample_example_id="sample_1",
        original_id="id1",
        outcomes={"task1": "yes", "task2": "good"},
    )

    # Test retrieval of previous outcomes
    result = session_manager.get_previous_outcome("task1", "id1")
    assert result == "yes"

    result = session_manager.get_previous_outcome("task2", "id1")
    assert result == "good"


def test_get_previous_outcome_no_existing_result(
    session_manager, sample_task_schemas, sample_expected_ids
):
    """Test get_previous_outcome when no existing result."""
    # Initialize user session properly
    session_manager.initialize_user_session(
        annotator_name="Test User",
        task_schemas=sample_task_schemas,
        expected_ids=sample_expected_ids,
    )

    # Don't create any results - test with empty results
    result = session_manager.get_previous_outcome("task1", "id1")
    assert result is None


def test_get_previous_outcome_missing_task(
    session_manager, sample_task_schemas, sample_expected_ids
):
    """Test get_previous_outcome when task attribute doesn't exist."""
    # Initialize user session properly
    session_manager.initialize_user_session(
        annotator_name="Test User",
        task_schemas=sample_task_schemas,
        expected_ids=sample_expected_ids,
    )

    # Create a complete result first (required by validation)
    session_manager.create_success_row(
        sample_example_id="sample_1",
        original_id="id1",
        outcomes={"task1": "yes", "task2": "good"},
    )

    # Try to get a task that doesn't exist in the schema
    result = session_manager.get_previous_outcome("nonexistent_task", "id1")
    assert result is None


# -------------------------
# Navigation Tests
# -------------------------


def test_navigation_methods(session_manager):
    """Test session navigation methods."""
    # Initial state
    assert session_manager.current_row == 0

    # Test next_row
    session_manager.next_row()
    assert session_manager.current_row == 1

    # Test previous_row
    session_manager.previous_row()
    assert session_manager.current_row == 0

    # Test previous_row at boundary
    session_manager.previous_row()
    assert session_manager.current_row == 0  # Should not go below 0

    # Test can_go_previous
    assert not session_manager.can_go_previous(5)
    session_manager.current_row = 1
    assert session_manager.can_go_previous(5)

    # Test can_go_next
    session_manager.current_row = 0
    assert session_manager.can_go_next(5)
    session_manager.current_row = 4
    assert not session_manager.can_go_next(5)


# -------------------------
# Results Storage Tests
# -------------------------


def test_create_success_row_no_session(session_manager):
    """Test create_success_row raises error when no user session."""
    with pytest.raises(AnnotatorInitializationError, match="User Session"):
        session_manager.create_success_row(
            sample_example_id="sample_1", original_id="id1", outcomes={"task1": "yes"}
        )


def test_create_success_row_calls_builder(
    session_manager, sample_task_schemas, sample_expected_ids
):
    """Test create_success_row properly creates and stores results."""
    # Initialize user session properly
    session_manager.initialize_user_session(
        annotator_name="Test User",
        task_schemas=sample_task_schemas,
        expected_ids=sample_expected_ids,
    )

    outcomes = {"task1": "yes", "task2": "good"}
    timestamp = datetime.now()

    # Verify no results initially
    assert session_manager.annotated_count == 0

    # Create success row
    session_manager.create_success_row(
        sample_example_id="sample_1",
        original_id="id1",
        outcomes=outcomes,
        annotation_timestamp=timestamp,
    )

    # Verify the result was created and stored
    assert session_manager.annotated_count == 1

    # Verify we can retrieve the stored outcome
    result = session_manager.get_previous_outcome("task1", "id1")
    assert result == "yes"

    result = session_manager.get_previous_outcome("task2", "id1")
    assert result == "good"


def test_create_success_row_removes_existing_result(
    session_manager, sample_task_schemas, sample_expected_ids
):
    """Test create_success_row replaces existing result when called again."""
    # Initialize user session properly
    session_manager.initialize_user_session(
        annotator_name="Test User",
        task_schemas=sample_task_schemas,
        expected_ids=sample_expected_ids,
    )

    # Create initial result
    session_manager.create_success_row(
        sample_example_id="sample_1",
        original_id="id1",
        outcomes={"task1": "yes", "task2": "good"},
    )

    # Verify initial result
    assert session_manager.annotated_count == 1
    assert session_manager.get_previous_outcome("task1", "id1") == "yes"
    assert session_manager.get_previous_outcome("task2", "id1") == "good"

    # Create new result for same ID (should replace the existing one)
    session_manager.create_success_row(
        sample_example_id="sample_2",
        original_id="id1",
        outcomes={"task1": "no", "task2": "bad"},
    )

    # Verify the result was replaced, not duplicated
    assert session_manager.annotated_count == 1  # Still only 1 result
    assert session_manager.get_previous_outcome("task1", "id1") == "no"
    assert session_manager.get_previous_outcome("task2", "id1") == "bad"


def test_complete_session_no_session(session_manager):
    """Test complete_session raises error when no user session."""
    with pytest.raises(AnnotatorInitializationError, match="User Session"):
        session_manager.complete_session()


def test_complete_session_calls_builder_complete(session_manager):
    """Test complete_session calls results_builder.complete()."""
    import streamlit as st

    mock_builder = Mock(spec=HumanAnnotationResultsBuilder)
    mock_results = Mock(spec=HumanAnnotationResults)
    mock_builder.complete.return_value = mock_results
    st.session_state["results_builder"] = mock_builder

    result = session_manager.complete_session()

    # Check that builder.complete was called
    mock_builder.complete.assert_called_once()
    assert result is mock_results


def test_reset_session(session_manager):
    """Test reset_session clears all session state."""
    import streamlit as st

    # Set up session state
    st.session_state["results_builder"] = Mock()
    st.session_state["current_run_id"] = "test_run"
    st.session_state["current_annotator_id"] = "test_annotator"
    st.session_state["current_row"] = 5

    # Reset session
    session_manager.reset_session()

    # Check that all relevant keys are removed
    assert "results_builder" not in st.session_state
    assert "current_run_id" not in st.session_state
    assert "current_annotator_id" not in st.session_state
    assert "current_row" not in st.session_state


def test_reset_session_partial_state(session_manager):
    """Test reset_session works when only some state exists."""
    import streamlit as st

    # Set up partial session state
    st.session_state["results_builder"] = Mock()
    st.session_state["other_key"] = "should_remain"

    # Reset session
    session_manager.reset_session()

    # Check that only relevant keys are removed
    assert "results_builder" not in st.session_state
    assert "other_key" in st.session_state


# -------------------------
# Integration Tests
# -------------------------


def test_full_session_workflow(
    session_manager, sample_task_schemas, sample_expected_ids
):
    """Test a complete session workflow from initialization to completion."""
    # Initialize session
    session_manager.initialize_user_session(
        annotator_name="Test User",
        task_schemas=sample_task_schemas,
        expected_ids=sample_expected_ids,
    )

    # Check session is initialized
    assert session_manager.has_user_session
    assert session_manager.annotated_count == 0

    # Navigate and create annotations
    session_manager.current_row = 0

    # Mock the results builder to track calls
    with (
        patch.object(
            session_manager.results_builder, "create_success_row"
        ) as mock_create,
        patch.object(session_manager.results_builder, "complete") as mock_complete,
    ):
        # Create first annotation
        session_manager.create_success_row(
            sample_example_id="sample_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
        )

        # Move to next row and create second annotation
        session_manager.next_row()
        session_manager.create_success_row(
            sample_example_id="sample_2",
            original_id="id2",
            outcomes={"task1": "no", "task2": "bad"},
        )

        # Complete session
        session_manager.complete_session()

        # Check that methods were called correctly
        assert mock_create.call_count == 2
        mock_complete.assert_called_once()
