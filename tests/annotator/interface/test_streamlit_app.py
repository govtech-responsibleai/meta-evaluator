"""Tests for the Streamlit app using AppTest.

This module provides comprehensive unit tests for the StreamlitAnnotator class and its
integration with Streamlit's UI components. It uses Streamlit's AppTest framework to
simulate user interactions and verify the behavior of the annotation interface.

Test Coverage:
- StreamlitAnnotator initialization and directory creation
- UI component rendering (name input, radio buttons, text areas, navigation)
- User interaction flows (name entry, annotation input, navigation)
- Annotation logic and data handling
- Results saving and export functionality
- Error handling and validation
- Exception handling for robustness

The tests use mocked dependencies to isolate the StreamlitAnnotator behavior and
ensure consistent, reliable testing without external dependencies.
"""

import os
import pytest
import polars as pl
from unittest.mock import Mock, patch
from datetime import datetime
from streamlit.testing.v1 import AppTest

from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.annotator.interface import StreamlitAnnotator
from meta_evaluator.annotator.exceptions import (
    AnnotationValidationError,
    NameValidationError,
    SaveError,
)


# -------------------------
# Fixtures for Real Objects
# -------------------------


@pytest.fixture
def sample_dataframe():
    """Create a sample Polars DataFrame for testing annotation workflows.

    Returns:
        pl.DataFrame: A DataFrame with sample question-answer pairs containing:
            - id: Unique identifiers (1, 2, 3)
            - question: Sample questions for annotation
            - answer: Corresponding answers to the questions
    """
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
    return mock_dataframe


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
                * sentiment: Radio button task with predefined options
                * quality: Radio button task with quality levels
                * comments: Free-form text task (None schema)
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
    return mock_task


@pytest.fixture
def temp_annotations_dir(tmp_path):
    """Create a temporary directory for storing annotation results.

    Args:
        tmp_path: pytest's temporary path fixture

    Returns:
        str: Path to a temporary annotations directory for test isolation
    """
    return str(tmp_path / "annotations")


def create_test_app():
    """Create a Streamlit test application with mocked dependencies.

    This function creates a complete Streamlit app for testing the StreamlitAnnotator
    interface. It mocks all external dependencies and provides a consistent test
    environment for UI component testing.

    Returns:
        function: A Streamlit app function that can be used with AppTest.from_function()
                 to create isolated test instances of the annotation interface.

    Mocked Components:
        - EvalData with sample question/answer pairs
        - EvalTask with mixed annotation task types
        - StreamlitSessionManager with default test state
        - File system operations for annotations directory
    """

    def app():
        import polars as pl
        from unittest.mock import Mock, patch
        from meta_evaluator.annotator.interface import StreamlitAnnotator
        from meta_evaluator.data import EvalData
        from meta_evaluator.eval_task import EvalTask

        # Create mock data
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

        mock_eval_data = Mock(spec=EvalData)
        mock_eval_data.name = "test_dataset"
        mock_eval_data.id_column = "id"
        mock_eval_data.data = mock_dataframe

        mock_eval_task = Mock(spec=EvalTask)
        mock_eval_task.task_schemas = {
            "sentiment": ["positive", "negative", "neutral"],
            "quality": ["high", "medium", "low"],
            "comments": None,  # Free form text
        }
        mock_eval_task.prompt_columns = ["question"]
        mock_eval_task.response_columns = ["answer"]
        mock_eval_task.answering_method = "structured"
        mock_eval_task.annotation_prompt = "Please evaluate the following response:"

        # Create temp directory
        temp_annotations_dir = "/tmp/test_annotations"

        # Create mock session manager
        mock_session_manager = Mock()
        mock_session_manager.current_row = 0
        mock_session_manager.annotated_count = 0
        mock_session_manager.has_user_session = False
        mock_session_manager.run_id = "test_run_123"
        mock_session_manager.annotator_id = "test_annotator_456"
        mock_session_manager.can_go_previous = Mock(return_value=True)
        mock_session_manager.can_go_next = Mock(return_value=True)
        mock_session_manager.get_input_key = Mock(side_effect=lambda x: f"key_{x}")
        mock_session_manager.get_previous_outcome = Mock(return_value=None)
        mock_session_manager.get_input_value = Mock(return_value="positive")
        mock_session_manager.initialize_user_session = Mock()
        mock_session_manager.create_success_row = Mock()
        mock_session_manager.complete_session = Mock()
        mock_session_manager.next_row = Mock()
        mock_session_manager.previous_row = Mock()

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )
            annotator.build_streamlit_app()

    return app


def create_complete_test_app():
    """Create a Streamlit test application with all annotations marked as complete.

    This function creates a test app similar to create_test_app() but with the
    session manager configured to indicate that all annotations have been completed.
    This is used to test the export functionality and completion UI states.

    Returns:
        function: A Streamlit app function configured with completed annotation state.
                 Used to test export button visibility and completion workflows.

    Key Differences from create_test_app():
        - annotated_count set to 3 (matching total samples)
        - has_user_session set to True
        - Enables testing of export button and completion states
    """

    def app():
        import polars as pl
        from unittest.mock import Mock, patch
        from meta_evaluator.annotator.interface import StreamlitAnnotator
        from meta_evaluator.data import EvalData
        from meta_evaluator.eval_task import EvalTask

        # Create mock data
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

        mock_eval_data = Mock(spec=EvalData)
        mock_eval_data.name = "test_dataset"
        mock_eval_data.id_column = "id"
        mock_eval_data.data = mock_dataframe

        mock_eval_task = Mock(spec=EvalTask)
        mock_eval_task.task_schemas = {
            "sentiment": ["positive", "negative", "neutral"],
            "quality": ["high", "medium", "low"],
            "comments": None,  # Free form text
        }
        mock_eval_task.prompt_columns = ["question"]
        mock_eval_task.response_columns = ["answer"]
        mock_eval_task.answering_method = "structured"
        mock_eval_task.annotation_prompt = "Please evaluate the following response:"

        # Create temp directory
        temp_annotations_dir = "/tmp/test_annotations"

        # Create mock session manager with all annotations complete
        mock_session_manager = Mock()
        mock_session_manager.current_row = 0
        mock_session_manager.annotated_count = 3  # All 3 samples annotated
        mock_session_manager.has_user_session = True
        mock_session_manager.run_id = "test_run_123"
        mock_session_manager.annotator_id = "test_annotator_456"
        mock_session_manager.can_go_previous = Mock(return_value=True)
        mock_session_manager.can_go_next = Mock(return_value=True)
        mock_session_manager.get_input_key = Mock(side_effect=lambda x: f"key_{x}")
        mock_session_manager.get_previous_outcome = Mock(return_value=None)
        mock_session_manager.get_input_value = Mock(return_value="positive")
        mock_session_manager.initialize_user_session = Mock()
        mock_session_manager.create_success_row = Mock()
        mock_session_manager.complete_session = Mock()
        mock_session_manager.next_row = Mock()
        mock_session_manager.previous_row = Mock()

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )
            annotator.build_streamlit_app()

    return app


# -------------------------
# Annotator Initialization
# -------------------------


class TestStreamlitAnnotatorInitialization:
    """Test cases for StreamlitAnnotator initialization and setup.

    This test class verifies that the StreamlitAnnotator properly initializes
    with the required dependencies and handles directory creation and error
    scenarios during initialization.
    """

    def test_init_creates_annotations_directory(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test that StreamlitAnnotator initialization creates the annotations directory.

        Verifies that when a StreamlitAnnotator is instantiated with a non-existent
        annotations directory path, the directory is automatically created during
        initialization and the annotator stores the correct directory path.
        """
        assert not os.path.exists(temp_annotations_dir)

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager"
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )

        assert os.path.exists(temp_annotations_dir)
        assert annotator.annotations_dir == temp_annotations_dir

    def test_init_with_invalid_directory_raises_save_error(
        self, mock_eval_data, mock_eval_task
    ):
        """Test that initialization raises SaveError when directory creation fails.

        Verifies that when StreamlitAnnotator is given an invalid directory path
        that cannot be created (e.g., due to permissions or invalid path structure),
        it raises a SaveError with appropriate error information.
        """
        invalid_dir = "/invalid/path/that/cannot/be/created"

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager"
        ):
            with pytest.raises(SaveError):
                StreamlitAnnotator(
                    eval_data=mock_eval_data,
                    eval_task=mock_eval_task,
                    annotations_dir=invalid_dir,
                )


# -------------------------
# App UI Components
# -------------------------


class TestStreamlitAppUI:
    """Test cases for Streamlit app UI components and user interactions.

    This test class uses Streamlit's AppTest framework to verify that UI components
    are rendered correctly, user interactions work as expected, and the annotation
    interface behaves properly under various scenarios.

    Testing Scope:
        - Initial UI state and component visibility
        - User input handling (name entry, annotation selection)
        - Navigation between samples
        - Dynamic UI updates based on annotation state
        - Error handling and exception scenarios
    """

    def test_app_shows_name_input_initially(self):
        """Test that the app displays the name input field on initial load."""
        app = create_test_app()
        at = AppTest.from_function(app)
        at.run()

        # Check that title is displayed
        assert len(at.title) > 0
        assert "Simple Annotation Interface" in at.title[0].value

        # Check that name input is displayed
        assert len(at.text_input) > 0
        assert "Hello," in at.text_input[0].label

    def test_app_shows_info_message_when_no_name(self):
        """Test that app shows info message when no name is entered."""
        app = create_test_app()
        at = AppTest.from_function(app)
        at.run()

        # Check that info message is displayed
        assert len(at.info) > 0
        assert "Please enter your name to start annotating." in at.info[0].value

    def test_app_shows_annotation_interface_with_name(self):
        """Test that app shows annotation interface when name is provided."""
        app = create_test_app()
        at = AppTest.from_function(app)
        at.run()

        # Enter a name
        at.text_input[0].input("Test User")
        at.run()

        # Check that subheader is displayed (Sample 1)
        assert len(at.subheader) > 0
        assert "Sample 1" in at.subheader[0].value

    def test_app_displays_prompt_and_response_columns(self):
        """Test that app displays prompt and response columns."""
        app = create_test_app()
        at = AppTest.from_function(app)
        at.run()

        # Enter a name
        at.text_input[0].input("Test User")
        at.run()

        # Check that prompt and response text are displayed
        text_elements = [elem.value for elem in at.text]
        assert "What is 2+2?" in text_elements
        assert "4" in text_elements

    def test_app_displays_radio_buttons_for_schema_tasks(self):
        """Test that app displays radio buttons for schema-based tasks."""
        app = create_test_app()
        at = AppTest.from_function(app)
        at.run()

        # Enter a name
        at.text_input[0].input("Test User")
        at.run()

        # Check that radio buttons are displayed for sentiment and quality tasks
        radio_labels = [radio.label for radio in at.radio]
        assert "sentiment:" in radio_labels
        assert "quality:" in radio_labels

        # Check radio button options
        sentiment_radio = next(
            radio for radio in at.radio if radio.label == "sentiment:"
        )
        assert sentiment_radio.options == ["positive", "negative", "neutral"]

    def test_app_displays_text_area_for_free_form_tasks(self):
        """Test that app displays text area for free-form tasks."""
        app = create_test_app()
        at = AppTest.from_function(app)
        at.run()

        # Enter a name
        at.text_input[0].input("Test User")
        at.run()

        # Check that text area is displayed for comments task
        text_area_labels = [ta.label for ta in at.text_area]
        assert "comments" in text_area_labels

    def test_app_displays_navigation_buttons(
        self,
    ):
        """Test that app displays navigation buttons."""
        app = create_test_app()
        at = AppTest.from_function(app)
        at.run()

        # Enter a name
        at.text_input[0].input("Test User")
        at.run()

        # Check that navigation buttons are displayed
        button_labels = [button.label for button in at.button]
        assert "Previous" in button_labels
        assert "Next" in button_labels

    def test_navigation_buttons_functionality(self):
        """Test navigation buttons functionality."""

        def test_app():
            import polars as pl
            import streamlit as st
            from unittest.mock import Mock, patch
            from meta_evaluator.annotator.interface import StreamlitAnnotator
            from meta_evaluator.data import EvalData
            from meta_evaluator.eval_task import EvalTask

            # Create mock data
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

            mock_eval_data = Mock(spec=EvalData)
            mock_eval_data.name = "test_dataset"
            mock_eval_data.id_column = "id"
            mock_eval_data.data = mock_dataframe

            mock_eval_task = Mock(spec=EvalTask)
            mock_eval_task.task_schemas = {
                "sentiment": ["positive", "negative", "neutral"],
                "quality": ["high", "medium", "low"],
                "comments": None,
            }
            mock_eval_task.prompt_columns = ["question"]
            mock_eval_task.response_columns = ["answer"]
            mock_eval_task.answering_method = "structured"
            mock_eval_task.annotation_prompt = "Please evaluate the following response:"

            temp_annotations_dir = "/tmp/test_annotations"

            # Create mock session manager and store it in session state
            mock_session_manager = Mock()
            mock_session_manager.current_row = 0
            mock_session_manager.annotated_count = 0
            mock_session_manager.has_user_session = True
            mock_session_manager.run_id = "test_run_123"
            mock_session_manager.annotator_id = "test_annotator_456"
            mock_session_manager.can_go_previous = Mock(return_value=True)
            mock_session_manager.can_go_next = Mock(return_value=True)
            mock_session_manager.get_input_key = Mock(side_effect=lambda x: f"key_{x}")
            mock_session_manager.get_previous_outcome = Mock(return_value=None)
            mock_session_manager.get_input_value = Mock(return_value="positive")
            mock_session_manager.initialize_user_session = Mock()
            mock_session_manager.create_success_row = Mock()
            mock_session_manager.complete_session = Mock()
            mock_session_manager.next_row = Mock()
            mock_session_manager.previous_row = Mock()

            # Store in session state for test access
            st.session_state["test_session_manager"] = mock_session_manager

            with patch(
                "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
                return_value=mock_session_manager,
            ):
                with patch("streamlit.rerun"):  # Mock st.rerun to prevent restart
                    annotator = StreamlitAnnotator(
                        eval_data=mock_eval_data,
                        eval_task=mock_eval_task,
                        annotations_dir=temp_annotations_dir,
                    )
                    annotator.build_streamlit_app()

        at = AppTest.from_function(test_app)
        at.run()

        # Enter a name
        at.text_input[0].input("Test User")
        at.run()

        # Click next button
        next_button = next(button for button in at.button if button.label == "Next")
        next_button.click()
        at.run()

        # Get the session manager from session state and verify next_row was called
        session_manager = at.session_state["test_session_manager"]
        session_manager.next_row.assert_called_once()

    def test_export_button_not_shown_when_incomplete(self):
        """Test that export button is not shown when annotations are incomplete."""
        app = create_test_app()
        at = AppTest.from_function(app)
        at.run()

        # Enter a name
        at.text_input[0].input("Test User")
        at.run()

        # Check that export button is not displayed
        button_labels = [button.label for button in at.button]
        assert "Export Annotations" not in button_labels

    def test_export_button_shown_when_complete(self):
        """Test that export button is shown when all annotations are complete."""
        app = create_complete_test_app()
        at = AppTest.from_function(app)
        at.run()

        # Enter a name
        at.text_input[0].input("Test User")
        at.run()

        # Check that export button is displayed
        button_labels = [button.label for button in at.button]
        assert "Export Annotations" in button_labels

    def test_export_button_error_when_no_name(
        self,
    ):
        """Test that info message is shown when no name is provided."""
        app = create_test_app()
        at = AppTest.from_function(app)
        at.run()

        # Don't enter a name, just run
        at.run()

        # Check that info message is displayed
        assert len(at.info) > 0
        assert "Please enter your name to start annotating." in at.info[0].value

    def test_radio_button_selection_updates_annotation(self):
        """Test that radio button selection updates annotation."""
        app = create_test_app()
        at = AppTest.from_function(app)
        at.run()

        # Enter a name
        at.text_input[0].input("Test User")
        at.run()

        # Select a radio button option
        sentiment_radio = next(
            radio for radio in at.radio if radio.label == "sentiment:"
        )
        sentiment_radio.set_value("positive")
        at.run()

        # Verify the input was processed (mock_session_manager.get_input_value would be called)
        assert sentiment_radio.value == "positive"

    def test_text_area_input_updates_annotation(self):
        """Test that text area input updates annotation."""
        app = create_test_app()
        at = AppTest.from_function(app)
        at.run()

        # Enter a name
        at.text_input[0].input("Test User")
        at.run()

        # Enter text in the comments text area
        comments_text_area = next(ta for ta in at.text_area if ta.label == "comments")
        comments_text_area.input("This is a test comment")
        at.run()

        # Verify the input was processed
        assert comments_text_area.value == "This is a test comment"

    def test_app_handles_exceptions(self):
        """Test that app handles exceptions."""

        # Create app that will raise exception
        def failing_app():
            import polars as pl
            from unittest.mock import Mock, patch
            from meta_evaluator.annotator.interface import StreamlitAnnotator
            from meta_evaluator.data import EvalData
            from meta_evaluator.eval_task import EvalTask

            # Create mock data
            sample_dataframe = pl.DataFrame(
                {
                    "id": ["sample_1", "sample_2", "sample_3"],
                    "prompt": ["What is AI?", "Explain ML", "Define DL"],
                    "response": ["AI is...", "ML is...", "DL is..."],
                    "extra_col": ["extra1", "extra2", "extra3"],
                }
            )

            mock_eval_data = Mock(spec=EvalData)
            mock_eval_data.data = sample_dataframe
            mock_eval_data.id_column = "id"

            mock_eval_task = Mock(spec=EvalTask)
            mock_eval_task.task_schemas = {
                "sentiment": ["positive", "negative", "neutral"],
                "quality": ["high", "medium", "low"],
                "comments": None,
            }
            mock_eval_task.prompt_columns = ["prompt"]
            mock_eval_task.response_columns = ["response"]
            mock_eval_task.annotation_prompt = "Please evaluate the following response:"

            temp_annotations_dir = "/tmp/test_annotations"

            mock_session_manager = Mock()
            mock_session_manager.has_user_session = True
            mock_session_manager.current_row = 0

            with patch(
                "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
                return_value=mock_session_manager,
            ):
                annotator = StreamlitAnnotator(
                    eval_data=mock_eval_data,
                    eval_task=mock_eval_task,
                    annotations_dir=temp_annotations_dir,
                )
                # Mock df.row to raise exception
                annotator.df.row = Mock(side_effect=Exception("Test error"))
                annotator.build_streamlit_app()

        at = AppTest.from_function(failing_app)
        at.run()

        # Enter a name to trigger the exception
        at.text_input[0].input("Test User")
        at.run()

        # Check that error message is displayed
        assert len(at.error) > 0
        assert "An error occurred while building the app:" in at.error[0].value


# -------------------------
# Annotation Logic
# -------------------------


class TestAnnotationLogic:
    """Test cases for annotation handling and processing logic.

    This test class verifies the core annotation logic of the StreamlitAnnotator,
    including how annotations are processed, validated, and stored. It focuses on
    the business logic rather than UI interactions.

    Testing Scope:
        - Annotation data processing and validation
        - Integration with session manager for annotation storage
        - Previous annotation state handling and restoration
        - Error handling for invalid annotation data
        - Success and failure scenarios in annotation workflows
    """

    def test_handle_annotation_with_complete_data(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test handle_annotation with complete annotation data."""
        mock_session_manager = Mock()
        mock_session_manager.get_input_key = Mock(side_effect=lambda x: f"key_{x}")
        mock_session_manager.get_previous_outcome = Mock(return_value=None)
        mock_session_manager.get_input_value = Mock(return_value="positive")
        mock_session_manager.create_success_row = Mock()
        mock_session_manager.current_row = 0

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )

            current_row = ("sample_1", "What is AI?", "AI is...", "extra1")

            with patch.object(annotator, "display_radio_buttons"):
                with patch.object(annotator, "display_free_form_text"):
                    annotation = annotator.handle_annotation(current_row)

            # Should create success row when all tasks are complete
            mock_session_manager.create_success_row.assert_called_once()
            assert "sentiment" in annotation
            assert "quality" in annotation
            assert "comments" in annotation

    def test_handle_annotation_with_previous_outcome(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test handle_annotation with previous outcome."""
        mock_session_manager = Mock()
        mock_session_manager.get_input_key = Mock(side_effect=lambda x: f"key_{x}")
        mock_session_manager.get_previous_outcome = Mock(return_value="positive")
        mock_session_manager.get_input_value = Mock(return_value="positive")
        mock_session_manager.create_success_row = Mock()
        mock_session_manager.current_row = 0

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )

            current_row = ("sample_1", "What is AI?", "AI is...", "extra1")

            with patch.object(annotator, "display_radio_buttons") as mock_radio:
                with patch.object(annotator, "display_free_form_text"):
                    annotator.handle_annotation(current_row)

            # Should call display_radio_buttons with selected_index=0 for "positive"
            mock_radio.assert_any_call(
                label="sentiment",
                outcomes=["positive", "negative", "neutral"],
                key="key_sentiment",
                selected_index=0,
            )

    def test_handle_annotation_raises_validation_error(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test that handle_annotation raises AnnotationValidationError on exception."""
        mock_session_manager = Mock()
        mock_session_manager.get_input_key = Mock(side_effect=Exception("Test error"))

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )

            current_row = ("sample_1", "What is AI?", "AI is...", "extra1")

            with pytest.raises(AnnotationValidationError):
                annotator.handle_annotation(current_row)


# -------------------------
# Results Saving
# -------------------------


class TestSaveResults:
    """Test cases for results saving and export functionality.

    This test class verifies the results saving and export capabilities of the
    StreamlitAnnotator, including successful saves, error handling, and validation
    of required prerequisites for saving.

    Testing Scope:
        - Successful results saving and export workflows
        - Validation of user session and annotator name requirements
        - Error handling for invalid save states
        - Integration with HumanAnnotationResults for data persistence
        - File system operations for results storage
    """

    def test_save_results_success(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test successful save results."""
        mock_session_manager = Mock()
        mock_results = Mock()
        mock_results.timestamp_local = datetime.now()
        mock_results.task_schemas = {"sentiment": ["positive", "negative"]}
        mock_results.get_task_success_rate = Mock(return_value=0.8)
        mock_results.save_state = Mock()

        mock_session_manager.has_user_session = True
        mock_session_manager.complete_session = Mock(return_value=mock_results)
        mock_session_manager.run_id = "test_run_123"
        mock_session_manager.annotator_id = "test_annotator_456"

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )
            annotator.annotator_name = "Test User"

            with patch("streamlit.success") as mock_success:
                with patch("streamlit.write"):
                    with patch("streamlit.progress"):
                        with patch("streamlit.markdown"):
                            with patch.object(annotator, "display_subheader"):
                                annotator.save_results()

            mock_results.save_state.assert_called_once()
            mock_success.assert_called_once()

    def test_save_results_no_user_session(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test save results with no user session."""
        mock_session_manager = Mock()
        mock_session_manager.has_user_session = False

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )

            with pytest.raises(SaveError):
                annotator.save_results()

    def test_save_results_no_annotator_name(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test save results with no annotator name."""
        mock_session_manager = Mock()
        mock_session_manager.has_user_session = True

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )
            annotator.annotator_name = None

            with pytest.raises(NameValidationError):
                annotator.save_results()
