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
from datetime import datetime
from unittest.mock import Mock, patch

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from meta_evaluator.annotator.exceptions import (
    AnnotationValidationError,
    NameValidationError,
    SaveError,
)
from meta_evaluator.annotator.interface import StreamlitAnnotator

# -------------------------
# Fixtures for Real Objects
# -------------------------


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
        from unittest.mock import Mock, patch

        import polars as pl

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
        mock_eval_task.required_tasks = ["sentiment", "quality"]
        mock_eval_task.prompt_columns = ["question"]
        mock_eval_task.response_columns = ["answer"]
        mock_eval_task.answering_method = "structured"
        mock_eval_task.annotation_prompt = "Please evaluate the following response:"
        mock_eval_task.get_required_tasks.return_value = ["sentiment", "quality"]

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

        # Mock initialize_user_session to set has_user_session = True after call
        def mock_initialize_user_session(*args, **kwargs):
            mock_session_manager.has_user_session = True

        mock_session_manager.initialize_user_session = mock_initialize_user_session
        mock_session_manager.create_success_row = Mock()
        mock_session_manager.complete_session = Mock()
        mock_session_manager.next_row = Mock()
        mock_session_manager.previous_row = Mock()
        mock_session_manager.get_incomplete_samples = Mock(return_value={})

        # Add results_builder for auto-save functionality
        mock_results_builder = Mock()
        mock_results_builder._results = {}
        mock_session_manager.results_builder = mock_results_builder

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
                test_environment=True,
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
        from unittest.mock import Mock, patch

        import polars as pl

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
        mock_eval_task.required_tasks = ["sentiment", "quality"]
        mock_eval_task.prompt_columns = ["question"]
        mock_eval_task.response_columns = ["answer"]
        mock_eval_task.answering_method = "structured"
        mock_eval_task.annotation_prompt = "Please evaluate the following response:"
        mock_eval_task.get_required_tasks.return_value = ["sentiment", "quality"]

        # Create temp directory
        temp_annotations_dir = "/tmp/test_annotations"

        # Create mock session manager with all annotations complete
        mock_session_manager = Mock()
        mock_session_manager.current_row = 2  # On last sample (index 2 of 3 samples)
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
        mock_session_manager.get_incomplete_samples = Mock(return_value={})

        # Add results_builder for auto-save functionality
        mock_results_builder = Mock()
        mock_results_builder._results = {}
        mock_session_manager.results_builder = mock_results_builder

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
                test_environment=True,
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

    def test_annotation_prompt_is_stored(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test that StreamlitAnnotator stores the custom annotation_prompt from EvalTask.

        Verifies that when a StreamlitAnnotator is initialized with an EvalTask
        that has a custom annotation_prompt, the annotator correctly stores and
        makes the prompt accessible via the annotation_prompt attribute.
        """
        # Set custom annotation prompt on mock_eval_task
        custom_prompt = "Please analyze the toxicity of the following text."
        mock_eval_task.annotation_prompt = custom_prompt

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager"
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )

        assert annotator.annotation_prompt == custom_prompt

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

        # Check that markdown header is displayed (Sample 1) - now uses display_subheader with markdown
        markdown_texts = [elem.value for elem in at.markdown]
        assert any("Sample 1" in text for text in markdown_texts)

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
        assert "sentiment" in radio_labels
        assert "quality" in radio_labels

        # Check radio button options
        sentiment_radio = next(
            radio for radio in at.radio if radio.label == "sentiment"
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

        # Check that navigation buttons are displayed (now using icon buttons)
        assert len(at.button) >= 2  # Should have at least 2 navigation buttons

    def test_navigation_buttons_functionality(self):
        """Test navigation buttons functionality."""

        def test_app():
            from unittest.mock import Mock, patch

            import polars as pl
            import streamlit as st

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
            mock_eval_task.required_tasks = ["sentiment", "quality"]
            mock_eval_task.prompt_columns = ["question"]
            mock_eval_task.response_columns = ["answer"]
            mock_eval_task.answering_method = "structured"
            mock_eval_task.annotation_prompt = "Please evaluate the following response:"
            mock_eval_task.get_required_tasks.return_value = ["sentiment", "quality"]

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

            # Add results_builder for auto-save functionality
            mock_results_builder = Mock()
            mock_results_builder._results = {}
            mock_session_manager.results_builder = mock_results_builder

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

        # Click the second button (should be next button with icon)
        if len(at.button) >= 2:
            next_button = at.button[1]  # Second button should be next
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
            radio for radio in at.radio if radio.label == "sentiment"
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
            from unittest.mock import Mock, patch

            import polars as pl

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
            mock_eval_task.get_required_tasks.return_value = ["sentiment", "quality"]

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
        mock_session_manager.has_user_session = (
            True  # Required for auto-save functionality
        )

        # Add results_builder for auto-save functionality
        mock_results_builder = Mock()
        mock_results_builder._results = {}
        mock_session_manager.results_builder = mock_results_builder

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
                    with patch.object(
                        annotator, "_auto_save_annotation"
                    ) as mock_auto_save:
                        annotation = annotator.handle_annotation(current_row)

            # Should create success row when all tasks are complete
            mock_session_manager.create_success_row.assert_called_once()
            # Should auto-save when annotation is complete
            mock_auto_save.assert_called_once_with("sample_1", annotation)
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

        # Add results_builder for auto-save functionality
        mock_results_builder = Mock()
        mock_results_builder._results = {}
        mock_session_manager.results_builder = mock_results_builder

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
                test_environment=True,
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
                is_required=True,
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


class TestAutoSave:
    """Test class for auto-save functionality in StreamlitAnnotator.

    This class provides comprehensive tests for the auto-save functionality including:
    - Auto-save filename generation and management
    - Loading existing auto-save files on initialization
    - Saving annotation data to auto-save files
    - Error handling for corrupted or missing auto-save files
    - Integration with the existing annotation workflow
    """

    def test_generate_auto_save_filename(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test auto-save filename generation.

        Verifies that the auto-save filename is generated correctly using the same
        pattern as the final save files but with an 'autosave' prefix. The filename
        should include run_id, annotator_id, and annotator_name in the expected format.
        """
        mock_session_manager = Mock()
        mock_session_manager.has_user_session = True
        mock_session_manager.run_id = "test_run_123"
        mock_session_manager.annotator_id = "test_annotator"

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )
            annotator.annotator_name = "test_user"

            filename = annotator._generate_auto_save_filename("test_user")
            expected_filename = os.path.join(
                temp_annotations_dir,
                "autosave_test_run_123_test_annotator_test_user_data.parquet",
            )
            assert filename == expected_filename

    def test_generate_auto_save_filename_without_session_raises_error(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test that generating auto-save filename without session raises error.

        Verifies that attempting to generate an auto-save filename when no user session
        is initialized raises a RuntimeError with an appropriate error message. This
        ensures that the filename generation is only called when the session is properly
        set up with run_id and annotator_id.
        """
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

            with pytest.raises(
                RuntimeError,
                match="Cannot generate auto-save filename without user session",
            ):
                annotator._generate_auto_save_filename("test_user")

    def test_set_auto_save_filename(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test setting auto-save filename.

        Verifies that the _set_auto_save_filename method correctly sets the auto_save_file
        attribute using the generated filename pattern. This ensures that the filename
        is properly initialized for subsequent auto-save operations.
        """
        mock_session_manager = Mock()
        mock_session_manager.has_user_session = True
        mock_session_manager.run_id = "test_run_123"
        mock_session_manager.annotator_id = "test_annotator"

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )
            annotator.annotator_name = "test_user"

            annotator._set_auto_save_filename()

            expected_filename = os.path.join(
                temp_annotations_dir,
                "autosave_test_run_123_test_annotator_test_user_data.parquet",
            )
            assert annotator.auto_save_file == expected_filename

    def test_load_auto_save_results_success(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test successful loading of auto-save results.

        Verifies that the _load_auto_save_results method correctly loads annotation data
        from a valid auto-save file and populates the session manager's results builder.
        This ensures that users can resume their work from previously saved annotations.
        """
        # Create a test auto-save file
        test_data = [
            {
                "sample_example_id": "sample_1",
                "original_id": "id1",
                "run_id": "test_run_123",
                "annotator_id": "test_annotator",
                "status": "success",
                "error_message": None,
                "error_details_json": None,
                "annotation_timestamp": "2024-01-01T12:00:00",
                "sentiment": "positive",
                "quality": "good",
                "comments": "Great response",
            }
        ]

        # Ensure directory exists
        os.makedirs(temp_annotations_dir, exist_ok=True)

        auto_save_file = os.path.join(temp_annotations_dir, "test_autosave.parquet")
        pl.DataFrame(test_data).write_parquet(auto_save_file)

        mock_session_manager = Mock()
        mock_session_manager.has_user_session = True
        mock_results_builder = Mock()
        mock_results_builder._results = {}
        mock_session_manager.results_builder = mock_results_builder

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )
            # Mock task_schemas to match test data
            annotator.task_schemas = {
                "sentiment": ["positive", "negative"],
                "quality": ["good", "bad"],
            }

            result = annotator._load_auto_save_results(auto_save_file)

            assert result is True
            # Verify that create_success_row was called correctly
            mock_session_manager.create_success_row.assert_called_once()
            call_args = mock_session_manager.create_success_row.call_args
            assert call_args[1]["original_id"] == "id1"
            assert call_args[1]["outcomes"]["sentiment"] == "positive"
            assert call_args[1]["outcomes"]["quality"] == "good"

    def test_load_auto_save_results_empty_file(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test loading empty auto-save file.

        Verifies that the _load_auto_save_results method handles empty auto-save files
        gracefully by returning False and not populating any results. This ensures
        robust handling of edge cases where auto-save files exist but contain no data.
        """
        # Ensure directory exists
        os.makedirs(temp_annotations_dir, exist_ok=True)

        auto_save_file = os.path.join(temp_annotations_dir, "empty_autosave.parquet")
        pl.DataFrame().write_parquet(auto_save_file)

        mock_session_manager = Mock()
        mock_session_manager.has_user_session = True
        mock_results_builder = Mock()
        mock_results_builder._results = {}
        mock_session_manager.results_builder = mock_results_builder

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )

            result = annotator._load_auto_save_results(auto_save_file)

            assert result is False
            assert len(mock_results_builder._results) == 0

    def test_load_auto_save_results_invalid_file(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test loading invalid auto-save file.

        Verifies that the _load_auto_save_results method handles corrupted or invalid
        auto-save files gracefully by returning False and not populating any results.
        This ensures robust error handling when auto-save files are corrupted or contain
        invalid JSON data.
        """
        # Ensure directory exists
        os.makedirs(temp_annotations_dir, exist_ok=True)

        auto_save_file = os.path.join(temp_annotations_dir, "invalid_autosave.parquet")
        with open(auto_save_file, "w") as f:
            f.write("invalid parquet content")

        mock_session_manager = Mock()
        mock_session_manager.has_user_session = True
        mock_results_builder = Mock()
        mock_results_builder._results = {}
        mock_session_manager.results_builder = mock_results_builder

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )

            result = annotator._load_auto_save_results(auto_save_file)

            assert result is False
            assert len(mock_results_builder._results) == 0

    def test_auto_save_annotation_saves_data(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test that auto-save annotation saves data correctly.

        Verifies that the _auto_save_annotation method correctly saves annotation data
        to the auto-save file in the same format as the final save. This includes
        creating the file, writing the data in JSON format, and showing appropriate
        success messages to the user.

        Raises:
            ValueError: If auto-save file is None
        """
        mock_session_manager = Mock()
        mock_session_manager.has_user_session = True
        mock_session_manager.run_id = "test_run_123"
        mock_session_manager.annotator_id = "test_annotator"
        mock_session_manager.current_row = 0

        # Mock result row
        mock_result_row = Mock()
        mock_result_row.model_dump.return_value = {
            "sample_example_id": "sample_1",
            "original_id": "id1",
            "sentiment": "positive",
            "quality": "good",
        }

        mock_results_builder = Mock()
        mock_results_builder._results = {"id1": mock_result_row}
        mock_session_manager.results_builder = mock_results_builder

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )
            annotator.annotator_name = "test_user"
            annotator._set_auto_save_filename()

            with patch("streamlit.success") as mock_success:
                annotator._auto_save_annotation(
                    "id1", {"sentiment": "positive", "quality": "good"}
                )

                # Check that file was created
                if annotator.auto_save_file is None:
                    raise ValueError("Auto-save file is None")

                assert os.path.exists(annotator.auto_save_file)

                # Check that success message was shown
                mock_success.assert_called_once()

                # Verify file content
                loaded_data = pl.read_parquet(annotator.auto_save_file)
                assert len(loaded_data) == 1
                assert loaded_data["original_id"][0] == "id1"

    def test_auto_save_annotation_without_session(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test auto-save annotation without user session.

        Verifies that the _auto_save_annotation method handles the case where no user
        session is initialized gracefully by returning early without attempting to save
        or raising errors. This ensures robust behavior when auto-save is called
        inappropriately.
        """
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

            # Should return early without error
            annotator._auto_save_annotation("id1", {"sentiment": "positive"})

            # No file should be created
            assert annotator.auto_save_file is None

    def test_auto_save_annotation_sets_filename_if_none(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test that auto-save sets filename if not already set.

        Verifies that the _auto_save_annotation method automatically sets the auto-save
        filename if it hasn't been set previously. This ensures that the auto-save
        functionality works correctly even when the filename hasn't been explicitly
        initialized, providing a seamless user experience.
        """
        mock_session_manager = Mock()
        mock_session_manager.has_user_session = True
        mock_session_manager.run_id = "test_run_123"
        mock_session_manager.annotator_id = "test_annotator"
        mock_session_manager.current_row = 0

        mock_results_builder = Mock()
        # Add a mock result so auto-save doesn't fail
        mock_result = Mock()
        mock_result.model_dump.return_value = {
            "sample_example_id": "sample_1",
            "original_id": "id1",
            "sentiment": "positive",
            "quality": "good",
            "annotation_timestamp": "2024-01-01T10:00:00",
        }
        mock_results_builder._results = {"id1": mock_result}
        mock_session_manager.results_builder = mock_results_builder

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )
            annotator.annotator_name = "test_user"
            # Don't set auto_save_file - should be set automatically

            with patch("streamlit.success"):
                annotator._auto_save_annotation("id1", {"sentiment": "positive"})

                # Check that filename was set
                expected_filename = os.path.join(
                    temp_annotations_dir,
                    "autosave_test_run_123_test_annotator_test_user_data.parquet",
                )
                assert annotator.auto_save_file == expected_filename

    def test_initialization_loads_existing_auto_save(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test that initialization loads existing auto-save file and resumes at correct position.

        Verifies that when a user session is initialized, the system automatically
        checks for and loads existing auto-save files, and resumes from the first
        incomplete sample. This ensures that users can seamlessly resume their
        annotation work from where they left off, providing a continuous and
        uninterrupted annotation experience.
        """
        # Create a test auto-save file with only first sample completed
        test_data = [
            {
                "sample_example_id": "sample_1",
                "original_id": 1,  # Match the actual ID from sample_dataframe
                "run_id": "test_run_123",
                "annotator_id": "test_annotator",
                "status": "success",
                "error_message": None,
                "error_details_json": None,
                "annotation_timestamp": datetime(2024, 1, 1, 12, 0, 0),
                "sentiment": "positive",
                "quality": "good",
                "comments": "Great response",
            }
        ]

        # Ensure directory exists
        os.makedirs(temp_annotations_dir, exist_ok=True)

        auto_save_file = os.path.join(
            temp_annotations_dir,
            "autosave_test_run_123_test_annotator_test_user_data.parquet",
        )
        pl.DataFrame(test_data).write_parquet(auto_save_file)

        mock_session_manager = Mock()
        mock_session_manager.has_user_session = True  # Set to True for this test
        mock_session_manager.run_id = "test_run_123"
        mock_session_manager.annotator_id = "test_annotator"
        mock_session_manager.current_row = 0  # Initial position

        mock_results_builder = Mock()
        mock_results_builder._results = {}
        mock_session_manager.results_builder = mock_results_builder

        # Mock get_incomplete_samples to return samples 2 and 3 as incomplete
        # (since sample 1 is completed in the autosave)
        mock_session_manager.get_incomplete_samples.return_value = {
            2: {"id": 2, "missing_fields": ["sentiment", "quality"]},
            3: {"id": 3, "missing_fields": ["sentiment", "quality"]},
        }

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )
            annotator.annotator_name = "test_user"
            # Mock task_schemas to match test data
            annotator.task_schemas = {
                "sentiment": ["positive", "negative"],
                "quality": ["good", "bad"],
            }

            # Test the auto-save loading directly
            result = annotator._load_auto_save_results(auto_save_file)

            # Verify that auto-save was loaded successfully
            assert result is True
            # Verify that create_success_row was called correctly
            mock_session_manager.create_success_row.assert_called_once()
            call_args = mock_session_manager.create_success_row.call_args
            assert call_args[1]["original_id"] == 1
            assert call_args[1]["outcomes"]["sentiment"] == "positive"
            assert call_args[1]["outcomes"]["quality"] == "good"

            # Test that resume position is set correctly when there are incomplete samples
            # Simulate what happens in build_streamlit_app after loading autosave
            expected_ids = [1, 2, 3]
            required_tasks = ["sentiment", "quality"]
            incomplete_samples = mock_session_manager.get_incomplete_samples(
                expected_ids, required_tasks
            )

            if incomplete_samples:
                first_incomplete = min(incomplete_samples.keys())
                mock_session_manager.current_row = first_incomplete - 1

            # Verify that current_row is set to 1 (0-based index for sample 2)
            assert mock_session_manager.current_row == 1

    def test_initialization_no_auto_save_found(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test initialization when no auto-save file exists.

        Verifies that when no existing auto-save file is found during initialization,
        the system correctly sets up a new auto-save filename for future use. This
        ensures that the auto-save functionality is properly initialized for new
        annotation sessions.
        """
        mock_session_manager = Mock()
        mock_session_manager.has_user_session = False
        mock_session_manager.run_id = "test_run_123"
        mock_session_manager.annotator_id = "test_annotator"
        mock_session_manager.initialize_user_session = Mock()

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )
            annotator.annotator_name = "test_user"

            # Simulate the initialization logic
            if not mock_session_manager.has_user_session:
                expected_ids = mock_eval_data.data[mock_eval_data.id_column].to_list()
                mock_session_manager.initialize_user_session(
                    annotator_name=annotator.annotator_name,
                    task_schemas=annotator.task_schemas,
                    expected_ids=expected_ids,
                )

                # After initialization, the session should exist
                mock_session_manager.has_user_session = True

                # Check for existing auto-save and load if found
                expected_auto_save_file = annotator._generate_auto_save_filename(
                    annotator.annotator_name
                )
                if os.path.exists(expected_auto_save_file):
                    # This should not happen in this test
                    pass
                else:
                    # Set up new auto-save filename
                    annotator._set_auto_save_filename()

            # Verify that new auto-save filename was set
            expected_filename = os.path.join(
                temp_annotations_dir,
                "autosave_test_run_123_test_annotator_test_user_data.parquet",
            )
            assert annotator.auto_save_file == expected_filename

    def test_save_annotations_data_parquet_format(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test the unified _save_annotations_data method with parquet format."""
        # Create mock session manager with results
        mock_session_manager = Mock()
        mock_session_manager.has_user_session = True

        mock_result = Mock()
        mock_result.model_dump.return_value = {
            "sample_example_id": "sample_1",
            "original_id": "id1",
            "sentiment": "positive",
            "quality": "good",
            "annotation_timestamp": "2024-01-01T10:00:00",
        }
        mock_results_builder = Mock()
        mock_results_builder._results = {"id1": mock_result}
        mock_session_manager.results_builder = mock_results_builder

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )

            # Test saving in parquet format
            test_file = os.path.join(temp_annotations_dir, "test_save.parquet")
            annotator._save_annotations_data(test_file, data_format="parquet")

            # Verify file was created and contains correct data
            assert os.path.exists(test_file)
            loaded_data = pl.read_parquet(test_file)
            assert len(loaded_data) == 1
            assert loaded_data["original_id"][0] == "id1"

    def test_find_existing_auto_save_file(
        self, mock_eval_data, mock_eval_task, temp_annotations_dir
    ):
        """Test finding existing auto-save files for an annotator."""
        # Create mock session manager
        mock_session_manager = Mock()
        mock_session_manager.has_user_session = True
        mock_session_manager.run_id = "test_run_123"
        mock_session_manager.annotator_id = "test_annotator"

        with patch(
            "meta_evaluator.annotator.interface.streamlit_app.StreamlitSessionManager",
            return_value=mock_session_manager,
        ):
            annotator = StreamlitAnnotator(
                eval_data=mock_eval_data,
                eval_task=mock_eval_task,
                annotations_dir=temp_annotations_dir,
            )

            # Create test auto-save files
            test_data = [{"id": "1", "data": "test"}]
            file1 = os.path.join(
                temp_annotations_dir, "autosave_run1_annotator1_john_data.parquet"
            )
            file2 = os.path.join(
                temp_annotations_dir, "autosave_run2_annotator2_john_data.parquet"
            )
            file3 = os.path.join(
                temp_annotations_dir, "autosave_run3_annotator3_jane_data.parquet"
            )

            pl.DataFrame(test_data).write_parquet(file1)
            pl.DataFrame(test_data).write_parquet(file2)
            pl.DataFrame(test_data).write_parquet(file3)

            # Test finding files for "john"
            found_file = annotator._find_existing_auto_save_file("john")
            assert found_file is not None
            assert "john" in found_file

            # Test finding files for "jane"
            found_file = annotator._find_existing_auto_save_file("jane")
            assert found_file is not None
            assert "jane" in found_file

            # Test finding files for non-existent user
            found_file = annotator._find_existing_auto_save_file("bob")
            assert found_file is None
