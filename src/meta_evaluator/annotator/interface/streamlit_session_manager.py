"""Session manager for Streamlit annotation interface."""

import re
import uuid
from datetime import datetime
from typing import Optional

import streamlit as st

from meta_evaluator.annotator.exceptions import AnnotatorInitializationError
from meta_evaluator.results import HumanAnnotationResultsBuilder


def generate_run_id() -> str:
    """Generate a unique run ID.

    Returns:
        str: Unique run ID in format 'annotation_run_YYYYMMDD_HHMMSS_XXXXXXXX'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"annotation_run_{timestamp}_{unique_id}"


def generate_annotator_id(annotator_name: str) -> str:
    """Generate an annotator ID from the annotator name.

    Args:
        annotator_name: Name of the annotator

    Returns:
        str: Sanitized annotator ID

    Raises:
        ValueError: If annotator_name is empty or contains invalid characters
    """
    if not annotator_name or not annotator_name.strip():
        raise ValueError("Annotator name cannot be empty")

    # Sanitize the name
    annotator_id = annotator_name.lower().strip()
    annotator_id = re.sub(r"[^a-z0-9_-]", "_", annotator_id)
    annotator_id = re.sub(
        r"_+", "_", annotator_id
    )  # Replace multiple underscores with single
    annotator_id = annotator_id.strip("_")  # Remove leading/trailing underscores

    # Ensure it starts with a letter
    if not annotator_id or not annotator_id[0].isalpha():
        annotator_id = f"user_{annotator_id}"

    return annotator_id


class StreamlitSessionManager:
    """Manages Streamlit session state for annotation interface."""

    def __init__(self):
        """Initialize the session manager."""
        self._ensure_basic_state()

    def _ensure_basic_state(self) -> None:
        """Ensure basic session state is initialized."""
        if "current_row" not in st.session_state:
            st.session_state.current_row = 0

    @property
    def current_row(self) -> int:
        """Get the current row index."""
        return st.session_state.current_row

    @current_row.setter
    def current_row(self, value: int) -> None:
        """Set the current row index."""
        st.session_state.current_row = value

    @property
    def has_user_session(self) -> bool:
        """Check if a user session is initialized."""
        return "results_builder" in st.session_state

    @property
    def results_builder(self) -> HumanAnnotationResultsBuilder:
        """Get the current results builder.

        Raises:
            AnnotatorInitializationError: If no user session is initialized
        """
        results_builder = st.session_state.get("results_builder")
        if not results_builder:
            raise AnnotatorInitializationError("User Session")
        return results_builder

    @property
    def run_id(self) -> Optional[str]:
        """Get the current run ID."""
        return st.session_state.get("current_run_id")

    @property
    def annotator_id(self) -> Optional[str]:
        """Get the current annotator ID."""
        return st.session_state.get("current_annotator_id")

    @property
    def annotated_count(self) -> int:
        """Get the number of completed annotations."""
        if not self.has_user_session:
            return 0
        return self.results_builder.completed_count

    def initialize_user_session(
        self,
        annotator_name: str,
        task_schemas: dict,
        expected_ids: list,
        required_tasks: list,
    ) -> None:
        """Initialize a new user session.

        Args:
            annotator_name: Name of the annotator
            task_schemas: Task schemas for the annotation
            expected_ids: List of expected IDs to annotate
            required_tasks: List of required task names for success rows
        """
        # Generate IDs using pure functions
        run_id = generate_run_id()
        annotator_id = generate_annotator_id(annotator_name)

        # Store in session state
        st.session_state.results_builder = HumanAnnotationResultsBuilder(
            run_id=run_id,
            annotator_id=annotator_id,
            task_schemas=task_schemas,
            expected_ids=expected_ids,
            required_tasks=required_tasks,
            is_sampled_run=False,
        )
        st.session_state.current_run_id = run_id
        st.session_state.current_annotator_id = annotator_id

    def get_input_key(self, task_name: str) -> str:
        """Generate input key for a task.

        Args:
            task_name: Name of the task

        Returns:
            str: Input key for the task
        """
        return f"outcome_{task_name}_{self.current_row}"

    def get_input_value(self, input_key: str) -> Optional[str]:
        """Get the value of an input from session state.

        Args:
            input_key: The key to look up in session state

        Returns:
            Optional[str]: The value from session state, or None if not found
        """
        return st.session_state.get(input_key)

    def get_previous_outcome(self, task_name: str, current_id: str) -> Optional[str]:
        """Get the previous outcome for a task.

        Args:
            task_name: Name of the task
            current_id: Current row ID

        Returns:
            Optional[str]: Previous outcome if exists
        """
        if not self.has_user_session:
            return None

        existing_result = self.results_builder._results.get(current_id)
        if existing_result:
            return getattr(existing_result, task_name, None)
        return None

    def create_success_row(
        self,
        sample_example_id: str,
        original_id: str,
        outcomes: dict,
        annotation_timestamp: Optional[datetime] = None,
    ) -> None:
        """Create a successful annotation row.

        Args:
            sample_example_id: Sample example ID
            original_id: Original row ID
            outcomes: Annotation outcomes
            annotation_timestamp: Timestamp of annotation

        Raises:
            AnnotatorInitializationError: If no user session is initialized
        """
        if not self.has_user_session:
            raise AnnotatorInitializationError("User Session")

        # Remove existing result if it exists
        if original_id in self.results_builder._results:
            del self.results_builder._results[original_id]

        # Add the new result
        self.results_builder.create_success_row(
            sample_example_id=sample_example_id,
            original_id=original_id,
            outcomes=outcomes,
            annotation_timestamp=annotation_timestamp,
        )

    def complete_session(self):
        """Complete the current session and return results.

        Returns:
            HumanAnnotationResults: Completed results

        Raises:
            AnnotatorInitializationError: If no user session is initialized
        """
        if not self.has_user_session:
            raise AnnotatorInitializationError("User Session")

        return self.results_builder.complete()

    def reset_session(self) -> None:
        """Reset the current session."""
        if "results_builder" in st.session_state:
            del st.session_state.results_builder
        if "current_run_id" in st.session_state:
            del st.session_state.current_run_id
        if "current_annotator_id" in st.session_state:
            del st.session_state.current_annotator_id
        if "current_row" in st.session_state:
            del st.session_state.current_row

    def next_row(self) -> None:
        """Move to the next row."""
        self.current_row += 1

    def previous_row(self) -> None:
        """Move to the previous row."""
        if self.current_row > 0:
            self.current_row -= 1

    def can_go_previous(self, total_rows: int) -> bool:
        """Check if can go to previous row.

        Args:
            total_rows: Total number of rows

        Returns:
            bool: True if can go previous
        """
        return self.current_row > 0

    def can_go_next(self, total_rows: int) -> bool:
        """Check if can go to next row.

        Args:
            total_rows: Total number of rows

        Returns:
            bool: True if can go next
        """
        return self.current_row < total_rows - 1
