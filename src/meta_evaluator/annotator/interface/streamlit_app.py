"""Streamlit-based annotation interface for meta-evaluator.

This module provides a Streamlit-based interface for annotating evaluation data.
It handles the display and collection of annotations through a web interface.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import streamlit as st
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask


@st.cache_data
def convert_annotations_for_download(annotations: Dict[str, Any]) -> bytes:
    """Convert annotations to JSON bytes for download. Defined outside of class due to caching.

    Args:
        annotations: Dictionary of annotations

    Returns:
        bytes: JSON data encoded in UTF-8
    """
    return json.dumps(annotations, indent=2).encode("utf-8")


class StreamlitAnnotator:
    """Streamlit-based interface for annotating evaluation data.

    This class provides a web interface for annotating evaluation data using Streamlit.
    It handles the display of data, collection of annotations, and saving of results.
    """

    def __init__(
        self,
        eval_data: EvalData,
        eval_task: EvalTask,
        annotations_dir: str,
    ):
        """Initialize the Streamlit annotator.

        Args:
            eval_data: EvalData object containing the evaluation data.
            eval_task: EvalTask object containing task configuration.
            annotations_dir: Path where annotations will be saved.
        """
        self.df: pl.DataFrame = eval_data.data
        # EvalData guarantees the ID column will be set after initialization. This is a case where the type system's Optional doesn't match the runtime guarantees of the class.
        self.id_col: str = eval_data.id_column  # type: ignore

        self.outcomes: Dict[str, Any] = eval_task.task_schemas
        self.prompt_columns: Optional[List[str]] = eval_task.prompt_columns
        self.response_columns: List[str] = eval_task.response_columns
        self.annotations_dir: str = annotations_dir

    @property
    def annotated_count(self) -> int:
        """Calculate the number of samples that have been annotated.

        Returns:
            int: Number of annotated samples
        """
        count = 0
        for row in self.df.iter_rows():
            row_id = row[self.df.columns.index(self.id_col)]
            annotation = st.session_state.annotations.get(row_id, {})
            if all(
                task_name in annotation and annotation[task_name] is not None
                for task_name in self.outcomes
            ):
                count += 1
        return count

    def initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if "annotations" not in st.session_state:
            st.session_state.annotations = {}
        if "current_row" not in st.session_state:
            st.session_state.current_row = 0

    def display_header(self, current_row_idx: int, total_samples: int) -> None:
        """Display the header section with sample number and progress.

        Args:
            current_row_idx: Current row index
            total_samples: Total number of samples
        """
        col1, col2 = st.columns([0.5, 0.5], vertical_alignment="center")
        with col1:
            st.subheader(f"Sample {current_row_idx + 1}")
        with col2:
            st.markdown(
                f"<div style='text-align: right'>Progress: {self.annotated_count}/{total_samples} samples annotated</div>",
                unsafe_allow_html=True,
            )

    def display_h4_header(self, text: str) -> None:
        """Display an h4 header with the given text.

        Args:
            text: The text to display in the header
        """
        st.markdown(f"<h4>{text}</h4>", unsafe_allow_html=True)

    def display_h5_header(self, text: str) -> None:
        """Display an h5 header with the given text.

        Args:
            text: The text to display in the header
        """
        st.markdown(f"<h5>{text}</h5>", unsafe_allow_html=True)

    def display_prompt_columns(self, current_row: Tuple[Any, ...]) -> None:
        """Display the input columns for the current sample.

        Args:
            current_row: Current row data as a tuple.
        """
        if self.prompt_columns is None:
            return
        for col in self.prompt_columns:
            st.write(f"**{col}:**")
            with st.container():
                st.markdown(
                    '<style>div[data-testid="stVerticalBlock"] > div:has(> div.stText) {background-color: #f0f2f6; padding: 20px; border-radius: 5px; margin-bottom: 20px;}</style>',
                    unsafe_allow_html=True,
                )
                st.text(current_row[self.df.columns.index(col)])

    def display_response_columns(self, current_row: Tuple[Any, ...]) -> None:
        """Display the response columns for the current sample.

        Args:
            current_row: Current row data as a tuple.
        """
        for col in self.response_columns:
            st.write(f"**{col}:**")
            with st.container():
                st.markdown(
                    '<style>div[data-testid="stVerticalBlock"] > div:has(> div.stText) {background-color: #f0f2f6; padding: 20px; border-radius: 5px; margin-bottom: 20px;}</style>',
                    unsafe_allow_html=True,
                )
                st.text(current_row[self.df.columns.index(col)])

    def display_radio_buttons(
        self,
        label: str,
        outcomes: List[str],
        key: str,
        selected_index: Optional[int] = None,
    ) -> None:
        """Display radio buttons for annotation.

        Args:
            label: Label for the radio button group.
            outcomes: List of possible outcomes.
            key: Unique key for the radio button group.
            selected_index: Index of the currently selected outcome.
        """
        st.radio(
            label=f"{label}:",
            options=outcomes,
            key=key,
            index=selected_index,
        )

    def display_free_form_text(
        self,
        label: str,
        value: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        """Display a free form text input.

        Args:
            label: Label for the text input.
            value: Initial value for the text input.
            key: Unique key for the text input.
        """
        st.text_area(label=label, value=value, key=key)

    def handle_annotation(
        self, current_row: Tuple[Any, ...]
    ) -> Dict[str, Optional[str]]:
        """Handle the annotation process for the current sample.

        Args:
            current_row: Current row data as a tuple.

        Returns:
            Dict[str, Optional[str]]: The annotation for the current sample.
        """
        annotation = {}
        current_id = current_row[self.df.columns.index(self.id_col)]

        if not self.outcomes:
            return annotation

        for task_name, schema in self.outcomes.items():
            input_key = f"outcome_{task_name}_{st.session_state.current_row}"
            prev_annotation = st.session_state.annotations.get(current_id, {})
            prev_outcome = prev_annotation.get(task_name)

            if isinstance(schema, list):
                prev_selected_index = (
                    schema.index(prev_outcome) if prev_outcome in schema else None
                )
                self.display_radio_buttons(
                    label=task_name,
                    outcomes=schema,
                    key=input_key,
                    selected_index=prev_selected_index,
                )

            else:
                self.display_free_form_text(
                    label=task_name,
                    value=prev_outcome,
                    key=input_key,
                )

            outcome = st.session_state.get(input_key)
            annotation[task_name] = outcome

        # Update annotated count if all outcomes are annotated
        st.session_state.annotations[current_id] = annotation
        return annotation

    def display_navigation_buttons(
        self,
        current_row: Tuple[Any, ...],
        annotation: Dict[str, Optional[str]],
    ) -> None:
        """Display navigation buttons and handle navigation logic.

        Args:
            current_row: Current row data as a tuple.
            annotation: Current annotation data.
        """
        col1, col2, col3 = st.columns([0.15, 0.7, 0.15], vertical_alignment="center")

        with col1:
            if st.button("Previous", disabled=st.session_state.current_row == 0):
                st.session_state.annotations[
                    current_row[self.df.columns.index(self.id_col)]
                ] = annotation
                st.session_state.current_row -= 1
                st.rerun()

        with col2:
            st.markdown(
                f"<div style='text-align: center'>Sample {st.session_state.current_row + 1} of {len(self.df)}</div>",
                unsafe_allow_html=True,
            )

        with col3:
            if st.button(
                "Next", disabled=st.session_state.current_row == len(self.df) - 1
            ):
                st.session_state.annotations[
                    current_row[self.df.columns.index(self.id_col)]
                ] = annotation
                st.session_state.current_row += 1
                st.rerun()

    def convert_annotations_to_json(self, annotations: Dict[str, Any]) -> str:
        """Convert annotations to JSON string.

        Args:
            annotations: Dictionary of annotations.

        Returns:
            str: JSON string representation of annotations.
        """
        return json.dumps(annotations, indent=2)

    def save_annotations_to_wd(self, save_path: str, json_data: str) -> None:
        """Save annotations to a file.

        Args:
            save_path: Path where to save the annotations.
            json_data: JSON data to save.
        """
        with open(save_path, "w") as f:
            f.write(json_data)

        if os.path.exists(save_path):
            st.success(
                "Annotations saved to local directory and host directory!",
                icon="🔥",
            )
        else:
            st.error("Failed to save annotations. Try again.", icon="🚨")

    def display_export_button(
        self,
        annotations_dir: str,
        total_samples: int,
        name: str,
    ) -> None:
        """Display export button when all samples are annotated.

        Args:
            annotations_dir: Path to save the annotations
            total_samples: Total number of samples
            name: Name of the annotator
        """
        if self.annotated_count != total_samples:
            return

        self.display_h5_header("All done here, export your annotations:")

        # Validate name
        if not name or not name.strip():
            st.error(
                "Please enter your name before downloading annotations.", icon="🚨"
            )
            return

        # Define user filename
        filename = f"annotations_{name}.json"

        # Define save path for filename file
        os.makedirs(annotations_dir, exist_ok=True)
        save_path = os.path.join(annotations_dir, filename)

        json_data = self.convert_annotations_to_json(st.session_state.annotations)

        st.download_button(
            "Save the annotations to the host's directory. (A copy will be saved on your local directory.)",
            data=json_data.encode("utf-8"),
            file_name=filename,
            icon=":material/download:",
            use_container_width=True,
            on_click=lambda: self.save_annotations_to_wd(save_path, json_data),
        )

    def build_streamlit_app(self) -> None:
        """Launch the Streamlit annotation interface."""
        st.title("Simple Annotation Interface")
        name = st.text_input(
            label="Hello,",
            placeholder="Your name here.",
            help="This will be used to identify your annotations!",
        )

        self.initialize_session_state()
        current_row = self.df.row(st.session_state.current_row)

        self.display_header(
            current_row_idx=st.session_state.current_row,
            total_samples=len(self.df),
        )
        st.markdown("---")
        self.display_h4_header(
            text="Label the following texts as positive, negative, or neutral (placeholder)"
        )
        self.display_prompt_columns(current_row=current_row)
        self.display_response_columns(current_row=current_row)
        st.markdown("---")

        self.display_h5_header(text="Your response:")
        annotation = self.handle_annotation(current_row=current_row)

        st.markdown("---")
        self.display_navigation_buttons(
            current_row=current_row,
            annotation=annotation,
        )
        st.markdown("---")

        self.display_export_button(
            annotations_dir=self.annotations_dir,
            total_samples=len(self.df),
            name=name,
        )
