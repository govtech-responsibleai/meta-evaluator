"""Streamlit-based annotation interface for meta-evaluator.

This module provides a Streamlit-based interface for annotating evaluation data.
It handles the display and collection of annotations through a web interface.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Literal
from datetime import datetime

import polars as pl
import streamlit as st
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.annotator.results import (
    HumanAnnotationResultsBuilder,
    HumanAnnotationResultsConfig,
)
from meta_evaluator.annotator.exceptions import (
    AnnotatorInitializationError,
    AnnotationValidationError,
    SaveError,
)


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

        Raises:
            SaveError: If the annotations directory cannot be created.
        """
        self.run_id: str = "run_id"
        self.annotator_id: str = "streamlit_annotator"
        self.annotator_name: str | None = None

        self.df: pl.DataFrame = eval_data.data
        # EvalData guarantees the ID column will be set after initialization. This is a case where the type system's Optional doesn't match the runtime guarantees of the class.
        self.id_col: str = eval_data.id_column  # type: ignore
        self.task_schemas: dict[str, List[str] | None] = eval_task.task_schemas
        self.prompt_columns: Optional[List[str]] = eval_task.prompt_columns
        self.response_columns: List[str] = eval_task.response_columns
        self.annotations_dir: str = annotations_dir

        # Validate annotations directory
        if not os.path.exists(self.annotations_dir):
            try:
                os.makedirs(self.annotations_dir, exist_ok=True)
            except Exception as e:
                raise SaveError(
                    f"Cannot create annotations directory: {e}", self.annotations_dir, e
                )

    @property
    def annotated_count(self) -> int:
        """Calculate the number of samples that have been annotated.

        Returns:
            int: Number of annotated samples
        """
        if "results_builder" not in st.session_state:
            return 0
        return st.session_state.results_builder.completed_count

    def initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables.

        Raises:
            AnnotatorInitializationError: If the annotator fails to initialize.
        """
        try:
            if "current_row" not in st.session_state:
                st.session_state.current_row = 0

            # Initialize the results builder if not already done
            if "results_builder" not in st.session_state:
                # Get expected IDs from the dataframe
                expected_ids = self.df[self.id_col].to_list()

                config = HumanAnnotationResultsConfig(
                    run_id="streamlit_annotation_run",
                    annotator_id="streamlit_annotator",
                    task_schemas=self.task_schemas,
                    timestamp_local=datetime.now(),
                    is_sampled_run=False,  # This could be determined from eval_data type
                    expected_ids=expected_ids,
                )
                st.session_state.results_builder = HumanAnnotationResultsBuilder(config)

        except Exception as e:
            raise AnnotatorInitializationError(
                f"Failed to initialize session state: {e}"
            )

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

    def display_columns_to_evaluate(
        self,
        col_type: Literal["prompt", "response"],
        current_row: Tuple[Any, ...],
    ) -> None:
        """Display the input columns for the current sample.

        Args:
            col_type: The type of column to display (prompt_columns or response_columns).
            current_row: Current row data as a tuple.
        """
        columns = self.prompt_columns if col_type == "prompt" else self.response_columns
        if not columns:
            return

        st.write(f"**{col_type.capitalize()}s to Evaluate:**")
        for col in columns:
            with st.container():
                # Add col name
                st.markdown(
                    f'<div style="position: absolute; top: 10px; left: 10px; font-size: 12px; color: #666; font-weight: bold;">{col}:</div>',
                    unsafe_allow_html=True,
                )
                # Add text
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

        Raises:
            AnnotationValidationError: If the annotation fails to process.
        """
        annotation = {}
        current_id = current_row[self.df.columns.index(self.id_col)]

        try:
            if not self.task_schemas:
                return annotation

            for task_name, schema in self.task_schemas.items():
                input_key = f"outcome_{task_name}_{st.session_state.current_row}"

                # Get previous annotation from the builder if it exists
                prev_outcome = None
                existing_result = st.session_state.results_builder._results.get(
                    current_id
                )
                if existing_result:
                    prev_outcome = getattr(existing_result, task_name, None)

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

            # Check if all tasks are completed for this row
            if all(
                task_name in annotation and annotation[task_name] is not None
                for task_name in self.task_schemas
            ):
                # All tasks completed - add/update the row in the builder
                sample_example_id = f"sample_{st.session_state.current_row + 1}"

                # Remove existing result if it exists (to update it)
                if current_id in st.session_state.results_builder._results:
                    del st.session_state.results_builder._results[current_id]

                # Add the new result
                st.session_state.results_builder.create_success_row(
                    sample_example_id=sample_example_id,
                    original_id=current_id,
                    outcomes=annotation,
                    annotation_timestamp=datetime.now(),
                )

            return annotation

        except Exception as e:
            raise AnnotationValidationError(
                f"Error processing annotation: {e}", "annotation"
            )

    def display_navigation_buttons(self) -> None:
        """Display navigation buttons and handle navigation logic.

        Args:
            current_row: Current row data as a tuple.
            annotation: Current annotation data.
        """
        col1, col2, col3 = st.columns([0.15, 0.7, 0.15], vertical_alignment="center")

        with col1:
            if st.button("Previous", disabled=st.session_state.current_row == 0):
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
                st.session_state.current_row += 1
                st.rerun()

    def save_results(self) -> None:
        """Save the results to the results builder.

        Raises:
            SaveError: If the results cannot be saved.
            AnnotationValidationError: If the annotator name is not provided.
        """
        try:
            # Validate that we have results to save
            if "results_builder" not in st.session_state:
                raise SaveError("No results to save")

            # Validate annotator name
            if not self.annotator_name or not self.annotator_name.strip():
                raise AnnotationValidationError(
                    "Annotator name is required", "annotator_name"
                )

            # Complete the builder to get HumanAnnotationResults
            results = st.session_state.results_builder.complete()
            metadata_filename = f"{results.run_id}_{results.annotator_id}_{self.annotator_name}_metadata.json"
            data_filename = f"{results.run_id}_{results.annotator_id}_{self.annotator_name}_data.json"

            # Save the results
            try:
                results.save_state(
                    state_file=os.path.join(self.annotations_dir, metadata_filename),
                    data_format="json",
                    data_filename=data_filename,
                )
            except Exception as e:
                raise SaveError(
                    f"Failed to save results: {e}",
                    os.path.join(self.annotations_dir, metadata_filename),
                    e,
                )

            # Show additional success information
            st.success("✅ Export completed successfully!", icon="🎉")

            # Show summary of exported data
            self.display_h5_header("Export Summary:")
            st.write(f"> **Run ID:** {results.run_id}")
            st.write(f"> **Annotator:** {results.annotator_id}")
            st.write(
                f"> **Time Completed:** {results.timestamp_local.strftime('%Y-%m-%d %H:%M')}"
            )

            # Show task completion rates
            self.display_h5_header("Task Completion Rates:")
            for task_name in results.task_schemas.keys():
                success_rate = results.get_task_success_rate(task_name)
                st.progress(
                    success_rate, text=f"{task_name}: {success_rate * 100:.1f}%"
                )

            # Provide next steps
            self.display_h5_header("Next Steps:")
            st.markdown(
                """
            - Your annotations have been saved successfully
            - You can close this browser tab
            - Thank you for your contribution!
            """
            )

        except Exception as e:
            st.error(f"Error creating results: {str(e)}", icon="🚨")
            st.markdown(
                """
            - Please check that all samples have been annotated
            - Ensure you have write permissions to the save directory
            - Try refreshing the page and re-trying
            - Contact the developer if the issue persists
            """
            )

    def display_export_button(
        self,
        total_samples: int,
    ) -> None:
        """Display export button when all samples are annotated.

        Args:
            annotations_dir: Path to save the annotations
            total_samples: Total number of samples
        """
        if self.annotated_count != total_samples:
            return

        self.display_h5_header("All done here, export your annotations:")

        # Validate name
        if not self.annotator_name or not self.annotator_name.strip():
            st.error(
                "Please enter your name before downloading annotations.", icon="🚨"
            )
            return

        # Add export button
        if st.button("Export Annotations", use_container_width=True):
            self.save_results()

    def build_streamlit_app(self) -> None:
        """Launch the Streamlit annotation interface."""
        try:
            st.title("Simple Annotation Interface")
            self.annotator_name = st.text_input(
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
            self.display_columns_to_evaluate(col_type="prompt", current_row=current_row)
            self.display_columns_to_evaluate(
                col_type="response", current_row=current_row
            )
            st.markdown("---")

            self.display_h5_header(text="Your response:")
            self.handle_annotation(current_row=current_row)

            st.markdown("---")
            self.display_navigation_buttons()
            st.markdown("---")

            self.display_export_button(
                total_samples=len(self.df),
            )

        except Exception as e:
            st.error(f"An error occurred while building the app: {e}")
