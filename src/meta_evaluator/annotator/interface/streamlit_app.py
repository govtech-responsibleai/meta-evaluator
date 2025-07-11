"""Streamlit-based annotation interface for meta-evaluator."""

import os
from typing import Any, Optional, Literal
from datetime import datetime

import polars as pl
import streamlit as st
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.annotator.exceptions import (
    AnnotationValidationError,
    NameValidationError,
    SaveError,
)
from .streamlit_session_manager import StreamlitSessionManager


class StreamlitAnnotator:
    """Streamlit-based interface for annotating evaluation data."""

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
        self.df: pl.DataFrame = eval_data.data
        self.id_col: str = eval_data.id_column  # type: ignore
        self.task_schemas: dict[str, list[str] | None] = eval_task.task_schemas
        self.prompt_columns: Optional[list[str]] = eval_task.prompt_columns
        self.response_columns: list[str] = eval_task.response_columns
        self.annotation_prompt: str = eval_task.annotation_prompt
        self.annotations_dir: str = annotations_dir
        self.annotator_name: str | None = None

        # Initialize session manager
        self.session_manager = StreamlitSessionManager()

        # Validate annotations directory
        if not os.path.exists(self.annotations_dir):
            try:
                os.makedirs(self.annotations_dir, exist_ok=True)
            except Exception as e:
                raise SaveError(
                    f"Cannot create annotations directory ({e})", self.annotations_dir
                )

    def display_header(self, current_row_idx: int, total_samples: int) -> None:
        """Display the header section with sample number and progress."""
        col1, col2 = st.columns([0.5, 0.5], vertical_alignment="center")
        with col1:
            st.subheader(f"Sample {current_row_idx + 1}")
        with col2:
            st.markdown(
                f"<div style='text-align: right'>Progress: {self.session_manager.annotated_count}/{total_samples} samples annotated</div>",
                unsafe_allow_html=True,
            )

    def display_h4_header(self, text: str) -> None:
        """Display an h4 header with the given text."""
        st.markdown(f"<h4>{text}</h4>", unsafe_allow_html=True)

    def display_h5_header(self, text: str) -> None:
        """Display an h5 header with the given text."""
        st.markdown(f"<h5>{text}</h5>", unsafe_allow_html=True)

    def display_columns_to_evaluate(
        self,
        col_type: Literal["prompt", "response"],
        current_row: tuple[Any, ...],
    ) -> None:
        """Display the input columns for the current sample."""
        columns = self.prompt_columns if col_type == "prompt" else self.response_columns
        if not columns:
            return

        st.write(f"**{col_type.capitalize()}s to Evaluate:**")
        for col in columns:
            with st.container():
                st.markdown(
                    f'<div style="position: absolute; top: 10px; left: 10px; font-size: 12px; color: #666; font-weight: bold;">{col}:</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<style>div[data-testid="stVerticalBlock"] > div:has(> div.stText) {background-color: #f0f2f6; padding: 20px; border-radius: 5px; margin-bottom: 20px;}</style>',
                    unsafe_allow_html=True,
                )
                st.text(current_row[self.df.columns.index(col)])

    def display_radio_buttons(
        self,
        label: str,
        outcomes: list[str],
        key: str,
        selected_index: Optional[int] = None,
    ) -> None:
        """Display radio buttons for annotation."""
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
        """Display a free form text input."""
        st.text_area(label=label, value=value, key=key)

    def handle_annotation(
        self, current_row: tuple[Any, ...]
    ) -> dict[str, Optional[str]]:
        """Handle the annotation process for the current sample.

        Returns:
            Dict[str, Optional[str]]: A dictionary of the annotation outcomes

        Raises:
            AnnotationValidationError: If there is an error processing the annotation
        """
        annotation = {}
        current_id = current_row[self.df.columns.index(self.id_col)]

        try:
            if not self.task_schemas:
                return annotation

            for task_name, schema in self.task_schemas.items():
                input_key = self.session_manager.get_input_key(task_name)

                # Get previous annotation
                prev_outcome = self.session_manager.get_previous_outcome(
                    task_name, current_id
                )

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

                outcome = self.session_manager.get_input_value(input_key)
                annotation[task_name] = outcome

            # Check if all tasks are completed for this row
            if all(
                task_name in annotation and annotation[task_name] is not None
                for task_name in self.task_schemas
            ):
                # All tasks completed - add/update the row
                sample_example_id = f"sample_{self.session_manager.current_row + 1}"

                self.session_manager.create_success_row(
                    sample_example_id=sample_example_id,
                    original_id=current_id,
                    outcomes=annotation,
                    annotation_timestamp=datetime.now(),
                )

            return annotation

        except Exception as e:
            raise AnnotationValidationError(current_id, e)

    def display_navigation_buttons(self) -> None:
        """Display navigation buttons and handle navigation logic."""
        col1, col2, col3 = st.columns([0.15, 0.7, 0.15], vertical_alignment="center")

        with col1:
            if st.button(
                "Previous",
                disabled=not self.session_manager.can_go_previous(len(self.df)),
            ):
                self.session_manager.previous_row()
                st.rerun()

        with col2:
            st.markdown(
                f"<div style='text-align: center'>Sample {self.session_manager.current_row + 1} of {len(self.df)}</div>",
                unsafe_allow_html=True,
            )

        with col3:
            if st.button(
                "Next", disabled=not self.session_manager.can_go_next(len(self.df))
            ):
                self.session_manager.next_row()
                st.rerun()

    def save_results(self) -> None:
        """Save the results to the results builder.

        Raises:
            SaveError: If there is an error saving the results
            NameValidationError: If there is an error validating the annotator name
        """
        # Validate that we have results to save
        if not self.session_manager.has_user_session:
            raise SaveError("No results to save")

        # Validate annotator name
        if not self.annotator_name or not self.annotator_name.strip():
            raise NameValidationError()

        try:
            # Complete the builder to get HumanAnnotationResults
            results = self.session_manager.complete_session()

            # Use the user-specific IDs
            run_id = self.session_manager.run_id
            annotator_id = self.session_manager.annotator_id

            metadata_filename = (
                f"{run_id}_{annotator_id}_{self.annotator_name}_metadata.json"
            )
            data_filename = f"{run_id}_{annotator_id}_{self.annotator_name}_data.json"

            # Save the results
            try:
                results.save_state(
                    state_file=os.path.join(self.annotations_dir, metadata_filename),
                    data_format="json",
                    data_filename=data_filename,
                )
            except Exception as e:
                raise SaveError(
                    f"Failed to save results ({e})",
                    os.path.join(self.annotations_dir, metadata_filename),
                )

            # Show success information
            st.success("✅ Export completed successfully!")
            self.display_h5_header("Export Summary:")
            st.write(f"> **Run ID:** {run_id}")
            st.write(f"> **Annotator:** {annotator_id}")
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

    def display_export_button(self, total_samples: int) -> None:
        """Display export button when all samples are annotated."""
        if self.session_manager.annotated_count != total_samples:
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

            # Check if user has entered a name and initialize their session
            if self.annotator_name and self.annotator_name.strip():
                # Initialize user-specific session if not already done
                if not self.session_manager.has_user_session:
                    expected_ids = self.df[self.id_col].to_list()
                    self.session_manager.initialize_user_session(
                        annotator_name=self.annotator_name,
                        task_schemas=self.task_schemas,
                        expected_ids=expected_ids,
                    )

                current_row = self.df.row(self.session_manager.current_row)

                self.display_header(
                    current_row_idx=self.session_manager.current_row,
                    total_samples=len(self.df),
                )
                st.markdown("---")
                self.display_h4_header(text=self.annotation_prompt)
                self.display_columns_to_evaluate(
                    col_type="prompt", current_row=current_row
                )
                self.display_columns_to_evaluate(
                    col_type="response", current_row=current_row
                )
                st.markdown("---")

                self.display_h5_header(text="Your response:")
                self.handle_annotation(current_row=current_row)

                st.markdown("---")
                self.display_navigation_buttons()
                st.markdown("---")

                self.display_export_button(total_samples=len(self.df))
            else:
                st.info("Please enter your name to start annotating.")

        except Exception as e:
            st.error(f"An error occurred while building the app: {e}")
