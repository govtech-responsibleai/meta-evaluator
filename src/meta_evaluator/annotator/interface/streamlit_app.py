"""Streamlit-based annotation interface for meta-evaluator."""

import logging
import os
from datetime import datetime
from typing import Any, Literal, Optional

import polars as pl
import streamlit as st

from meta_evaluator.annotator.exceptions import (
    AnnotationValidationError,
    NameValidationError,
    SaveError,
)
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask

from .streamlit_session_manager import StreamlitSessionManager


class StreamlitAnnotator:
    """Streamlit-based interface for annotating evaluation data."""

    def __init__(
        self,
        eval_data: EvalData,
        eval_task: EvalTask,
        annotations_dir: str,
        required_tasks: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        test_environment: bool = False,
    ):
        """Initialize the Streamlit annotator.

        Args:
            eval_data: EvalData object containing the evaluation data.
            eval_task: EvalTask object containing task configuration.
            annotations_dir: Path where annotations will be saved.
            required_tasks: List of columns that must be filled (default: all except remarks).
            metadata: Metadata to include in saved annotations.
            test_environment: Whether running in test environment (disables auto-rerun).

        Raises:
            SaveError: If the annotations directory cannot be created.
        """
        # Set page config
        st.set_page_config(layout="wide")

        self.df: pl.DataFrame = eval_data.data
        self.id_col: str = eval_data.id_column  # type: ignore
        self.eval_task: EvalTask = eval_task  # Store the full eval_task object
        self.task_schemas: dict[str, list[str] | None] = eval_task.task_schemas
        self.prompt_columns: Optional[list[str]] = eval_task.prompt_columns
        self.response_columns: list[str] = eval_task.response_columns
        self.annotation_prompt: str = eval_task.annotation_prompt
        self.annotations_dir: str = annotations_dir
        self.annotator_name: str | None = None
        self.required_tasks: list[str] = required_tasks or [
            col for col in self.response_columns
        ]
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Store metadata for use in exports
        self.metadata = metadata or {}

        # Initialize auto-save file path (will be set when session is initialized)
        self.auto_save_file = None

        # Store test environment flag
        self.test_environment = test_environment

        # Initialize session manager
        self.session_manager = StreamlitSessionManager()

        # Validate annotations directory
        if not os.path.exists(self.annotations_dir):
            try:
                os.makedirs(self.annotations_dir, exist_ok=True)
            except OSError as e:
                raise SaveError(
                    f"Cannot create annotations directory ({e})", self.annotations_dir
                ) from e

    def set_style(self) -> None:
        """Set the style for the Streamlit app."""
        css = """
        <style>
            [data-testid="stProgress"] > div:nth-child(2) {
                padding-bottom: 25px;
            }
            [data-testid="stExpanderDetails"] {
                overflow: auto;
                max-height: 400px;
                scrollbar-color: rgba(0, 0, 0, .5) rgba(0, 0, 0, .025) !important;
            }
            [data-testid="stVerticalBlock"] > div:has(> div.stText) {
                padding: 5px;
            }
            ::-webkit-scrollbar {
                -webkit-appearance: none;
                width: 7px;
            }
            ::-webkit-scrollbar-thumb {
                border-radius: 4px;
                background-color: rgba(0, 0, 0, .5);
            }
            ::-webkit-scrollbar-track {
                background: rgba(0, 0, 0, .025);
            }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    def display_annotation_progress(self, total_samples: int) -> None:
        """Display the progress bar with navigation buttons."""
        col1, col2 = st.columns([0.8, 0.2], vertical_alignment="center")
        progress = self.session_manager.annotated_count / total_samples
        with col1:
            st.progress(progress, text="Your progress:")
        with col2:
            st.caption(
                f"{self.session_manager.annotated_count}/{total_samples} ({progress * 100:.1f}%)"
            )

    def display_annotation_navigation(self) -> None:
        """Display navigation buttons and handle navigation logic."""
        left, right = st.columns(2, vertical_alignment="center")
        if left.button(
            "",
            icon=":material/arrow_back:",
            disabled=not self.session_manager.can_go_previous(len(self.df)),
            width="stretch",
        ):
            self.session_manager.previous_row()
            st.rerun()

        if right.button(
            "",
            icon=":material/arrow_forward:",
            disabled=not self.session_manager.can_go_next(len(self.df)),
            width="stretch",
        ):
            self.session_manager.next_row()
            st.rerun()

    def display_subheader(self, text: str, level: int = 3) -> None:
        """Display a header with the given text."""
        st.markdown(f"<h{level}>{text}</h{level}>", unsafe_allow_html=True)

    def display_annotation_prompt(self) -> None:
        """Display the annotation prompt in a nicely formatted container with instructions title."""
        if len(self.annotation_prompt) <= 500:
            with st.container(border=True):
                self.display_subheader(self.annotation_prompt, level=4)
        else:
            with st.expander("Instructions", expanded=True):
                st.code(
                    self.annotation_prompt,
                    language=None,
                    line_numbers=True,
                    wrap_lines=False,
                )

    def display_columns_to_evaluate(
        self,
        col_type: Literal["prompt", "response"],
        current_row: tuple[Any, ...],
    ) -> None:
        """Display the input columns for the current sample."""
        columns = self.prompt_columns if col_type == "prompt" else self.response_columns
        if not columns:
            return

        st.write(f"**{col_type.capitalize()}s to evaluate**")
        for col in columns:
            with st.expander(col, expanded=True):
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
        st.markdown(
            "<div style='color: green; font-size: 12px; margin-top: -25px;'>💾  Responses are automatically saved when all required fields are filled in.</div>",
            unsafe_allow_html=True,
        )

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

            # Check if required tasks are completed for this row (flexible validation)
            if self._is_annotation_valid(annotation):
                # Required tasks completed - add/update the row
                sample_example_id = f"sample_{self.session_manager.current_row + 1}"

                # Check if this is a new annotation
                is_new_annotation = (
                    current_id not in self.session_manager.results_builder._results
                )

                self.session_manager.create_success_row(
                    sample_example_id=sample_example_id,
                    original_id=current_id,
                    outcomes=annotation,
                    annotation_timestamp=datetime.now(),
                )

                # Auto-save annotations
                try:
                    self._auto_save_annotation(current_id, annotation)
                except Exception:
                    # Re-raise to see original behavior
                    raise

                # Refresh the page to update the progress bar for new annotation
                if is_new_annotation and not self.test_environment:
                    st.rerun()

            return annotation

        except Exception as e:
            raise AnnotationValidationError(current_id, e)

    def _is_annotation_valid(self, annotation: dict[str, Any]) -> bool:
        """Check if annotation is valid based on required columns.

        Args:
            annotation: Dictionary of annotation outcomes

        Returns:
            bool: True if annotation is valid, False otherwise
        """
        missing_fields = []

        # Get required columns from the eval task
        required_tasks = self.eval_task.get_required_tasks()

        # Only check required fields
        for task_name in required_tasks:
            if (
                task_name not in annotation
                or not annotation[task_name]
                or str(annotation[task_name]).strip() == ""
            ):
                missing_fields.append(task_name)

        if missing_fields:
            self.logger.debug(f"Missing required fields: {missing_fields}")
            return False

        return True

    def _generate_auto_save_filename(self, annotator_name: str | None) -> str:
        """Generate the auto-save filename using the same pattern as final save.

        Args:
            annotator_name: Name of the annotator

        Returns:
            str: Full path to the auto-save file

        Raises:
            RuntimeError: If there is no user session
        """
        if not self.session_manager.has_user_session:
            raise RuntimeError(
                "Cannot generate auto-save filename without user session"
            )

        run_id = self.session_manager.run_id
        annotator_id = self.session_manager.annotator_id

        # Use same pattern as final save but with 'autosave' prefix
        return os.path.join(
            self.annotations_dir,
            f"autosave_{run_id}_{annotator_id}_{annotator_name}_data.parquet",
        )

    def _find_existing_auto_save_file(self, annotator_name: str) -> Optional[str]:
        """Find existing auto-save file for the given annotator.

        Args:
            annotator_name: Name of the annotator

        Returns:
            Optional[str]: Path to existing auto-save file, or None if not found
        """
        import glob

        # Create pattern to match auto-save files for this annotator
        # Pattern: autosave_*_{annotator_id}_{annotator_name}_data.parquet
        pattern = os.path.join(
            self.annotations_dir, f"autosave_*_*_{annotator_name}_data.parquet"
        )
        matching_files = glob.glob(pattern)

        if matching_files:
            # Return the most recent file (by modification time)
            most_recent = max(matching_files, key=os.path.getmtime)
            self.logger.info(f"Found existing auto-save file: {most_recent}")
            return most_recent

        return None

    def _set_auto_save_filename(self) -> None:
        """Set the auto-save filename using the same pattern as final save."""
        self.auto_save_file = self._generate_auto_save_filename(self.annotator_name)

    def _save_annotations_data(
        self, filepath: str | None, data_format: str = "parquet"
    ) -> None:
        """Unified function to save annotation data in specified format.

        Args:
            filepath: Path to save the data file
            data_format: Format to save in ("parquet" or "json")

        Raises:
            SaveError: If filepath is None or if there is no user session
        """
        if filepath is None:
            raise SaveError(
                "Error: filepath is None but it should have been automatically set. Try again."
            )

        if not self.session_manager.has_user_session:
            raise SaveError("No results to save")

        # Get current results from the session manager
        results_builder = self.session_manager.results_builder
        current_results = results_builder._results.copy()

        if not current_results:
            raise SaveError("No annotation data to save")

        # Convert results to DataFrame format
        row_dicts = [row.model_dump() for row in current_results.values()]
        results_data = pl.DataFrame(row_dicts)

        # Save in the specified format
        if data_format == "parquet":
            results_data.write_parquet(filepath)
        elif data_format == "json":
            results_data.write_json(filepath)
        else:
            raise SaveError(f"Unsupported data format: {data_format}")

        self.logger.info(
            f"Saved {len(results_data)} annotations to {filepath} in {data_format} format"
        )

    def _load_auto_save_results(self, auto_save_file: str) -> bool:
        """Load results from auto-save file into the session manager.

        Args:
            auto_save_file: Path to the auto-save file

        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            # Load the auto-save data (support both parquet and json for backward compatibility)
            if auto_save_file.endswith(".parquet"):
                results_data = pl.read_parquet(auto_save_file)
            else:
                results_data = pl.read_json(auto_save_file)

            if results_data.is_empty():
                return False

            # Convert DataFrame back to result rows using proper builder methods
            for row_dict in results_data.to_dicts():
                # Extract the original_id to use as key
                original_id = row_dict.get("original_id")
                if not original_id:
                    continue

                # Extract task outcomes (only the actual task results)
                outcomes = {}
                for task_name in self.task_schemas.keys():
                    if task_name in row_dict and row_dict[task_name] is not None:
                        outcomes[task_name] = row_dict[task_name]

                # Skip rows that don't have any outcomes
                if not outcomes:
                    continue

                # Parse annotation timestamp
                from datetime import datetime

                annotation_timestamp = row_dict.get("annotation_timestamp")
                if isinstance(annotation_timestamp, str):
                    try:
                        annotation_timestamp = datetime.fromisoformat(
                            annotation_timestamp.replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        annotation_timestamp = datetime.now()
                elif annotation_timestamp is None:
                    annotation_timestamp = datetime.now()

                # Use session manager's proper method (handles duplicates correctly)
                self.session_manager.create_success_row(
                    sample_example_id=row_dict.get(
                        "sample_example_id", f"sample_{original_id}"
                    ),
                    original_id=original_id,
                    outcomes=outcomes,
                    annotation_timestamp=annotation_timestamp,
                )

            self.logger.info(
                f"Loaded {len(results_data)} annotations from auto-save file: {os.path.basename(auto_save_file)}"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to load auto-save file {auto_save_file}: {str(e)}"
            )
            return False

    def _auto_save_annotation(
        self, current_id: str | int, annotation: dict[str, Any]
    ) -> None:
        """Auto-save annotation using the same format as final save (parquet format)."""
        try:
            if not self.session_manager.has_user_session:
                return

            # Set auto-save filename if not already set
            if self.auto_save_file is None:
                self._set_auto_save_filename()

            # Use the unified save function to save in parquet format
            self._save_annotations_data(self.auto_save_file, data_format="parquet")

            # Show success message
            st.success(
                f"Auto-saved annotation (backup) for sample {self.session_manager.current_row + 1}"
            )

        except Exception as e:
            self.logger.error(f"Auto-save failed: {str(e)}")
            st.error(f"Auto-save failed: {str(e)}")

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
            data_filename = (
                f"{run_id}_{annotator_id}_{self.annotator_name}_data.parquet"
            )

            # Save the results
            try:
                results.save_state(
                    state_file=os.path.join(self.annotations_dir, metadata_filename),
                    data_format="parquet",
                    data_filename=data_filename,
                )
            except Exception as e:
                raise SaveError(
                    f"Failed to save results ({e})",
                    os.path.join(self.annotations_dir, metadata_filename),
                ) from e

            # Show success information
            st.success("✅ Export completed successfully!")
            self.display_subheader("Export Summary:", level=5)
            st.write(f"> **Run ID:** {run_id}")
            st.write(f"> **Annotator:** {annotator_id}")
            st.write(
                f"> **Time Completed:** {results.timestamp_local.strftime('%Y-%m-%d %H:%M')}"
            )

            # Show task completion rates
            self.display_subheader("Task Completion Rates:", level=5)
            for task_name in results.task_schemas.keys():
                success_rate = results.get_task_success_rate(task_name)
                st.progress(
                    success_rate, text=f"{task_name}: {success_rate * 100:.1f}%"
                )

            # Provide next steps
            self.display_subheader("Next Steps:", level=5)
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

    def display_annotation_export(self, total_samples: int) -> None:
        """Display export button when all samples are annotated."""
        if self.session_manager.annotated_count != total_samples:
            return

        self.display_subheader("All done here, export your annotations:", level=5)

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
            # Set style (important for the UI)
            self.set_style()

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
                    required_tasks = self.eval_task.get_required_tasks()
                    self.session_manager.initialize_user_session(
                        annotator_name=self.annotator_name,
                        task_schemas=self.task_schemas,
                        expected_ids=expected_ids,
                        required_tasks=required_tasks,
                    )

                    # Check for existing auto-save and load if found
                    try:
                        # Look for existing auto-save files for this annotator
                        existing_auto_save_file = self._find_existing_auto_save_file(
                            self.annotator_name
                        )

                        if existing_auto_save_file:
                            if self._load_auto_save_results(existing_auto_save_file):
                                st.info(
                                    f"Resumed from auto-save: {os.path.basename(existing_auto_save_file)}"
                                )
                                # Set the auto-save file to the existing one
                                self.auto_save_file = existing_auto_save_file
                            else:
                                st.warning(
                                    "Found auto-save file but failed to load it. Starting fresh."
                                )
                                # Set up new auto-save filename
                                self._set_auto_save_filename()
                        else:
                            # Set up new auto-save filename
                            self._set_auto_save_filename()
                    except Exception as e:
                        # If auto-save setup fails, continue without auto-save
                        self.logger.warning(f"Auto-save setup failed: {e}")
                        self._set_auto_save_filename()

                current_row = self.df.row(self.session_manager.current_row)

                # Display the annotation interface
                st.markdown("---")

                col1, col2 = st.columns((0.6, 0.4), gap="large")
                with col1:
                    self.display_subheader(
                        f"Sample {self.session_manager.current_row + 1}", level=3
                    )
                    self.display_annotation_prompt()

                    self.display_columns_to_evaluate(
                        col_type="prompt", current_row=current_row
                    )
                    self.display_columns_to_evaluate(
                        col_type="response", current_row=current_row
                    )

                with col2:
                    self.display_subheader(text="Annotations", level=4)
                    self.display_annotation_progress(total_samples=len(self.df))
                    self.handle_annotation(current_row=current_row)
                    self.display_annotation_navigation()

                    st.markdown("---")
                    self.display_annotation_export(total_samples=len(self.df))

            else:
                st.info("Please enter your name to start annotating.")

        except Exception as e:
            st.error(f"An error occurred while building the app: {e}")
