"""In-memory session state manager for annotation API."""

import glob
import logging
import os
import re
import uuid
from datetime import datetime

import polars as pl

from meta_evaluator.annotator.api.schemas import (
    CreateSessionResponse,
    ExportResponse,
    ProgressResponse,
    SampleResponse,
    SubmitAnnotationResponse,
)
from meta_evaluator.annotator.exceptions import AnnotationValidationError
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask, MultiLabelSchema
from meta_evaluator.results import HumanAnnotationResultsBuilder

logger = logging.getLogger(__name__)


def _generate_run_id() -> str:
    """Generate a unique run ID.

    Returns:
        str: Run ID in format 'annotation_run_YYYYMMDD_HHMMSS_XXXXXXXX'.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"annotation_run_{timestamp}_{unique_id}"


def _generate_annotator_id(annotator_name: str) -> str:
    """Generate sanitized annotator ID from name.

    Args:
        annotator_name: Raw annotator name.

    Returns:
        str: Sanitized annotator ID.

    Raises:
        ValueError: If annotator_name is empty.
    """
    if not annotator_name or not annotator_name.strip():
        raise ValueError("Annotator name cannot be empty")
    annotator_id = annotator_name.lower().strip()
    annotator_id = re.sub(r"[^a-z0-9_-]", "_", annotator_id)
    annotator_id = re.sub(r"_+", "_", annotator_id)
    annotator_id = annotator_id.strip("_")
    if not annotator_id or not annotator_id[0].isalpha():
        annotator_id = f"user_{annotator_id}"
    return annotator_id


class _Session:
    """Internal session state."""

    def __init__(
        self,
        run_id: str,
        annotator_id: str,
        annotator_name: str,
        builder: HumanAnnotationResultsBuilder,
        sample_ids: list[str],
    ):
        """Initialize session.

        Args:
            run_id: Unique run identifier.
            annotator_id: Sanitized annotator ID.
            annotator_name: Original annotator name.
            builder: Results builder instance.
            sample_ids: Ordered list of sample IDs.
        """
        self.run_id = run_id
        self.annotator_id = annotator_id
        self.annotator_name = annotator_name
        self.builder = builder
        self.sample_ids = sample_ids


class SessionStore:
    """Manages annotation sessions backed by HumanAnnotationResultsBuilder."""

    def __init__(
        self,
        eval_task: EvalTask,
        eval_data: EvalData,
        annotations_dir: str,
    ):
        """Initialize session store.

        Args:
            eval_task: Task configuration.
            eval_data: Evaluation data.
            annotations_dir: Directory for saving annotations.
        """
        self._eval_task = eval_task
        self._eval_data = eval_data
        self._annotations_dir = annotations_dir
        self._sessions: dict[str, _Session] = {}

        os.makedirs(self._annotations_dir, exist_ok=True)

    def _get_sample_ids(self) -> list[str]:
        """Get all sample IDs from eval_data.

        Returns:
            list[str]: Ordered sample IDs.
        """
        id_col = self._eval_data.id_column
        assert id_col is not None
        return [str(v) for v in self._eval_data.data[id_col].to_list()]

    def _find_auto_save(self, annotator_name: str) -> str | None:
        """Find existing auto-save file for annotator.

        Args:
            annotator_name: Annotator name to search for.

        Returns:
            str | None: Path to most recent auto-save, or None.
        """
        pattern = os.path.join(
            self._annotations_dir,
            f"autosave_*_*_{annotator_name}_data.parquet",
        )
        matching = glob.glob(pattern)
        if matching:
            return max(matching, key=os.path.getmtime)
        return None

    def _auto_save(self, session: _Session) -> None:
        """Save current annotations to parquet.

        Args:
            session: Session to auto-save.
        """
        results = session.builder._results.copy()
        if not results:
            return
        row_dicts = [row.model_dump() for row in results.values()]
        df = pl.DataFrame(row_dicts)
        filepath = os.path.join(
            self._annotations_dir,
            f"autosave_{session.run_id}_{session.annotator_id}_{session.annotator_name}_data.parquet",
        )
        df.write_parquet(filepath)

    def create_session(self, annotator_name: str) -> CreateSessionResponse:
        """Create or resume an annotation session.

        Args:
            annotator_name: Name of the annotator.

        Returns:
            CreateSessionResponse: Session info.

        Raises:
            ValueError: If annotator_name is empty.
        """
        if not annotator_name or not annotator_name.strip():
            raise ValueError("Annotator name cannot be empty")

        run_id = _generate_run_id()
        annotator_id = _generate_annotator_id(annotator_name)
        sample_ids = self._get_sample_ids()
        required_tasks = self._eval_task.get_required_tasks()

        expected_ids: list[str | int] = list(sample_ids)
        builder = HumanAnnotationResultsBuilder(
            run_id=run_id,
            annotator_id=annotator_id,
            task_schemas=self._eval_task.task_schemas,
            expected_ids=expected_ids,
            required_tasks=required_tasks,
            is_sampled_run=False,
        )

        resumed = False
        annotated_count = 0
        auto_save_file = self._find_auto_save(annotator_name)
        if auto_save_file:
            try:
                saved_df = pl.read_parquet(auto_save_file)
                for row in saved_df.iter_rows(named=True):
                    if row.get("status") == "success":
                        outcomes = {
                            k: v
                            for k, v in row.items()
                            if k in self._eval_task.task_schemas and v is not None
                        }
                        original_id = str(row["original_id"])
                        if original_id in builder._results:
                            del builder._results[original_id]
                        builder.create_success_row(
                            sample_example_id=row["sample_example_id"],
                            original_id=original_id,
                            outcomes=outcomes,
                            annotation_timestamp=row.get("annotation_timestamp"),
                        )
                resumed = True
                annotated_count = builder.completed_count
            except Exception as e:
                logger.warning(f"Failed to load auto-save: {e}")

        session = _Session(
            run_id=run_id,
            annotator_id=annotator_id,
            annotator_name=annotator_name,
            builder=builder,
            sample_ids=sample_ids,
        )
        self._sessions[run_id] = session

        return CreateSessionResponse(
            run_id=run_id,
            annotator_id=annotator_id,
            total_samples=len(sample_ids),
            resumed=resumed,
            annotated_count=annotated_count,
        )

    def get_session(self, run_id: str) -> CreateSessionResponse | None:
        """Get session info by run_id.

        Args:
            run_id: Session run ID.

        Returns:
            CreateSessionResponse | None: Session info or None if not found.
        """
        session = self._sessions.get(run_id)
        if session is None:
            return None
        return CreateSessionResponse(
            run_id=session.run_id,
            annotator_id=session.annotator_id,
            total_samples=len(session.sample_ids),
            resumed=False,
            annotated_count=session.builder.completed_count,
        )

    def get_sample(self, run_id: str, index: int) -> SampleResponse:
        """Get sample data at index.

        Args:
            run_id: Session run ID.
            index: Sample index.

        Returns:
            SampleResponse: Sample data.

        Raises:
            KeyError: If session not found.
            IndexError: If index out of bounds.
        """
        session = self._sessions.get(run_id)
        if session is None:
            raise KeyError(f"Session not found: {run_id}")
        if index < 0 or index >= len(session.sample_ids):
            raise IndexError(f"Sample index out of bounds: {index}")

        sample_id = session.sample_ids[index]
        df = self._eval_data.data
        id_col = self._eval_data.id_column
        assert id_col is not None
        row = df.filter(pl.col(id_col).cast(str) == sample_id)

        prompt_data = None
        if self._eval_task.prompt_columns:
            prompt_data = {
                col: str(row[col][0])
                for col in self._eval_task.prompt_columns
                if col in row.columns
            }
        response_data = {
            col: str(row[col][0])
            for col in self._eval_task.response_columns
            if col in row.columns
        }

        previous_annotation = None
        existing = session.builder._results.get(sample_id)
        if existing:
            previous_annotation = {
                k: getattr(existing, k)
                for k in self._eval_task.task_schemas
                if getattr(existing, k, None) is not None
            }

        return SampleResponse(
            index=index,
            total=len(session.sample_ids),
            sample_id=sample_id,
            prompt_data=prompt_data,
            response_data=response_data,
            previous_annotation=previous_annotation,
        )

    def submit_annotation(
        self, run_id: str, sample_index: int, outcomes: dict[str, str | list[str]]
    ) -> SubmitAnnotationResponse:
        """Submit annotation for a sample.

        Args:
            run_id: Session run ID.
            sample_index: Index of sample being annotated.
            outcomes: Task outcomes dict. A multi-label outcome is the full
                ordered vector (list); single-select and free-form are strings.

        Returns:
            SubmitAnnotationResponse: Submission result.

        Raises:
            KeyError: If session not found.
            IndexError: If index out of bounds.
            AnnotationValidationError: If a multi-label value violates its schema.
        """
        session = self._sessions.get(run_id)
        if session is None:
            raise KeyError(f"Session not found: {run_id}")
        if sample_index < 0 or sample_index >= len(session.sample_ids):
            raise IndexError(f"Sample index out of bounds: {sample_index}")

        for task_name, value in outcomes.items():
            schema = self._eval_task.task_schemas.get(task_name)
            if isinstance(schema, MultiLabelSchema):
                try:
                    schema.validate_value(task_name, value)
                except ValueError as error:
                    raise AnnotationValidationError(task_name, error) from error

        sample_id = session.sample_ids[sample_index]

        if sample_id in session.builder._results:
            del session.builder._results[sample_id]

        session.builder.create_success_row(
            sample_example_id=f"{session.run_id}_{sample_index}",
            original_id=sample_id,
            outcomes=outcomes,
            annotation_timestamp=datetime.now(),
        )

        auto_saved = False
        try:
            self._auto_save(session)
            auto_saved = True
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")

        return SubmitAnnotationResponse(
            success=True,
            annotated_count=session.builder.completed_count,
            auto_saved=auto_saved,
        )

    def get_progress(self, run_id: str) -> ProgressResponse:
        """Get annotation progress.

        Args:
            run_id: Session run ID.

        Returns:
            ProgressResponse: Current progress.

        Raises:
            KeyError: If session not found.
        """
        session = self._sessions.get(run_id)
        if session is None:
            raise KeyError(f"Session not found: {run_id}")

        annotated_ids = set(session.builder._results.keys())
        incomplete = [
            i for i, sid in enumerate(session.sample_ids) if sid not in annotated_ids
        ]

        return ProgressResponse(
            run_id=run_id,
            annotated_count=session.builder.completed_count,
            total_samples=len(session.sample_ids),
            incomplete_indices=incomplete,
        )

    def export(self, run_id: str, annotator_name: str) -> ExportResponse:
        """Export completed annotations.

        Args:
            run_id: Session run ID.
            annotator_name: Annotator name for file naming.

        Returns:
            ExportResponse: Export result with file paths.

        Raises:
            KeyError: If session not found.
        """
        session = self._sessions.get(run_id)
        if session is None:
            raise KeyError(f"Session not found: {run_id}")

        results = session.builder.complete()

        metadata_filename = (
            f"{session.run_id}_{session.annotator_id}_{annotator_name}_metadata.json"
        )
        data_filename = (
            f"{session.run_id}_{session.annotator_id}_{annotator_name}_data.parquet"
        )

        results.save_state(
            state_file=os.path.join(self._annotations_dir, metadata_filename),
            data_format="parquet",
            data_filename=data_filename,
        )

        return ExportResponse(
            metadata_file=metadata_filename,
            data_file=data_filename,
            total_count=results.total_count,
            succeeded_count=results.succeeded_count,
            error_count=results.error_count,
        )
