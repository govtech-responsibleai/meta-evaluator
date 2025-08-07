"""Human annotation results implementation using the shared base classes."""

import json
import logging
from datetime import datetime
from typing import Dict, List, Literal, Optional, cast

import polars as pl
from pydantic import Field, ValidationError

from ..common.error_constants import (
    INVALID_JSON_MSG,
    INVALID_JSON_STRUCTURE_MSG,
    STATE_FILE_NOT_FOUND_MSG,
)
from .base import (
    BaseEvaluationResults,
    BaseEvaluationResultsBuilder,
)
from .enums import HumanAnnotationStatusEnum
from .exceptions import (
    IncompleteResultsError,
    InvalidFileError,
    MismatchedTasksError,
)
from .models import BaseResultRow, HumanAnnotationResultRow
from .serialization import (
    BaseResultsSerializedState,
    HumanAnnotationResultsSerializedState,
)

logger = logging.getLogger(__name__)


class HumanAnnotationResults(BaseEvaluationResults):
    """Immutable container for human annotation results from a single annotation run."""

    annotator_id: str = Field(
        ..., description="ID of the human annotator who performed the annotations."
    )
    error_count: int = Field(
        ..., description="Number of examples that failed due to unexpected errors."
    )

    def get_evaluator_id(self) -> str:
        """Get the ID of the evaluator (annotator_id).

        Returns:
            str: The annotator ID.
        """
        return self.annotator_id

    def get_error_count(self) -> int:
        """Get the total error count for this evaluation.

        Returns:
            int: The total error count.
        """
        return self.error_count

    def get_failed_results(self) -> pl.DataFrame:
        """Get all failed annotation results.

        Returns:
            pl.DataFrame: DataFrame containing only failed results.
        """
        return self.results_data.filter(
            pl.col("status") == HumanAnnotationStatusEnum.ERROR.value
        )

    def get_base_result_row_class(self) -> type[BaseResultRow]:
        """Get the base result row class for this evaluation type.

        Returns:
            type[BaseResultRow]: The HumanAnnotationResultRow class.
        """
        return HumanAnnotationResultRow

    def serialize(
        self,
        data_format: Literal["json", "csv", "parquet"],
        data_filename: str,
    ) -> HumanAnnotationResultsSerializedState:
        """Serialize HumanAnnotationResults to metadata.

        Args:
            data_format: Format for data serialization.
            data_filename: Name of data file.

        Returns:
            HumanAnnotationResultsSerializedState: Serialized state for HumanAnnotationResults.
        """
        return HumanAnnotationResultsSerializedState(
            run_id=self.run_id,
            annotator_id=self.annotator_id,
            task_schemas=self.task_schemas,
            timestamp_local=self.timestamp_local,
            total_count=self.total_count,
            succeeded_count=self.succeeded_count,
            error_count=self.error_count,
            is_sampled_run=self.is_sampled_run,
            data_file=data_filename,
            data_format=data_format,
        )

    @classmethod
    def deserialize(
        cls,
        results_data: pl.DataFrame,
        state: BaseResultsSerializedState,
    ) -> "HumanAnnotationResults":
        """Deserialize HumanAnnotationResults from serialized state.

        Args:
            results_data: The loaded DataFrame.
            state: Serialized state for HumanAnnotationResults.

        Returns:
            HumanAnnotationResults: Reconstructed HumanAnnotationResults instance.
        """
        human_state = cast(HumanAnnotationResultsSerializedState, state)
        return cls(
            run_id=human_state.run_id,
            annotator_id=human_state.annotator_id,
            task_schemas=human_state.task_schemas,
            timestamp_local=human_state.timestamp_local,
            total_count=human_state.total_count,
            succeeded_count=human_state.succeeded_count,
            error_count=human_state.error_count,
            is_sampled_run=human_state.is_sampled_run,
            results_data=results_data,
        )

    @classmethod
    def _load_json_state(cls, state_file: str) -> HumanAnnotationResultsSerializedState:
        """Load and validate JSON state file.

        Args:
            state_file: Path to the JSON state file.

        Returns:
            HumanAnnotationResultsSerializedState: The loaded and validated state object.

        Raises:
            InvalidFileError: If the state file doesn't exist or the JSON structure is invalid.
        """
        try:
            with open(state_file, "r") as f:
                return HumanAnnotationResultsSerializedState.model_validate_json(
                    f.read()
                )
        except FileNotFoundError as e:
            raise InvalidFileError(f"{STATE_FILE_NOT_FOUND_MSG}: {state_file}", e)
        except ValidationError as e:
            raise InvalidFileError(INVALID_JSON_STRUCTURE_MSG, e)
        except json.JSONDecodeError as e:
            raise InvalidFileError(INVALID_JSON_MSG, e)

    def get_successful_results(self) -> pl.DataFrame:
        """Get all successful annotation results.

        Returns:
            pl.DataFrame: DataFrame containing only successful results.
        """
        return self.results_data.filter(
            pl.col("status") == HumanAnnotationStatusEnum.SUCCESS.value
        )

    def _get_status_counts(self) -> Dict[str, int]:
        """Get status counts for count consistency validation.

        Returns:
            Dict[str, int]: Dictionary mapping status values to their counts.
        """
        return {
            HumanAnnotationStatusEnum.SUCCESS.value: self.succeeded_count,
            HumanAnnotationStatusEnum.ERROR.value: self.error_count,
        }

    def _get_valid_status_values(self) -> List[str]:
        """Get valid status values for this evaluation type.

        Returns:
            List[str]: List of valid status values.
        """
        return [status.value for status in HumanAnnotationStatusEnum]


class HumanAnnotationResultsBuilder(BaseEvaluationResultsBuilder):
    """Builder for HumanAnnotationResults with annotation-specific functionality."""

    def __init__(
        self,
        run_id: str,
        annotator_id: str,
        task_schemas: Dict[str, List[str] | None],
        expected_ids: List[str | int],
        is_sampled_run: bool = False,
    ):
        """Initialize the human annotation results builder.

        Args:
            run_id: Unique identifier for this evaluation run.
            annotator_id: ID of the human annotator.
            task_schemas: Dictionary mapping task names to their allowed outcome values.
            expected_ids: List of expected original IDs.
            is_sampled_run: True if input was sampled data.
        """
        super().__init__(
            run_id, annotator_id, task_schemas, expected_ids, is_sampled_run
        )

    def create_success_row(
        self,
        sample_example_id: str,
        original_id: str | int,
        outcomes: dict[str, str],
        annotation_timestamp: datetime | None = None,
        **kwargs,
    ) -> HumanAnnotationResultRow:
        """Create a success row for human annotation.

        Args:
            sample_example_id: Unique identifier for the example within this run.
            original_id: Original identifier from the source data.
            outcomes: Dictionary of task outcomes.
            annotation_timestamp: Timestamp when annotation was completed.
            **kwargs: Additional keyword arguments

        Returns:
            HumanAnnotationResultRow: The created success row.

        Raises:
            MismatchedTasksError: If outcomes do not contain all tasks.
        """
        # Validate that outcomes contain exactly all tasks
        expected_tasks = set(self.task_schemas.keys())
        outcome_tasks = set(outcomes.keys())

        if expected_tasks != outcome_tasks:
            raise MismatchedTasksError(
                list(expected_tasks - outcome_tasks),
                "Success row must contain outcomes for ALL tasks",
            )

        row = HumanAnnotationResultRow(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.run_id,
            annotator_id=self.evaluator_id,
            status=HumanAnnotationStatusEnum.SUCCESS.value,
            error_message=None,
            error_details_json=None,
            annotation_timestamp=annotation_timestamp or datetime.now(),
            **outcomes,
        )
        self._validate_and_store(row)
        return row

    def create_error_row(
        self,
        sample_example_id: str,
        original_id: str | int,
        error_message: str,
        error_details_json: Optional[str] = None,
        annotation_timestamp: Optional[datetime] = None,
    ) -> HumanAnnotationResultRow:
        """Create an error row for human annotation.

        Args:
            sample_example_id: Unique identifier for the example within this run.
            original_id: Original identifier from the source data.
            error_message: Error message describing the failure.
            error_details_json: Optional JSON string with error details.
            annotation_timestamp: Timestamp when annotation was attempted.

        Returns:
            HumanAnnotationResultRow: The created error row.
        """
        # Set all task outcomes to None
        outcomes = {task_name: None for task_name in self.task_schemas.keys()}

        row = HumanAnnotationResultRow(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.run_id,
            annotator_id=self.evaluator_id,
            status=HumanAnnotationStatusEnum.ERROR.value,
            error_message=error_message,
            error_details_json=error_details_json,
            annotation_timestamp=annotation_timestamp or datetime.now(),
            **outcomes,
        )
        self._validate_and_store(row)
        return row

    def complete(self) -> HumanAnnotationResults:
        """Complete the building process and return the HumanAnnotationResults.

        Returns:
            HumanAnnotationResults: The completed annotation results.

        Raises:
            IncompleteResultsError: If no rows were added to the builder or missing expected results.
        """
        if not self._results:
            raise IncompleteResultsError("No rows added to builder")

        # Check for missing results
        if not self.is_complete:
            received_ids = set(self._results.keys())
            expected_ids = self._expected_ids
            missing_ids = expected_ids - received_ids
            raise IncompleteResultsError(
                f"Missing results for IDs: {sorted(missing_ids)}"
            )

        # Create DataFrame
        results_data = self._create_dataframe()

        # Calculate counts
        status_count_map = self._calculate_counts()

        return HumanAnnotationResults(
            run_id=self.run_id,
            annotator_id=self.evaluator_id,
            task_schemas=self.task_schemas,
            timestamp_local=datetime.now(),
            total_count=self.total_count,
            succeeded_count=status_count_map.get(
                HumanAnnotationStatusEnum.SUCCESS.value, 0
            ),
            error_count=status_count_map.get(HumanAnnotationStatusEnum.ERROR.value, 0),
            is_sampled_run=self.is_sampled_run,
            results_data=results_data,
        )
