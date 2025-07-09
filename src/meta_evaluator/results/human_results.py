"""Human annotation results implementation using the shared base classes."""

from enum import Enum
import polars as pl
from pydantic import Field, ConfigDict
from datetime import datetime
from typing import List, Optional, Annotated, Dict
import logging

from .base import (
    BaseEvaluationResults,
    BaseEvaluationResultsBuilder,
    BaseResultRow,
    FieldTags,
)


logger = logging.getLogger(__name__)


class HumanAnnotationStatusEnum(str, Enum):
    """Enumeration of possible human annotation outcomes for a single example."""

    SUCCESS = "success"
    ERROR = "error"


class HumanAnnotationResultRow(BaseResultRow):
    """Result row for human annotation with annotation-specific fields."""

    annotator_id: Annotated[
        str,
        Field(description="ID of the human annotator"),
        FieldTags(tags=["metadata"]),
    ]

    annotation_timestamp: Annotated[
        Optional[datetime],
        Field(default=None, description="Timestamp when annotation was completed"),
        FieldTags(tags=["annotation_diagnostic"]),
    ]

    model_config = ConfigDict(extra="allow", frozen=False)

    @classmethod
    def get_annotation_diagnostic_fields(cls) -> list[str]:
        """Get all annotation diagnostic field names.

        Returns:
            list[str]: List of annotation diagnostic field names.
        """
        return cls.get_fields_by_tag("annotation_diagnostic")


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
        is_sampled_run: bool = False,
        expected_ids: Optional[List[str | int]] = None,
    ):
        """Initialize the human annotation results builder.

        Args:
            run_id: Unique identifier for this evaluation run.
            annotator_id: ID of the human annotator.
            task_schemas: Dictionary mapping task names to their allowed outcome values.
            is_sampled_run: True if input was sampled data.
            expected_ids: Optional list of expected original IDs.
        """
        super().__init__(
            run_id, annotator_id, task_schemas, is_sampled_run, expected_ids
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
        """
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
            ValueError: If no rows were added to the builder.
        """
        if not self._results:
            raise ValueError("No rows added to builder")

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
