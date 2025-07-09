"""Judge results implementation using the shared base classes."""

from enum import Enum
import json
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
from ..llm_client import LLMClientEnum


logger = logging.getLogger(__name__)


class EvaluationStatusEnum(str, Enum):
    """Enumeration of possible evaluation outcomes for a single example."""

    SUCCESS = "success"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    LLM_ERROR = "llm_error"
    PARSING_ERROR = "parsing_error"
    OTHER_ERROR = "other_error"


class JudgeResultRow(BaseResultRow):
    """Result row for Judge evaluation with LLM-specific fields."""

    judge_id: Annotated[
        str,
        Field(description="ID of the judge configuration used"),
        FieldTags(tags=["metadata"]),
    ]

    llm_raw_response_content: Annotated[
        Optional[str],
        Field(default=None, description="Raw response content from LLM"),
        FieldTags(tags=["llm_diagnostic"]),
    ]

    llm_prompt_tokens: Annotated[
        Optional[int],
        Field(default=None, description="Number of tokens used in the prompt"),
        FieldTags(tags=["llm_diagnostic"]),
    ]

    llm_completion_tokens: Annotated[
        Optional[int],
        Field(default=None, description="Number of tokens used in the completion"),
        FieldTags(tags=["llm_diagnostic"]),
    ]

    llm_total_tokens: Annotated[
        Optional[int],
        Field(default=None, description="Total number of tokens used"),
        FieldTags(tags=["llm_diagnostic"]),
    ]

    llm_call_duration_seconds: Annotated[
        Optional[float],
        Field(default=None, description="Duration of the LLM call in seconds"),
        FieldTags(tags=["llm_diagnostic"]),
    ]

    model_config = ConfigDict(extra="allow", frozen=False)

    @classmethod
    def get_llm_diagnostic_fields(cls) -> list[str]:
        """Get all LLM diagnostic field names.

        Returns:
            list[str]: List of LLM diagnostic field names.
        """
        return cls.get_fields_by_tag("llm_diagnostic")


class JudgeResults(BaseEvaluationResults):
    """Immutable container for evaluation results from a single Judge run."""

    judge_id: str = Field(
        ..., description="ID of the Judge configuration used for this run."
    )
    llm_client_enum: LLMClientEnum = Field(
        ..., description="The LLM client provider used for this run."
    )
    model_used: str = Field(..., description="Name of the LLM model used for this run.")
    skipped_count: int = Field(
        ...,
        description="Number of examples skipped by the evaluation task's skip function.",
    )
    partial_count: int = Field(
        ...,
        description="Number of examples where LLM call succeeded but only some task outcomes were parsed successfully.",
    )
    llm_error_count: int = Field(
        ..., description="Number of examples that failed due to an LLM API error."
    )
    parsing_error_count: int = Field(
        ...,
        description="Number of examples where LLM call succeeded but response parsing failed completely.",
    )
    other_error_count: int = Field(
        ...,
        description="Number of examples that failed due to unexpected errors not covered by other categories.",
    )

    def get_evaluator_id(self) -> str:
        """Get the ID of the evaluator (judge_id).

        Returns:
            str: The judge ID.
        """
        return self.judge_id

    def get_error_count(self) -> int:
        """Get the total error count for this evaluation.

        Returns:
            int: The total error count.
        """
        return self.llm_error_count + self.parsing_error_count + self.other_error_count

    def get_failed_results(self) -> pl.DataFrame:
        """Get all failed evaluation results.

        Returns:
            pl.DataFrame: DataFrame containing only failed results.
        """
        return self.results_data.filter(
            pl.col("status").is_in(
                [
                    EvaluationStatusEnum.LLM_ERROR.value,
                    EvaluationStatusEnum.PARSING_ERROR.value,
                    EvaluationStatusEnum.OTHER_ERROR.value,
                ]
            )
        )

    def get_base_result_row_class(self) -> type[BaseResultRow]:
        """Get the base result row class for this evaluation type.

        Returns:
            type[BaseResultRow]: The JudgeResultRow class.
        """
        return JudgeResultRow

    def get_successful_results(self) -> pl.DataFrame:
        """Get all successful evaluation results.

        Returns:
            pl.DataFrame: DataFrame containing only successful results.
        """
        return self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.SUCCESS.value
        )

    def get_partial_results(self) -> pl.DataFrame:
        """Get all partial evaluation results.

        Returns:
            pl.DataFrame: DataFrame containing only partial results.
        """
        return self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.PARTIAL.value
        )

    def get_skipped_results(self) -> pl.DataFrame:
        """Get all skipped evaluation results.

        Returns:
            pl.DataFrame: DataFrame containing only skipped results.
        """
        return self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.SKIPPED.value
        )

    def _get_status_counts(self) -> Dict[str, int]:
        """Get status counts for count consistency validation.

        Returns:
            Dict[str, int]: Dictionary mapping status values to their counts.
        """
        return {
            EvaluationStatusEnum.SUCCESS.value: self.succeeded_count,
            EvaluationStatusEnum.PARTIAL.value: self.partial_count,
            EvaluationStatusEnum.SKIPPED.value: self.skipped_count,
            EvaluationStatusEnum.LLM_ERROR.value: self.llm_error_count,
            EvaluationStatusEnum.PARSING_ERROR.value: self.parsing_error_count,
            EvaluationStatusEnum.OTHER_ERROR.value: self.other_error_count,
        }

    def _get_valid_status_values(self) -> List[str]:
        """Get valid status values for this evaluation type.

        Returns:
            List[str]: List of valid status values.
        """
        return [status.value for status in EvaluationStatusEnum]

    def _validate_success_constraints(
        self, success_df: pl.DataFrame, task_names: List[str]
    ) -> None:
        """Validate constraints for success status rows.

        Args:
            success_df: DataFrame containing only success status rows
            task_names: List of task names

        Raises:
            ValueError: If success constraint validation fails.
        """
        # Call parent validation first
        super()._validate_success_constraints(success_df, task_names)

        # Additional judge-specific validation: LLM diagnostic columns MUST NOT be null
        llm_diagnostic_fields = JudgeResultRow.get_llm_diagnostic_fields()
        for llm_field in llm_diagnostic_fields:
            if success_df[llm_field].is_null().any():
                raise ValueError(
                    f"LLM diagnostic field '{llm_field}' for SUCCESS status contains null values."
                )


class JudgeResultsBuilder(BaseEvaluationResultsBuilder):
    """Builder for JudgeResults with LLM-specific functionality."""

    def __init__(
        self,
        run_id: str,
        judge_id: str,
        llm_client_enum: LLMClientEnum,
        model_used: str,
        task_schemas: Dict[str, List[str] | None],
        expected_ids: List[str | int],
        is_sampled_run: bool = False,
    ):
        """Initialize the judge results builder.

        Args:
            run_id: Unique identifier for this evaluation run.
            judge_id: ID of the judge configuration.
            llm_client_enum: The LLM client provider used.
            model_used: Name of the LLM model used.
            task_schemas: Dictionary mapping task names to their allowed outcome values.
            expected_ids: List of expected original IDs.
            is_sampled_run: True if input was sampled data.
        """
        super().__init__(run_id, judge_id, task_schemas, expected_ids, is_sampled_run)
        self.llm_client_enum = llm_client_enum
        self.model_used = model_used

    def create_success_row(
        self,
        sample_example_id: str,
        original_id: str | int,
        outcomes: dict[str, str],
        llm_raw_response_content: str | None = None,
        llm_prompt_tokens: int | None = None,
        llm_completion_tokens: int | None = None,
        llm_total_tokens: int | None = None,
        llm_call_duration_seconds: float | None = None,
        **kwargs,
    ) -> JudgeResultRow:
        """Create a success row for judge evaluation.

        Args:
            sample_example_id: Unique identifier for the example within this run.
            original_id: Original identifier from the source data.
            outcomes: Dictionary of task outcomes.
            llm_raw_response_content: Raw response content from LLM.
            llm_prompt_tokens: Number of tokens used in the prompt.
            llm_completion_tokens: Number of tokens used in the completion.
            llm_total_tokens: Total number of tokens used.
            llm_call_duration_seconds: Duration of the LLM call in seconds.
            **kwargs: Additional keyword arguments

        Returns:
            JudgeResultRow: The created success row.

        Raises:
            ValueError: If outcomes don't contain exactly all expected tasks.
        """
        # Validate that outcomes contain exactly all tasks
        expected_tasks = set(self.task_schemas.keys())
        outcome_tasks = set(outcomes.keys())

        if expected_tasks != outcome_tasks:
            raise ValueError("Success row must contain outcomes for ALL tasks")

        row = JudgeResultRow(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.run_id,
            judge_id=self.evaluator_id,
            status=EvaluationStatusEnum.SUCCESS.value,
            error_message=None,
            error_details_json=None,
            llm_raw_response_content=llm_raw_response_content,
            llm_prompt_tokens=llm_prompt_tokens,
            llm_completion_tokens=llm_completion_tokens,
            llm_total_tokens=llm_total_tokens,
            llm_call_duration_seconds=llm_call_duration_seconds,
            **outcomes,
        )
        self._validate_and_store(row)
        return row

    def create_partial_row(
        self,
        sample_example_id: str,
        original_id: str | int,
        outcomes: dict[str, str],
        error_message: str,
        llm_raw_response_content: str,
        llm_prompt_tokens: int,
        llm_completion_tokens: int,
        llm_total_tokens: int,
        llm_call_duration_seconds: float,
    ) -> JudgeResultRow:
        """Create a partial row for judge evaluation.

        Args:
            sample_example_id: Unique identifier for the example within this run.
            original_id: Original identifier from the source data.
            outcomes: Dictionary of task outcomes.
            error_message: Error message describing the parsing issue.
            llm_raw_response_content: Raw response content from LLM.
            llm_prompt_tokens: Number of tokens used in the prompt.
            llm_completion_tokens: Number of tokens used in the completion.
            llm_total_tokens: Total number of tokens used.
            llm_call_duration_seconds: Duration of the LLM call in seconds.

        Returns:
            JudgeResultRow: The created partial row.

        Raises:
            ValueError: If outcomes contain invalid task names.
        """
        # Validate that all outcome task names are valid
        expected_tasks = set(self.task_schemas.keys())
        invalid_tasks = set(outcomes.keys()) - expected_tasks

        if invalid_tasks:
            raise ValueError(f"Invalid task names: {invalid_tasks}")

        row = JudgeResultRow(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.run_id,
            judge_id=self.evaluator_id,
            status=EvaluationStatusEnum.PARTIAL.value,
            error_message=error_message,
            error_details_json=None,
            llm_raw_response_content=llm_raw_response_content,
            llm_prompt_tokens=llm_prompt_tokens,
            llm_completion_tokens=llm_completion_tokens,
            llm_total_tokens=llm_total_tokens,
            llm_call_duration_seconds=llm_call_duration_seconds,
            **outcomes,
        )
        self._validate_and_store(row)
        return row

    def create_skipped_row(
        self,
        sample_example_id: str,
        original_id: str | int,
    ) -> JudgeResultRow:
        """Create a skipped row for judge evaluation.

        Args:
            sample_example_id: Unique identifier for the example within this run.
            original_id: Original identifier from the source data.

        Returns:
            JudgeResultRow: The created skipped row.
        """
        # Set all task outcomes to None
        outcomes = {task_name: None for task_name in self.task_schemas.keys()}

        row = JudgeResultRow(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.run_id,
            judge_id=self.evaluator_id,
            status=EvaluationStatusEnum.SKIPPED.value,
            error_message=None,
            error_details_json=None,
            llm_raw_response_content=None,
            llm_prompt_tokens=None,
            llm_completion_tokens=None,
            llm_total_tokens=None,
            llm_call_duration_seconds=None,
            **outcomes,
        )
        self._validate_and_store(row)
        return row

    def create_llm_error_row(
        self,
        sample_example_id: str,
        original_id: str | int,
        error: Exception,
    ) -> None:
        """Create and store an LLM error row.

        Args:
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
            error: The exception that occurred during LLM call
        """
        # Set all task outcomes to None
        outcomes = {task_name: None for task_name in self.task_schemas.keys()}

        row = JudgeResultRow(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.run_id,
            judge_id=self.evaluator_id,
            error_message=str(error),
            error_details_json=json.dumps(
                {"error": str(error), "type": type(error).__name__}
            ),
            status=EvaluationStatusEnum.LLM_ERROR,
            llm_raw_response_content=None,
            llm_prompt_tokens=None,
            llm_completion_tokens=None,
            llm_total_tokens=None,
            llm_call_duration_seconds=None,
            **outcomes,
        )
        self._validate_and_store(row)

    def create_parsing_error_row(
        self,
        sample_example_id: str,
        original_id: str | int,
        error: Exception,
        llm_raw_response_content: str,
        llm_prompt_tokens: int,
        llm_completion_tokens: int,
        llm_total_tokens: int,
        llm_call_duration_seconds: float,
    ) -> None:
        """Create and store a parsing error row.

        Args:
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
            error: The exception that occurred during parsing
            llm_raw_response_content: Raw response content from LLM
            llm_prompt_tokens: Number of prompt tokens used
            llm_completion_tokens: Number of completion tokens used
            llm_total_tokens: Total number of tokens used
            llm_call_duration_seconds: Duration of the LLM call in seconds
        """
        # Set all task outcomes to None
        outcomes = {task_name: None for task_name in self.task_schemas.keys()}

        row = JudgeResultRow(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.run_id,
            judge_id=self.evaluator_id,
            error_message=str(error),
            error_details_json=json.dumps(
                {"error": str(error), "type": type(error).__name__}
            ),
            status=EvaluationStatusEnum.PARSING_ERROR,
            llm_raw_response_content=llm_raw_response_content,
            llm_prompt_tokens=llm_prompt_tokens,
            llm_completion_tokens=llm_completion_tokens,
            llm_total_tokens=llm_total_tokens,
            llm_call_duration_seconds=llm_call_duration_seconds,
            **outcomes,
        )
        self._validate_and_store(row)

    def create_other_error_row(
        self,
        sample_example_id: str,
        original_id: str | int,
        error: Exception,
    ) -> None:
        """Create and store an other error row.

        Args:
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
            error: The exception that occurred
        """
        # Set all task outcomes to None
        outcomes = {task_name: None for task_name in self.task_schemas.keys()}

        row = JudgeResultRow(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.run_id,
            judge_id=self.evaluator_id,
            error_message=str(error),
            error_details_json=json.dumps(
                {"error": str(error), "type": type(error).__name__}
            ),
            status=EvaluationStatusEnum.OTHER_ERROR.value,
            llm_raw_response_content=None,
            llm_prompt_tokens=None,
            llm_completion_tokens=None,
            llm_total_tokens=None,
            llm_call_duration_seconds=None,
            **outcomes,
        )
        self._validate_and_store(row)

    def complete(self) -> JudgeResults:
        """Complete the building process and return the JudgeResults.

        Returns:
            JudgeResults: The completed judge results.

        Raises:
            ValueError: If no rows were added to the builder or missing expected results.
        """
        if not self._results:
            raise ValueError("No rows added to builder")

        # Check for missing results
        if not self.is_complete:
            received_ids = set(self._results.keys())
            expected_ids = self._expected_ids
            missing_ids = expected_ids - received_ids
            raise ValueError(f"Missing results for IDs: {sorted(missing_ids)}")

        # Create DataFrame
        results_data = self._create_dataframe()

        # Calculate counts
        status_count_map = self._calculate_counts()

        return JudgeResults(
            run_id=self.run_id,
            judge_id=self.evaluator_id,
            task_schemas=self.task_schemas,
            llm_client_enum=self.llm_client_enum,
            model_used=self.model_used,
            timestamp_local=datetime.now(),
            total_count=self.total_count,
            succeeded_count=status_count_map.get(EvaluationStatusEnum.SUCCESS.value, 0),
            skipped_count=status_count_map.get(EvaluationStatusEnum.SKIPPED.value, 0),
            partial_count=status_count_map.get(EvaluationStatusEnum.PARTIAL.value, 0),
            llm_error_count=status_count_map.get(
                EvaluationStatusEnum.LLM_ERROR.value, 0
            ),
            parsing_error_count=status_count_map.get(
                EvaluationStatusEnum.PARSING_ERROR.value, 0
            ),
            other_error_count=status_count_map.get(
                EvaluationStatusEnum.OTHER_ERROR.value, 0
            ),
            is_sampled_run=self.is_sampled_run,
            results_data=results_data,
        )
