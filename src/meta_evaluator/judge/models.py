"""Models used by the judge module."""

from enum import Enum
from types import UnionType
import polars as pl
from pydantic import BaseModel, Field, ConfigDict, model_validator
from datetime import datetime
from typing import List, Any
import logging
import json

from ..llm_client import LLMClientEnum

_BASE_REQUIRED_COLUMNS: list[tuple[str, type | UnionType]] = [
    ("sample_example_id", str),
    ("original_id", str),
    ("run_id", str),
    ("judge_id", str),
    ("task_name", str),
    ("status", str),
    ("error_type", str | None),
    ("error_message", str | None),
    ("error_details_json", str | None),
    ("llm_raw_response_content", str | None),
    ("llm_prompt_tokens", int | None),
    ("llm_completion_tokens", int | None),
    ("llm_total_tokens", int | None),
    ("llm_call_duration_seconds", float | None),
]


class EvaluationStatusEnum(str, Enum):
    """Enumeration of possible evaluation outcomes for a single example."""

    SUCCESS = "success"
    SKIPPED = "skipped"
    LLM_ERROR = "llm_error"
    PARSING_ERROR = "parsing_error"
    OTHER_ERROR = "other_error"


logger = logging.getLogger(__name__)


class JudgeResults(BaseModel):
    """Immutable container for evaluation results from a single Judge run.

    This class stores the detailed outcomes for each example from an EvalData
    instance after being processed by a specific Judge configuration. It also
    includes summary metadata about the entire run.

    The primary data is stored in a Polars DataFrame, where each row corresponds
    to an example from the original EvalData (linked by 'original_id' and identified
    uniquely within the run by 'sample_example_id'), and additional columns represent
    the evaluation status, outcomes, and any errors.

    Attributes:
        run_id (str): Unique identifier for this specific evaluation run.
        judge_id (str): ID of the Judge configuration used for this run.
        task_name (str): Name of the evaluation task performed.
        task_outcomes (list[str]): List of possible outcomes defined by the evaluation task.
        llm_client_enum (LLMClientEnum): The LLM client provider used for this run.
        model_used (str): Name of the LLM model used for this run.
        timestamp_local (datetime): Local timestamp of when this evaluation run completed.
        total_examples_count (int): Total number of examples from the input EvalData that were attempted for this run.
        skipped_examples_count (int): Number of examples skipped by the evaluation task's skip function.
        succeeded_examples_count (int): Number of examples where LLM call and parsing were fully successful.
        llm_error_examples_count (int): Number of examples that failed due to an LLM API error.
        parsing_error_examples_count (int): Number of examples where LLM call succeeded but response parsing failed.
        other_error_examples_count (int): Number of examples that failed due to unexpected errors not covered by other categories.
        is_sampled_run (bool): True if the EvalData provided for this run was a SampledEvalData instance.
        results_data (pl.DataFrame): A Polars DataFrame containing per-example results.

    model_config (ConfigDict): Pydantic configuration dictionary.
        - `frozen=True`: Makes the JudgeResults instance immutable after creation.
        - `arbitrary_types_allowed=True`: Allows Polars DataFrame as an attribute.

    Validation:
        - Ensures all top-level metadata attributes are present and correctly typed.
        - Ensures `results_data` is a Polars DataFrame and is not empty.
        - Verifies count consistency: `total_examples_count` equals sum of status counts.
        - Validates presence of all required `results_data` columns.
        - Checks that 'sample_example_id' column contains unique values.
        - Validates 'status' column contains only valid `EvaluationStatusEnum` values.
        - Enforces conditional null/non-null states for columns based on 'status'.
    """

    # Schema definition: (column_name, type)
    run_id: str = Field(
        ..., description="Unique identifier for this specific evaluation run."
    )
    judge_id: str = Field(
        ..., description="ID of the Judge configuration used for this run."
    )
    task_name: str = Field(..., description="Name of the evaluation task performed.")
    task_outcomes: List[str] = Field(
        ..., description="List of possible outcomes defined by the evaluation task."
    )
    llm_client_enum: LLMClientEnum = Field(
        ..., description="The LLM client provider used for this run."
    )
    model_used: str = Field(..., description="Name of the LLM model used for this run.")
    timestamp_local: datetime = Field(
        ..., description="Local timestamp of when this evaluation run completed."
    )
    total_examples_count: int = Field(
        ...,
        description="Total number of examples from the input EvalData that were attempted for this run.",
    )
    skipped_examples_count: int = Field(
        ...,
        description="Number of examples skipped by the evaluation task's skip function.",
    )
    succeeded_examples_count: int = Field(
        ...,
        description="Number of examples where LLM call and parsing were fully successful.",
    )
    llm_error_examples_count: int = Field(
        ..., description="Number of examples that failed due to an LLM API error."
    )
    parsing_error_examples_count: int = Field(
        ...,
        description="Number of examples where LLM call succeeded but response parsing failed.",
    )
    other_error_examples_count: int = Field(
        ...,
        description="Number of examples that failed due to unexpected errors not covered by other categories.",
    )
    is_sampled_run: bool = Field(
        ...,
        description="True if the EvalData provided for this run was a SampledEvalData instance.",
    )
    results_data: pl.DataFrame = Field(
        ..., description="A Polars DataFrame containing per-example results."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @classmethod
    def get_required_columns(cls, task_name: str) -> list[tuple[str, type | UnionType]]:
        """Get full column schema including dynamic outcome column.

        Args:
            task_name: Name of the evaluation task, used as the dynamic outcome column name.

        Returns:
            List of (column_name, type) tuples defining the complete DataFrame schema.
        """
        return _BASE_REQUIRED_COLUMNS + [(task_name, str)]

    @classmethod
    def create_empty_row_dict(cls, task_name: str) -> dict[str, Any]:
        """Create properly typed empty dict from schema.

        Args:
            task_name: Name of the evaluation task, used as the dynamic outcome column name.

        Returns:
            Dictionary with all required columns initialized to None.
        """
        schema = cls.get_required_columns(task_name)
        return {col_name: None for col_name, col_type in schema}

    @classmethod
    def create_success_row(
        cls,
        task_name: str,
        sample_example_id: str,
        original_id: str,
        run_id: str,
        judge_id: str,
        outcome: str,
        llm_raw_response: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        call_duration: float,
    ) -> dict[str, Any]:
        """Create a properly structured success row dict.

        Args:
            task_name: Name of the evaluation task
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
            run_id: Unique identifier for the evaluation run
            judge_id: ID of the judge configuration
            outcome: The evaluation outcome/result
            llm_raw_response: Raw response content from LLM
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            total_tokens: Total number of tokens used
            call_duration: Duration of the LLM call in seconds

        Returns:
            Dictionary with all fields properly set for SUCCESS status
        """
        row = cls.create_empty_row_dict(task_name)
        row.update(
            {
                "sample_example_id": sample_example_id,
                "original_id": original_id,
                "run_id": run_id,
                "judge_id": judge_id,
                "task_name": task_name,
                "status": EvaluationStatusEnum.SUCCESS.value,
                task_name: outcome,  # Dynamic outcome column
                # Error fields stay None
                "error_type": None,
                "error_message": None,
                "error_details_json": None,
                # LLM diagnostic fields are populated
                "llm_raw_response_content": llm_raw_response,
                "llm_prompt_tokens": prompt_tokens,
                "llm_completion_tokens": completion_tokens,
                "llm_total_tokens": total_tokens,
                "llm_call_duration_seconds": call_duration,
            }
        )
        return row

    @classmethod
    def create_skipped_row(
        cls,
        task_name: str,
        sample_example_id: str,
        original_id: str,
        run_id: str,
        judge_id: str,
    ) -> dict[str, Any]:
        """Create a properly structured skipped row dict.

        Args:
            task_name: Name of the evaluation task
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
            run_id: Unique identifier for the evaluation run
            judge_id: ID of the judge configuration

        Returns:
            Dictionary with all fields properly set for SKIPPED status
        """
        row = cls.create_empty_row_dict(task_name)
        row.update(
            {
                "sample_example_id": sample_example_id,
                "original_id": original_id,
                "run_id": run_id,
                "judge_id": judge_id,
                "task_name": task_name,
                "status": EvaluationStatusEnum.SKIPPED.value,
                # Outcome column stays None
                task_name: None,
                # Error fields stay None
                "error_type": None,
                "error_message": None,
                "error_details_json": None,
                # LLM diagnostic fields stay None
                "llm_raw_response_content": None,
                "llm_prompt_tokens": None,
                "llm_completion_tokens": None,
                "llm_total_tokens": None,
                "llm_call_duration_seconds": None,
            }
        )
        return row

    @classmethod
    def create_llm_error_row(
        cls,
        task_name: str,
        sample_example_id: str,
        original_id: str,
        run_id: str,
        judge_id: str,
        error: Exception,
    ) -> dict[str, Any]:
        """Create a properly structured LLM error row dict.

        Args:
            task_name: Name of the evaluation task
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
            run_id: Unique identifier for the evaluation run
            judge_id: ID of the judge configuration
            error: The exception that occurred during LLM call

        Returns:
            Dictionary with all fields properly set for LLM_ERROR status
        """
        row = cls.create_empty_row_dict(task_name)
        row.update(
            {
                "sample_example_id": sample_example_id,
                "original_id": original_id,
                "run_id": run_id,
                "judge_id": judge_id,
                "task_name": task_name,
                "status": EvaluationStatusEnum.LLM_ERROR.value,
                # Outcome column stays None
                task_name: None,
                # Error fields are populated
                "error_type": "LLM_API_ERROR",
                "error_message": str(error),
                "error_details_json": json.dumps(
                    {"error": str(error), "type": type(error).__name__}
                ),
                # LLM diagnostic fields stay None (call failed)
                "llm_raw_response_content": None,
                "llm_prompt_tokens": None,
                "llm_completion_tokens": None,
                "llm_total_tokens": None,
                "llm_call_duration_seconds": None,
            }
        )
        return row

    @classmethod
    def create_parsing_error_row(
        cls,
        task_name: str,
        sample_example_id: str,
        original_id: str,
        run_id: str,
        judge_id: str,
        error: Exception,
        llm_raw_response: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        call_duration: float,
    ) -> dict[str, Any]:
        """Create a properly structured parsing error row dict.

        Args:
            task_name: Name of the evaluation task
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
            run_id: Unique identifier for the evaluation run
            judge_id: ID of the judge configuration
            error: The exception that occurred during response parsing
            llm_raw_response: Raw response content from LLM
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            total_tokens: Total number of tokens used
            call_duration: Duration of the LLM call in seconds

        Returns:
            Dictionary with all fields properly set for PARSING_ERROR status
        """
        row = cls.create_empty_row_dict(task_name)
        row.update(
            {
                "sample_example_id": sample_example_id,
                "original_id": original_id,
                "run_id": run_id,
                "judge_id": judge_id,
                "task_name": task_name,
                "status": EvaluationStatusEnum.PARSING_ERROR.value,
                # Outcome column stays None
                task_name: None,
                # Error fields are populated (but NOT "LLM_API_ERROR")
                "error_type": "PARSING_ERROR",
                "error_message": str(error),
                "error_details_json": json.dumps(
                    {"error": str(error), "type": type(error).__name__}
                ),
                # LLM diagnostic fields are populated (call succeeded)
                "llm_raw_response_content": llm_raw_response,
                "llm_prompt_tokens": prompt_tokens,
                "llm_completion_tokens": completion_tokens,
                "llm_total_tokens": total_tokens,
                "llm_call_duration_seconds": call_duration,
            }
        )
        return row

    @classmethod
    def create_other_error_row(
        cls,
        task_name: str,
        sample_example_id: str,
        original_id: str,
        run_id: str,
        judge_id: str,
        error: Exception,
    ) -> dict[str, Any]:
        """Create a properly structured other error row dict.

        Args:
            task_name: Name of the evaluation task
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
            run_id: Unique identifier for the evaluation run
            judge_id: ID of the judge configuration
            error: The exception that occurred (not LLM API or parsing related)

        Returns:
            Dictionary with all fields properly set for OTHER_ERROR status
        """
        row = cls.create_empty_row_dict(task_name)
        row.update(
            {
                "sample_example_id": sample_example_id,
                "original_id": original_id,
                "run_id": run_id,
                "judge_id": judge_id,
                "task_name": task_name,
                "status": EvaluationStatusEnum.OTHER_ERROR.value,
                # Outcome column stays None
                task_name: None,
                # Error fields are populated
                "error_type": "OTHER_ERROR",
                "error_message": str(error),
                "error_details_json": json.dumps(
                    {"error": str(error), "type": type(error).__name__}
                ),
                # LLM diagnostic fields stay None (unknown if LLM was called)
                "llm_raw_response_content": None,
                "llm_prompt_tokens": None,
                "llm_completion_tokens": None,
                "llm_total_tokens": None,
                "llm_call_duration_seconds": None,
            }
        )
        return row

    @model_validator(mode="after")
    def validate_judge_results(self) -> "JudgeResults":
        """Performs comprehensive validation of the JudgeResults instance.

        Returns:
            The validated JudgeResults instance

        Raises:
            ValueError: if validation fails
            TypeError: if results_data is not a Polars DataFrame
        """
        # 1. Validate count consistency
        if self.total_examples_count != (
            self.skipped_examples_count
            + self.succeeded_examples_count
            + self.llm_error_examples_count
            + self.parsing_error_examples_count
            + self.other_error_examples_count
        ):
            raise ValueError(
                f"""Count mismatch: total_examples_count does not equal the sum of skipped, succeeded, llm_error, parsing_error, and other_error counts.
                
                Number of {self.task_name} examples: {self.total_examples_count}
                Number of skipped examples: {self.skipped_examples_count}
                Number of succeeded examples: {self.succeeded_examples_count}
                Number of LLM API error examples: {self.llm_error_examples_count}
                Number of parsing error examples: {self.parsing_error_examples_count}
                Number of other error examples: {self.other_error_examples_count}
                """
            )

        # 2. Validate results_data DataFrame basic properties
        if not isinstance(self.results_data, pl.DataFrame):
            raise TypeError("results_data must be a Polars DataFrame.")

        if self.results_data.is_empty():
            raise ValueError("results_data DataFrame cannot be empty.")

        # 3. Validate presence of required DataFrame columns using schema
        schema = self.get_required_columns(self.task_name)
        required_col_names = [col_name for col_name, _ in schema]
        missing_cols = [
            col for col in required_col_names if col not in self.results_data.columns
        ]
        if missing_cols:
            raise ValueError(
                f"results_data DataFrame missing required columns: {missing_cols}"
            )

        # 4. Validate 'sample_example_id' uniqueness
        if self.results_data["sample_example_id"].n_unique() != len(self.results_data):
            duplicate_ids = (
                self.results_data.group_by("sample_example_id")
                .len()
                .filter(pl.col("len") > 1)
                .select("sample_example_id")
                .to_series()
                .to_list()
            )
            raise ValueError(
                f"sample_example_id column in results_data must contain unique values. Duplicates found: {duplicate_ids}"
            )

        # 5. Validate 'status' column contains only valid enum values
        valid_statuses = [e.value for e in EvaluationStatusEnum]
        if not self.results_data["status"].is_in(valid_statuses).all():
            invalid_statuses = (
                self.results_data.filter(~pl.col("status").is_in(valid_statuses))
                .select("status")
                .unique()
                .to_series()
                .to_list()
            )
            raise ValueError(
                f"Status column contains invalid values: {invalid_statuses}. Must be one of {valid_statuses}"
            )

        # 6. Enforce conditional null/non-null states based on 'status'

        # Success status checks
        success_df = self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.SUCCESS.value
        )
        if not success_df.is_empty():
            # Outcome column MUST NOT be null
            if success_df[self.task_name].is_null().any():
                raise ValueError(
                    f"Outcome column '{self.task_name}' for SUCCESS status contains null values."
                )
            # Error columns MUST be null
            if (
                success_df["error_type"].is_not_null().any()
                or success_df["error_message"].is_not_null().any()
                or success_df["error_details_json"].is_not_null().any()
            ):
                raise ValueError(
                    "Error columns for SUCCESS status contain non-null values."
                )
            # LLM diagnostic columns MUST NOT be null
            if (
                success_df["llm_raw_response_content"].is_null().any()
                or success_df["llm_prompt_tokens"].is_null().any()
                or success_df["llm_completion_tokens"].is_null().any()
                or success_df["llm_total_tokens"].is_null().any()
                or success_df["llm_call_duration_seconds"].is_null().any()
            ):
                raise ValueError(
                    "LLM diagnostic columns for SUCCESS status contain null values."
                )

        # Skipped status checks
        skipped_df = self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.SKIPPED.value
        )
        if not skipped_df.is_empty():
            # Outcome column MUST be null
            if skipped_df[self.task_name].is_not_null().any():
                raise ValueError(
                    f"Outcome column '{self.task_name}' for SKIPPED status contains non-null values."
                )
            # Error columns MUST be null
            if (
                skipped_df["error_type"].is_not_null().any()
                or skipped_df["error_message"].is_not_null().any()
                or skipped_df["error_details_json"].is_not_null().any()
            ):
                raise ValueError(
                    "Error columns for SKIPPED status contain non-null values."
                )
            # LLM diagnostic columns MUST be null
            if (
                skipped_df["llm_raw_response_content"].is_not_null().any()
                or skipped_df["llm_prompt_tokens"].is_not_null().any()
                or skipped_df["llm_completion_tokens"].is_not_null().any()
                or skipped_df["llm_total_tokens"].is_not_null().any()
                or skipped_df["llm_call_duration_seconds"].is_not_null().any()
            ):
                raise ValueError(
                    "LLM diagnostic columns for SKIPPED status contain non-null values."
                )

        # LLM_ERROR status checks
        llm_error_df = self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.LLM_ERROR.value
        )
        if not llm_error_df.is_empty():
            # Outcome column MUST be null
            if llm_error_df[self.task_name].is_not_null().any():
                raise ValueError(
                    f"Outcome column '{self.task_name}' for LLM_ERROR status contains non-null values."
                )
            # Error type MUST NOT be null and must be "LLM_API_ERROR"
            if (
                llm_error_df["error_type"].is_null().any()
                or (llm_error_df["error_type"] != "LLM_API_ERROR").any()
            ):
                raise ValueError(
                    "error_type for LLM_ERROR status is null or not 'LLM_API_ERROR'."
                )
            # Error message MUST NOT be null
            if llm_error_df["error_message"].is_null().any():
                raise ValueError(
                    "error_message for LLM_ERROR status contains null values."
                )
            # Error details JSON MUST NOT be null (can be empty string, but not null)
            if llm_error_df["error_details_json"].is_null().any():
                raise ValueError(
                    "error_details_json for LLM_ERROR status contains null values."
                )
            # LLM diagnostic columns MUST be null (call failed)
            if (
                llm_error_df["llm_raw_response_content"].is_not_null().any()
                or llm_error_df["llm_prompt_tokens"].is_not_null().any()
                or llm_error_df["llm_completion_tokens"].is_not_null().any()
                or llm_error_df["llm_total_tokens"].is_not_null().any()
                or llm_error_df["llm_call_duration_seconds"].is_not_null().any()
            ):
                raise ValueError(
                    "LLM diagnostic columns for LLM_ERROR status contain non-null values."
                )

        # PARSING_ERROR status checks
        parsing_error_df = self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.PARSING_ERROR.value
        )
        if not parsing_error_df.is_empty():
            # Outcome column MUST be null
            if parsing_error_df[self.task_name].is_not_null().any():
                raise ValueError(
                    f"Outcome column '{self.task_name}' for PARSING_ERROR status contains non-null values."
                )
            # Error type MUST NOT be null and NOT be "LLM_API_ERROR"
            if (
                parsing_error_df["error_type"].is_null().any()
                or (parsing_error_df["error_type"] == "LLM_API_ERROR").any()
            ):
                raise ValueError(
                    "error_type for PARSING_ERROR status is null or is 'LLM_API_ERROR'."
                )
            # Error message MUST NOT be null
            if parsing_error_df["error_message"].is_null().any():
                raise ValueError(
                    "error_message for PARSING_ERROR status contains null values."
                )
            # Error details JSON MUST NOT be null
            if parsing_error_df["error_details_json"].is_null().any():
                raise ValueError(
                    "error_details_json for PARSING_ERROR status contains null values."
                )
            # LLM diagnostic columns MUST NOT be null (call succeeded, but parsing failed)
            if (
                parsing_error_df["llm_raw_response_content"].is_null().any()
                or parsing_error_df["llm_prompt_tokens"].is_null().any()
                or parsing_error_df["llm_completion_tokens"].is_null().any()
                or parsing_error_df["llm_total_tokens"].is_null().any()
                or parsing_error_df["llm_call_duration_seconds"].is_null().any()
            ):
                raise ValueError(
                    "LLM diagnostic columns for PARSING_ERROR status contain null values."
                )

        # OTHER_ERROR status checks
        other_error_df = self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.OTHER_ERROR.value
        )
        if not other_error_df.is_empty():
            # Outcome column MUST be null
            if other_error_df[self.task_name].is_not_null().any():
                raise ValueError(
                    f"Outcome column '{self.task_name}' for OTHER_ERROR status contains non-null values."
                )
            # Error type MUST NOT be null and must be "OTHER_ERROR"
            if (
                other_error_df["error_type"].is_null().any()
                or (other_error_df["error_type"] != "OTHER_ERROR").any()
            ):
                raise ValueError(
                    "error_type for OTHER_ERROR status is null or not 'OTHER_ERROR'."
                )
            # Error message MUST NOT be null
            if other_error_df["error_message"].is_null().any():
                raise ValueError(
                    "error_message for OTHER_ERROR status contains null values."
                )
            # Error details JSON MUST NOT be null
            if other_error_df["error_details_json"].is_null().any():
                raise ValueError(
                    "error_details_json for OTHER_ERROR status contains null values."
                )
            # LLM diagnostic columns MUST be null (unknown if LLM was called)
            if (
                other_error_df["llm_raw_response_content"].is_not_null().any()
                or other_error_df["llm_prompt_tokens"].is_not_null().any()
                or other_error_df["llm_completion_tokens"].is_not_null().any()
                or other_error_df["llm_total_tokens"].is_not_null().any()
                or other_error_df["llm_call_duration_seconds"].is_not_null().any()
            ):
                raise ValueError(
                    "LLM diagnostic columns for OTHER_ERROR status contain non-null values."
                )

        return self

    def __len__(self) -> int:
        """Return the number of examples in the results."""
        return len(self.results_data)

    def get_successful_results(self) -> pl.DataFrame:
        """Filter and return a DataFrame of successful evaluation results.

        Returns:
            pl.DataFrame: A DataFrame of successful evaluation results.
        """
        return self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.SUCCESS.value
        )

    def get_failed_results(self) -> pl.DataFrame:
        """Filter and return a DataFrame of failed evaluation results (LLM, parsing, or other errors).

        Returns:
            pl.DataFrame: A DataFrame of failed evaluation results.
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

    def get_skipped_results(self) -> pl.DataFrame:
        """Filter and return a DataFrame of skipped evaluation results.

        Returns:
            pl.DataFrame: A DataFrame of skipped evaluation results.
        """
        return self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.SKIPPED.value
        )
