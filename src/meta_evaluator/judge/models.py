"""Models used by the judge module."""

from enum import Enum
import polars as pl
from pydantic import BaseModel, Field, ConfigDict, create_model, model_validator
from datetime import datetime
from typing import List, Any, Literal, Optional, Annotated
import logging
import re
import json
from dataclasses import dataclass

from ..llm_client import LLMClientEnum


class EvaluationStatusEnum(str, Enum):
    """Enumeration of possible evaluation outcomes for a single example."""

    SUCCESS = "success"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    LLM_ERROR = "llm_error"
    PARSING_ERROR = "parsing_error"
    OTHER_ERROR = "other_error"


@dataclass
class FieldTags:
    """Metadata for field tags."""

    tags: list[str]


class BaseResultRow(BaseModel):
    """Base model for evaluation result rows containing all fixed columns."""

    sample_example_id: Annotated[
        str,
        Field(
            description="Unique identifier for this sample within the evaluation run"
        ),
        FieldTags(tags=["metadata"]),
    ]

    original_id: Annotated[
        str | int,
        Field(description="Original ID from the source data"),
        FieldTags(tags=["metadata"]),
    ]

    run_id: Annotated[
        str,
        Field(description="Unique identifier for the evaluation run"),
        FieldTags(tags=["metadata"]),
    ]

    judge_id: Annotated[
        str,
        Field(description="ID of the judge configuration used"),
        FieldTags(tags=["metadata"]),
    ]

    status: Annotated[
        str,
        Field(
            description="Evaluation status (success, partial, skipped, llm_error, parsing_error, other_error)"
        ),
        FieldTags(tags=["metadata"]),
    ]

    error_message: Annotated[
        Optional[str],
        Field(default=None, description="Error message if evaluation failed"),
        FieldTags(tags=["error"]),
    ]

    error_details_json: Annotated[
        Optional[str],
        Field(
            default=None, description="JSON-encoded error details if evaluation failed"
        ),
        FieldTags(tags=["error"]),
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
    def get_fields_by_tag(cls, tag: str) -> list[str]:
        """Get field names that have a specific tag.

        Args:
            tag: The tag to filter by

        Returns:
            list[str]: List of field names with the specified tag
        """
        field_names = []
        for field_name, field_info in cls.model_fields.items():
            # Look for FieldTags in the metadata
            for metadata in field_info.metadata:
                if isinstance(metadata, FieldTags) and tag in metadata.tags:
                    field_names.append(field_name)
                    break
        return field_names

    @classmethod
    def get_metadata_fields(cls) -> list[str]:
        """Get all metadata field names.

        Returns:
            list[str]: List of field names tagged as metadata
        """
        return cls.get_fields_by_tag("metadata")

    @classmethod
    def get_error_fields(cls) -> list[str]:
        """Get all error field names.

        Returns:
            list[str]: List of field names tagged as error
        """
        return cls.get_fields_by_tag("error")

    @classmethod
    def get_llm_diagnostic_fields(cls) -> list[str]:
        """Get all LLM diagnostic field names.

        Returns:
            list[str]: List of field names tagged as llm_diagnostic
        """
        return cls.get_fields_by_tag("llm_diagnostic")

    @classmethod
    def get_all_base_fields(cls) -> list[str]:
        """Get all base field names.

        Returns:
            list[str]: List of all base field names
        """
        return list(cls.model_fields.keys())

    @classmethod
    def get_required_columns_with_tasks(cls, task_names: list[str]) -> list[str]:
        """Get all required columns including base fields and task columns.

        Args:
            task_names: List of task names to include as columns

        Returns:
            list[str]: Complete list of required column names
        """
        return cls.get_all_base_fields() + task_names


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
        task_schema (dict[str, List[str]]): Dictionary mapping task names to their allowed outcome values.
        llm_client_enum (LLMClientEnum): The LLM client provider used for this run.
        model_used (str): Name of the LLM model used for this run.
        timestamp_local (datetime): Local timestamp of when this evaluation run completed.
        total_count (int): Total number of examples from the input EvalData that were attempted for this run.
        skipped_count (int): Number of examples skipped by the evaluation task's skip function.
        succeeded_count (int): Number of examples where LLM call and parsing were fully successful.
        partial_count (int): Number of examples where LLM call succeeded but only some task outcomes were parsed successfully.
        llm_error_count (int): Number of examples that failed due to an LLM API error.
        parsing_error_count (int): Number of examples where LLM call succeeded but response parsing failed completely.
        other_error_count (int): Number of examples that failed due to unexpected errors not covered by other categories.
        is_sampled_run (bool): True if the EvalData provided for this run was a SampledEvalData instance.
        results_data (pl.DataFrame): A Polars DataFrame containing per-example results.

    model_config (ConfigDict): Pydantic configuration dictionary.
        - `frozen=True`: Makes the JudgeResults instance immutable after creation.
        - `arbitrary_types_allowed=True`: Allows Polars DataFrame as an attribute.

    Validation:
        - Ensures all top-level metadata attributes are present and correctly typed.
        - Ensures `results_data` is a Polars DataFrame and is not empty.
        - Verifies count consistency: `total_count` equals sum of all status counts.
        - Validates presence of all required `results_data` columns including all task outcome columns.
        - Checks that 'sample_example_id' column contains unique values.
        - Validates 'status' column contains only valid `EvaluationStatusEnum` values.
        - Enforces conditional null/non-null states for columns based on 'status'.
    """

    run_id: str = Field(
        ..., description="Unique identifier for this specific evaluation run."
    )
    judge_id: str = Field(
        ..., description="ID of the Judge configuration used for this run."
    )
    task_schema: dict[str, List[str]] = Field(
        ...,
        description="Dictionary mapping task names to their allowed outcome values.",
    )
    llm_client_enum: LLMClientEnum = Field(
        ..., description="The LLM client provider used for this run."
    )
    model_used: str = Field(..., description="Name of the LLM model used for this run.")
    timestamp_local: datetime = Field(
        ..., description="Local timestamp of when this evaluation run completed."
    )
    total_count: int = Field(
        ...,
        description="Total number of examples from the input EvalData that were attempted for this run.",
    )
    skipped_count: int = Field(
        ...,
        description="Number of examples skipped by the evaluation task's skip function.",
    )
    succeeded_count: int = Field(
        ...,
        description="Number of examples where LLM call and parsing were fully successful.",
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
    is_sampled_run: bool = Field(
        ...,
        description="True if the EvalData provided for this run was a SampledEvalData instance.",
    )
    results_data: pl.DataFrame = Field(
        ..., description="A Polars DataFrame containing per-example results."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_judge_results(self) -> "JudgeResults":
        """Performs comprehensive validation of the JudgeResults instance.

        Returns:
            The validated JudgeResults instance

        Raises:
            ValueError: if validation fails
        """
        # 1. Validate count consistency
        if self.total_count != (
            self.skipped_count
            + self.succeeded_count
            + self.partial_count
            + self.llm_error_count
            + self.parsing_error_count
            + self.other_error_count
        ):
            raise ValueError(
                f"""Count mismatch: total_count does not equal the sum of all status counts.
                
                Total examples: {self.total_count}
                Skipped examples: {self.skipped_count}
                Succeeded examples: {self.succeeded_count}
                Partial examples: {self.partial_count}
                LLM error examples: {self.llm_error_count}
                Parsing error examples: {self.parsing_error_count}
                Other error examples: {self.other_error_count}
                Sum: {self.skipped_count + self.succeeded_count + self.partial_count + self.llm_error_count + self.parsing_error_count + self.other_error_count}
                """
            )

        if self.results_data.is_empty():
            raise ValueError("results_data DataFrame cannot be empty.")

        # 3. Validate presence of required DataFrame columns
        task_names = list(self.task_schema.keys())
        required_columns = BaseResultRow.get_required_columns_with_tasks(task_names)

        missing_cols = [
            col for col in required_columns if col not in self.results_data.columns
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

        # Get field categories for validation
        error_fields = BaseResultRow.get_error_fields()
        llm_diagnostic_fields = BaseResultRow.get_llm_diagnostic_fields()

        # 6. Enforce conditional null/non-null states based on 'status'

        # Success status checks - ALL task columns must be non-null
        success_df = self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.SUCCESS.value
        )
        if not success_df.is_empty():
            # Check that ALL task outcome columns are non-null
            for task_name in task_names:
                if success_df[task_name].is_null().any():
                    raise ValueError(
                        f"Task outcome column '{task_name}' for SUCCESS status contains null values."
                    )

            # Error columns MUST be null
            for error_field in error_fields:
                if success_df[error_field].is_not_null().any():
                    raise ValueError(
                        f"Error field '{error_field}' for SUCCESS status contains non-null values."
                    )

            # LLM diagnostic columns MUST NOT be null
            for llm_field in llm_diagnostic_fields:
                if success_df[llm_field].is_null().any():
                    raise ValueError(
                        f"LLM diagnostic field '{llm_field}' for SUCCESS status contains null values."
                    )

        # Partial status checks - SOME task columns non-null, SOME null
        partial_df = self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.PARTIAL.value
        )
        if not partial_df.is_empty():
            # Check that at least one task column is non-null and at least one is null
            for i in range(len(partial_df)):
                row = partial_df.row(i, named=True)
                task_values = [row[task_name] for task_name in task_names]
                non_null_count = sum(1 for val in task_values if val is not None)

                if non_null_count == 0:
                    raise ValueError(
                        f"PARTIAL status row must have at least one non-null task outcome. Row: {row['sample_example_id']}"
                    )
                if non_null_count == len(task_names):
                    raise ValueError(
                        f"PARTIAL status row cannot have all task outcomes non-null (should be SUCCESS). Row: {row['sample_example_id']}"
                    )

            # Error message MUST NOT be null for partial
            if partial_df["error_message"].is_null().any():
                raise ValueError(
                    "error_message for PARTIAL status contains null values."
                )

            # LLM diagnostic columns MUST NOT be null (call succeeded)
            for llm_field in llm_diagnostic_fields:
                if partial_df[llm_field].is_null().any():
                    raise ValueError(
                        f"LLM diagnostic field '{llm_field}' for PARTIAL status contains null values."
                    )

        # Skipped status checks - ALL task columns must be null
        skipped_df = self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.SKIPPED.value
        )
        if not skipped_df.is_empty():
            # Check that ALL task outcome columns are null
            for task_name in task_names:
                if skipped_df[task_name].is_not_null().any():
                    raise ValueError(
                        f"Task outcome column '{task_name}' for SKIPPED status contains non-null values."
                    )

            # Error columns MUST be null
            for error_field in error_fields:
                if skipped_df[error_field].is_not_null().any():
                    raise ValueError(
                        f"Error field '{error_field}' for SKIPPED status contains non-null values."
                    )

            # LLM diagnostic columns MUST be null
            for llm_field in llm_diagnostic_fields:
                if skipped_df[llm_field].is_not_null().any():
                    raise ValueError(
                        f"LLM diagnostic field '{llm_field}' for SKIPPED status contains non-null values."
                    )

        # LLM_ERROR status checks - ALL task columns must be null
        llm_error_df = self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.LLM_ERROR.value
        )
        if not llm_error_df.is_empty():
            # Check that ALL task outcome columns are null
            for task_name in task_names:
                if llm_error_df[task_name].is_not_null().any():
                    raise ValueError(
                        f"Task outcome column '{task_name}' for LLM_ERROR status contains non-null values."
                    )

            # Error message MUST NOT be null
            if llm_error_df["error_message"].is_null().any():
                raise ValueError(
                    "error_message for LLM_ERROR status contains null values."
                )

            # Error details JSON MUST NOT be null
            if llm_error_df["error_details_json"].is_null().any():
                raise ValueError(
                    "error_details_json for LLM_ERROR status contains null values."
                )

            # LLM diagnostic columns MUST be null (call failed)
            for llm_field in llm_diagnostic_fields:
                if llm_error_df[llm_field].is_not_null().any():
                    raise ValueError(
                        f"LLM diagnostic field '{llm_field}' for LLM_ERROR status contains non-null values."
                    )

        # PARSING_ERROR status checks - ALL task columns must be null
        parsing_error_df = self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.PARSING_ERROR.value
        )
        if not parsing_error_df.is_empty():
            # Check that ALL task outcome columns are null
            for task_name in task_names:
                if parsing_error_df[task_name].is_not_null().any():
                    raise ValueError(
                        f"Task outcome column '{task_name}' for PARSING_ERROR status contains non-null values."
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
            for llm_field in llm_diagnostic_fields:
                if parsing_error_df[llm_field].is_null().any():
                    raise ValueError(
                        f"LLM diagnostic field '{llm_field}' for PARSING_ERROR status contains null values."
                    )

        # OTHER_ERROR status checks - ALL task columns must be null
        other_error_df = self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.OTHER_ERROR.value
        )
        if not other_error_df.is_empty():
            # Check that ALL task outcome columns are null
            for task_name in task_names:
                if other_error_df[task_name].is_not_null().any():
                    raise ValueError(
                        f"Task outcome column '{task_name}' for OTHER_ERROR status contains non-null values."
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
            for llm_field in llm_diagnostic_fields:
                if other_error_df[llm_field].is_not_null().any():
                    raise ValueError(
                        f"LLM diagnostic field '{llm_field}' for OTHER_ERROR status contains non-null values."
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

    def get_partial_results(self) -> pl.DataFrame:
        """Filter and return a DataFrame of partial evaluation results.

        Returns:
            pl.DataFrame: A DataFrame of partial evaluation results.
        """
        return self.results_data.filter(
            pl.col("status") == EvaluationStatusEnum.PARTIAL.value
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

    def get_task_success_rate(self, task_name: str) -> float:
        """Get the success rate for a specific task.

        Args:
            task_name: Name of the task to calculate success rate for

        Returns:
            float: Success rate between 0.0 and 1.0

        Raises:
            ValueError: If task_name is not in task_schema
        """
        if task_name not in self.task_schema:
            raise ValueError(f"Task '{task_name}' not found in task_schema")

        # Count rows where this specific task has a non-null outcome
        successful_task_count = self.results_data.filter(
            pl.col(task_name).is_not_null()
        ).height

        return successful_task_count / self.total_count if self.total_count > 0 else 0.0


class JudgeResultsConfig(BaseModel):
    """Configuration for initializing a JudgeResultsBuilder.

    This class validates all the static metadata required to start building
    evaluation results. It ensures proper setup before any rows are added
    to the builder.

    Attributes:
        run_id (str): Unique identifier for this specific evaluation run.
            Must contain only alphanumeric characters and underscores.
        judge_id (str): ID of the Judge configuration used for this run.
            Must contain only alphanumeric characters and underscores.
        task_schemas (Dict[str, List[str]]): Dictionary mapping task names to
            their allowed outcome values. Each task must have at least one
            possible outcome. Task names must follow naming conventions.
        llm_client_enum (LLMClientEnum): The LLM client provider used for this run.
        model_used (str): Name of the LLM model used for this run. Cannot be empty.
        timestamp_local (datetime): Local timestamp of when this evaluation run
            completed. Cannot be in the future.
        is_sampled_run (bool): True if the EvalData provided for this run was
            a SampledEvalData instance.
        expected_ids (List[str | int]): List of ID values that evaluation results are expected for
    """

    run_id: str = Field(
        ..., description="Unique identifier for this specific evaluation run"
    )
    judge_id: str = Field(
        ..., description="ID of the Judge configuration used for this run"
    )
    task_schemas: dict[str, List[str]] = Field(
        ..., description="Dictionary mapping task names to their allowed outcome values"
    )
    llm_client_enum: LLMClientEnum = Field(
        ..., description="The LLM client provider used for this run"
    )
    model_used: str = Field(..., description="Name of the LLM model used for this run")
    timestamp_local: datetime = Field(
        ..., description="Local timestamp of when this evaluation run completed"
    )
    is_sampled_run: bool = Field(
        ...,
        description="True if the EvalData provided for this run was a SampledEvalData instance",
    )

    expected_ids: list[str | int] = Field(
        ..., description="List of ID values that evaluation results are expected for"
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_config(self) -> "JudgeResultsConfig":
        """Validate the configuration parameters.

        Returns:
            JudgeResultsConfig: The validated configuration instance

        Raises:
            ValueError: If validation fails
        """
        # Validate run_id format
        if not re.fullmatch(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self.run_id):
            raise ValueError(
                "run_id must only contain alphanumeric characters and underscores"
            )

        # Validate judge_id format
        if not re.fullmatch(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self.judge_id):
            raise ValueError(
                "judge_id must only contain alphanumeric characters and underscores"
            )

        # Validate task_schemas is not empty
        if not self.task_schemas:
            raise ValueError("task_schemas cannot be empty")

        # Validate model_used is not empty
        if not self.model_used.strip():
            raise ValueError("model_used cannot be empty")

        now = datetime.now()
        if self.timestamp_local > now:
            raise ValueError("timestamp_local cannot be in the future")

        return self


class JudgeResultsBuilder:
    """Builder class for JudgeResults."""

    def __init__(self, config: JudgeResultsConfig) -> None:
        """Initialize a JudgeResultsBuilder.

        Args:
            config (JudgeResultsConfig): The configuration for the JudgeResultsBuilder.
        """
        self.config = config
        self._result_row_class = self._create_result_row_class()

        self._results: dict[str | int, BaseResultRow] = {}

        # Convert expected_ids to set for O(1) lookup
        self._expected_ids = set(self.config.expected_ids)

    def _create_result_row_class(self) -> type[BaseResultRow]:
        """Create the result row class based on the configuration.

        Returns:
            Type[BaseResultRow]: The result row class.
        """
        fields: dict[str, Any] = {}

        for task_name, outcomes in self.config.task_schemas.items():
            literal_type = Literal[tuple(outcomes)]
            fields[task_name] = (Optional[literal_type], None)

        output_class = create_model(
            "JudgeResultsRow",
            __base__=BaseResultRow,
            **fields,
        )

        return output_class

    def _validate_and_store(self, result_row: BaseResultRow) -> None:
        """Validate and store a result row.

        Args:
            result_row: The validated Pydantic result row instance

        Raises:
            ValueError: If original_id is not in expected IDs or already exists
        """
        original_id = result_row.original_id

        if original_id not in self._expected_ids:
            raise ValueError(
                f"Unexpected original_id '{original_id}' not in expected IDs"
            )

        if original_id in self._results:
            raise ValueError(f"Result for original_id '{original_id}' already exists")

        self._results[original_id] = result_row

    @property
    def completed_count(self) -> int:
        """Number of results completed."""
        return len(self._results)

    @property
    def total_count(self) -> int:
        """Total number of expected results."""
        return len(self._expected_ids)

    @property
    def is_complete(self) -> bool:
        """Whether all expected results have been received."""
        return self.completed_count == self.total_count

    def create_success_row(
        self,
        sample_example_id: str,
        original_id: str | int,
        outcomes: dict[str, str],
        llm_raw_response: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        call_duration: float,
    ) -> None:
        """Create and store a success row with outcomes for ALL tasks.

        Args:
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
            outcomes: Dictionary with outcomes for ALL tasks
            llm_raw_response: Raw response content from LLM
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            total_tokens: Total number of tokens used
            call_duration: Duration of the LLM call in seconds

        Raises:
            ValueError: If outcomes don't match ALL tasks or validation fails
        """
        # Validate all tasks have outcomes
        expected_tasks = set(self.config.task_schemas.keys())
        provided_tasks = set(outcomes.keys())
        if provided_tasks != expected_tasks:
            raise ValueError(
                f"Success row must contain outcomes for ALL tasks. "
                f"Expected: {expected_tasks}, Provided: {provided_tasks}"
            )

        # Create validated Pydantic instance
        result_row = self._result_row_class(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.config.run_id,
            judge_id=self.config.judge_id,
            status=EvaluationStatusEnum.SUCCESS.value,
            error_message=None,
            error_details_json=None,
            llm_raw_response_content=llm_raw_response,
            llm_prompt_tokens=prompt_tokens,
            llm_completion_tokens=completion_tokens,
            llm_total_tokens=total_tokens,
            llm_call_duration_seconds=call_duration,
            **outcomes,  # Set all task outcome columns
        )

        self._validate_and_store(result_row)

    def create_partial_row(
        self,
        sample_example_id: str,
        original_id: str | int,
        outcomes: dict[str, str],
        error_message: str,
        llm_raw_response: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        call_duration: float,
    ) -> None:
        """Create and store a partial success row with outcomes for SOME tasks.

        Args:
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
            outcomes: Dictionary with outcomes for SOME tasks
            error_message: Description of what went wrong with missing tasks
            llm_raw_response: Raw response content from LLM
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            total_tokens: Total number of tokens used
            call_duration: Duration of the LLM call in seconds

        Raises:
            ValueError: If outcomes contain invalid task names or validation fails
        """
        # Validate provided tasks exist in schema
        valid_tasks = set(self.config.task_schemas.keys())
        provided_tasks = set(outcomes.keys())
        if not provided_tasks.issubset(valid_tasks):
            invalid_tasks = provided_tasks - valid_tasks
            raise ValueError(f"Invalid task names: {invalid_tasks}")

        # Create field dict with provided outcomes
        field_values: dict[str, Optional[str]] = {
            task: None for task in self.config.task_schemas.keys()
        }
        field_values.update(outcomes)

        # Create validated Pydantic instance
        result_row = self._result_row_class(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.config.run_id,
            judge_id=self.config.judge_id,
            status=EvaluationStatusEnum.PARTIAL.value,
            error_message=error_message,
            error_details_json=json.dumps({"partial_outcomes": list(outcomes.keys())}),
            llm_raw_response_content=llm_raw_response,
            llm_prompt_tokens=prompt_tokens,
            llm_completion_tokens=completion_tokens,
            llm_total_tokens=total_tokens,
            llm_call_duration_seconds=call_duration,
            **field_values,  # Set provided task outcomes (others None)
        )

        self._validate_and_store(result_row)

    def create_skipped_row(
        self,
        sample_example_id: str,
        original_id: str | int,
    ) -> None:
        """Create and store a skipped row.

        Args:
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
        """
        # Create field dict with all task outcomes as None
        field_values: dict[str, Optional[str]] = {
            task: None for task in self.config.task_schemas.keys()
        }

        # Create validated Pydantic instance
        result_row = self._result_row_class(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.config.run_id,
            judge_id=self.config.judge_id,
            status=EvaluationStatusEnum.SKIPPED.value,
            error_message=None,
            error_details_json=None,
            llm_raw_response_content=None,
            llm_prompt_tokens=None,
            llm_completion_tokens=None,
            llm_total_tokens=None,
            llm_call_duration_seconds=None,
            **field_values,  # All task outcomes None
        )

        self._validate_and_store(result_row)

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
        # Create field dict with all task outcomes as None
        field_values: dict[str, Optional[str]] = {
            task: None for task in self.config.task_schemas.keys()
        }

        # Create validated Pydantic instance
        result_row = self._result_row_class(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.config.run_id,
            judge_id=self.config.judge_id,
            status=EvaluationStatusEnum.LLM_ERROR.value,
            error_message=str(error),
            error_details_json=json.dumps(
                {"error": str(error), "type": type(error).__name__}
            ),
            llm_raw_response_content=None,
            llm_prompt_tokens=None,
            llm_completion_tokens=None,
            llm_total_tokens=None,
            llm_call_duration_seconds=None,
            **field_values,  # All task outcomes None
        )

        self._validate_and_store(result_row)

    def create_parsing_error_row(
        self,
        sample_example_id: str,
        original_id: str | int,
        error: Exception,
        llm_raw_response: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        call_duration: float,
    ) -> None:
        """Create and store a parsing error row.

        Args:
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
            error: The exception that occurred during parsing
            llm_raw_response: Raw response content from LLM
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            total_tokens: Total number of tokens used
            call_duration: Duration of the LLM call in seconds
        """
        # Create field dict with all task outcomes as None
        field_values: dict[str, Optional[str]] = {
            task: None for task in self.config.task_schemas.keys()
        }

        # Create validated Pydantic instance
        result_row = self._result_row_class(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.config.run_id,
            judge_id=self.config.judge_id,
            status=EvaluationStatusEnum.PARSING_ERROR.value,
            error_message=str(error),
            error_details_json=json.dumps(
                {"error": str(error), "type": type(error).__name__}
            ),
            llm_raw_response_content=llm_raw_response,
            llm_prompt_tokens=prompt_tokens,
            llm_completion_tokens=completion_tokens,
            llm_total_tokens=total_tokens,
            llm_call_duration_seconds=call_duration,
            **field_values,  # All task outcomes None
        )

        self._validate_and_store(result_row)

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
        # Create field dict with all task outcomes as None
        field_values: dict[str, Optional[str]] = {
            task: None for task in self.config.task_schemas.keys()
        }

        # Create validated Pydantic instance
        result_row = self._result_row_class(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.config.run_id,
            judge_id=self.config.judge_id,
            status=EvaluationStatusEnum.OTHER_ERROR.value,
            error_message=str(error),
            error_details_json=json.dumps(
                {"error": str(error), "type": type(error).__name__}
            ),
            llm_raw_response_content=None,
            llm_prompt_tokens=None,
            llm_completion_tokens=None,
            llm_total_tokens=None,
            llm_call_duration_seconds=None,
            **field_values,  # All task outcomes None
        )

        self._validate_and_store(result_row)

    def complete(self) -> JudgeResults:
        """Complete the building process and return JudgeResults.

        Returns:
            JudgeResults: The final results with auto-calculated counts

        Raises:
            ValueError: If not all expected results have been received
        """
        if not self.is_complete:
            missing_ids = self._expected_ids - set(self._results.keys())
            raise ValueError(f"Missing results for IDs: {sorted(missing_ids)}")

        # Convert Pydantic instances to dicts for DataFrame
        row_dicts = [instance.model_dump() for instance in self._results.values()]
        results_df = pl.DataFrame(row_dicts)

        # Calculate counts by status
        status_counts = results_df.group_by("status").len().to_dict(as_series=False)
        status_count_map = dict(zip(status_counts["status"], status_counts["len"]))

        return JudgeResults(
            run_id=self.config.run_id,
            judge_id=self.config.judge_id,
            task_schema=self.config.task_schemas,
            llm_client_enum=self.config.llm_client_enum,
            model_used=self.config.model_used,
            timestamp_local=self.config.timestamp_local,
            total_count=self.total_count,
            skipped_count=status_count_map.get(EvaluationStatusEnum.SKIPPED.value, 0),
            succeeded_count=status_count_map.get(EvaluationStatusEnum.SUCCESS.value, 0),
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
            is_sampled_run=self.config.is_sampled_run,
            results_data=results_df,
        )
