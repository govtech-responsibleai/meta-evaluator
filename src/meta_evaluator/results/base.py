"""Base classes for evaluation results that provide common functionality."""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, TypeVar

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .serialization import BaseResultsSerializedState

logger = logging.getLogger(__name__)

# Type variable for child classes of BaseEvaluationResults
EvaluationResultsType = TypeVar("EvaluationResultsType", bound="BaseEvaluationResults")


class BaseEvaluationStatusEnum(str, Enum):
    """Base enumeration for evaluation status types."""

    SUCCESS = "success"
    ERROR = "error"


@dataclass
class FieldTags:
    """Metadata for field tags."""

    tags: list[str]


class BaseResultRow(BaseModel):
    """Base class for result row structures."""

    sample_example_id: Annotated[
        str,
        Field(..., description="Unique identifier for the example within this run"),
        FieldTags(tags=["metadata"]),
    ]
    original_id: Annotated[
        str | int,
        Field(..., description="Original identifier from the source data"),
        FieldTags(tags=["metadata"]),
    ]
    run_id: Annotated[
        str,
        Field(..., description="Unique identifier for this evaluation run"),
        FieldTags(tags=["metadata"]),
    ]
    status: Annotated[
        str,
        Field(..., description="Status of the evaluation for this example"),
        FieldTags(tags=["metadata"]),
    ]
    error_message: Annotated[
        Optional[str],
        Field(None, description="Error message if evaluation failed"),
        FieldTags(tags=["error"]),
    ]
    error_details_json: Annotated[
        Optional[str],
        Field(None, description="JSON string with error details"),
        FieldTags(tags=["error"]),
    ]

    model_config = ConfigDict(extra="allow")

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
    def get_all_base_fields(cls) -> list[str]:
        """Get all base field names.

        Returns:
            list[str]: List of all base field names
        """
        return list(cls.model_fields.keys())

    @classmethod
    def get_required_columns_with_tasks(cls, task_names: List[str]) -> List[str]:
        """Get all required columns including task columns.

        Args:
            task_names: List of task names to include as columns.

        Returns:
            List[str]: Combined list of base columns and task columns.
        """
        base_columns = list(cls.model_fields.keys())
        return base_columns + task_names


class BaseEvaluationResults(BaseModel, ABC):
    """Base class for evaluation results with common functionality."""

    run_id: str = Field(..., description="Unique identifier for this evaluation run")
    task_schemas: Dict[str, List[str] | None] = Field(
        ..., description="Dictionary mapping task names to their allowed outcome values"
    )
    timestamp_local: datetime = Field(
        ..., description="Local timestamp when evaluation completed"
    )
    total_count: int = Field(..., description="Total number of examples attempted")
    succeeded_count: int = Field(..., description="Number of successful evaluations")
    is_sampled_run: bool = Field(..., description="True if input was sampled data")
    results_data: pl.DataFrame = Field(
        ..., description="DataFrame containing per-example results"
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @abstractmethod
    def get_evaluator_id(self) -> str:
        """Get the ID of the evaluator (judge_id or annotator_id)."""
        pass

    @abstractmethod
    def get_error_count(self) -> int:
        """Get the total error count for this evaluation."""
        pass

    @abstractmethod
    def get_failed_results(self) -> pl.DataFrame:
        """Get all failed evaluation results."""
        pass

    @model_validator(mode="after")
    def validate_base_results(self) -> "BaseEvaluationResults":
        """Validate common aspects of evaluation results.

        Returns:
            BaseEvaluationResults: The validated instance.

        Raises:
            ValueError: If validation fails.
        """
        # Validate results_data is not empty
        if self.results_data.is_empty():
            raise ValueError("results_data DataFrame cannot be empty.")

        # Validate presence of required DataFrame columns
        task_names = list(self.task_schemas.keys())
        required_columns = (
            self.get_base_result_row_class().get_required_columns_with_tasks(task_names)
        )

        missing_cols = [
            col for col in required_columns if col not in self.results_data.columns
        ]
        if missing_cols:
            raise ValueError(
                f"results_data DataFrame missing required columns: {missing_cols}"
            )

        # Validate 'sample_example_id' uniqueness
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
                f"Duplicate sample_example_id values found: {duplicate_ids}"
            )

        # Consolidated validation methods
        self._validate_count_consistency()
        self._validate_status_values()
        self._validate_status_constraints()

        return self

    def _validate_count_consistency(self) -> None:
        """Validate that total_count equals sum of all status counts.

        Raises:
            ValueError: If count consistency validation fails.
        """
        # Get all status counts from the subclass
        status_counts = self._get_status_counts()
        expected_total = sum(status_counts.values())

        if self.total_count != expected_total:
            raise ValueError(
                "Count mismatch: total_count does not equal the sum of all status counts."
            )

    def _validate_status_values(self) -> None:
        """Validate that status column contains only valid enum values.

        Raises:
            ValueError: If invalid status values are found.
        """
        valid_statuses = self._get_valid_status_values()
        actual_statuses = self.results_data["status"].unique().to_list()
        invalid_statuses = [
            status for status in actual_statuses if status not in valid_statuses
        ]
        if invalid_statuses:
            raise ValueError(f"Invalid status values found: {invalid_statuses}")

    def _validate_status_constraints(self) -> None:
        """Validate constraints for each status type."""
        task_names = list(self.task_schemas.keys())
        success_status = BaseEvaluationStatusEnum.SUCCESS.value
        error_status = BaseEvaluationStatusEnum.ERROR.value

        # Validate success status constraints
        success_df = self.results_data.filter(pl.col("status") == success_status)
        if not success_df.is_empty():
            self._validate_success_constraints(success_df, task_names)

        # Validate error status constraints
        error_df = self.results_data.filter(pl.col("status") == error_status)
        if not error_df.is_empty():
            self._validate_error_constraints(error_df, task_names)

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
        # Check that ALL task outcome columns are non-null
        for task_name in task_names:
            if success_df[task_name].is_null().any():
                raise ValueError(
                    f"Task outcome column '{task_name}' for SUCCESS status contains null values."
                )

        # Error columns MUST be null
        error_fields = self.get_base_result_row_class().get_error_fields()
        for error_field in error_fields:
            if success_df[error_field].is_not_null().any():
                raise ValueError(
                    f"Error field '{error_field}' for SUCCESS status contains non-null values."
                )

    def _validate_error_constraints(
        self, error_df: pl.DataFrame, task_names: List[str]
    ) -> None:
        """Validate constraints for error status rows.

        Args:
            error_df: DataFrame containing only error status rows
            task_names: List of task names

        Raises:
            ValueError: If error constraint validation fails.
        """
        # Check that ALL task outcome columns are null
        for task_name in task_names:
            if error_df[task_name].is_not_null().any():
                raise ValueError(
                    f"Task outcome column '{task_name}' for ERROR status contains non-null values."
                )

    @abstractmethod
    def _get_status_counts(self) -> Dict[str, int]:
        """Get status counts for count consistency validation.

        Returns:
            Dict[str, int]: Dictionary mapping status values to their counts.
        """
        pass

    @abstractmethod
    def _get_valid_status_values(self) -> List[str]:
        """Get valid status values for this evaluation type.

        Returns:
            List[str]: List of valid status values.
        """
        pass

    @abstractmethod
    def get_base_result_row_class(self) -> type[BaseResultRow]:
        """Get the base result row class for this evaluation type.

        Returns:
            type[BaseResultRow]: The base result row class.
        """
        pass

    @abstractmethod
    def serialize(
        self, data_format: Literal["json", "csv", "parquet"], data_filename: str
    ) -> BaseModel:
        """Serialize the evaluation results to a Pydantic model.

        Args:
            data_format: Format for data serialization.
            data_filename: Name of data file.

        Returns:
            BaseModel: Serialized state as a Pydantic model.
        """
        pass

    @classmethod
    @abstractmethod
    def deserialize(
        cls: type[EvaluationResultsType],
        results_data: pl.DataFrame,
        state: BaseResultsSerializedState,
    ) -> EvaluationResultsType:
        """Deserialize evaluation results from serialized state.

        Args:
            results_data: The loaded DataFrame.
            state: Serialized state as a Pydantic model.

        Returns:
            EvaluationResultsType: Reconstructed evaluation results instance.
        """
        pass

    @classmethod
    @abstractmethod
    def _load_json_state(cls, state_file: str) -> BaseResultsSerializedState:
        """Load and validate JSON state file.

        Args:
            state_file: Path to the JSON state file.

        Returns:
            BaseResultsSerializedState: The loaded and validated state object.

        Raises:
            FileNotFoundError: If the state file doesn't exist.
            ValueError: If the JSON structure is invalid.
        """
        pass

    def __len__(self) -> int:
        """Return the number of results in the DataFrame.

        Returns:
            int: Number of results in the DataFrame.
        """
        return len(self.results_data)

    def get_successful_results(self) -> pl.DataFrame:
        """Get all successful evaluation results.

        Returns:
            pl.DataFrame: DataFrame containing only successful results.
        """
        return self.results_data.filter(
            pl.col("status") == BaseEvaluationStatusEnum.SUCCESS.value
        )

    def get_task_success_rate(self, task_name: str) -> float:
        """Calculate the success rate for a specific task.

        Args:
            task_name: Name of the task to calculate success rate for.

        Returns:
            float: Success rate as a value between 0.0 and 1.0.

        Raises:
            ValueError: If task_name is not found in task schema.
        """
        if task_name not in self.task_schemas:
            raise ValueError(f"Task '{task_name}' not found in task schema")

        if self.total_count == 0:
            return 0.0

        # Count non-null values for this task
        non_null_count = self.results_data.filter(
            pl.col(task_name).is_not_null()
        ).height

        return non_null_count / self.total_count

    def write_data(
        self, filepath: str, data_format: Literal["json", "csv", "parquet"]
    ) -> None:
        """Write the data to a file in the specified format.

        Args:
            filepath: Path to write the data file to.
            data_format: Format to write the data in.

        Raises:
            ValueError: If data_format is not supported.
        """
        match data_format:
            case "parquet":
                self.results_data.write_parquet(filepath)
            case "csv":
                self.results_data.write_csv(filepath)
            case "json":
                self.results_data.write_json(filepath)
            case _:
                raise ValueError(f"Unsupported data format: {data_format}")

    @staticmethod
    def load_data(
        filepath: str, data_format: Literal["json", "csv", "parquet"]
    ) -> pl.DataFrame:
        """Load data from a file in the specified format.

        Args:
            filepath: Path to the data file.
            data_format: Format of the data file.

        Returns:
            pl.DataFrame: Loaded DataFrame.

        Raises:
            ValueError: If data_format is not supported.
        """
        match data_format:
            case "parquet":
                return pl.read_parquet(filepath)
            case "csv":
                return pl.read_csv(filepath)
            case "json":
                return pl.read_json(filepath)
            case _:
                raise ValueError(f"Unsupported data format: {data_format}")

    def save_state(
        self,
        state_file: str,
        data_format: Literal["json", "csv", "parquet"] = "json",
        data_filename: Optional[str] = None,
    ) -> None:
        """Save the current state of the evaluation results to a file.

        Args:
            state_file: The path to the file to save the state to.
            data_format: The format to save the data in (json, csv, or parquet).
            data_filename: The name of the data file to save. If None, auto-generated.

        Raises:
            ValueError: If data_filename extension does not match data_format.
        """
        state_path = Path(state_file)
        base_name = state_path.stem
        directory = state_path.parent

        final_data_filename = data_filename or f"{base_name}_data.{data_format}"

        if data_filename and not data_filename.endswith(f".{data_format}"):
            raise ValueError(
                f"data_filename extension for '{data_filename}' must match data_format '{data_format}'"
            )

        directory.mkdir(parents=True, exist_ok=True)

        # Serialize results to state object
        serialized_state = self.serialize(data_format, final_data_filename)

        # Write state to JSON
        with open(state_file, "w") as f:
            json.dump(serialized_state.model_dump(mode="json"), f, indent=2)

        # Write DataFrame to separate file using write_data()
        data_filepath = directory / final_data_filename
        self.write_data(filepath=str(data_filepath), data_format=data_format)

        logger.info(f"State saved to {state_file} with data in {data_filepath}")

    @classmethod
    def load_state(
        cls: type[EvaluationResultsType], state_file: str
    ) -> EvaluationResultsType:
        """Load evaluation results from a state file.

        Args:
            state_file: Path to the JSON state file.

        Returns:
            EvaluationResultsType: Loaded evaluation results of children of BaseEvaluationResults.
        """
        # Load JSON state from disk
        serialized_state = cls._load_json_state(state_file)

        # Extract data file information
        data_format = serialized_state.data_format
        data_filename = serialized_state.data_file

        state_path = Path(state_file)
        data_filepath = state_path.parent / data_filename

        # Load the data from the referenced file using load_data()
        results_data = cls.load_data(str(data_filepath), data_format=data_format)

        # Deserialize using the state and data
        return cls.deserialize(results_data, serialized_state)


class BaseEvaluationResultsBuilder(ABC):
    """Base class for building evaluation results with common functionality."""

    def __init__(
        self,
        run_id: str,
        evaluator_id: str,
        task_schemas: Dict[str, List[str] | None],
        expected_ids: List[str | int],
        is_sampled_run: bool = False,
    ):
        """Initialize the base evaluation results builder.

        Args:
            run_id: Unique identifier for this evaluation run.
            evaluator_id: ID of the evaluator (judge_id or annotator_id).
            task_schemas: Dictionary mapping task names to their allowed outcome values.
            expected_ids: List of expected original IDs.
            is_sampled_run: True if input was sampled data.

        Raises:
            ValueError: If expected_ids is empty.
        """
        self.run_id = run_id
        self.evaluator_id = evaluator_id
        self.task_schemas = task_schemas
        self.is_sampled_run = is_sampled_run
        self._results: Dict[str | int, BaseResultRow] = {}

        # Validate and convert expected_ids to set for O(1) lookup
        if len(expected_ids) == 0:
            raise ValueError("expected_ids cannot be empty.")
        self._expected_ids = set(expected_ids)

    def _validate_and_store(self, result_row: BaseResultRow) -> None:
        """Validate and store a result row.

        Args:
            result_row: The result row to validate and store

        Raises:
            ValueError: If validation fails
        """
        # Validate original_id is in expected_ids
        if result_row.original_id not in self._expected_ids:
            raise ValueError(
                f"Unexpected original_id '{result_row.original_id}' not in expected IDs"
            )

        # Validate no duplicate original_id
        if result_row.original_id in self._results:
            raise ValueError(
                f"Result for original_id '{result_row.original_id}' already exists"
            )

        # Store the result
        self._results[result_row.original_id] = result_row

    @property
    def completed_count(self) -> int:
        """Get the number of completed results.

        Returns:
            int: Number of completed results.
        """
        return len(self._results)

    @property
    def total_count(self) -> int:
        """Get the total number of expected results.

        Returns:
            int: Total number of expected results.
        """
        return len(self._expected_ids)

    @property
    def is_complete(self) -> bool:
        """Check if all expected results have been received.

        Returns:
            bool: True if all expected results are received.
        """
        return self.completed_count == len(self._expected_ids)

    @abstractmethod
    def create_success_row(
        self,
        sample_example_id: str,
        original_id: str | int,
        outcomes: dict[str, str],
        **kwargs,
    ) -> BaseResultRow:
        """Create a success row for this evaluation type.

        Args:
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
            outcomes: Dictionary with outcomes for ALL tasks
            **kwargs: Additional keyword arguments
        """
        pass

    @abstractmethod
    def complete(self) -> BaseEvaluationResults:
        """Complete the building process and return the results."""
        pass

    def _create_dataframe(self) -> pl.DataFrame:
        """Create a DataFrame from the collected rows.

        Returns:
            pl.DataFrame: DataFrame created from collected rows.

        Raises:
            ValueError: If no rows are available to create DataFrame.
        """
        if not self._results:
            raise ValueError("No rows to create DataFrame from")

        # Convert rows to dictionaries
        row_dicts = [row.model_dump() for row in self._results.values()]

        return pl.DataFrame(row_dicts)

    def _calculate_counts(self) -> dict:
        """Calculate various counts from the collected rows.

        Returns:
            dict: Dictionary containing counts for all status types.
        """
        # Create a status count map
        status_count_map = {}
        for row in self._results.values():
            status = row.status
            status_count_map[status] = status_count_map.get(status, 0) + 1

        return status_count_map
