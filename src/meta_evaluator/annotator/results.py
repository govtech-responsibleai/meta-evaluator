"""Models for human annotation results that follow the same structure as JudgeResults."""

from enum import Enum
import polars as pl
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    create_model,
    model_validator,
    ValidationError,
)
from datetime import datetime
from pathlib import Path
from typing import List, Any, Literal, Optional, Annotated
import logging
import re
import json
from dataclasses import dataclass

# Error message constants
INVALID_JSON_STRUCTURE_MSG = "Invalid JSON structure in state file"
INVALID_JSON_MSG = "Invalid JSON in state file"
STATE_FILE_NOT_FOUND_MSG = "State file not found"


class HumanAnnotationStatusEnum(str, Enum):
    """Enumeration of possible human annotation outcomes for a single example."""

    SUCCESS = "success"
    ERROR = "error"


@dataclass
class FieldTags:
    """Metadata for field tags."""

    tags: list[str]


class BaseResultRow(BaseModel):
    """Base model for human annotation result rows containing all fixed columns."""

    sample_example_id: Annotated[
        str,
        Field(
            description="Unique identifier for this sample within the annotation run"
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
        Field(description="Unique identifier for the annotation run"),
        FieldTags(tags=["metadata"]),
    ]

    annotator_id: Annotated[
        str,
        Field(description="ID of the human annotator"),
        FieldTags(tags=["metadata"]),
    ]

    status: Annotated[
        str,
        Field(description="Annotation status (success, error)"),
        FieldTags(tags=["metadata"]),
    ]

    error_message: Annotated[
        Optional[str],
        Field(default=None, description="Error message if annotation failed"),
        FieldTags(tags=["error"]),
    ]

    error_details_json: Annotated[
        Optional[str],
        Field(
            default=None, description="JSON-encoded error details if annotation failed"
        ),
        FieldTags(tags=["error"]),
    ]

    annotation_timestamp: Annotated[
        Optional[datetime],
        Field(default=None, description="Timestamp when annotation was completed"),
        FieldTags(tags=["annotation_diagnostic"]),
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
    def get_annotation_diagnostic_fields(cls) -> list[str]:
        """Get all annotation diagnostic field names.

        Returns:
            list[str]: List of field names tagged as annotation_diagnostic
        """
        return cls.get_fields_by_tag("annotation_diagnostic")

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


class HumanAnnotationResults(BaseModel):
    """Immutable container for human annotation results from a single annotation run.

    This class stores the detailed outcomes for each example from an EvalData
    instance after being annotated by a human annotator. It follows the same
    structure as JudgeResults for easy comparison and analysis.

    The primary data is stored in a Polars DataFrame, where each row corresponds
    to an example from the original EvalData (linked by 'original_id' and identified
    uniquely within the run by 'sample_example_id'), and additional columns represent
    the annotation status, outcomes, and any errors.

    Attributes:
        run_id (str): Unique identifier for this specific annotation run.
        annotator_id (str): ID of the human annotator who performed the annotations.
        task_schemas (dict[str, List[str]]): Dictionary mapping task names to their allowed outcome values.
        timestamp_local (datetime): Local timestamp of when this annotation run completed.
        total_count (int): Total number of examples from the input EvalData that were attempted for this run.
        succeeded_count (int): Number of examples where annotation was fully successful.
        error_count (int): Number of examples that failed due to unexpected errors.
        is_sampled_run (bool): True if the EvalData provided for this run was a SampledEvalData instance.
        results_data (pl.DataFrame): A Polars DataFrame containing per-example results.
    """

    run_id: str = Field(
        ..., description="Unique identifier for this specific annotation run."
    )
    annotator_id: str = Field(
        ..., description="ID of the human annotator who performed the annotations."
    )
    task_schemas: dict[str, List[str] | None] = Field(
        ...,
        description="Dictionary mapping task names to their allowed outcome values. Use None for free form text outputs.",
    )
    timestamp_local: datetime = Field(
        ..., description="Local timestamp of when this annotation run completed."
    )
    total_count: int = Field(
        ...,
        description="Total number of examples from the input EvalData that were attempted for this run.",
    )
    succeeded_count: int = Field(
        ..., description="Number of examples where annotation was fully successful."
    )
    error_count: int = Field(
        ..., description="Number of examples that failed due to unexpected errors."
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
    def validate_human_annotation_results(self) -> "HumanAnnotationResults":
        """Performs comprehensive validation of the HumanAnnotationResults instance.

        Returns:
            The validated HumanAnnotationResults instance

        Raises:
            ValueError: if validation fails
        """
        # 1. Validate count consistency
        if self.total_count != (self.succeeded_count + self.error_count):
            raise ValueError(
                f"""Count mismatch: total_count does not equal the sum of all status counts.

                Total examples: {self.total_count}
                Succeeded examples: {self.succeeded_count}
                Other error examples: {self.error_count}
                Sum: {self.succeeded_count + self.error_count}
                """
            )

        # 2. Validate results_data is not empty
        if self.results_data.is_empty():
            raise ValueError("results_data DataFrame cannot be empty.")

        # 3. Validate presence of required DataFrame columns
        task_names = list(self.task_schemas.keys())
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
        valid_statuses = [e.value for e in HumanAnnotationStatusEnum]
        if not self.results_data["status"].is_in(valid_statuses).all():
            invalid_statuses = (
                self.results_data.filter(~pl.col("status").is_in(valid_statuses))
                .select("status")
                .unique()
                .to_series()
                .to_list()
            )
            raise ValueError(
                f"status column in results_data contains invalid values: {invalid_statuses}. Valid values are: {valid_statuses}"
            )

        # 6. Validate conditional null/non-null states based on status
        for row in self.results_data.iter_rows(named=True):
            status = row["status"]

            if status == HumanAnnotationStatusEnum.SUCCESS.value:
                # For success, all task outcomes should be non-null
                for task_name in task_names:
                    if task_name in row and row[task_name] is None:
                        raise ValueError(
                            f"Task outcome '{task_name}' is null for successful annotation (sample_example_id: {row['sample_example_id']})"
                        )
                # Error fields should be null
                if (
                    row.get("error_message") is not None
                    or row.get("error_details_json") is not None
                ):
                    raise ValueError(
                        f"Error fields should be null for successful annotation (sample_example_id: {row['sample_example_id']})"
                    )

            elif status == HumanAnnotationStatusEnum.ERROR.value:
                # For error, all task outcomes should be null
                for task_name in task_names:
                    if task_name in row and row[task_name] is not None:
                        raise ValueError(
                            f"Task outcome '{task_name}' should be null for error annotation (sample_example_id: {row['sample_example_id']})"
                        )

        return self

    def __len__(self) -> int:
        """Return the number of results in the DataFrame.

        Returns:
            int: Number of results
        """
        return len(self.results_data)

    def get_successful_results(self) -> pl.DataFrame:
        """Get all successful annotation results.

        Returns:
            pl.DataFrame: DataFrame containing only successful results
        """
        return self.results_data.filter(
            pl.col("status") == HumanAnnotationStatusEnum.SUCCESS.value
        )

    def get_failed_results(self) -> pl.DataFrame:
        """Get all failed annotation results.

        Returns:
            pl.DataFrame: DataFrame containing only failed results
        """
        return self.results_data.filter(
            pl.col("status") == HumanAnnotationStatusEnum.ERROR.value
        )

    def get_task_success_rate(self, task_name: str) -> float:
        """Calculate the success rate for a specific task.

        Args:
            task_name: Name of the task to calculate success rate for

        Returns:
            float: Success rate as a percentage (0.0 to 1.0)

        Raises:
            ValueError: If task_name is not in the task schema
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
            data_format: Format to write the data in (json, csv, or parquet).

        Raises:
            ValueError: If data_format is not one of json, csv, or parquet.
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
        """Load DataFrame from file in specified format.

        Args:
            filepath: Path to write the data file to.
            data_format: Format to write the data in (json, csv, or parquet).

        Returns:
            Loaded DataFrame (polars).

        Raises:
            FileNotFoundError: If the data file doesn't exist.
            ValueError: If the data format is not supported.
        """
        try:
            match data_format:
                case "parquet":
                    return pl.read_parquet(filepath)
                case "csv":
                    return pl.read_csv(filepath)
                case "json":
                    return pl.read_json(filepath)
                case _:
                    raise ValueError(f"Unsupported data format: {data_format}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {filepath}")

    def save_state(
        self,
        state_file: str,
        data_format: Literal["json", "csv", "parquet"] = "json",
        data_filename: str = "data.json",
    ) -> None:
        """Save the current state of the HumanAnnotationResults to a file.

        Args:
            state_file: The path to the file to save the state to.
            data_format: The format to save the data in (json, csv, or parquet).
            data_filename: The name of the data file to save.

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

        # Create a final state object for serialization
        state_to_save = {
            "metadata": self.model_dump(mode="json", exclude={"results_data"}),
            "data_format": data_format,
            "data_filename": final_data_filename,
        }

        directory.mkdir(parents=True, exist_ok=True)

        # Write state to JSON
        with open(state_file, "w") as f:
            json.dump(state_to_save, f, indent=2)

        # Write DataFrame to separate file
        data_filepath = directory / final_data_filename
        self.write_data(filepath=str(data_filepath), data_format=data_format)

    @classmethod
    def load_state(cls, state_file: str) -> "HumanAnnotationResults":
        """Load HumanAnnotationResults from a state file.

        Args:
            state_file: Path to the JSON state file.

        Returns:
            A new HumanAnnotationResults instance.

        Raises:
            FileNotFoundError: If the state or data file doesn't exist.
            ValueError: If the JSON is invalid or missing required keys.
        """
        try:
            with open(state_file, "r") as f:
                state_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"{STATE_FILE_NOT_FOUND_MSG}: {state_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"{INVALID_JSON_MSG}: {e}")

        try:
            metadata = state_data["metadata"]
            data_format = state_data["data_format"]
            data_filename = state_data["data_filename"]
        except KeyError as e:
            raise ValueError(f"State file missing required key: {e}")

        state_path = Path(state_file)
        data_filepath = state_path.parent / data_filename

        # Load the data from the referenced file
        results_data = cls.load_data(str(data_filepath), data_format=data_format)

        # Reconstruct the HumanAnnotationResults object
        try:
            return cls(**metadata, results_data=results_data)
        except ValidationError as e:
            raise ValueError(f"{INVALID_JSON_STRUCTURE_MSG}: {e}")


class HumanAnnotationResultsConfig(BaseModel):
    """Configuration for initializing a HumanAnnotationResultsBuilder.

    This class validates all the static metadata required to start building
    human annotation results. It ensures proper setup before any rows are added
    to the builder.

    Attributes:
        run_id (str): Unique identifier for this specific annotation run.
            Must contain only alphanumeric characters and underscores.
        annotator_id (str): ID of the human annotator performing the annotations.
            Must contain only alphanumeric characters and underscores.
        task_schemas (Dict[str, List[str]]): Dictionary mapping task names to
            their allowed outcome values. Each task must have at least one
            possible outcome. Task names must follow naming conventions.
        timestamp_local (datetime): Local timestamp of when this annotation run
            completed. Cannot be in the future.
        is_sampled_run (bool): True if the EvalData provided for this run was
            a SampledEvalData instance.
        expected_ids (List[str | int]): List of ID values that annotation results are expected for
    """

    run_id: str = Field(
        ..., description="Unique identifier for this specific annotation run"
    )
    annotator_id: str = Field(
        ..., description="ID of the human annotator performing the annotations"
    )
    task_schemas: dict[str, List[str] | None] = Field(
        ...,
        description="Dictionary mapping task names to their allowed outcome values. Use None for free form text outputs.",
    )
    timestamp_local: datetime = Field(
        ..., description="Local timestamp of when this annotation run completed"
    )
    is_sampled_run: bool = Field(
        ...,
        description="True if the EvalData provided for this run was a SampledEvalData instance",
    )
    expected_ids: list[str | int] = Field(
        ..., description="List of ID values that annotation results are expected for"
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_config(self) -> "HumanAnnotationResultsConfig":
        """Validate the configuration parameters.

        Returns:
            HumanAnnotationResultsConfig: The validated configuration instance

        Raises:
            ValueError: If validation fails
        """
        # Validate run_id format
        if not re.fullmatch(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self.run_id):
            raise ValueError(
                "run_id must only contain alphanumeric characters and underscores"
            )

        # Validate annotator_id format
        if not re.fullmatch(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self.annotator_id):
            raise ValueError(
                "annotator_id must only contain alphanumeric characters and underscores"
            )

        # Validate task_schemas is not empty
        if not self.task_schemas:
            raise ValueError("task_schemas cannot be empty")

        now = datetime.now()
        if self.timestamp_local > now:
            raise ValueError("timestamp_local cannot be in the future")

        return self


class HumanAnnotationResultsBuilder:
    """Builder class for HumanAnnotationResults."""

    def __init__(self, config: HumanAnnotationResultsConfig) -> None:
        """Initialize a HumanAnnotationResultsBuilder.

        Args:
            config (HumanAnnotationResultsConfig): The configuration for the HumanAnnotationResultsBuilder.
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

        for task_name, schemas in self.config.task_schemas.items():
            if schemas is None:
                # Free form text field
                fields[task_name] = (Optional[str], None)
            else:
                # Predefined outcomes
                literal_type = Literal[tuple(schemas)]
                fields[task_name] = (Optional[literal_type], None)

        output_class = create_model(
            "HumanAnnotationResultsRow",
            __base__=BaseResultRow,
            **fields,
        )

        return output_class

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
            int: Number of completed results
        """
        return len(self._results)

    @property
    def total_count(self) -> int:
        """Get the total number of expected results.

        Returns:
            int: Total number of expected results
        """
        return len(self._expected_ids)

    @property
    def is_complete(self) -> bool:
        """Check if all expected results have been received.

        Returns:
            bool: True if all expected results have been received
        """
        return self.completed_count == self.total_count

    def create_success_row(
        self,
        sample_example_id: str,
        original_id: str | int,
        outcomes: dict[str, str],
        annotation_timestamp: Optional[datetime] = None,
    ) -> None:
        """Create a successful annotation row.

        Args:
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
            outcomes: Dictionary mapping task names to their outcomes
            annotation_timestamp: Timestamp when annotation was completed

        Raises:
            ValueError: If validation fails
        """
        # Validate all tasks have outcomes
        if set(outcomes.keys()) != set(self.config.task_schemas.keys()):
            raise ValueError("Success row must contain outcomes for ALL tasks")

        # Create the result row
        result_row = self._result_row_class(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.config.run_id,
            annotator_id=self.config.annotator_id,
            status=HumanAnnotationStatusEnum.SUCCESS.value,
            annotation_timestamp=annotation_timestamp,
            **outcomes,
        )

        self._validate_and_store(result_row)

    def create_error_row(
        self,
        sample_example_id: str,
        original_id: str | int,
        error: Exception,
    ) -> None:
        """Create an error annotation row.

        Args:
            sample_example_id: Unique identifier for this sample
            original_id: Original ID from the source data
            error: The exception that occurred
        """
        result_row = self._result_row_class(
            sample_example_id=sample_example_id,
            original_id=original_id,
            run_id=self.config.run_id,
            annotator_id=self.config.annotator_id,
            status=HumanAnnotationStatusEnum.ERROR.value,
            error_message=str(error),
            error_details_json=json.dumps(
                {"error_type": type(error).__name__, "error_args": error.args}
            ),
            annotation_timestamp=datetime.now(),
        )

        self._validate_and_store(result_row)

    def complete(self) -> HumanAnnotationResults:
        """Complete the building process and return HumanAnnotationResults.

        Returns:
            HumanAnnotationResults: The final results with auto-calculated counts

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

        return HumanAnnotationResults(
            run_id=self.config.run_id,
            annotator_id=self.config.annotator_id,
            task_schemas=self.config.task_schemas,
            timestamp_local=self.config.timestamp_local,
            total_count=self.total_count,
            succeeded_count=status_count_map.get(
                HumanAnnotationStatusEnum.SUCCESS.value, 0
            ),
            error_count=status_count_map.get(HumanAnnotationStatusEnum.ERROR.value, 0),
            is_sampled_run=self.config.is_sampled_run,
            results_data=results_df,
        )
