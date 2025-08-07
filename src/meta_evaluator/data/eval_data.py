"""Immutable container for evaluation data with column categorization and validation.

This module provides the EvalData class, which serves as a structured container for
LLM evaluation datasets. It enforces column categorization, validates data integrity,
and provides type-safe access to different data categories while preserving original
column names and ensuring immutability after initialization.
"""

import json
import logging
from typing import Any, Literal, Optional, cast

import polars as pl
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .exceptions import (
    DataFileError,
    DuplicateInIDColumnError,
    EmptyColumnListError,
    EmptyDataFrameError,
    IdColumnExistsError,
    InvalidColumnNameError,
    InvalidInIDColumnError,
    InvalidNameError,
    NoDataLeftError,
    NullValuesInDataError,
)
from .serialization import DataMetadata

ID_COLUMN_NAME = "id"


class EvalData(BaseModel):
    """Immutable container for evaluation data with strict column categorization and validation.

    This class provides a structured interface for organizing evaluation datasets while
    enforcing data integrity constraints. Once initialized, the container is immutable, ensuring
    thread-safe access and predictable behavior throughout evaluation workflows.

    The class follows a "fail-fast" validation approach where all column specifications are
    validated at initialization time, preventing runtime errors during data access. Column
    uniqueness is enforced across all categories to avoid ambiguous data interpretations.

    Uncategorized columns are automatically identified as DataFrame columns that don't belong
    to any user-specified category. A warning is logged when uncategorized columns are present
    to encourage explicit categorization for clearer evaluation semantics.

    Design Philosophy:
        - Immutable after initialization for thread safety and predictable behavior
        - Preserve original column names without transformation or normalization
        - Strict validation with clear error messages for invalid configurations
        - Clean separation between data storage and column semantics
        - Type-safe access methods that return properly categorized DataFrames
        - Encourage explicit column categorization through logging warnings

    Attributes:
        data (pl.DataFrame): The complete evaluation dataset as a Polars DataFrame.
            Contains all columns regardless of categorization. This is the single
            source of truth for the actual data values.
        id_column (Optional[str]): Column name containing unique identifiers for each
            evaluation example.  If not provided (defaults to None), an ID column will be automatically generated
            with row indices. After initialization, this will always contain the name of the ID column.
        name (str): Name of the evaluation dataset.
    """

    data: pl.DataFrame
    name: str
    id_column: Optional[str] = None  # id_column is never None after initalisation
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger(f"{__name__}.EvalData")
    )
    _initialized: bool = PrivateAttr(default=False)
    _user_set_id: bool = PrivateAttr(default=False)

    model_config = {
        "arbitrary_types_allowed": True,  # Allow Polars DataFrame and Logger
    }

    @staticmethod
    def check_single_column_name(column_name: str) -> None:
        """Validate a single column name.

        1. The column name must not be empty.
        2. The column name must start with a letter or underscore.
        3. The column name must only contain letters, numbers, and underscores.

        Args:
            column_name (str): The column name to validate.

        Raises:
            InvalidColumnNameError: If the column name is invalid.
        """
        if len(column_name) == 0 or not column_name:
            raise InvalidColumnNameError(column_name, "Column name is empty")

        if not column_name[0].isalpha() and column_name[0] != "_":
            raise InvalidColumnNameError(
                column_name, "Column name must start with a letter or underscore"
            )

        if any(not (char.isalnum() or char == "_") for char in column_name):
            raise InvalidColumnNameError(
                column_name,
                "Column name must only contain letters, numbers, and underscores",
            )

        return

    @staticmethod
    def check_dataset_name(name: str) -> None:
        """Validate a dataset name.

        1. The dataset name must not be empty.
        2. The dataset name must not be whitespace-only.
        3. The dataset name must start with a letter or underscore.
        4. The dataset name must only contain letters, numbers, and underscores.

        Args:
            name (str): The dataset name to validate.

        Raises:
            InvalidNameError: If the dataset name is invalid.
        """
        if len(name) == 0 or not name:
            raise InvalidNameError(name, "Dataset name is empty")

        if name.strip() == "":
            raise InvalidNameError(name, "Dataset name contains only whitespace")

        stripped_name = name.strip()
        if not stripped_name[0].isalpha() and stripped_name[0] != "_":
            raise InvalidNameError(
                name, "Dataset name must start with a letter or underscore"
            )

        if any(not (char.isalnum() or char == "_") for char in stripped_name):
            raise InvalidNameError(
                name,
                "Dataset name must only contain letters, numbers, and underscores",
            )

        return

    @model_validator(mode="before")
    @classmethod
    def validate_dataframe(cls, data: Any) -> Any:
        """Validate the DataFrame before it is set in the model.

        This function is called before the model is initialized, and it checks the
        given DataFrame for the following conditions:

        - The DataFrame must be a Polars DataFrame.
        - The DataFrame must not be empty.

        If the DataFrame does not meet any of the above conditions, an appropriate
        exception is raised.

        Args:
            data (dict): The data to validate.

        Raises:
            TypeError: If the given data is not a dictionary or the DataFrame is not a Polars DataFrame.
            EmptyDataFrameError: If the DataFrame is empty.

        Returns:
            dict: The validated data.
        """
        if not isinstance(data, dict):
            return data
        dataframe = data.get("data")
        if not isinstance(dataframe, pl.DataFrame):
            raise TypeError("data must be a Polars DataFrame")

        if len(dataframe) == 0:
            raise EmptyDataFrameError()

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_dataset_name(cls, data: Any) -> Any:
        """Validate the dataset name before the model is initialized.

        This function is called before the model is initialized, and it checks the
        given dataset name for the following conditions:

        - The dataset name must not be empty.
        - The dataset name must not be whitespace-only.
        - The dataset name must start with a letter or underscore.
        - The dataset name must only contain letters, numbers, and underscores.

        Args:
            data (dict): The data to validate.

        Raises:
            InvalidNameError: If the dataset name is invalid.

        Returns:
            dict: The validated data.
        """
        if not isinstance(data, dict):
            return data

        name = data.get("name")
        if name is not None:  # Allow Pydantic to handle None case
            try:
                cls.check_dataset_name(name)
            except InvalidNameError:
                raise  # For ruff's docstring checker

        return data

    def _create_id_column_if_needed(self) -> "EvalData":
        """Create an ID column if none was provided by the user.

        If no id_column is specified, this method automatically generates one using
        row indices prefixed with 'id-' (e.g., 'id-1', 'id-2', 'id-3').

        The generated ID column will be named using the ID_COLUMN_NAME constant
        and will be added as the first column in the DataFrame.

        Sets _user_set_id to track whether the ID column was user-provided (True)
        or auto-generated (False).

        Raises:
            IdColumnExistsError: If the ID_COLUMN_NAME already exists in the DataFrame
                when trying to auto-generate an ID column.

        Returns:
            EvalData: The instance with ID column created if needed.
        """
        if self.id_column is None:
            # Check if the ID column name already exists in the DataFrame
            if ID_COLUMN_NAME in self.data.columns:
                raise IdColumnExistsError(ID_COLUMN_NAME)

            # Step 1: Add row indices with the final column name and start from 1
            data_with_indices = self.data.with_row_index(name=ID_COLUMN_NAME, offset=1)

            # Step 2: Convert the index column to "id-N" format in place
            data_with_id_strings = data_with_indices.with_columns(
                pl.concat_str([pl.lit("id-"), pl.col(ID_COLUMN_NAME)]).alias(
                    ID_COLUMN_NAME
                )
            )

            # Step 3: Reorder columns to put ID first
            final_data = data_with_id_strings.select(
                [ID_COLUMN_NAME] + self.data.columns
            )

            # Update the instance
            self.data = final_data
            self.id_column = ID_COLUMN_NAME
            self._user_set_id = False

            self.logger.info(
                f"Auto-generated ID column '{ID_COLUMN_NAME}' with {len(final_data)} entries"
            )
        else:
            # User provided an ID column
            self._user_set_id = True
            self.logger.info(f"Using user-provided ID column: '{self.id_column}'")

        return self

    def _final_validation_checks(self) -> None:
        """Final validation checks for data integrity.

        Raises:
            InvalidInIDColumnError: If the ID column contains null or invalid values.
            DuplicateInIDColumnError: If the ID column contains duplicate values.
            RuntimeError: If the ID column is not set at this point or if there is a bug in the automatic id setting pipeline.
            NullValuesInDataError: If any non-ID columns contain null values.

        """
        # Step 1: Ensure ID column is set
        if self.id_column is None:
            raise RuntimeError(
                "ID column should be set by this point. This indicates a bug in the validation pipeline."
            )

        # Step 2: ID column specific validation

        # Validate ID column name format
        self.check_single_column_name(self.id_column)

        # Check for nulls in ID column
        id_null_mask = self.data[self.id_column].is_null()
        if id_null_mask.any():
            null_rows = (
                self.data.with_row_index("row_idx")
                .filter(id_null_mask)
                .select("row_idx")
                .to_series()
                .to_list()
            )
            if not self._user_set_id:
                raise RuntimeError(
                    f"Auto-generated ID column '{self.id_column}' contains null values at rows {null_rows}. "
                    "This indicates a bug in ID generation."
                )
            else:
                raise InvalidInIDColumnError(null_rows, self.id_column)

        # Check for duplicates in ID column
        duplicate_groups_df = (
            self.data.with_row_index("row_idx")
            .group_by(self.id_column)
            .agg(pl.col("row_idx"))
            .filter(pl.col("row_idx").list.len() > 1)
        )

        if duplicate_groups_df.height > 0:
            duplicate_dict = {}
            for row in duplicate_groups_df.iter_rows(named=True):
                duplicate_dict[row[self.id_column]] = row["row_idx"]

            if not self._user_set_id:
                raise RuntimeError(
                    f"Auto-generated ID column '{self.id_column}' contains duplicates: {duplicate_dict}. "
                    "This indicates a bug in ID generation."
                )
            else:
                raise DuplicateInIDColumnError(duplicate_dict, self.id_column)

        # Check for whitespace/empty in ID column (only for string columns)
        if self.data[self.id_column].dtype == pl.String:
            # Check for empty strings
            id_empty_mask = self.data[self.id_column].str.len_chars() == 0
            if id_empty_mask.any():
                empty_rows = (
                    self.data.with_row_index("row_idx")
                    .filter(id_empty_mask)
                    .select("row_idx")
                    .to_series()
                    .to_list()
                )
                if not self._user_set_id:
                    raise RuntimeError(
                        f"Auto-generated ID column '{self.id_column}' contains empty strings at rows {empty_rows}. "
                        "This indicates a bug in ID generation."
                    )
                else:
                    raise InvalidInIDColumnError(empty_rows, self.id_column)

            # Check for whitespace-only strings
            id_whitespace_mask = (
                self.data[self.id_column].str.strip_chars().str.len_chars() == 0
            ) & (self.data[self.id_column].str.len_chars() > 0)
            if id_whitespace_mask.any():
                whitespace_rows = (
                    self.data.with_row_index("row_idx")
                    .filter(id_whitespace_mask)
                    .select("row_idx")
                    .to_series()
                    .to_list()
                )
                if not self._user_set_id:
                    raise RuntimeError(
                        f"Auto-generated ID column '{self.id_column}' contains whitespace-only values at rows {whitespace_rows}. "
                        "This indicates a bug in ID generation."
                    )
                else:
                    raise InvalidInIDColumnError(whitespace_rows, self.id_column)

        # Step 3: Validate all other columns
        non_id_columns = [col for col in self.data.columns if col != self.id_column]

        # Validate all column names
        for col in non_id_columns:
            self.check_single_column_name(col)

        # Check for nulls in all non-ID columns
        null_issues = []
        for col in non_id_columns:
            col_null_mask = self.data[col].is_null()
            if col_null_mask.any():
                null_rows = (
                    self.data.with_row_index("row_idx")
                    .filter(col_null_mask)
                    .select("row_idx")
                    .to_series()
                    .to_list()
                )
                coordinates = [(row, col) for row in null_rows]
                null_issues.append(
                    f"Column '{col}' has null values at rows {null_rows} (coordinates: {coordinates})"
                )

        if null_issues:
            raise NullValuesInDataError(null_issues)

        # Check for empty/whitespace strings in non-ID columns (log only)
        string_columns = [
            col for col in non_id_columns if self.data[col].dtype == pl.String
        ]

        for col in string_columns:
            # Check for empty strings
            col_empty_mask = self.data[col].str.len_chars() == 0
            if col_empty_mask.any():
                empty_rows = (
                    self.data.with_row_index("row_idx")
                    .filter(col_empty_mask)
                    .select("row_idx")
                    .to_series()
                    .to_list()
                )
                coordinates = [(row, col) for row in empty_rows]
                self.logger.warning(
                    f"Column '{col}' has empty strings at rows {empty_rows} (coordinates: {coordinates})"
                )

            # Check for whitespace-only strings
            col_whitespace_mask = (
                self.data[col].str.strip_chars().str.len_chars() == 0
            ) & (self.data[col].str.len_chars() > 0)
            if col_whitespace_mask.any():
                whitespace_rows = (
                    self.data.with_row_index("row_idx")
                    .filter(col_whitespace_mask)
                    .select("row_idx")
                    .to_series()
                    .to_list()
                )
                coordinates = [(row, col) for row in whitespace_rows]
                self.logger.warning(
                    f"Column '{col}' has whitespace-only values at rows {whitespace_rows} (coordinates: {coordinates})"
                )

    @model_validator(mode="after")
    def complete_data_validation(self) -> "EvalData":
        """Complete data validation pipeline in correct order.

        Returns:
            EvalData: The instance with all validation checks completed.
        """
        self._create_id_column_if_needed()

        self._final_validation_checks()

        self._initialized = True

        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Override __setattr__ to prevent setting attributes after initialization.

        Raises:
            TypeError: If attempting to set an attribute after initialization.
        """
        if getattr(self, "_initialized", False):
            raise TypeError(
                f"Cannot modify attribute '{name}' on immutable EvalData instance after initialization"
            )
        super().__setattr__(name, value)

    def stratified_sample_by_columns(
        self,
        columns: list[str],
        sample_percentage: float = 0.1,
        sample_name: Optional[str] = None,
        seed: int = 42,
    ) -> "SampleEvalData":
        """Perform stratified sampling based on specified columns.

        Creates a stratified sample by grouping data based on unique combinations
        of the specified columns and sampling the specified percentage from each stratum.
        This preserves the original distribution of column combinations in the sample.

        Args:
            columns: List of column names to use for stratification. Must be non-empty
                and all columns must exist in the DataFrame.
            sample_percentage: Fraction of data to sample from each stratum (0.0 to 1.0).
                Defaults to 0.1 (10% of each column combination group).
            sample_name: Human-readable name for this sample. If None, auto-generated based
                on stratification columns and sample percentage.
            seed: Random seed for reproducible sampling. Defaults to 42. This ensures
                identical sampling results when called with the same parameters.

        Returns:
            SampleEvalData: A new instance containing the stratified sample with
                comprehensive sampling metadata attached.

        Raises:
            ValueError: If sample_percentage is not between 0 and 1, or if any specified
                columns don't exist in the DataFrame.
            EmptyColumnListError: If columns list is empty.
            NoDataLeftError: If the sampling operation results in an empty dataset.
        """
        # Validate sample_percentage
        if not 0 < sample_percentage <= 1:
            raise ValueError(
                f"sample_percentage must be between 0 and 1, got {sample_percentage}"
            )

        # Validate columns parameter
        if not columns:
            raise EmptyColumnListError("columns list cannot be empty")

        # Validate that specified columns exist in the DataFrame
        missing_cols = set(columns) - set(self.data.columns)
        if missing_cols:
            raise EmptyColumnListError(
                f"columns contains non-existent columns: {list(missing_cols)}"
            )

        # Generate sample name if not provided
        if sample_name is None:
            cols_str = "_".join(columns)
            sample_name = f"Stratified Sample ({self.name}, {cols_str}, {sample_percentage * 100:.1f}%)"

        self.logger.info(
            f"Starting stratified sampling: {sample_percentage * 100:.1f}% from {len(self.data)} rows using columns {columns}"
        )

        # Perform stratified sampling
        # Partition by specified columns and sample from each partition
        partitioned = self.data.partition_by(columns, as_dict=False)
        self.logger.info(f"Created {len(partitioned)} strata for sampling")
        sampled_partitions = []

        for partition in partitioned:
            # Calculate sample size for this partition (at least 1 if partition exists)
            sample_size = max(1, int(len(partition) * sample_percentage))
            # Ensure we don't try to sample more rows than exist in the partition
            sample_size = min(sample_size, len(partition))
            sampled_partition = partition.sample(n=sample_size, seed=seed)
            sampled_partitions.append(sampled_partition)

        # Combine all sampled partitions
        if sampled_partitions:
            sampled_df = pl.concat(sampled_partitions)
        else:
            # Edge case: no partitions (shouldn't happen with valid data)
            sampled_df = self.data.clear()

        # Check if we have any data left after sampling
        if len(sampled_df) == 0:
            raise NoDataLeftError("stratified sampling", len(self.data))

        sample_result = SampleEvalData(
            data=sampled_df,
            name=self.name,  # Inherit the original dataset name
            id_column=self.id_column,
            sample_name=sample_name,
            stratification_columns=columns,
            sample_percentage=sample_percentage,
            seed=seed,
            sampling_method="stratified_by_columns",
        )

        self.logger.info(
            f"Completed stratified sampling: created '{sample_name}' with {len(sampled_df)} rows"
        )
        return sample_result

    def write_data(
        self, filepath: str, data_format: Literal["json", "csv", "parquet"]
    ) -> None:
        """Write the data to a file in the specified format.

        Args:
            filepath: Path to write the data file to.
            data_format: Format to write the data in (json, csv, or parquet).

        Raises:
            DataFileError: If data_format is not one of json, csv, or parquet.
        """
        self.logger.info(
            f"Writing data to {filepath} in {data_format} format ({len(self.data)} rows)"
        )

        match data_format:
            case "parquet":
                self.data.write_parquet(filepath)
            case "csv":
                self.data.write_csv(filepath)
            case "json":
                data_dict = self.data.to_dict(as_series=False)
                with open(filepath, "w") as f:
                    json.dump(data_dict, f, indent=2)
            case _:
                raise DataFileError(f"Unsupported data format: {data_format}")

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
            DataFileError: If the data format is not supported or if there are parsing errors.
        """
        try:
            match data_format:
                case "parquet":
                    return pl.read_parquet(filepath)
                case "csv":
                    return pl.read_csv(filepath)
                case "json":
                    with open(filepath, "r") as f:
                        data_dict = json.load(f)
                    return pl.DataFrame(data_dict)
                case _:
                    raise DataFileError(f"Unsupported data format: {data_format}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {filepath}")
        except (
            pl.exceptions.ComputeError,
            pl.exceptions.NoDataError,
            json.JSONDecodeError,
        ) as e:
            raise DataFileError(f"Failed to parse {data_format} file: {str(e)}")

    def serialize_metadata(
        self,
        data_format: Optional[Literal["json", "csv", "parquet"]],
        data_filename: Optional[str],
    ) -> DataMetadata:
        """Serialize EvalData metadata.

        Args:
            data_format: Format for data serialization.
            data_filename: Name of data file if applicable.

        Returns:
            DataMetadata: Metadata object for this EvalData.

        Raises:
            TypeError: If id_column is None during serialization.
        """
        self.logger.info(
            f"Serializing metadata for {self.name} in {data_format} format"
        )

        if self.id_column is None:
            raise TypeError(
                "Cannot serialize EvalData: id_column is None. "
                "Data should be properly initialized before serialization."
            )
        return DataMetadata(
            name=self.name,
            id_column=self.id_column,
            data_file=cast(str, data_filename),
            data_format=cast(Literal["json", "csv", "parquet"], data_format),
            type="EvalData",
        )

    @classmethod
    def deserialize(
        cls,
        data: pl.DataFrame,
        metadata: DataMetadata,
    ) -> "EvalData":
        """Deserialize EvalData from data file metadata.

        Args:
            data: The loaded DataFrame.
            metadata: DataMetadata object containing data metadata.

        Returns:
            EvalData: Reconstructed EvalData instance.
        """
        return cls(
            data=data,
            name=metadata.name,
            id_column=metadata.id_column,
        )


class SampleEvalData(EvalData):
    """Immutable container for sampled evaluation data with stratified sampling information.

    SampleEvalData extends EvalData to represent datasets that have been created through
    the stratified_sample_by_columns sampling process. It enriches the base EvalData functionality
    with detailed metadata about the sampling operation, enabling full traceability,
    reproducibility, and auditability of stratified sampling workflows.

    This class maintains all the core functionality of EvalData while adding sampling-specific
    context that is crucial for understanding how the sample relates to its source dataset
    and for reproducing sampling results.

    Inheritance Design:
        SampleEvalData IS an EvalData instance, meaning it can be used anywhere an EvalData
        is expected. All EvalData methods, properties, and validation logic are inherited
        unchanged. The sampling metadata is stored as additional object-level attributes,
        not as DataFrame columns, preserving the semantic clarity of the underlying data.

    ## Inherited EvalData Functionality

    All of the following EvalData features are fully inherited and available:

    ### Data Organization & Validation:
        - Comprehensive data integrity validation with clear error messages
        - Automatic ID column generation if not provided
        - Immutability after initialization for thread safety

    ### Inherited Properties:
        - data: Complete evaluation dataset as Polars DataFrame
        - name: Name/identifier of the evaluation dataset
        - id_column: Column containing unique identifiers for each row

    ## New SampleEvalData Attributes

    The following attributes are specific to SampleEvalData and provide comprehensive context
    about the stratified sampling operation:

    Attributes:
        sample_name (str): Human-readable identifier for this specific sample.
            Used to distinguish between different samples from the same source dataset.
            Examples: "Balanced Topic Sample", "Difficulty Stratified 30%", "Q3 Data Sample"

        stratification_columns (list[str]): List of column names that were used
            to define the unique combinations for stratified sampling. These columns determine
            how the original data was grouped before sampling. Can be any valid columns
            from the DataFrame.
            Examples: ["topic", "difficulty"], ["user_type"], ["region", "language"]

        sample_percentage (float): The exact percentage (as a float between 0.0 and 1.0)
            of rows that were sampled from each stratification combination group. This ensures
            proportional representation across all identified stratification categories.
            Examples: 0.5 (50%), 0.2 (20%), 0.75 (75%)

        seed (int): The random seed value used during the sampling process. This integer
            enables exact reproducibility of the sampling operation when the same source
            data and parameters are used.
            Examples: 42, 12345, 999

        sampling_method (str): Programmatic identifier indicating the specific sampling
            strategy used. Defaults to "stratified_by_columns" for samples created
            through the stratified_sample_by_columns method. This field enables filtering and
            categorizing samples by their creation methodology.

    ## Sampling Metadata Design Philosophy

    The sampling metadata is stored as object-level attributes rather than DataFrame
    columns for several important reasons:

    1. **Semantic Clarity**: DataFrame columns describe properties of individual evaluation
       examples, while sampling metadata describes properties of the dataset container itself.

    2. **No Data Redundancy**: Avoids storing the same metadata value in every row,
       which would be wasteful and potentially confusing.

    3. **Clean Separation**: Maintains the distinction between data content (rows/columns)
       and data context (how the container was created).

    4. **Backward Compatibility**: Any code expecting EvalData can work with SampleEvalData
       without modification, as the DataFrame structure is unchanged.

    ## Usage Examples

    Creating a SampleEvalData through stratified_sample_by_columns:
        ```python
        original = EvalData(...)
        sample = original.stratified_sample_by_columns(
            columns=["topic", "difficulty"],
            sample_percentage=0.3,
            sample_name="Balanced Topic-Difficulty Sample",
            seed=42
        )
        # sample is now a SampleEvalData instance
        ```

    Accessing sampling metadata:
        ```python
        print(f"Sample: {sample.sample_name}")
        print(f"Strategy: {sample.sampling_method}")
        print(f"Stratification columns: {sample.stratification_columns}")
        print(f"Percentage sampled: {sample.sample_percentage}")
        print(f"Seed: {sample.seed}")
        ```

    Getting comprehensive sampling information:
        ```python
        info = sample.sampling_info
        print(f"Complete sampling details: {info}")
        ```

    ## Reproducibility

    The combination of seed, stratification_columns, and sample_percentage values
    stored in SampleEvalData instances enables exact reproduction of sampling results:

        ```python
        # Reproduce the exact same sample
        reproduced_sample = original.stratified_sample_by_columns(
            columns=sample.stratification_columns,
            sample_percentage=sample.sample_percentage,
            seed=sample.seed
        )
        # reproduced_sample will contain identical rows to the original sample
        ```

    ## Thread Safety & Immutability

    Like EvalData, SampleEvalData instances are immutable after initialization.
    All sampling metadata attributes are set during construction and cannot be
    modified afterward, ensuring predictable behavior in concurrent environments.

    ## Validation

    SampleEvalData inherits all EvalData validation logic, ensuring that sampled
    datasets maintain the same data integrity guarantees as their source datasets.
    The sampling metadata attributes are validated by Pydantic's type system but
    do not affect the core data validation pipeline.
    """

    sample_name: str
    stratification_columns: list[str]
    sample_percentage: float
    seed: int
    sampling_method: str = "stratified_by_columns"

    @property
    def sampling_info(self) -> dict:
        """Get comprehensive sampling operation details as a dictionary.

        Returns a dictionary containing all sampling-related metadata, providing
        a complete picture of how this sample was created. This is useful for
        logging, debugging, and documenting sampling workflows.

        Returns:
            dict: Dictionary containing all sampling metadata with the following keys:
                - sample_name (str): Human-readable sample identifier
                - sampling_method (str): Sampling strategy used
                - stratification_columns (list[str]): Columns used for grouping
                - sample_percentage (float): Percentage sampled from each group
                - seed (int): Random seed for reproducibility
                - sampled_rows (int): Total number of rows in this sample

        Example:
            ```python
            sample = original.stratified_sample_by_columns(
                columns=["topic"],
                sample_percentage=0.4,
                sample_name="Topic Balanced Sample",
                seed=123
            )

            info = sample.sampling_info
            print(f"Sample '{info['sample_name']}' contains {info['sampled_rows']} rows")
            print(f"Created using {info['sample_percentage']:.1%} from each {info['stratification_columns']} group")
            print(f"Reproducible with seed {info['seed']}")
            ```
        """
        return {
            "sample_name": self.sample_name,
            "sampling_method": self.sampling_method,
            "stratification_columns": self.stratification_columns,
            "sample_percentage": self.sample_percentage,
            "seed": self.seed,
            "sampled_rows": len(self.data),
        }

    def serialize_metadata(
        self,
        data_format: Optional[Literal["json", "csv", "parquet"]],
        data_filename: Optional[str],
    ) -> DataMetadata:
        """Serialize SampleEvalData metadata.

        Args:
            data_format: Format for data serialization.
            data_filename: Name of data file if applicable. Indicates the name of the df if data is written to disk.

        Returns:
            DataMetadata: Metadata object for this SampleEvalData.

        Raises:
            TypeError: If id_column is None during serialization.
        """
        self.logger.info(
            f"Serializing metadata for {self.name} in {data_format} format"
        )

        if self.id_column is None:
            raise TypeError(
                "Cannot serialize SampleEvalData: id_column is None. "
                "Data should be properly initialized before serialization."
            )
        return DataMetadata(
            name=self.name,
            id_column=self.id_column,
            data_file=cast(str, data_filename),
            data_format=cast(Literal["json", "csv", "parquet"], data_format),
            type="SampleEvalData",
            sample_name=self.sample_name,
            stratification_columns=self.stratification_columns,
            sample_percentage=self.sample_percentage,
            seed=self.seed,
            sampling_method=self.sampling_method,
        )

    @classmethod
    def deserialize(
        cls,
        data: pl.DataFrame,
        metadata: DataMetadata,
    ) -> "SampleEvalData":
        """Deserialize EvalData from data file metadata.

        Args:
            data: The loaded DataFrame.
            metadata: DataMetadata object containing data metadata.

        Returns:
            SampleEvalData: Reconstructed SampleEvalData instance.

        Raises:
            DataFileError: If the metadata type is not "SampleEvalData".
            DataFileError: If the metadata sample_name is None.
        """
        if metadata.sample_name is None:
            raise DataFileError("Reload data. MetaData missing required fields")

        return cls(
            data=data,
            name=metadata.name,
            id_column=metadata.id_column,
            sample_name=cast(str, metadata.sample_name),
            stratification_columns=cast(list, metadata.stratification_columns),
            sample_percentage=cast(float, metadata.sample_percentage),
            seed=cast(int, metadata.seed),
            sampling_method=cast(str, metadata.sampling_method),
        )
