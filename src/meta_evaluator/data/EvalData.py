"""Immutable container for evaluation data with column categorization and validation.

This module provides the EvalData class, which serves as a structured container for
LLM evaluation datasets. It enforces column categorization, validates data integrity,
and provides type-safe access to different data categories while preserving original
column names and ensuring immutability after initialization.
"""

from typing import Any, Optional
import polars as pl
from pydantic import BaseModel, PrivateAttr, model_validator
import logging
from .exceptions import (
    ColumnNotFoundError,
    DuplicateColumnError,
    EmptyDataFrameError,
    InvalidColumnNameError,
    EmptyColumnListError,
    IdColumnExistsError,
    DuplicateInIDColumnError,
    InvalidInIDColumnError,
    NullValuesInDataError,
    InvalidNameError,
)

logger = logging.getLogger(__name__)
ID_COLUMN_NAME = "id"


class EvalData(BaseModel):
    """Immutable container for evaluation data with strict column categorization and validation.

    This class provides a structured interface for organizing evaluation datasets by categorizing
    columns into semantic groups (inputs, outputs, metadata, human labels, uncategorized) while
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
        input_columns (list[str]): Column names containing input data for evaluation.
            These typically represent prompts, questions, or other stimuli sent to
            models being evaluated.
        output_columns (list[str]): Column names containing model outputs or responses.
            These represent the generated content that will be evaluated for quality,
            correctness, or other metrics.
        metadata_columns (list[str]): Column names containing contextual information
            about each evaluation example. May include identifiers, timestamps,
            difficulty ratings, or other descriptive attributes.
        human_label_columns (list[str]): Column names containing human-provided
            ground truth labels, ratings, or annotations used as evaluation targets.

    Column Categorization Rules:
        - Every specified column must exist in the provided DataFrame
        - No column name may appear more than once – either within a category list or across lists (strict uniqueness).
        - At minimum, input_columns and output_columns must be non-empty
        - metadata_columns and human_label_columns are optional and default to empty lists
        - _uncategorized_columns is automatically computed and cannot be specified by users
        - Column names are preserved exactly as provided (no normalization or transformation)
    """

    data: pl.DataFrame
    name: str
    id_column: Optional[str] = None
    input_columns: list[str]
    output_columns: list[str]
    metadata_columns: list[str] = []
    human_label_columns: list[str] = []
    _uncategorized_columns: list[str] = PrivateAttr(
        default_factory=list
    )  # Computed during validation, not user-specified
    _initialized: bool = PrivateAttr(default=False)
    _user_set_id: bool = PrivateAttr(default=False)

    model_config = {
        "arbitrary_types_allowed": True,  # Allow Polars DataFrame
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
    def validate_given_column_names(cls, data: Any) -> Any:
        """Validate the given column names.

        This function is called before the model is initialized, and it checks the
        given column names for the following conditions:

        - The column name must not be empty.
        - The column name must start with a letter or underscore.
        - The column name must only contain letters, numbers, and underscores.

        If the column name is invalid, an InvalidColumnNameError is raised.

        It only requires input and output columns to be non-empty.

        Args:
            data (dict): The data to validate.

        Raises:
            EmptyColumnListError: If any of the column lists are empty.
            InvalidColumnNameError: If any of the column names are invalid.

        Returns:
            dict: The validated data.
        """
        if not isinstance(data, dict):
            return data

        input_columns = data.get("input_columns")
        if not input_columns:
            raise EmptyColumnListError("input")  # For ruff's docstring checker
        for column_name in input_columns:
            try:
                cls.check_single_column_name(column_name)
            except InvalidColumnNameError:
                raise  # For ruff's docstring checker

        output_columns = data.get("output_columns")
        if not output_columns:
            raise EmptyColumnListError("output")  # For ruff's docstring checker
        for column_name in output_columns:
            try:
                cls.check_single_column_name(column_name)
            except InvalidColumnNameError:
                raise  # For ruff's docstring checker

        metadata_columns = data.get("metadata_columns", [])
        for column_name in metadata_columns:
            try:
                cls.check_single_column_name(column_name)
            except InvalidColumnNameError:
                raise  # For ruff's docstring checker

        human_label_columns = data.get("human_label_columns", [])
        for column_name in human_label_columns:
            try:
                cls.check_single_column_name(column_name)
            except InvalidColumnNameError:
                raise  # For ruff's docstring checker
        return data

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

    def _validate_data_integrity(self) -> "EvalData":
        """Validate data integrity after all fields are set.

        This method validates the data integrity in the following ways:

        1. Checks if all specified columns are present in the DataFrame.
        2. Checks if any column is specified more than once.
        3. Computes uncategorized columns (i.e., columns present in the DataFrame
            but not specified in any category).

        Returns:
            EvalData: The validated EvalData

        Raises:
            ColumnNotFoundError: If any of the specified columns are missing from the DataFrame.
            DuplicateColumnError: If any column is specified more than once.

        Note:
            Error vs Logging:
            - Missing or duplicate columns trigger exceptions, halting execution and requiring user intervention.
            - Uncategorized columns are logged as warnings, indicating non-critical issues that may warrant attention.
        """
        all_specified_columns = (
            self.input_columns
            + self.output_columns
            + self.metadata_columns
            + self.human_label_columns
            + ([self.id_column] if self.id_column else [])
        )

        missing_cols = set(all_specified_columns) - set(self.data.columns)
        if missing_cols:
            raise ColumnNotFoundError(list(missing_cols))

        # 2. Check for duplicates across all categories
        column_counts = {}
        for col in all_specified_columns:
            column_counts[col] = column_counts.get(col, 0) + 1

        duplicates = {col: count for col, count in column_counts.items() if count > 1}
        duplicates_list = list(duplicates.keys())
        if duplicates:
            raise DuplicateColumnError(duplicates_list)

        # 3. Compute uncategorized columns
        all_df_columns = set(self.data.columns)
        categorized_columns = set(all_specified_columns)
        uncategorized = list(all_df_columns - categorized_columns)

        if uncategorized:
            logger.warning(f"Uncategorized columns detected: {uncategorized}")

        # 4. Set private attributes
        self._uncategorized_columns = uncategorized

        return self

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

            logger.info(
                f"Auto-generated ID column '{ID_COLUMN_NAME}' with {len(final_data)} entries"
            )
        else:
            # User provided an ID column
            self._user_set_id = True
            logger.debug(f"Using user-provided ID column: '{self.id_column}'")

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
                logger.warning(
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
                logger.warning(
                    f"Column '{col}' has whitespace-only values at rows {whitespace_rows} (coordinates: {coordinates})"
                )

    @model_validator(mode="after")
    def complete_data_validation(self) -> "EvalData":
        """Complete data validation pipeline in correct order.

        Returns:
            EvalData: The instance with all validation checks completed.
        """
        # Step 1: Validate data integrity
        self._validate_data_integrity()

        # Step 2: Create ID column if needed
        self._create_id_column_if_needed()

        # Step 3: Final validation checks
        self._final_validation_checks()

        self._initialized = True

        return self

    @property
    def input_data(self) -> pl.DataFrame:
        """Extract input columns as a DataFrame."""
        return self.data.select(self.input_columns)

    @property
    def output_data(self) -> pl.DataFrame:
        """Extract output columns as a DataFrame."""
        return self.data.select(self.output_columns)

    @property
    def metadata_data(self) -> pl.DataFrame:
        """Extract metadata columns as a DataFrame."""
        return self.data.select(self.metadata_columns)

    @property
    def human_label_data(self) -> pl.DataFrame:
        """Extract human label columns as a DataFrame."""
        return self.data.select(self.human_label_columns)

    @property
    def uncategorized_column_names(self) -> list[str]:
        """Return the list of uncategorized column names."""
        return self._uncategorized_columns

    @property
    def uncategorized_data(self) -> pl.DataFrame:
        """Extract uncategorized columns as a DataFrame.

        Returns an empty DataFrame if no uncategorized columns exist.
        """
        if not self._uncategorized_columns:
            # Return empty DataFrame with same schema structure
            return self.data.select([]).clear()
        return self.data.select(self._uncategorized_columns)

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
