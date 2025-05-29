"""Immutable container for evaluation data with column categorization and validation.

This module provides the EvalData class, which serves as a structured container for
LLM evaluation datasets. It enforces column categorization, validates data integrity,
and provides type-safe access to different data categories while preserving original
column names and ensuring immutability after initialization.
"""

import polars as pl
from pydantic import BaseModel, model_validator
import logging
from .exceptions import (
    EmptyColumnsError,
    EmptyDataFrameError,
    ColumnNotFoundError,
    DuplicateColumnError,
)

logger = logging.getLogger(__name__)


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
        uncategorized_columns (list[str]): Column names that exist in the DataFrame
            but were not assigned to any explicit category. Automatically computed
            during validation. A warning is logged when uncategorized columns are present.

    Column Categorization Rules:
        - Every specified column must exist in the provided DataFrame
        - No column can appear in multiple categories (strict uniqueness)
        - At minimum, input_columns and output_columns must be non-empty
        - metadata_columns and human_label_columns are optional and default to empty lists
        - uncategorized_columns is automatically computed and cannot be specified by users
        - Column names are preserved exactly as provided (no normalization or transformation)
    """

    data: pl.DataFrame
    input_columns: list[str]
    output_columns: list[str]
    metadata_columns: list[str] = []
    human_label_columns: list[str] = []
    uncategorized_columns: list[
        str
    ] = []  # Computed during validation, not user-specified

    model_config = {
        "frozen": True,
        "arbitrary_types_allowed": True,  # Allow Polars DataFrame
    }

    @model_validator(mode="after")
    def validate_eval_data(self) -> "EvalData":
        """Validate column specifications and data integrity after initialization.

        This method performs comprehensive validation of the column categorization
        against the provided DataFrame. It ensures data consistency and prevents
        common configuration errors that would cause runtime failures later.

        The uncategorized_columns attribute is automatically populated with DataFrame
        columns that don't belong to any user-specified category. A warning is logged
        when uncategorized columns are present to encourage explicit categorization.

        Validation Steps:
            1. Validate that input_columns and output_columns are non-empty
            2. Validate DataFrame is not empty (has at least one row)
            3. Check that all specified columns exist in the DataFrame
            4. Compute uncategorized_columns from remaining DataFrame columns
            5. Enforce column uniqueness across all categories (including uncategorized)
            6. Log warning if uncategorized columns are present

        The validation follows a "fail-fast" approach where the first encountered
        error immediately raises an exception with a descriptive message.

        Returns:
            EvalData: The validated instance with computed uncategorized_columns

        Raises:
            EmptyColumnsError: If input_columns or output_columns are empty.
            EmptyDataFrameError: If the DataFrame contains no rows of data.
            ColumnNotFoundError: If specified columns don't exist in the DataFrame.
            DuplicateColumnError: If columns appear in multiple categories.
        """

        def _validate_required_columns_non_empty():
            """Validate that required column categories are non-empty.

            Raises:
                EmptyColumnsError: If input_columns or output_columns are empty.
            """
            """Validate that required column categories are non-empty."""
            if not self.input_columns:
                raise EmptyColumnsError(
                    "input_columns cannot be empty - at least one input column is required"
                )
            if not self.output_columns:
                raise EmptyColumnsError(
                    "output_columns cannot be empty - at least one output column is required"
                )

        def _validate_dataframe_not_empty():
            """Validate DataFrame has at least one row of data.

            Raises:
                EmptyDataFrameError: If the DataFrame contains no rows of data.
            """
            if len(self.data) == 0:
                raise EmptyDataFrameError(
                    "DataFrame cannot be empty - at least one row of data is required"
                )

        def _validate_columns_exist_in_dataframe():
            """Check that all user-specified columns exist in the DataFrame.

            Raises:
                ColumnNotFoundError: If any specified column is not found in the DataFrame.
            """
            available_columns = set(self.data.columns)
            user_specified_columns = (
                self.input_columns
                + self.output_columns
                + self.metadata_columns
                + self.human_label_columns
            )

            for column in user_specified_columns:
                if column not in available_columns:
                    raise ColumnNotFoundError(
                        f"Column '{column}' specified in categorization but not found in DataFrame. "
                        f"Available columns: {sorted(available_columns)}"
                    )

        def _compute_uncategorized_columns():
            """Compute uncategorized columns and update the instance.

            Raises:
                AttributeError: If the attempt to set `uncategorized_columns` fails due to
                restrictions in the model's immutability.
            """
            user_specified_columns = set(
                self.input_columns
                + self.output_columns
                + self.metadata_columns
                + self.human_label_columns
            )
            available_columns = set(self.data.columns)
            uncategorized = sorted(available_columns - user_specified_columns)

            try:
                object.__setattr__(self, "uncategorized_columns", uncategorized)
            except AttributeError as error:
                raise AttributeError("Failed to set uncategorized_columns") from error

        def _validate_column_uniqueness():
            """Enforce column uniqueness across all categories including uncategorized.

            Checks that no column appears in more than one category. Categories include
            input_columns, output_columns, metadata_columns, human_label_columns, and
            uncategorized_columns.

            Raises:
                DuplicateColumnError: If any column appears in more than one category.
            """
            column_category_map = {}

            # Check input columns
            for column in self.input_columns:
                column_category_map[column] = "input_columns"

            # Check output columns
            for column in self.output_columns:
                if column in column_category_map:
                    raise DuplicateColumnError(
                        f"Column '{column}' appears in both {column_category_map[column]} "
                        f"and output_columns. Each column can only belong to one category."
                    )
                column_category_map[column] = "output_columns"

            # Check metadata columns
            for column in self.metadata_columns:
                if column in column_category_map:
                    raise DuplicateColumnError(
                        f"Column '{column}' appears in both {column_category_map[column]} "
                        f"and metadata_columns. Each column can only belong to one category."
                    )
                column_category_map[column] = "metadata_columns"

            # Check human label columns
            for column in self.human_label_columns:
                if column in column_category_map:
                    raise DuplicateColumnError(
                        f"Column '{column}' appears in both {column_category_map[column]} "
                        f"and human_label_columns. Each column can only belong to one category."
                    )
                column_category_map[column] = "human_label_columns"

            # Check uncategorized columns (should not conflict by definition, but validate anyway)
            for column in self.uncategorized_columns:
                if column in column_category_map:
                    raise DuplicateColumnError(
                        f"Column '{column}' appears in both {column_category_map[column]} "
                        f"and uncategorized_columns. This should not happen - please report as a bug."
                    )
                column_category_map[column] = "uncategorized_columns"

        def _warn_about_uncategorized_columns():
            """Log warning if uncategorized columns are present."""
            if self.uncategorized_columns:
                logger.warning(
                    f"Found {len(self.uncategorized_columns)} uncategorized columns: {self.uncategorized_columns}. "
                    f"Consider categorizing these columns for clearer evaluation semantics."
                )

        # Run all validations in order
        try:
            _validate_required_columns_non_empty()
        except EmptyColumnsError:
            raise  # re-raise to ensure visibility to ruff's docstring checker

        try:
            _validate_dataframe_not_empty()
        except EmptyDataFrameError:
            raise  # re-raise to ensure visibility to ruff's docstring checker

        try:
            _validate_columns_exist_in_dataframe()
        except ColumnNotFoundError:
            raise  # re-raise to ensure visibility to ruff's docstring checker

        # Compute uncategorized columns before uniqueness validation
        _compute_uncategorized_columns()

        try:
            _validate_column_uniqueness()
        except DuplicateColumnError:
            raise  # re-raise to ensure visibility to ruff's docstring checker

        # Warn about uncategorized columns after all validations pass
        _warn_about_uncategorized_columns()

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
    def uncategorized_data(self) -> pl.DataFrame:
        """Extract uncategorized columns as a DataFrame.

        Returns an empty DataFrame if no uncategorized columns exist.
        """
        if not self.uncategorized_columns:
            # Return empty DataFrame with same schema structure
            return self.data.select([]).clear()
        return self.data.select(self.uncategorized_columns)
