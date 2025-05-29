"""Custom exceptions for data validation and processing errors in LLM evaluation workflows.

This module defines a hierarchy of specialized exceptions for the data ingestion and
organization components of the LLM evaluation system. These exceptions provide clear,
actionable error messages and enable precise error handling for different validation
failure scenarios.

The exception hierarchy is designed to allow both granular error handling (catching
specific exception types) and broad error handling (catching the base EvalDataError).
All exceptions include detailed context about what went wrong and suggestions for
resolution where applicable.

Exception Hierarchy:
    EvalDataError: Base exception for all data validation errors
    ├── EmptyColumnsError: Required column categories are empty
    ├── EmptyDataFrameError: DataFrame contains no data rows
    ├── ColumnNotFoundError: Specified columns don't exist in DataFrame
    └── DuplicateColumnError: Columns appear in multiple categories

Design Philosophy:
    - Fail-fast validation with clear error messages
    - Specific exception types for different failure modes
    - Actionable error messages that guide users toward solutions
    - Consistent error context including available alternatives where relevant

Usage Patterns:
    Catching specific errors for targeted handling:
        >>> try:
        ...     eval_data = EvalData(df, input_columns=[], output_columns=["response"])
        ... except EmptyColumnsError as e:
        ...     print(f"Column configuration error: {e}")
        ...     # Handle empty columns specifically

    Catching category-specific errors:
        >>> try:
        ...     eval_data = EvalData(df, input_columns=["missing"], output_columns=["response"])
        ... except ColumnNotFoundError as e:
        ...     print(f"Available columns: {e}")
        ...     # Suggest valid column names to user

    Broad error handling for all data validation issues:
        >>> try:
        ...     eval_data = EvalData(df, input_columns=["prompt"], output_columns=["prompt"])
        ... except EvalDataError as e:
        ...     print(f"Data validation failed: {e}")
        ...     # Handle any data validation error generically

Integration with DataLoader:
    These exceptions are primarily raised during EvalData initialization, which occurs
    within DataLoader methods. Users typically encounter them when calling DataLoader
    factory methods rather than constructing EvalData directly:

        >>> try:
        ...     eval_data = DataLoader.from_csv(
        ...         "data.csv",
        ...         input_columns=["nonexistent"],
        ...         output_columns=["response"]
        ...     )
        ... except ColumnNotFoundError as e:
        ...     print(f"CSV column error: {e}")

Error Message Quality:
    All exceptions are designed to provide actionable feedback:
    - ColumnNotFoundError includes the list of available columns
    - DuplicateColumnError specifies which categories conflict
    - EmptyColumnsError clarifies which categories need values
    - Clear suggestions for resolution where applicable
"""


class EvalDataError(Exception):
    """Base exception for EvalData validation errors."""

    pass


class EmptyColumnsError(EvalDataError):
    """Raised when required column categories are empty."""

    pass


class EmptyDataFrameError(EvalDataError):
    """Raised when DataFrame has no rows."""

    pass


class ColumnNotFoundError(EvalDataError):
    """Raised when specified columns don't exist in DataFrame."""

    pass


class DuplicateColumnError(EvalDataError):
    """Raised when columns appear in multiple categories."""

    pass
