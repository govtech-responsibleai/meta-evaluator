"""Placeholder file for exceptions in data."""

from abc import ABC, abstractmethod
from typing import Union


class DataException(Exception, ABC):
    """Base class for data-related exceptions. Do not instantiate directly."""

    @abstractmethod
    def __init__(self, message: str = ""):
        """Initializes the DataException with an optional message.

        Args:
            message (str): The exception message. Defaults to an empty string.
        """
        super().__init__(message)


class InvalidColumnNameError(DataException):
    """Exception raised when a column name is invalid."""

    def __init__(self, column_name: str, error: str):
        """Initializes the InvalidColumnNameError with the invalid column name.

        Args:
            column_name (str): The invalid column name.
            error (str): The error message.
        """
        message = f"Invalid column name: '{column_name}' with error: {error}"
        super().__init__(message)


class EmptyColumnListError(DataException):
    """Exception raised when the list of columns is empty."""

    def __init__(self, column_type_name: str):
        """Initializes the EmptyColumnList exception with the column type name.

        Args:
            column_type_name (str): The type name of the column (e.g. "input", "output", etc.).
        """
        message = f"List of {column_type_name} columns is empty."
        super().__init__(message)


class IdColumnExistsError(DataException):
    """Exception raised when the ID column already exists in the dataset."""

    def __init__(self, column_name: str):
        """Initializes the IdColumnExistsError.

        Args:
            column_name (str): The name of the ID column.
        """
        message = f"Id Column with name {column_name} already exists in dataset."
        super().__init__(message)


class ColumnNotFoundError(DataException):
    """Exception raised when a column is not found in the dataset."""

    def __init__(self, column_names: Union[str, list[str]]):
        """Initializes the ColumnNotFoundError.

        Args:
            column_names (Union[str, list[str]]): The name of the column or list of column names.
        """
        if isinstance(column_names, str):
            column_names = [column_names]
        message = f"Column(s) {column_names} not found in dataset."
        super().__init__(message)


class DuplicateColumnError(DataException):
    """Exception raised when a column is found multiple times in the dataset."""

    def __init__(self, column_names: Union[str, list[str]]):
        """Initializes the DuplicateColumnError.

        Args:
            column_names (Union[str, list[str]]): The name of the column or list of column names.
        """
        if isinstance(column_names, str):
            column_names = [column_names]
        message = f"Column(s) {column_names} found multiple times in dataset."
        super().__init__(message)


class InvalidInIDColumnError(DataException):
    """Exception raised when the ID column contains invalid values."""

    def __init__(self, row_numbers: list[int], id_column_name: str):
        """Initializes the NullExistsException.

        Args:
            row_numbers (list[int]): The row numbers where invalid values are found.
            id_column_name (str): The name of the ID column.
        """
        message = f"Invalid values found in ID column {id_column_name} at row numbers: {row_numbers}."
        super().__init__(message)


class DuplicateInIDColumnError(DataException):
    """Exception raised when the ID column contains duplicate values."""

    def __init__(self, duplicate_groups: dict[str, list[int]], id_column_name: str):
        """Initializes the DuplicateInIDColumnError.

        Args:
            duplicate_groups (dict[str, list[int]]): Dictionary mapping duplicate ID values
                to lists of row numbers where they appear.
            id_column_name (str): The name of the ID column.
        """
        details = []
        for value, rows in duplicate_groups.items():
            details.append(f"  - '{value}' appears at rows {rows}")

        message = (
            f"Duplicate values found in ID column '{id_column_name}':\n"
            + "\n".join(details)
        )
        super().__init__(message)


class NullValuesInDataError(DataException):
    """Exception raised when null values are found in non-ID data columns."""

    def __init__(self, null_issues: list[str]):
        """Initializes the NullValuesInDataError.

        Args:
            null_issues (list[str]): List of detailed descriptions of null value locations
                in the format "Column 'name' has null values at rows [x, y] (coordinates: [(x, 'name'), ...])"
        """
        message = f"Null values found in data columns: {'; '.join(null_issues)}"
        super().__init__(message)


class EmptyDataFrameError(DataException):
    """Exception raised when the DataFrame is empty."""

    def __init__(self):
        """Initializes the EmptyDataFrameError.

        This exception is raised when an operation is attempted on an empty DataFrame.
        """
        message = "DataFrame is empty."
        super().__init__(message)


class DataFileError(DataException):
    """Exception raised for file-related errors during data loading."""

    def __init__(self, message: str):
        """Initializes the DataFileError with a descriptive message.

        Args:
            message (str): The error message describing the file-related issue.
        """
        super().__init__(message)
