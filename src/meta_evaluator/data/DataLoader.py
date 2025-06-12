"""File for DataLoader Class."""

import os
from pathlib import Path
from typing import Optional
import polars as pl

from .EvalData import EvalData
from .exceptions import DataFileError, DataException


class DataLoader:
    """Universal data ingestion from multiple sources, returns structured EvalData."""

    def __init__(self) -> None:
        """Initialize the DataLoader with default settings."""
        pass

    @staticmethod
    def load_csv(
        file_path: str,
        name: str,
        id_column: Optional[str] = None,
    ) -> EvalData:
        """Load a CSV file and return an EvalData object.

        Loads data from a given CSV file and returns an EvalData object. The EvalData
        object is immutable and enforces strict data validation.

        Performs upfront validation of file existence and permissions before
        attempting to parse the CSV. All file-related errors are wrapped in
        DataFileError for consistent error handling.

        Args:
            file_path: The path to the CSV file to load.
            name: The name of the evaluation dataset.
            id_column: The name of the column containing unique identifiers for each
                example. If not provided (or None), an ID column will be automatically
                generated with row indices.

        Returns:
            EvalData: An immutable container with strict validation.

        Raises:
            DataFileError: If there are file-related issues (file not found,
                permissions, or CSV parsing errors).
            DataException: If there is an error validating the loaded data structure.
        """
        # Convert to Path object for easier manipulation
        file_path_obj = Path(file_path)

        # Upfront file validation
        if not file_path_obj.exists():
            raise DataFileError(f"File not found: {file_path}")

        if not file_path_obj.is_file():
            raise DataFileError(f"Path is not a file: {file_path}")

        if not os.access(file_path_obj, os.R_OK):
            raise DataFileError(f"No read permission for file: {file_path}")

        # Attempt CSV parsing with error wrapping
        try:
            data = pl.read_csv(file_path, has_header=True)
        except Exception as e:
            raise DataFileError(f"Failed to parse CSV file '{file_path}': {e}") from e

        # Create EvalData object with validation
        try:
            output = EvalData(
                data=data,
                name=name,
                id_column=id_column,
            )
        except DataException:
            # Re-raise DataExceptions from EvalData validation unchanged
            raise

        return output
