"""File for DataLoader Class."""

import logging
import os
from pathlib import Path
from typing import Optional
import polars as pl

from .eval_data import EvalData
from .exceptions import DataFileError

logger = logging.getLogger(__name__)


class DataLoader:
    """Universal data ingestion from multiple sources, returns structured EvalData.

    The DataLoader class provides a simple interface for loading evaluation data
    from various file formats and sources. It serves as an abstraction layer between
    data sources and the EvalData container, making it easier to add support for
    new data sources in the future.

    Current Features:
    - Supports multiple file formats (CSV, JSON, Parquet)
    - Automatic data validation and preprocessing
    - Consistent error handling across formats
    - Direct loading from polars DataFrame

    Future Development:
    This class is designed to be extensible for supporting additional data sources
    and integrations, such as:
    - HuggingFace Datasets integration
    - Database connections (SQL, MongoDB, etc.)
    - Cloud storage services (S3, GCS, etc.)
    - Streaming data sources
    - Custom data formats and protocols

    The abstraction provided by this class ensures that regardless of the data source,
    users always receive a validated EvalData object with consistent behavior and
    guarantees. This makes it easier to add new data sources without changing how
    users work with the evaluation data.
    """

    def __init__(self) -> None:
        """Initialize the DataLoader with default settings."""
        pass

    @staticmethod
    def _validate_file(file_path: str) -> Path:
        """Validate file existence and permissions.

        Args:
            file_path: Path to the data file

        Returns:
            Path object for the validated file

        Raises:
            DataFileError: If file validation fails
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise DataFileError(f"File not found: {file_path}")

        if not file_path_obj.is_file():
            raise DataFileError(f"Path is not a file: {file_path}")

        if not os.access(file_path_obj, os.R_OK):
            raise DataFileError(f"No read permission for file: {file_path}")

        return file_path_obj

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
        attempting to parse the CSV.

        Args:
            file_path: The path to the CSV file to load.
            name: The name of the evaluation dataset.
            id_column: The name of the column containing unique identifiers for each
                example. If not provided (or None), an ID column will be automatically
                generated with row indices.

        Returns:
            EvalData: An immutable container with strict validation.
        """
        logger.info(f"Loading CSV file: {file_path} as dataset '{name}'")

        # Validate file
        DataLoader._validate_file(file_path)

        # Load data using EvalData's static method
        data = EvalData.load_data(file_path, "csv")
        logger.info(f"Successfully loaded {data.height} rows and {data.width} columns")

        # Create EvalData
        output = EvalData(
            data=data,
            name=name,
            id_column=id_column,
        )
        logger.info(f"Created EvalData '{name}' with {len(output.data)} rows")

        return output

    @staticmethod
    def load_json(
        file_path: str,
        name: str,
        id_column: Optional[str] = None,
    ) -> EvalData:
        """Load a JSON file and return an EvalData object.

        Args:
            file_path: The path to the JSON file to load.
            name: The name of the evaluation dataset.
            id_column: The name of the column containing unique identifiers.
                If not provided, an ID column will be automatically generated.

        Returns:
            EvalData: An immutable container with strict validation.
        """
        logger.info(f"Loading JSON file: {file_path} as dataset '{name}'")

        # Validate file
        DataLoader._validate_file(file_path)

        # Load data using EvalData's static method
        data = EvalData.load_data(file_path, "json")
        logger.info(f"Successfully loaded {data.height} rows and {data.width} columns")

        # Create EvalData
        output = EvalData(
            data=data,
            name=name,
            id_column=id_column,
        )
        logger.info(f"Created EvalData '{name}' with {len(output.data)} rows")

        return output

    @staticmethod
    def load_parquet(
        file_path: str,
        name: str,
        id_column: Optional[str] = None,
    ) -> EvalData:
        """Load a Parquet file and return an EvalData object.

        Args:
            file_path: The path to the Parquet file to load.
            name: The name of the evaluation dataset.
            id_column: The name of the column containing unique identifiers.
                If not provided, an ID column will be automatically generated.

        Returns:
            EvalData: An immutable container with strict validation.
        """
        logger.info(f"Loading Parquet file: {file_path} as dataset '{name}'")

        # Validate file
        DataLoader._validate_file(file_path)

        # Load data using EvalData's static method
        data = EvalData.load_data(file_path, "parquet")
        logger.info(f"Successfully loaded {data.height} rows and {data.width} columns")

        # Create EvalData
        output = EvalData(
            data=data,
            name=name,
            id_column=id_column,
        )
        logger.info(f"Created EvalData '{name}' with {len(output.data)} rows")

        return output

    @staticmethod
    def load_from_dataframe(
        data: pl.DataFrame,
        name: str,
        id_column: Optional[str] = None,
    ) -> EvalData:
        """Create an EvalData object from an existing polars DataFrame.

        This method allows direct creation of EvalData from an in-memory DataFrame,
        useful when data is already loaded or generated programmatically.

        Args:
            data: A polars DataFrame containing the evaluation data.
            name: The name of the evaluation dataset.
            id_column: The name of the column containing unique identifiers.
                If not provided, an ID column will be automatically generated.

        Returns:
            EvalData: An immutable container with strict validation.
        """
        logger.info(
            f"Creating EvalData '{name}' from DataFrame with {data.height} rows and {data.width} columns"
        )

        output = EvalData(
            data=data,
            name=name,
            id_column=id_column,
        )
        logger.info(f"Created EvalData '{name}' with {len(output.data)} rows")

        return output
