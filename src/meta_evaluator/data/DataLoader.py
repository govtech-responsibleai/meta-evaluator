"""File for DataLoader Class."""

from typing import Optional
import polars as pl

from . import EvalData, DataException


class DataLoader:
    """Universal data ingestion from multiple sources, returns structured EvalData."""

    def __init__(self) -> None:
        """Initialize the DataLoader with default settings."""
        pass

    @staticmethod
    def load_csv(
        file_path: str,
        input_columns: list[str],
        output_columns: list[str],
        id_column: Optional[str] = None,
        metadata_columns: list[str] = [],
        label_columns: list[str] = [],
    ) -> EvalData:
        """Load a CSV file and return an EvalData object.

        Loads data from a given CSV file and returns an EvalData object with the
        specified columns categorized into input, output, metadata, and human
        labels. The EvalData object is immutable and enforces strict column
        categorization and validation.

        Args:
            file_path: The path to the CSV file to load.
            input_columns: The list of column names corresponding to input data.
            output_columns: The list of column names corresponding to output data.
            id_column: The name of the column containing unique identifiers for each
                example. If not provided (or None), an ID column will be automatically
                generated with row indices.
            metadata_columns: The list of column names containing metadata.
            label_columns: The list of column names containing human labels.

        Returns:
            EvalData: An immutable container with categorized columns and strict
                validation.

        Raises:
            DataException: If there is an error loading the data or validating it.
        """
        data = pl.read_csv(file_path, has_header=True)
        try:
            output = EvalData(
                data=data,
                input_columns=input_columns,
                output_columns=output_columns,
                id_column=id_column,
                metadata_columns=metadata_columns,
                human_label_columns=label_columns,
            )
        except DataException:
            raise

        return output
