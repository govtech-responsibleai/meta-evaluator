"""File for testing the EvalData module."""

import pytest
import polars as pl
from unittest.mock import MagicMock
from pydantic import ValidationError

from meta_evaluator.data import EvalData
from meta_evaluator.data.exceptions import (
    EmptyColumnsError,
    EmptyDataFrameError,
    ColumnNotFoundError,
    DuplicateColumnError,
)


class TestEvalData:
    """Test suite for the EvalData class."""

    @pytest.fixture
    def minimal_valid_data(self) -> pl.DataFrame:
        """Provides minimal valid DataFrame with 2 columns, 1 row.

        This is the simplest valid DataFrame that can pass validation.

        Returns:
            pl.DataFrame: DataFrame with one input and one output column.
        """
        return pl.DataFrame(
            {
                "input_col": ["test input"],
                "output_col": ["test output"],
            }
        )

    @pytest.fixture
    def rich_valid_data(self) -> pl.DataFrame:
        """Provides rich DataFrame with all column types and uncategorized columns.

        This DataFrame contains multiple rows and representatives of all
        possible column categories plus some uncategorized columns.

        Returns:
            pl.DataFrame: DataFrame with diverse column types and data.
        """
        return pl.DataFrame(
            {
                "prompt": ["What is AI?", "Explain ML"],
                "response": ["AI is...", "ML is..."],
                "timestamp": ["2024-01-01", "2024-01-02"],
                "rating": [5, 4],
                "uncategorized_field": ["extra1", "extra2"],
                "another_extra": ["extra3", "extra4"],
            }
        )

    @pytest.fixture
    def empty_dataframe(self) -> pl.DataFrame:
        """Provides empty DataFrame for testing empty data validation.

        Returns:
            pl.DataFrame: Empty DataFrame with proper schema but no rows.
        """
        return pl.DataFrame(
            {"input_col": [], "output_col": []},
            schema={"input_col": pl.Utf8, "output_col": pl.Utf8},
        )

    @pytest.fixture
    def single_column_data(self) -> pl.DataFrame:
        """Provides DataFrame with only one column for testing missing columns.

        Returns:
            pl.DataFrame: DataFrame with only one column.
        """
        return pl.DataFrame({"only_col": ["test data"]})

    @pytest.fixture
    def mock_logger(self, mocker) -> MagicMock:
        """Mocks the logger instance used by EvalData.

        This fixture patches the logging.getLogger method to return a mock logger
        instance, allowing verification of logging calls.

        Args:
            mocker: The pytest-mock mocker fixture.

        Returns:
            MagicMock: A mock logger instance.
        """
        return mocker.patch("meta_evaluator.data.EvalData.logger")

    # Happy Path Tests

    def test_initialization_minimal_valid_data(self, minimal_valid_data):
        """Verify initialization with minimal valid data."""
        eval_data = EvalData(
            data=minimal_valid_data,
            input_columns=["input_col"],
            output_columns=["output_col"],
        )

        assert eval_data.data.equals(minimal_valid_data)
        assert eval_data.input_columns == ["input_col"]
        assert eval_data.output_columns == ["output_col"]
        assert eval_data.metadata_columns == []
        assert eval_data.human_label_columns == []
        assert eval_data._uncategorized_columns == []

    def test_initialization_rich_data_with_all_categories(self, rich_valid_data):
        """Verify initialization with all column categories."""
        eval_data = EvalData(
            data=rich_valid_data,
            input_columns=["prompt"],
            output_columns=["response"],
            metadata_columns=["timestamp"],
            human_label_columns=["rating"],
        )

        assert len(eval_data._uncategorized_columns) == 2
        assert "uncategorized_field" in eval_data._uncategorized_columns
        assert "another_extra" in eval_data._uncategorized_columns
        assert eval_data.input_columns == ["prompt"]
        assert eval_data.output_columns == ["response"]
        assert eval_data.metadata_columns == ["timestamp"]
        assert eval_data.human_label_columns == ["rating"]

    def test_initialization_all_columns_categorized_no_warning(
        self, rich_valid_data, mock_logger
    ):
        """Verify no warning when all columns are categorized."""
        eval_data = EvalData(
            data=rich_valid_data,
            input_columns=["prompt"],
            output_columns=["response"],
            metadata_columns=["timestamp", "uncategorized_field"],
            human_label_columns=["rating", "another_extra"],
        )

        assert eval_data._uncategorized_columns == []
        mock_logger.warning.assert_not_called()

    def test_initialization_multiple_columns_per_category(self, rich_valid_data):
        """Verify initialization with multiple columns per category."""
        eval_data = EvalData(
            data=rich_valid_data,
            input_columns=["prompt", "timestamp"],
            output_columns=["response", "rating"],
            metadata_columns=["uncategorized_field"],
            human_label_columns=["another_extra"],
        )

        assert len(eval_data.input_columns) == 2
        assert len(eval_data.output_columns) == 2
        assert len(eval_data.metadata_columns) == 1
        assert len(eval_data.human_label_columns) == 1
        assert eval_data._uncategorized_columns == []

    # Required Columns Validation Tests

    def test_empty_input_columns_raises_error(self, minimal_valid_data):
        """Empty input_columns raises EmptyColumnsError."""
        with pytest.raises(EmptyColumnsError) as excinfo:
            EvalData(
                data=minimal_valid_data,
                input_columns=[],  # Empty!
                output_columns=["output_col"],
            )

        assert "input_columns cannot be empty" in str(excinfo.value)

    def test_empty_output_columns_raises_error(self, minimal_valid_data):
        """Empty output_columns raises EmptyColumnsError."""
        with pytest.raises(EmptyColumnsError) as excinfo:
            EvalData(
                data=minimal_valid_data,
                input_columns=["input_col"],
                output_columns=[],  # Empty!
            )

        assert "output_columns cannot be empty" in str(excinfo.value)

    def test_both_input_and_output_empty_raises_input_error_first(
        self, minimal_valid_data
    ):
        """Both empty raises input error first (fail-fast behavior)."""
        with pytest.raises(EmptyColumnsError) as excinfo:
            EvalData(
                data=minimal_valid_data,
                input_columns=[],  # Empty!
                output_columns=[],  # Also empty!
            )

        # Should get input error first due to fail-fast validation order
        assert "input_columns cannot be empty" in str(excinfo.value)

    # DataFrame Empty Validation Tests

    def test_empty_dataframe_raises_error(self, empty_dataframe):
        """Empty DataFrame raises EmptyDataFrameError."""
        with pytest.raises(EmptyDataFrameError) as excinfo:
            EvalData(
                data=empty_dataframe,
                input_columns=["input_col"],
                output_columns=["output_col"],
            )

        assert "DataFrame cannot be empty" in str(excinfo.value)

    # Column Existence Validation Tests

    def test_missing_input_column_raises_error(self, minimal_valid_data):
        """Missing input column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError) as excinfo:
            EvalData(
                data=minimal_valid_data,
                input_columns=["nonexistent_input"],
                output_columns=["output_col"],
            )

        assert (
            "Column 'nonexistent_input' specified in categorization but not found"
            in str(excinfo.value)
        )
        assert "Available columns:" in str(excinfo.value)

    def test_missing_output_column_raises_error(self, minimal_valid_data):
        """Missing output column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError) as excinfo:
            EvalData(
                data=minimal_valid_data,
                input_columns=["input_col"],
                output_columns=["nonexistent_output"],
            )

        assert (
            "Column 'nonexistent_output' specified in categorization but not found"
            in str(excinfo.value)
        )

    def test_missing_metadata_column_raises_error(self, minimal_valid_data):
        """Missing metadata column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError) as excinfo:
            EvalData(
                data=minimal_valid_data,
                input_columns=["input_col"],
                output_columns=["output_col"],
                metadata_columns=["nonexistent_metadata"],
            )

        assert (
            "Column 'nonexistent_metadata' specified in categorization but not found"
            in str(excinfo.value)
        )

    def test_missing_human_label_column_raises_error(self, minimal_valid_data):
        """Missing human label column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError) as excinfo:
            EvalData(
                data=minimal_valid_data,
                input_columns=["input_col"],
                output_columns=["output_col"],
                human_label_columns=["nonexistent_label"],
            )

        assert (
            "Column 'nonexistent_label' specified in categorization but not found"
            in str(excinfo.value)
        )

    def test_multiple_missing_columns_reports_first_one(self, single_column_data):
        """Multiple missing columns reports first one found (fail-fast)."""
        with pytest.raises(ColumnNotFoundError) as excinfo:
            EvalData(
                data=single_column_data,
                input_columns=["missing_input"],
                output_columns=["missing_output"],
                metadata_columns=["missing_metadata"],
            )

        # Should report the first missing column encountered during validation
        error_message = str(excinfo.value)
        assert "specified in categorization but not found" in error_message

    # Duplicate Column Tests (All Combinations)

    def test_duplicate_input_output_raises_error(self, minimal_valid_data):
        """Column in both input and output raises DuplicateColumnError."""
        with pytest.raises(DuplicateColumnError) as excinfo:
            EvalData(
                data=minimal_valid_data,
                input_columns=["input_col"],
                output_columns=["input_col"],  # Duplicate!
            )

        assert (
            "Column 'input_col' appears in both input_columns and output_columns"
            in str(excinfo.value)
        )

    def test_duplicate_input_metadata_raises_error(self, rich_valid_data):
        """Column in both input and metadata raises DuplicateColumnError."""
        with pytest.raises(DuplicateColumnError) as excinfo:
            EvalData(
                data=rich_valid_data,
                input_columns=["prompt"],
                output_columns=["response"],
                metadata_columns=["prompt"],  # Duplicate!
            )

        assert (
            "Column 'prompt' appears in both input_columns and metadata_columns"
            in str(excinfo.value)
        )

    def test_duplicate_input_human_label_raises_error(self, rich_valid_data):
        """Column in both input and human_label raises DuplicateColumnError."""
        with pytest.raises(DuplicateColumnError) as excinfo:
            EvalData(
                data=rich_valid_data,
                input_columns=["prompt"],
                output_columns=["response"],
                human_label_columns=["prompt"],  # Duplicate!
            )

        assert (
            "Column 'prompt' appears in both input_columns and human_label_columns"
            in str(excinfo.value)
        )

    def test_duplicate_output_metadata_raises_error(self, rich_valid_data):
        """Column in both output and metadata raises DuplicateColumnError."""
        with pytest.raises(DuplicateColumnError) as excinfo:
            EvalData(
                data=rich_valid_data,
                input_columns=["prompt"],
                output_columns=["response"],
                metadata_columns=["response"],  # Duplicate!
            )

        assert (
            "Column 'response' appears in both output_columns and metadata_columns"
            in str(excinfo.value)
        )

    def test_duplicate_output_human_label_raises_error(self, rich_valid_data):
        """Column in both output and human_label raises DuplicateColumnError."""
        with pytest.raises(DuplicateColumnError) as excinfo:
            EvalData(
                data=rich_valid_data,
                input_columns=["prompt"],
                output_columns=["response"],
                human_label_columns=["response"],  # Duplicate!
            )

        assert (
            "Column 'response' appears in both output_columns and human_label_columns"
            in str(excinfo.value)
        )

    def test_duplicate_metadata_human_label_raises_error(self, rich_valid_data):
        """Column in both metadata and human_label raises DuplicateColumnError."""
        with pytest.raises(DuplicateColumnError) as excinfo:
            EvalData(
                data=rich_valid_data,
                input_columns=["prompt"],
                output_columns=["response"],
                metadata_columns=["rating"],
                human_label_columns=["rating"],  # Duplicate!
            )

        assert (
            "Column 'rating' appears in both metadata_columns and human_label_columns"
            in str(excinfo.value)
        )

    def test_input_data_property(self, rich_valid_data):
        """Verify input_data property returns correct columns."""
        eval_data = EvalData(
            data=rich_valid_data, input_columns=["prompt"], output_columns=["response"]
        )

        input_data = eval_data.input_data
        assert input_data.columns == ["prompt"]
        assert len(input_data) == 2
        assert input_data["prompt"].to_list() == ["What is AI?", "Explain ML"]

    def test_output_data_property(self, rich_valid_data):
        """Verify output_data property returns correct columns."""
        eval_data = EvalData(
            data=rich_valid_data, input_columns=["prompt"], output_columns=["response"]
        )

        output_data = eval_data.output_data
        assert output_data.columns == ["response"]
        assert len(output_data) == 2
        assert output_data["response"].to_list() == ["AI is...", "ML is..."]

    def test_metadata_data_property_with_columns(self, rich_valid_data):
        """Verify metadata_data property with non-empty metadata_columns."""
        eval_data = EvalData(
            data=rich_valid_data,
            input_columns=["prompt"],
            output_columns=["response"],
            metadata_columns=["timestamp", "rating"],
        )

        metadata_data = eval_data.metadata_data
        assert set(metadata_data.columns) == {"timestamp", "rating"}
        assert len(metadata_data) == 2

    def test_metadata_data_property_empty_columns(self, minimal_valid_data):
        """Verify metadata_data property with empty metadata_columns."""
        eval_data = EvalData(
            data=minimal_valid_data,
            input_columns=["input_col"],
            output_columns=["output_col"],
            metadata_columns=[],  # Empty
        )

        metadata_data = eval_data.metadata_data
        assert len(metadata_data.columns) == 0
        assert len(metadata_data) == 0  # Same number of rows as original data

    def test_human_label_data_property_with_columns(self, rich_valid_data):
        """Verify human_label_data property with non-empty human_label_columns."""
        eval_data = EvalData(
            data=rich_valid_data,
            input_columns=["prompt"],
            output_columns=["response"],
            human_label_columns=["rating"],
        )

        human_label_data = eval_data.human_label_data
        assert human_label_data.columns == ["rating"]
        assert len(human_label_data) == 2
        assert human_label_data["rating"].to_list() == [5, 4]

    def test_human_label_data_property_empty_columns(self, minimal_valid_data):
        """Verify human_label_data property with empty human_label_columns."""
        eval_data = EvalData(
            data=minimal_valid_data,
            input_columns=["input_col"],
            output_columns=["output_col"],
            human_label_columns=[],  # Empty
        )

        human_label_data = eval_data.human_label_data
        assert len(human_label_data.columns) == 0
        assert len(human_label_data) == 0  # Same number of rows as original data

    def test_uncategorized_data_property_empty(self, minimal_valid_data):
        """Verify uncategorized_data returns empty DataFrame when no uncategorized columns."""
        eval_data = EvalData(
            data=minimal_valid_data,
            input_columns=["input_col"],
            output_columns=["output_col"],
        )

        uncategorized_data = eval_data.uncategorized_data
        assert len(uncategorized_data.columns) == 0
        assert len(uncategorized_data) == 0  # Empty DataFrame

    def test_uncategorized_data_property_with_data(self, rich_valid_data):
        """Verify uncategorized_data returns correct columns."""
        eval_data = EvalData(
            data=rich_valid_data, input_columns=["prompt"], output_columns=["response"]
        )

        uncategorized_data = eval_data.uncategorized_data
        assert "uncategorized_field" in uncategorized_data.columns
        assert "another_extra" in uncategorized_data.columns
        assert "timestamp" in uncategorized_data.columns
        assert "rating" in uncategorized_data.columns
        assert len(uncategorized_data) == 2

    # Immutability Tests

    def test_immutability_data_attribute(self, minimal_valid_data):
        """Verify data attribute cannot be modified."""
        eval_data = EvalData(
            data=minimal_valid_data,
            input_columns=["input_col"],
            output_columns=["output_col"],
        )

        with pytest.raises((ValidationError, AttributeError)):
            eval_data.data = pl.DataFrame({"new": ["data"]})

    def test_immutability_input_columns_attribute(self, minimal_valid_data):
        """Verify input_columns cannot be modified."""
        eval_data = EvalData(
            data=minimal_valid_data,
            input_columns=["input_col"],
            output_columns=["output_col"],
        )

        with pytest.raises((ValidationError, AttributeError)):
            eval_data.input_columns = ["new_col"]

    def test_immutability_output_columns_attribute(self, minimal_valid_data):
        """Verify output_columns cannot be modified."""
        eval_data = EvalData(
            data=minimal_valid_data,
            input_columns=["input_col"],
            output_columns=["output_col"],
        )

        with pytest.raises((ValidationError, AttributeError)):
            eval_data.output_columns = ["new_col"]

    def test_immutability_metadata_columns_attribute(self, minimal_valid_data):
        """Verify metadata_columns cannot be modified."""
        eval_data = EvalData(
            data=minimal_valid_data,
            input_columns=["input_col"],
            output_columns=["output_col"],
        )

        with pytest.raises((ValidationError, AttributeError)):
            eval_data.metadata_columns = ["new_col"]

    def test_immutability_human_label_columns_attribute(self, minimal_valid_data):
        """Verify human_label_columns cannot be modified."""
        eval_data = EvalData(
            data=minimal_valid_data,
            input_columns=["input_col"],
            output_columns=["output_col"],
        )

        with pytest.raises((ValidationError, AttributeError)):
            eval_data.human_label_columns = ["new_col"]

    # Logging Tests

    def test_uncategorized_columns_warning_logged(self, rich_valid_data, mock_logger):
        """Verify warning is logged for uncategorized columns."""
        EvalData(
            data=rich_valid_data, input_columns=["prompt"], output_columns=["response"]
        )

        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Found 4 uncategorized columns" in warning_call
        assert "uncategorized_field" in warning_call
        assert "another_extra" in warning_call
        assert "Consider categorizing these columns" in warning_call

    def test_no_warning_when_no_uncategorized_columns(
        self, minimal_valid_data, mock_logger
    ):
        """Verify no warning when all columns are categorized."""
        EvalData(
            data=minimal_valid_data,
            input_columns=["input_col"],
            output_columns=["output_col"],
        )

        mock_logger.warning.assert_not_called()

    def test_uncategorized_columns_count_in_warning(self, rich_valid_data, mock_logger):
        """Verify warning includes correct count of uncategorized columns."""
        EvalData(
            data=rich_valid_data,
            input_columns=["prompt"],
            output_columns=["response"],
            metadata_columns=["timestamp"],  # Categorize one, leaving 3 uncategorized
        )

        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Found 3 uncategorized columns" in warning_call

    # Edge Cases and Boundary Conditions

    def test_special_character_column_names(self):
        """Verify handling of special characters in column names."""
        data = pl.DataFrame(
            {
                "input-col": ["test"],
                "output_col": ["test"],
                "meta@col": ["test"],
                "human.label": ["test"],
            }
        )

        eval_data = EvalData(
            data=data,
            input_columns=["input-col"],
            output_columns=["output_col"],
            metadata_columns=["meta@col"],
            human_label_columns=["human.label"],
        )

        assert eval_data.input_columns == ["input-col"]
        assert eval_data.metadata_columns == ["meta@col"]
        assert eval_data.human_label_columns == ["human.label"]
        assert eval_data._uncategorized_columns == []

    def test_numeric_column_names(self):
        """Verify handling of numeric column names."""
        data = pl.DataFrame({"1": ["input1"], "2": ["output1"], "3": ["metadata1"]})

        eval_data = EvalData(
            data=data, input_columns=["1"], output_columns=["2"], metadata_columns=["3"]
        )

        assert eval_data.input_columns == ["1"]
        assert eval_data.output_columns == ["2"]
        assert eval_data.metadata_columns == ["3"]

    def test_very_long_column_names(self):
        """Verify handling of very long column names."""
        long_name = "a" * 1000
        data = pl.DataFrame({long_name: ["input"], "output": ["output"]})

        eval_data = EvalData(
            data=data, input_columns=[long_name], output_columns=["output"]
        )

        assert eval_data.input_columns == [long_name]

    def test_large_dataframe_performance(self):
        """Verify performance with larger DataFrame."""
        # Create a larger DataFrame to test performance
        n_rows = 10000
        data = pl.DataFrame(
            {
                "input": [f"input_{i}" for i in range(n_rows)],
                "output": [f"output_{i}" for i in range(n_rows)],
                "metadata": [f"meta_{i}" for i in range(n_rows)],
            }
        )

        eval_data = EvalData(
            data=data,
            input_columns=["input"],
            output_columns=["output"],
            metadata_columns=["metadata"],
        )

        assert len(eval_data.data) == n_rows
        assert eval_data._uncategorized_columns == []

    def test_different_polars_dtypes(self):
        """Verify handling of different Polars data types."""
        data = pl.DataFrame(
            {
                "str_col": ["text1", "text2"],
                "int_col": [1, 2],
                "float_col": [1.1, 2.2],
                "bool_col": [True, False],
                "date_col": ["2024-01-01", "2024-01-02"],
            }
        ).with_columns([pl.col("date_col").str.strptime(pl.Date, "%Y-%m-%d")])

        eval_data = EvalData(
            data=data,
            input_columns=["str_col"],
            output_columns=["int_col"],
            metadata_columns=["float_col", "bool_col"],
            human_label_columns=["date_col"],
        )

        assert eval_data._uncategorized_columns == []
        assert len(eval_data.input_data.columns) == 1
        assert len(eval_data.output_data.columns) == 1
        assert len(eval_data.metadata_data.columns) == 2
        assert len(eval_data.human_label_data.columns) == 1
