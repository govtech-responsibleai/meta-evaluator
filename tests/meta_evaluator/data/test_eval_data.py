"""Test suite for the EvalData class with comprehensive path coverage."""

import pytest
from unittest.mock import MagicMock
import polars as pl
import logging
from meta_evaluator.data import EvalData
from meta_evaluator.data.exceptions import (
    EmptyDataFrameError,
    InvalidColumnNameError,
    IdColumnExistsError,
    DuplicateInIDColumnError,
    InvalidInIDColumnError,
    NullValuesInDataError,
    InvalidNameError,
)


class TestEvalData:
    """Comprehensive test suite for EvalData class achieving 100% path coverage."""

    @pytest.fixture
    def valid_dataframe(self) -> pl.DataFrame:
        """Provides a valid DataFrame for testing.

        Returns:
            pl.DataFrame: A DataFrame with sample evaluation data.
        """
        return pl.DataFrame(
            {
                "question": ["What is 2+2?", "What is 3+3?"],
                "answer": ["4", "6"],
                "model_response": ["Four", "Six"],
                "difficulty": ["easy", "easy"],
                "human_rating": [5, 4],
                "extra_col": ["a", "b"],
            }
        )

    @pytest.fixture
    def minimal_dataframe(self) -> pl.DataFrame:
        """Provides a minimal valid DataFrame.

        Returns:
            pl.DataFrame: A DataFrame with minimal test data columns.
        """
        return pl.DataFrame(
            {
                "input": ["test1", "test2"],
                "output": ["result1", "result2"],
            }
        )

    @pytest.fixture
    def mock_logger(self, mocker) -> MagicMock:
        """Mocks the logger for testing logging behavior.

        Returns:
            MagicMock: A mock logger instance.
        """
        return mocker.patch("logging.getLogger").return_value

    # === Column Name Validation Tests ===

    def test_check_single_column_name_valid(self):
        """Test valid column names pass validation."""
        # Valid names should not raise
        EvalData.check_single_column_name("valid_name")
        EvalData.check_single_column_name("_starts_with_underscore")
        EvalData.check_single_column_name("has123numbers")
        EvalData.check_single_column_name("CamelCase")

    def test_check_single_column_name_empty(self):
        """Test empty column names raise InvalidColumnNameError."""
        with pytest.raises(InvalidColumnNameError, match="Column name is empty"):
            EvalData.check_single_column_name("")

        with pytest.raises(
            InvalidColumnNameError,
            match="Column name must start with a letter or underscore",
        ):
            EvalData.check_single_column_name("   ")  # Whitespace only

    def test_check_single_column_name_invalid_first_char(self):
        """Test column names with invalid first characters."""
        with pytest.raises(
            InvalidColumnNameError, match="must start with a letter or underscore"
        ):
            EvalData.check_single_column_name("123invalid")

        with pytest.raises(
            InvalidColumnNameError, match="must start with a letter or underscore"
        ):
            EvalData.check_single_column_name("-invalid")

    def test_check_single_column_name_invalid_chars(self):
        """Test column names with invalid characters."""
        with pytest.raises(
            InvalidColumnNameError,
            match="must only contain letters, numbers, and underscores",
        ):
            EvalData.check_single_column_name("invalid-name")

        with pytest.raises(
            InvalidColumnNameError,
            match="must only contain letters, numbers, and underscores",
        ):
            EvalData.check_single_column_name("invalid name")

    # === Dataset Name Validation Tests ===

    def test_check_dataset_name_valid(self):
        """Test valid dataset names pass validation."""
        # Valid names should not raise
        EvalData.check_dataset_name("valid_name")
        EvalData.check_dataset_name("_starts_with_underscore")
        EvalData.check_dataset_name("has123numbers")
        EvalData.check_dataset_name("CamelCase")

    def test_check_dataset_name_empty(self):
        """Test empty dataset names raise InvalidNameError."""
        with pytest.raises(InvalidNameError, match="Dataset name is empty"):
            EvalData.check_dataset_name("")

    def test_check_dataset_name_whitespace_only(self):
        """Test whitespace-only dataset names raise InvalidNameError."""
        with pytest.raises(InvalidNameError, match="contains only whitespace"):
            EvalData.check_dataset_name("   ")

        with pytest.raises(InvalidNameError, match="contains only whitespace"):
            EvalData.check_dataset_name("\t\n  ")

    def test_check_dataset_name_invalid_first_char(self):
        """Test dataset names with invalid first characters."""
        with pytest.raises(
            InvalidNameError, match="must start with a letter or underscore"
        ):
            EvalData.check_dataset_name("123invalid")

        with pytest.raises(
            InvalidNameError, match="must start with a letter or underscore"
        ):
            EvalData.check_dataset_name("-invalid")

    def test_check_dataset_name_invalid_chars(self):
        """Test dataset names with invalid characters."""
        with pytest.raises(
            InvalidNameError,
            match="must only contain letters, numbers, and underscores",
        ):
            EvalData.check_dataset_name("invalid-name")

        with pytest.raises(
            InvalidNameError,
            match="must only contain letters, numbers, and underscores",
        ):
            EvalData.check_dataset_name("invalid name")

    def test_name_validation_in_constructor_empty(self, valid_dataframe):
        """Test empty name in constructor raises InvalidNameError."""
        with pytest.raises(InvalidNameError, match="Dataset name is empty"):
            EvalData(
                name="",
                data=valid_dataframe,
            )

    def test_name_validation_in_constructor_whitespace_only(self, valid_dataframe):
        """Test whitespace-only name in constructor raises InvalidNameError."""
        with pytest.raises(InvalidNameError, match="contains only whitespace"):
            EvalData(
                name="   ",
                data=valid_dataframe,
            )

    def test_name_validation_in_constructor_invalid_chars(self, valid_dataframe):
        """Test invalid characters in name raises InvalidNameError."""
        with pytest.raises(
            InvalidNameError,
            match="must only contain letters, numbers, and underscores",
        ):
            EvalData(
                name="invalid-name",
                data=valid_dataframe,
            )

    # === Basic Functionality Tests ===

    # === DataFrame Validation Tests ===

    def test_validate_dataframe_empty(self):
        """Test empty DataFrame raises EmptyDataFrameError."""
        empty_df = pl.DataFrame({"col1": []})
        with pytest.raises(EmptyDataFrameError):
            EvalData(
                name="test",
                data=empty_df,
            )

    # === Data Integrity Validation Tests ===

    def test_clean_data_no_issues(self, minimal_dataframe):
        """Test clean data with no integrity issues."""
        eval_data = EvalData(
            name="test",
            data=minimal_dataframe,
        )

        assert eval_data is not None
        assert len(eval_data.data.columns) == 3  # input, output, id

    # === ID Column Creation Tests ===

    def test_auto_generate_id_column(self, minimal_dataframe):
        """Test automatic ID column generation."""
        eval_data = EvalData(
            name="test",
            data=minimal_dataframe,
        )

        assert eval_data.id_column == "id"
        assert "id" in eval_data.data.columns
        assert eval_data.data["id"].to_list() == ["id-1", "id-2"]

    def test_id_column_name_conflict(self):
        """Test IdColumnExistsError when ID column name conflicts."""
        df_with_id = pl.DataFrame(
            {
                "id": ["existing1", "existing2"],
                "input": ["test1", "test2"],
                "output": ["result1", "result2"],
            }
        )

        with pytest.raises(IdColumnExistsError):
            EvalData(
                name="test",
                data=df_with_id,
                # id_column is None, will try to auto-generate "id"
            )

    def test_user_provided_id_column(self, valid_dataframe):
        """Test using user-provided ID column."""
        eval_data = EvalData(
            name="test",
            data=valid_dataframe,
            id_column="question",  # Use existing column as ID
        )

        assert eval_data.id_column == "question"

    # === Final Validation Tests ===

    def test_user_provided_id_with_nulls(self):
        """Test user-provided ID column with null values."""
        df_with_null_id = pl.DataFrame(
            {
                "custom_id": ["id1", None],
                "input": ["test1", "test2"],
                "output": ["result1", "result2"],
            }
        )

        with pytest.raises(InvalidInIDColumnError):
            EvalData(
                name="test",
                data=df_with_null_id,
                id_column="custom_id",
            )

    def test_user_provided_id_with_duplicates(self):
        """Test user-provided ID column with duplicate values."""
        df_with_duplicate_id = pl.DataFrame(
            {
                "custom_id": ["id1", "id1"],
                "input": ["test1", "test2"],
                "output": ["result1", "result2"],
            }
        )

        with pytest.raises(DuplicateInIDColumnError):
            EvalData(
                name="test",
                data=df_with_duplicate_id,
                id_column="custom_id",
            )

    def test_user_provided_string_id_with_empty_strings(self):
        """Test user-provided string ID column with empty strings."""
        df_with_empty_id = pl.DataFrame(
            {
                "custom_id": ["id1", ""],
                "input": ["test1", "test2"],
                "output": ["result1", "result2"],
            }
        )

        with pytest.raises(InvalidInIDColumnError):
            EvalData(
                name="test",
                data=df_with_empty_id,
                id_column="custom_id",
            )

    def test_user_provided_string_id_with_whitespace_only(self):
        """Test user-provided string ID column with whitespace-only values."""
        df_with_whitespace_id = pl.DataFrame(
            {
                "custom_id": ["id1", "   "],
                "input": ["test1", "test2"],
                "output": ["result1", "result2"],
            }
        )

        with pytest.raises(InvalidInIDColumnError):
            EvalData(
                name="test",
                data=df_with_whitespace_id,
                id_column="custom_id",
            )

    def test_non_string_id_column_skips_string_validation(self):
        """Test non-string ID columns skip string-specific validations."""
        df_with_int_id = pl.DataFrame(
            {
                "custom_id": [1, 2],
                "input": ["test1", "test2"],
                "output": ["result1", "result2"],
            }
        )

        eval_data = EvalData(
            name="test",
            data=df_with_int_id,
            id_column="custom_id",
        )

        assert eval_data.id_column == "custom_id"

    def test_null_values_in_non_id_columns(self):
        """Test null values in non-ID columns raise NullValuesInDataError."""
        df_with_nulls = pl.DataFrame(
            {
                "input": ["test1", None],
                "output": ["result1", "result2"],
            }
        )

        with pytest.raises(NullValuesInDataError):
            EvalData(
                name="test",
                data=df_with_nulls,
            )

    def test_empty_strings_in_non_id_columns_warning(self, caplog):
        """Test empty strings in non-ID columns trigger warnings."""
        df_with_empty_strings = pl.DataFrame(
            {
                "input": ["test1", ""],
                "output": ["result1", "result2"],
            }
        )

        with caplog.at_level(logging.WARNING):
            eval_data = EvalData(
                name="test",
                data=df_with_empty_strings,
            )

        assert eval_data is not None
        assert "has empty strings" in caplog.text

    def test_whitespace_only_strings_warning(self, caplog):
        """Test whitespace-only strings in non-ID columns trigger warnings."""
        df_with_whitespace = pl.DataFrame(
            {
                "input": ["test1", "   "],
                "output": ["result1", "result2"],
            }
        )

        with caplog.at_level(logging.WARNING):
            eval_data = EvalData(
                name="test",
                data=df_with_whitespace,
            )

        assert eval_data is not None
        assert "has whitespace-only values" in caplog.text

    # === Property Access Tests ===

    def test_basic_data_access(self, minimal_dataframe):
        """Test basic data access works correctly."""
        eval_data = EvalData(
            name="test",
            data=minimal_dataframe,
        )

        assert eval_data.data is not None
        assert len(eval_data.data.columns) == 3  # input, output, id
        assert "id" in eval_data.data.columns

    # === Immutability Tests ===

    def test_immutability_after_initialization(self, minimal_dataframe):
        """Test that attributes cannot be modified after initialization."""
        eval_data = EvalData(
            name="test",
            data=minimal_dataframe,
        )

        with pytest.raises(
            TypeError, match="Cannot modify attribute.*on immutable EvalData instance"
        ):
            eval_data.name = "new_name"

        with pytest.raises(
            TypeError, match="Cannot modify attribute.*on immutable EvalData instance"
        ):
            eval_data.data = minimal_dataframe

    # === Integration/Happy Path Tests ===

    def test_complete_happy_path(self, valid_dataframe):
        """Test complete happy path."""
        eval_data = EvalData(
            name="test_dataset",
            data=valid_dataframe,
        )

        assert eval_data.name == "test_dataset"
        assert eval_data.id_column == "id"
        assert len(eval_data.data.columns) == 7  # 6 original + 1 id column

    def test_happy_path_with_user_provided_id(self, valid_dataframe):
        """Test happy path with user-provided ID column."""
        eval_data = EvalData(
            name="test",
            data=valid_dataframe,
            id_column="extra_col",
        )

        assert eval_data.id_column == "extra_col"
        assert len(eval_data.data.columns) == 6  # Original columns, no new id added

    def test_invalid_column_names_in_dataframe(self):
        """Test that invalid column names in DataFrame raise InvalidColumnNameError during initialization."""
        # Test with invalid column starting with number
        df_invalid_col = pl.DataFrame(
            {"123invalid": ["value1", "value2"], "valid_col": ["value3", "value4"]}
        )

        with pytest.raises(
            InvalidColumnNameError, match="must start with a letter or underscore"
        ):
            EvalData(
                name="test",
                data=df_invalid_col,
            )

        # Test with invalid column containing special characters
        df_invalid_col2 = pl.DataFrame(
            {"invalid-col": ["value1", "value2"], "valid_col": ["value3", "value4"]}
        )

        with pytest.raises(
            InvalidColumnNameError,
            match="must only contain letters, numbers, and underscores",
        ):
            EvalData(
                name="test",
                data=df_invalid_col2,
            )

    def test_invalid_id_column_name(self):
        """Test that invalid ID column names raise InvalidColumnNameError during initialization."""
        valid_df = pl.DataFrame(
            {"123invalid": ["id1", "id2"], "valid_col": ["value1", "value2"]}
        )

        with pytest.raises(
            InvalidColumnNameError, match="must start with a letter or underscore"
        ):
            EvalData(
                name="test",
                data=valid_df,
                id_column="123invalid",
            )
