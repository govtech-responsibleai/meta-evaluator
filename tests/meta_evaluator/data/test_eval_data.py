"""Test suite for the EvalData class with comprehensive path coverage."""

import pytest
from unittest.mock import MagicMock
import polars as pl
import logging
from meta_evaluator.data import EvalData
from meta_evaluator.data.exceptions import (
    ColumnNotFoundError,
    DuplicateColumnError,
    EmptyDataFrameError,
    InvalidColumnNameError,
    EmptyColumnListError,
    IdColumnExistsError,
    DuplicateInIDColumnError,
    InvalidInIDColumnError,
    NullValuesInDataError,
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
            pl.DataFrame: A DataFrame with minimal input and output columns.
        """
        return pl.DataFrame(
            {"input": ["test1", "test2"], "output": ["result1", "result2"]}
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

    # === Pre-validation Tests ===

    def test_validate_given_column_names_empty_input_columns(self, valid_dataframe):
        """Test empty input_columns raises EmptyColumnListError."""
        with pytest.raises(EmptyColumnListError):
            EvalData(data=valid_dataframe, input_columns=[], output_columns=["answer"])

    def test_validate_given_column_names_empty_output_columns(self, valid_dataframe):
        """Test empty output_columns raises EmptyColumnListError."""
        with pytest.raises(EmptyColumnListError):
            EvalData(
                data=valid_dataframe, input_columns=["question"], output_columns=[]
            )

    def test_validate_given_column_names_invalid_input_column(self, valid_dataframe):
        """Test invalid input column name raises InvalidColumnNameError."""
        with pytest.raises(InvalidColumnNameError):
            EvalData(
                data=valid_dataframe,
                input_columns=["123invalid"],
                output_columns=["answer"],
            )

    def test_validate_given_column_names_invalid_output_column(self, valid_dataframe):
        """Test invalid output column name raises InvalidColumnNameError."""
        with pytest.raises(InvalidColumnNameError):
            EvalData(
                data=valid_dataframe,
                input_columns=["question"],
                output_columns=["invalid-name"],
            )

    def test_validate_given_column_names_invalid_metadata_column(self, valid_dataframe):
        """Test invalid metadata column name raises InvalidColumnNameError."""
        with pytest.raises(InvalidColumnNameError):
            EvalData(
                data=valid_dataframe,
                input_columns=["question"],
                output_columns=["answer"],
                metadata_columns=["123invalid"],
            )

    def test_validate_given_column_names_invalid_human_label_column(
        self, valid_dataframe
    ):
        """Test invalid human label column name raises InvalidColumnNameError."""
        with pytest.raises(InvalidColumnNameError):
            EvalData(
                data=valid_dataframe,
                input_columns=["question"],
                output_columns=["answer"],
                human_label_columns=["invalid-name"],
            )

    # === DataFrame Validation Tests ===

    def test_validate_dataframe_empty(self):
        """Test empty DataFrame raises EmptyDataFrameError."""
        empty_df = pl.DataFrame({"col1": []})
        with pytest.raises(EmptyDataFrameError):
            EvalData(data=empty_df, input_columns=["col1"], output_columns=["col1"])

    # === Data Integrity Validation Tests ===

    def test_missing_columns_error(self, valid_dataframe):
        """Test missing columns raise ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError):
            EvalData(
                data=valid_dataframe,
                input_columns=["nonexistent_column"],
                output_columns=["answer"],
            )

    def test_duplicate_columns_error(self, valid_dataframe):
        """Test duplicate column specifications raise DuplicateColumnError."""
        with pytest.raises(DuplicateColumnError):
            EvalData(
                data=valid_dataframe,
                input_columns=["question"],
                output_columns=["question"],  # Duplicate
            )

    def test_uncategorized_columns_warning(self, valid_dataframe, caplog):
        """Test uncategorized columns trigger warning."""
        with caplog.at_level(logging.WARNING):
            eval_data = EvalData(
                data=valid_dataframe,
                input_columns=["question"],
                output_columns=["answer"],
            )

        # Should have uncategorized columns
        assert len(eval_data.uncategorized_column_names) > 0
        assert "Uncategorized columns detected" in caplog.text

    def test_clean_data_no_issues(self, minimal_dataframe):
        """Test clean data with no integrity issues."""
        eval_data = EvalData(
            data=minimal_dataframe, input_columns=["input"], output_columns=["output"]
        )

        assert eval_data is not None
        assert len(eval_data.uncategorized_column_names) == 0

    # === ID Column Creation Tests ===

    def test_auto_generate_id_column(self, minimal_dataframe):
        """Test automatic ID column generation."""
        eval_data = EvalData(
            data=minimal_dataframe, input_columns=["input"], output_columns=["output"]
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
                data=df_with_id,
                input_columns=["input"],
                output_columns=["output"],
                # id_column is None, will try to auto-generate "id"
            )

    def test_user_provided_id_column(self, valid_dataframe):
        """Test using user-provided ID column."""
        eval_data = EvalData(
            data=valid_dataframe,
            id_column="question",  # Use existing column as ID
            input_columns=["answer"],
            output_columns=["model_response"],
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
                data=df_with_null_id,
                id_column="custom_id",
                input_columns=["input"],
                output_columns=["output"],
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
                data=df_with_duplicate_id,
                id_column="custom_id",
                input_columns=["input"],
                output_columns=["output"],
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
                data=df_with_empty_id,
                id_column="custom_id",
                input_columns=["input"],
                output_columns=["output"],
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
                data=df_with_whitespace_id,
                id_column="custom_id",
                input_columns=["input"],
                output_columns=["output"],
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
            data=df_with_int_id,
            id_column="custom_id",
            input_columns=["input"],
            output_columns=["output"],
        )

        assert eval_data.id_column == "custom_id"

    def test_null_values_in_non_id_columns(self):
        """Test null values in non-ID columns raise NullValuesInDataError."""
        df_with_nulls = pl.DataFrame(
            {"input": ["test1", None], "output": ["result1", "result2"]}
        )

        with pytest.raises(NullValuesInDataError):
            EvalData(
                data=df_with_nulls, input_columns=["input"], output_columns=["output"]
            )

    def test_empty_strings_in_non_id_columns_warning(self, caplog):
        """Test empty strings in non-ID columns trigger warnings."""
        df_with_empty_strings = pl.DataFrame(
            {"input": ["test1", ""], "output": ["result1", "result2"]}
        )

        with caplog.at_level(logging.WARNING):
            eval_data = EvalData(
                data=df_with_empty_strings,
                input_columns=["input"],
                output_columns=["output"],
            )

        assert eval_data is not None
        assert "has empty strings" in caplog.text

    def test_whitespace_only_strings_warning(self, caplog):
        """Test whitespace-only strings in non-ID columns trigger warnings."""
        df_with_whitespace = pl.DataFrame(
            {"input": ["test1", "   "], "output": ["result1", "result2"]}
        )

        with caplog.at_level(logging.WARNING):
            eval_data = EvalData(
                data=df_with_whitespace,
                input_columns=["input"],
                output_columns=["output"],
            )

        assert eval_data is not None
        assert "has whitespace-only values" in caplog.text

    # === Property Access Tests ===

    def test_uncategorized_data_with_no_uncategorized_columns(self, minimal_dataframe):
        """Test uncategorized_data property with no uncategorized columns."""
        eval_data = EvalData(
            data=minimal_dataframe, input_columns=["input"], output_columns=["output"]
        )

        uncategorized = eval_data.uncategorized_data
        assert isinstance(uncategorized, pl.DataFrame)
        assert len(uncategorized.columns) == 0

    def test_uncategorized_data_with_uncategorized_columns(self, valid_dataframe):
        """Test uncategorized_data property with uncategorized columns."""
        eval_data = EvalData(
            data=valid_dataframe, input_columns=["question"], output_columns=["answer"]
        )

        uncategorized = eval_data.uncategorized_data
        assert isinstance(uncategorized, pl.DataFrame)
        assert len(uncategorized.columns) > 0

    def test_input_data_property(self, valid_dataframe):
        """Test input_data property returns correct columns."""
        eval_data = EvalData(
            data=valid_dataframe, input_columns=["question"], output_columns=["answer"]
        )

        input_data = eval_data.input_data
        assert list(input_data.columns) == ["question"]

    def test_output_data_property(self, valid_dataframe):
        """Test output_data property returns correct columns."""
        eval_data = EvalData(
            data=valid_dataframe, input_columns=["question"], output_columns=["answer"]
        )

        output_data = eval_data.output_data
        assert list(output_data.columns) == ["answer"]

    def test_metadata_data_property(self, valid_dataframe):
        """Test metadata_data property returns correct columns."""
        eval_data = EvalData(
            data=valid_dataframe,
            input_columns=["question"],
            output_columns=["answer"],
            metadata_columns=["difficulty"],
        )

        metadata_data = eval_data.metadata_data
        assert list(metadata_data.columns) == ["difficulty"]

    def test_human_label_data_property(self, valid_dataframe):
        """Test human_label_data property returns correct columns."""
        eval_data = EvalData(
            data=valid_dataframe,
            input_columns=["question"],
            output_columns=["answer"],
            human_label_columns=["human_rating"],
        )

        human_label_data = eval_data.human_label_data
        assert list(human_label_data.columns) == ["human_rating"]

    # === Immutability Tests ===

    def test_immutability_after_initialization(self, minimal_dataframe):
        """Test that attributes cannot be modified after initialization."""
        eval_data = EvalData(
            data=minimal_dataframe, input_columns=["input"], output_columns=["output"]
        )

        with pytest.raises(
            TypeError, match="Cannot modify attribute.*on immutable EvalData instance"
        ):
            eval_data.input_columns = ["new_input"]

        with pytest.raises(
            TypeError, match="Cannot modify attribute.*on immutable EvalData instance"
        ):
            eval_data.data = minimal_dataframe

    # === Integration/Happy Path Tests ===

    def test_complete_happy_path_with_all_categories(self, valid_dataframe):
        """Test complete happy path with all column categories specified."""
        eval_data = EvalData(
            data=valid_dataframe,
            input_columns=["question"],
            output_columns=["answer", "model_response"],
            metadata_columns=["difficulty"],
            human_label_columns=["human_rating"],
        )

        assert eval_data.id_column == "id"
        assert eval_data.input_columns == ["question"]
        assert eval_data.output_columns == ["answer", "model_response"]
        assert eval_data.metadata_columns == ["difficulty"]
        assert eval_data.human_label_columns == ["human_rating"]
        assert len(eval_data.uncategorized_column_names) == 1  # "extra_col"

        # Test all data access methods work
        assert len(eval_data.input_data.columns) == 1
        assert len(eval_data.output_data.columns) == 2
        assert len(eval_data.metadata_data.columns) == 1
        assert len(eval_data.human_label_data.columns) == 1

    def test_happy_path_with_user_provided_id(self, valid_dataframe):
        """Test happy path with user-provided ID column."""
        eval_data = EvalData(
            data=valid_dataframe,
            id_column="extra_col",
            input_columns=["question"],
            output_columns=["answer"],
        )

        assert eval_data.id_column == "extra_col"
        assert "extra_col" not in eval_data.uncategorized_column_names
