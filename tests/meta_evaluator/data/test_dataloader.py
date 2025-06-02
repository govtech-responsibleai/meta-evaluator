"""Test suite for DataLoader class with file validation and error handling."""

import pytest
import os
from pathlib import Path

from meta_evaluator.data import DataLoader, EvalData
from meta_evaluator.data.exceptions import (
    DataFileError,
    ColumnNotFoundError,
    DuplicateColumnError,
    EmptyDataFrameError,
)


class TestDataLoader:
    """Comprehensive test suite for DataLoader class."""

    # === FIXTURES ===

    @pytest.fixture
    def valid_csv_file(self, tmp_path):
        """Create a valid CSV file for testing.

        Returns:
            str: Path to the created CSV file.
        """
        csv_content = """question,answer,model_response,difficulty,rating
What is 2+2?,4,Four,easy,5
What is 3+3?,6,Six,medium,4"""
        csv_file = tmp_path / "valid_data.csv"
        csv_file.write_text(csv_content)
        return str(csv_file)

    @pytest.fixture
    def minimal_csv_file(self, tmp_path):
        """Create minimal CSV with only input/output columns.

        Returns:
            str: Path to the created CSV file.
        """
        csv_content = """input,output
test1,result1
test2,result2"""
        csv_file = tmp_path / "minimal.csv"
        csv_file.write_text(csv_content)
        return str(csv_file)

    @pytest.fixture
    def empty_csv_file(self, tmp_path):
        """Create empty CSV file.

        Returns:
            str: Path to the created empty CSV file.
        """
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        return str(csv_file)

    @pytest.fixture
    def headers_only_csv_file(self, tmp_path):
        """Create CSV with headers but no data.

        Returns:
            str: Path to the created CSV file with headers only.
        """
        csv_content = "input,output\n"
        csv_file = tmp_path / "headers_only.csv"
        csv_file.write_text(csv_content)
        return str(csv_file)

    @pytest.fixture
    def malformed_quotes_csv(self, tmp_path):
        """Create CSV with malformed quotes that cause parsing errors.

        Returns:
            str: Path to the created CSV file.
        """
        # Improperly escaped quotes cause ComputeError
        csv_content = """input,output
"test test" test,result1
test2,result2"""
        csv_file = tmp_path / "malformed_quotes.csv"
        csv_file.write_text(csv_content)
        return str(csv_file)

    # === HAPPY PATH TESTS ===

    def test_load_csv_minimal_success(self, minimal_csv_file):
        """Test successful loading with minimal configuration."""
        result = DataLoader.load_csv(
            file_path=minimal_csv_file,
            input_columns=["input"],
            output_columns=["output"],
        )

        assert isinstance(result, EvalData)
        assert result.input_columns == ["input"]
        assert result.output_columns == ["output"]
        assert len(result.data) == 2

    def test_load_csv_full_features_success(self, valid_csv_file):
        """Test successful loading with all column types."""
        result = DataLoader.load_csv(
            file_path=valid_csv_file,
            input_columns=["question"],
            output_columns=["answer", "model_response"],
            metadata_columns=["difficulty"],
            label_columns=["rating"],
        )

        assert isinstance(result, EvalData)
        assert result.input_columns == ["question"]
        assert result.output_columns == ["answer", "model_response"]
        assert result.metadata_columns == ["difficulty"]
        assert result.human_label_columns == ["rating"]

    def test_load_csv_with_user_id_column(self, valid_csv_file):
        """Test successful loading with user-provided ID column."""
        result = DataLoader.load_csv(
            file_path=valid_csv_file,
            input_columns=["question"],
            output_columns=["answer"],
            id_column="difficulty",  # Now has unique values: "easy", "medium"
        )

        assert result.id_column == "difficulty"

    # === FILE VALIDATION ERROR TESTS ===

    def test_load_csv_file_not_found(self):
        """Test DataFileError when file doesn't exist."""
        with pytest.raises(DataFileError, match="File not found"):
            DataLoader.load_csv(
                file_path="nonexistent_file.csv",
                input_columns=["input"],
                output_columns=["output"],
            )

    def test_load_csv_path_is_directory(self, tmp_path):
        """Test DataFileError when path points to directory."""
        directory = tmp_path / "not_a_file"
        directory.mkdir()

        with pytest.raises(DataFileError, match="Path is not a file"):
            DataLoader.load_csv(
                file_path=str(directory),
                input_columns=["input"],
                output_columns=["output"],
            )

    def test_load_csv_no_read_permission(self, minimal_csv_file):
        """Test DataFileError when file lacks read permissions."""
        # Remove read permissions
        os.chmod(minimal_csv_file, 0o000)

        try:
            with pytest.raises(DataFileError, match="No read permission"):
                DataLoader.load_csv(
                    file_path=minimal_csv_file,
                    input_columns=["input"],
                    output_columns=["output"],
                )
        finally:
            # Restore permissions for cleanup
            os.chmod(minimal_csv_file, 0o644)

    # === CSV PARSING ERROR TESTS ===

    def test_load_csv_empty_file(self, empty_csv_file):
        """Test DataFileError with empty CSV file."""
        with pytest.raises(DataFileError, match="Failed to parse CSV"):
            DataLoader.load_csv(
                file_path=empty_csv_file,
                input_columns=["input"],
                output_columns=["output"],
            )

    def test_load_csv_malformed_quotes(self, malformed_quotes_csv):
        """Test DataFileError with malformed quotes."""
        with pytest.raises(DataFileError, match="Failed to parse CSV"):
            DataLoader.load_csv(
                file_path=malformed_quotes_csv,
                input_columns=["input"],
                output_columns=["output"],
            )

    # === EVALDATA INTEGRATION TESTS ===

    def test_load_csv_missing_specified_columns(self, minimal_csv_file):
        """Test that EvalData validation errors bubble up unchanged."""
        with pytest.raises(ColumnNotFoundError):  # Not wrapped in DataFileError
            DataLoader.load_csv(
                file_path=minimal_csv_file,
                input_columns=["nonexistent_column"],
                output_columns=["output"],
            )

    def test_load_csv_duplicate_column_specification(self, minimal_csv_file):
        """Test EvalData duplicate column error bubbles up."""
        with pytest.raises(DuplicateColumnError):  # Not wrapped in DataFileError
            DataLoader.load_csv(
                file_path=minimal_csv_file,
                input_columns=["input"],
                output_columns=["input"],  # Duplicate
            )

    def test_load_csv_headers_only_creates_empty_evaldata(self, headers_only_csv_file):
        """Test headers-only CSV creates empty EvalData (triggers EmptyDataFrameError)."""
        with pytest.raises(EmptyDataFrameError):  # From EvalData, not wrapped
            DataLoader.load_csv(
                file_path=headers_only_csv_file,
                input_columns=["input"],
                output_columns=["output"],
            )

    # === EDGE CASE TESTS ===

    def test_load_csv_relative_path(self, minimal_csv_file):
        """Test loading with relative file path."""
        # Convert to relative path
        relative_path = Path(minimal_csv_file).name

        # Change to directory containing the file
        original_cwd = os.getcwd()
        try:
            os.chdir(Path(minimal_csv_file).parent)
            result = DataLoader.load_csv(
                file_path=relative_path,
                input_columns=["input"],
                output_columns=["output"],
            )
            assert isinstance(result, EvalData)
        finally:
            os.chdir(original_cwd)

    def test_load_csv_absolute_path(self, minimal_csv_file):
        """Test loading with absolute file path."""
        absolute_path = Path(minimal_csv_file).resolve()
        result = DataLoader.load_csv(
            file_path=str(absolute_path),
            input_columns=["input"],
            output_columns=["output"],
        )
        assert isinstance(result, EvalData)

    # === ADDITIONAL EDGE CASES ===

    def test_load_csv_with_auto_generated_id(self, minimal_csv_file):
        """Test that auto-generated ID works correctly."""
        result = DataLoader.load_csv(
            file_path=minimal_csv_file,
            input_columns=["input"],
            output_columns=["output"],
            # No id_column specified, should auto-generate
        )

        assert result.id_column == "id"
        assert "id" in result.data.columns
        assert result.data["id"].to_list() == ["id-1", "id-2"]

    def test_load_csv_with_empty_optional_lists(self, minimal_csv_file):
        """Test loading with explicitly empty metadata and label columns."""
        result = DataLoader.load_csv(
            file_path=minimal_csv_file,
            input_columns=["input"],
            output_columns=["output"],
            metadata_columns=[],  # Explicitly empty
            label_columns=[],  # Explicitly empty
        )

        assert result.metadata_columns == []
        assert result.human_label_columns == []
        assert len(result.uncategorized_column_names) == 0  # All columns categorized

    def test_load_csv_file_with_special_characters_in_path(self, tmp_path):
        """Test loading file with special characters in path."""
        special_name = "test file with spaces & symbols!.csv"
        csv_content = """input,output
test1,result1"""
        csv_file = tmp_path / special_name
        csv_file.write_text(csv_content)

        result = DataLoader.load_csv(
            file_path=str(csv_file), input_columns=["input"], output_columns=["output"]
        )

        assert isinstance(result, EvalData)
        assert len(result.data) == 1
