"""Test suite for DataLoader class with file validation and error handling."""

import pytest
import os
from pathlib import Path

from meta_evaluator.data import DataLoader, EvalData
from meta_evaluator.data.exceptions import (
    DataFileError,
    DuplicateInIDColumnError,
    EmptyDataFrameError,
    IdColumnExistsError,
    InvalidNameError,
    NullValuesInDataError,
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
            name="test",
            file_path=minimal_csv_file,
        )

        assert isinstance(result, EvalData)
        assert len(result.data) == 2

    def test_load_csv_full_features_success(self, valid_csv_file):
        """Test successful loading with all features."""
        result = DataLoader.load_csv(
            name="test",
            file_path=valid_csv_file,
        )

        assert isinstance(result, EvalData)
        assert len(result.data) == 2

    def test_load_csv_with_user_id_column(self, valid_csv_file):
        """Test successful loading with user-provided ID column."""
        result = DataLoader.load_csv(
            name="test",
            file_path=valid_csv_file,
            id_column="difficulty",  # Now has unique values: "easy", "medium"
        )

        assert result.id_column == "difficulty"

    # === FILE VALIDATION ERROR TESTS ===

    def test_load_csv_file_not_found(self):
        """Test DataFileError when file doesn't exist."""
        with pytest.raises(DataFileError, match="File not found"):
            DataLoader.load_csv(
                name="test",
                file_path="nonexistent_file.csv",
            )

    def test_load_csv_path_is_directory(self, tmp_path):
        """Test DataFileError when path points to directory."""
        directory = tmp_path / "not_a_file"
        directory.mkdir()

        with pytest.raises(DataFileError, match="Path is not a file"):
            DataLoader.load_csv(
                name="test",
                file_path=str(directory),
            )

    def test_load_csv_no_read_permission(self, minimal_csv_file):
        """Test DataFileError when file lacks read permissions."""
        # Remove read permissions
        os.chmod(minimal_csv_file, 0o000)

        try:
            with pytest.raises(DataFileError, match="No read permission"):
                DataLoader.load_csv(
                    name="test",
                    file_path=minimal_csv_file,
                )
        finally:
            # Restore permissions for cleanup
            os.chmod(minimal_csv_file, 0o644)

    # === CSV PARSING ERROR TESTS ===

    def test_load_csv_empty_file(self, empty_csv_file):
        """Test DataFileError with empty CSV file."""
        with pytest.raises(DataFileError, match="Failed to parse CSV"):
            DataLoader.load_csv(
                name="test",
                file_path=empty_csv_file,
            )

    def test_load_csv_malformed_quotes(self, malformed_quotes_csv):
        """Test DataFileError with malformed quotes."""
        with pytest.raises(DataFileError, match="Failed to parse CSV"):
            DataLoader.load_csv(
                name="test",
                file_path=malformed_quotes_csv,
            )

    # === EVALDATA INTEGRATION TESTS ===

    def test_load_csv_headers_only_creates_empty_evaldata(self, headers_only_csv_file):
        """Test headers-only CSV creates empty EvalData (triggers EmptyDataFrameError)."""
        with pytest.raises(EmptyDataFrameError):  # From EvalData, not wrapped
            DataLoader.load_csv(
                name="test",
                file_path=headers_only_csv_file,
            )

    def test_load_csv_empty_name_error(self, minimal_csv_file):
        """Test that InvalidNameError is raised when name is empty."""
        with pytest.raises(InvalidNameError):  # From EvalData, not wrapped
            DataLoader.load_csv(
                name="",  # Empty name should trigger InvalidNameError
                file_path=minimal_csv_file,
            )

    def test_load_csv_whitespace_only_name_error(self, minimal_csv_file):
        """Test that InvalidNameError is raised when name is whitespace only."""
        with pytest.raises(InvalidNameError):
            DataLoader.load_csv(
                name="   ",
                file_path=minimal_csv_file,
            )

    def test_load_csv_id_column_exists_conflict_error(self, tmp_path):
        """Test that EvalData validation errors bubble up unchanged when auto-generated ID column conflicts."""
        # Create CSV that already has an 'id' column
        csv_content = """id,input,output
existing1,test1,result1
existing2,test2,result2"""
        csv_file = tmp_path / "id_conflict.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(IdColumnExistsError):  # Not wrapped in DataFileError
            DataLoader.load_csv(
                name="test",
                file_path=str(csv_file),
                # No id_column specified, will try to auto-generate "id" but it already exists
            )

    def test_load_csv_id_column_with_duplicates_error(self, tmp_path):
        """Test that EvalData ID column validation errors bubble up unchanged."""
        # Create CSV with duplicate values in a column
        csv_content = """custom_id,input,output
id1,test1,result1
id1,test2,result2"""
        csv_file = tmp_path / "duplicate_id.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(DuplicateInIDColumnError):  # Not wrapped in DataFileError
            DataLoader.load_csv(
                name="test",
                file_path=str(csv_file),
                id_column="custom_id",  # Column with duplicate values
            )

    def test_load_csv_null_values_in_data_error(self, tmp_path):
        """Test that EvalData data integrity validation errors bubble up unchanged."""
        # Create CSV with null values (empty cells)
        csv_content = """input,output
test1,result1
,result2"""  # Empty input field creates null
        csv_file = tmp_path / "null_data.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(NullValuesInDataError):  # Not wrapped in DataFileError
            DataLoader.load_csv(
                name="test",
                file_path=str(csv_file),
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
                name="test",
                file_path=relative_path,
            )
            assert isinstance(result, EvalData)
        finally:
            os.chdir(original_cwd)

    def test_load_csv_absolute_path(self, minimal_csv_file):
        """Test loading with absolute file path."""
        absolute_path = Path(minimal_csv_file).resolve()
        result = DataLoader.load_csv(
            name="test",
            file_path=str(absolute_path),
        )
        assert isinstance(result, EvalData)

    # === ADDITIONAL EDGE CASES ===

    def test_load_csv_with_auto_generated_id(self, minimal_csv_file):
        """Test that auto-generated ID works correctly."""
        result = DataLoader.load_csv(
            name="test",
            file_path=minimal_csv_file,
            # No id_column specified, should auto-generate
        )

        assert result.id_column == "id"
        assert "id" in result.data.columns
        assert result.data["id"].to_list() == ["id-1", "id-2"]

    def test_load_csv_with_minimal_data(self, minimal_csv_file):
        """Test loading with minimal data."""
        result = DataLoader.load_csv(
            name="test",
            file_path=minimal_csv_file,
        )

        assert len(result.data.columns) == 3  # input, output, and auto-generated id

    def test_load_csv_file_with_special_characters_in_path(self, tmp_path):
        """Test loading file with special characters in path."""
        special_name = "test file with spaces & symbols!.csv"
        csv_content = """input,output
test1,result1"""
        csv_file = tmp_path / special_name
        csv_file.write_text(csv_content)

        result = DataLoader.load_csv(
            name="test",
            file_path=str(csv_file),
        )

        assert isinstance(result, EvalData)
        assert len(result.data) == 1
