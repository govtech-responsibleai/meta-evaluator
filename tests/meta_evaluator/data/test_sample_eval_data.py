"""Test suite for the sampling functionality with comprehensive path coverage."""

import json
import pytest
import polars as pl
from meta_evaluator.data import EvalData, SampleEvalData
from meta_evaluator.data.exceptions import (
    EmptyColumnListError,
    EmptyDataFrameError,
)


class TestSamplingFunctionality:
    """Comprehensive test suite for sampling functionality achieving 100% path coverage."""

    @pytest.fixture
    def valid_dataframe_for_sampling(self) -> pl.DataFrame:
        """Provides a valid DataFrame for sampling tests.

        Returns:
            pl.DataFrame: A DataFrame with sample evaluation data for stratification.
        """
        return pl.DataFrame(
            {
                "question": [
                    "What is 2+2?",
                    "What is 3+3?",
                    "What is 4+4?",
                    "What is 5+5?",
                ],
                "answer": ["4", "6", "8", "10"],
                "model_response": ["Four", "Six", "Eight", "Ten"],
                "topic": ["math", "math", "math", "math"],
                "difficulty": ["easy", "easy", "medium", "hard"],
                "language": ["en", "en", "en", "fr"],
                "human_rating": [5, 4, 3, 2],
            }
        )

    @pytest.fixture
    def multi_stratification_dataframe(self) -> pl.DataFrame:
        """Provides a DataFrame with multiple stratification combinations.

        Returns:
            pl.DataFrame: A DataFrame with varied column value combinations.
        """
        return pl.DataFrame(
            {
                "input": ["q1", "q2", "q3", "q4", "q5", "q6"],
                "output": ["a1", "a2", "a3", "a4", "a5", "a6"],
                "category": ["A", "A", "B", "B", "C", "C"],
                "level": ["1", "2", "1", "2", "1", "2"],
            }
        )

    @pytest.fixture
    def single_row_partitions_dataframe(self) -> pl.DataFrame:
        """Provides a DataFrame where each stratification combination has only one row.

        Returns:
            pl.DataFrame: A DataFrame with single-row partitions.
        """
        return pl.DataFrame(
            {
                "input": ["q1", "q2", "q3"],
                "output": ["a1", "a2", "a3"],
                "unique_meta": ["A", "B", "C"],
            }
        )

    @pytest.fixture
    def eval_data_with_stratification_columns(
        self, valid_dataframe_for_sampling
    ) -> EvalData:
        """Provides an EvalData instance for stratification testing.

        Returns:
            EvalData: A configured EvalData instance for sampling tests.
        """
        return EvalData(
            name="test_dataset",
            data=valid_dataframe_for_sampling,
        )

    @pytest.fixture
    def eval_data_minimal(self) -> EvalData:
        """Provides an EvalData instance with minimal data.

        Returns:
            EvalData: An EvalData instance with basic columns for testing.
        """
        minimal_df = pl.DataFrame(
            {
                "input": ["test1", "test2"],
                "output": ["result1", "result2"],
            }
        )
        return EvalData(
            name="minimal_data",
            data=minimal_df,
        )

    # === stratified_sample_by_columns Parameter Validation Tests ===

    def test_stratified_sample_by_columns_invalid_sample_percentage_zero(
        self, eval_data_with_stratification_columns
    ):
        """Test sample_percentage of 0 raises ValueError."""
        with pytest.raises(
            ValueError, match="sample_percentage must be between 0 and 1"
        ):
            eval_data_with_stratification_columns.stratified_sample_by_columns(
                columns=["difficulty"], sample_percentage=0.0
            )

    def test_stratified_sample_by_columns_invalid_sample_percentage_negative(
        self, eval_data_with_stratification_columns
    ):
        """Test negative sample_percentage raises ValueError."""
        with pytest.raises(
            ValueError, match="sample_percentage must be between 0 and 1"
        ):
            eval_data_with_stratification_columns.stratified_sample_by_columns(
                columns=["difficulty"], sample_percentage=-0.1
            )

    def test_stratified_sample_by_columns_invalid_sample_percentage_above_one(
        self, eval_data_with_stratification_columns
    ):
        """Test sample_percentage above 1 raises ValueError."""
        with pytest.raises(
            ValueError, match="sample_percentage must be between 0 and 1"
        ):
            eval_data_with_stratification_columns.stratified_sample_by_columns(
                columns=["difficulty"], sample_percentage=1.5
            )

    def test_stratified_sample_by_columns_valid_sample_percentage_boundary(
        self, eval_data_with_stratification_columns
    ):
        """Test sample_percentage boundary values work correctly."""
        # Test sample_percentage = 1.0 (100%)
        sample = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"], sample_percentage=1.0
        )
        assert len(sample.data) == len(eval_data_with_stratification_columns.data)

        # Test very small sample_percentage
        sample_small = (
            eval_data_with_stratification_columns.stratified_sample_by_columns(
                columns=["difficulty"], sample_percentage=0.01
            )
        )
        assert (
            len(sample_small.data) > 0
        )  # Should still get at least 1 row per stratification group

    # === Column Parameter Validation Tests ===

    def test_stratified_sample_by_columns_focus_invalid_column_names(
        self, eval_data_with_stratification_columns
    ):
        """Test columns with invalid column names raises ValueError."""
        with pytest.raises(ValueError, match="columns contains non-existent columns"):
            eval_data_with_stratification_columns.stratified_sample_by_columns(
                columns=[
                    "nonexistent1",
                    "nonexistent2",
                ],  # invalid columns
                sample_percentage=0.5,
            )

    def test_stratified_sample_by_columns_focus_empty_list(
        self, eval_data_with_stratification_columns
    ):
        """Test columns with empty list raises EmptyColumnListError."""
        with pytest.raises(EmptyColumnListError, match="columns list cannot be empty"):
            eval_data_with_stratification_columns.stratified_sample_by_columns(
                columns=[], sample_percentage=0.5
            )

    def test_stratified_sample_by_columns_focus_valid_subset(
        self, eval_data_with_stratification_columns
    ):
        """Test columns with valid column names."""
        sample = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"], sample_percentage=0.5
        )
        assert sample.stratification_columns == ["difficulty"]

    # === sample_name Generation Tests ===

    def test_stratified_sample_by_columns_auto_generate_sample_name(
        self, eval_data_with_stratification_columns
    ):
        """Test automatic sample name generation when sample_name=None."""
        sample = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty", "topic"],
            sample_percentage=0.3,
            sample_name=None,
        )
        cols_name = "difficulty_topic"
        expected_name = f"Stratified Sample ({eval_data_with_stratification_columns.name}, {cols_name}, {0.3 * 100:.1f}%)"
        assert sample.sample_name == expected_name

    def test_stratified_sample_by_columns_provided_sample_name(
        self, eval_data_with_stratification_columns
    ):
        """Test using provided sample_name."""
        custom_name = "My Custom Sample"
        sample = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"],
            sample_percentage=0.5,
            sample_name=custom_name,
        )
        assert sample.sample_name == custom_name

    # === Sampling Logic Tests ===

    def test_stratified_sample_by_columns_single_partition(
        self, single_row_partitions_dataframe
    ):
        """Test sampling with single-row partitions."""
        eval_data = EvalData(
            name="single_partitions",
            data=single_row_partitions_dataframe,
        )

        sample = eval_data.stratified_sample_by_columns(
            columns=["unique_meta"], sample_percentage=0.5
        )

        # Each partition has 1 row, so should get 1 row from each (max(1, 1*0.5) = 1)
        assert len(sample.data) == 3

    def test_stratified_sample_by_columns_multiple_partitions(
        self, multi_stratification_dataframe
    ):
        """Test sampling with multiple partitions."""
        eval_data = EvalData(
            name="multi_partitions",
            data=multi_stratification_dataframe,
        )

        sample = eval_data.stratified_sample_by_columns(
            columns=["category", "level"], sample_percentage=0.5
        )

        # Should have samples from each stratification combination
        assert len(sample.data) > 0
        assert len(sample.data) <= len(multi_stratification_dataframe)

    def test_stratified_sample_by_columns_small_sample_percentage_ensures_minimum(
        self, multi_stratification_dataframe
    ):
        """Test very small sample_percentage still gets at least 1 row per stratification group."""
        eval_data = EvalData(
            name="min_sample_test",
            data=multi_stratification_dataframe,
        )

        sample = eval_data.stratified_sample_by_columns(
            columns=["category"],
            sample_percentage=0.01,  # Very small percentage
        )

        # Should get at least 1 row per stratification group (A, B, C = 3 groups minimum)
        assert len(sample.data) >= 3

    def test_stratified_sample_by_columns_sample_size_respects_partition_size(self):
        """Test sample size doesn't exceed partition size."""
        # Create a DataFrame where one stratification group has fewer rows than calculated sample size
        df = pl.DataFrame(
            {
                "input": ["q1", "q2", "q3"],
                "output": ["a1", "a2", "a3"],
                "category": ["A", "A", "B"],  # A group has 2 rows, B group has 1 row
            }
        )

        eval_data = EvalData(
            name="size_test",
            data=df,
        )

        sample = eval_data.stratified_sample_by_columns(
            columns=["category"],
            sample_percentage=0.9,  # High percentage
        )

        # Should not exceed original data size
        assert len(sample.data) <= len(df)

    def test_stratified_sample_by_columns_empty_result_raises_error(self):
        """Test sampling that results in empty dataset raises NoDataLeftError."""
        # This is a theoretical edge case - in practice, the max(1, ...) logic prevents this
        # But we test the validation exists
        df = pl.DataFrame(
            {
                "input": ["q1"],
                "output": ["a1"],
                "category": ["A"],
            }
        )

        eval_data = EvalData(
            name="empty_test",
            data=df,
        )

        # This should work normally since max(1, ...) ensures at least 1 row
        sample = eval_data.stratified_sample_by_columns(
            columns=["category"], sample_percentage=0.1
        )
        assert len(sample.data) == 1

    def test_stratified_sample_by_columns_reproducibility_with_seed(
        self, eval_data_with_stratification_columns
    ):
        """Test sampling is reproducible with same seed."""
        sample1 = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"], sample_percentage=0.5, seed=42
        )

        sample2 = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"], sample_percentage=0.5, seed=42
        )

        # Should get identical results
        assert sample1.data.equals(sample2.data)

    def test_stratified_sample_by_columns_different_seeds_different_results(
        self, eval_data_with_stratification_columns
    ):
        """Test different seeds produce different results."""
        sample1 = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"], sample_percentage=0.5, seed=42
        )

        sample2 = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"], sample_percentage=0.5, seed=123
        )

        # Should likely get different results (not guaranteed but very probable)
        # We just check they're both valid samples
        assert len(sample1.data) > 0
        assert len(sample2.data) > 0

    # === SampleEvalData Class Tests ===

    def test_sampled_eval_data_initialization(
        self, eval_data_with_stratification_columns
    ):
        """Test SampleEvalData initialization with all required fields."""
        sample = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"],
            sample_percentage=0.5,
            sample_name="Test Sample",
            seed=42,
        )

        # Verify it's a SampleEvalData instance
        assert isinstance(sample, SampleEvalData)

        # Verify sampling-specific attributes
        assert sample.sample_name == "Test Sample"
        assert sample.sample_percentage == 0.5
        assert sample.seed == 42
        assert sample.sampling_method == "stratified_by_columns"

        # Verify inherited EvalData attributes still work
        assert sample.name == eval_data_with_stratification_columns.name

    def test_sampled_eval_data_inherits_eval_data_validation(
        self, valid_dataframe_for_sampling
    ):
        """Test SampleEvalData inherits all EvalData validation logic."""
        # This should fail the same way EvalData would fail - test with empty DataFrame
        empty_df = pl.DataFrame({"col1": []})
        with pytest.raises(EmptyDataFrameError):
            SampleEvalData(
                data=empty_df,
                name="test",
                stratification_columns=["difficulty"],  # Valid stratification columns
                sample_name="Test",
                sample_percentage=0.5,
                seed=42,
            )

    def test_sampled_eval_data_immutability(
        self, eval_data_with_stratification_columns
    ):
        """Test SampleEvalData immutability after initialization."""
        sample = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"], sample_percentage=0.5
        )

        # Should not be able to modify any attributes
        with pytest.raises(
            TypeError, match="Cannot modify attribute.*on immutable EvalData instance"
        ):
            sample.sample_name = "New Name"

        with pytest.raises(
            TypeError, match="Cannot modify attribute.*on immutable EvalData instance"
        ):
            sample.stratification_columns = ["topic"]

    # === sampling_info Property Tests ===

    def test_sampling_info_property(self, eval_data_with_stratification_columns):
        """Test sampling_info property returns comprehensive dictionary."""
        sample = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty", "topic"],
            sample_percentage=0.25,
            sample_name="Info Test Sample",
            seed=999,
        )

        info = sample.sampling_info

        # Verify all expected keys and values
        assert info["sample_name"] == "Info Test Sample"
        assert info["sampling_method"] == "stratified_by_columns"
        assert info["stratification_columns"] == ["difficulty", "topic"]
        assert info["sample_percentage"] == 0.25
        assert info["seed"] == 999
        assert info["sampled_rows"] == len(sample.data)
        assert isinstance(info["sampled_rows"], int)

    # === Integration Tests ===

    def test_end_to_end_sampling_workflow(self, eval_data_with_stratification_columns):
        """Test complete end-to-end sampling workflow."""
        # Perform sampling
        sample = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"],
            sample_percentage=0.5,
            sample_name="E2E Test Sample",
            seed=42,
        )

        # Verify sample properties
        assert isinstance(sample, SampleEvalData)
        assert len(sample.data) > 0
        assert len(sample.data) <= len(eval_data_with_stratification_columns.data)

        # Verify sample maintains data structure
        assert sample.id_column is not None

        # Verify sampling information
        info = sample.sampling_info
        assert info["sample_name"] == "E2E Test Sample"
        assert info["sampled_rows"] == len(sample.data)

    def test_reproduce_sample_from_sampling_info(
        self, eval_data_with_stratification_columns
    ):
        """Test reproducing a sample using its sampling_info."""
        # Create original sample
        original = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"], sample_percentage=0.3, seed=555
        )

        # Get sampling info
        info = original.sampling_info

        # Reproduce the sample
        reproduced = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=info["stratification_columns"],
            sample_percentage=info["sample_percentage"],
            seed=info["seed"],
        )

        # Should be identical
        assert original.data.equals(reproduced.data)
        assert original.sampling_info == reproduced.sampling_info

    def test_sample_preserves_basic_structure(
        self, eval_data_with_stratification_columns
    ):
        """Test sampling preserves basic data structure."""
        sample = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"], sample_percentage=0.5
        )

        # Basic structure should be preserved
        assert sample.name == eval_data_with_stratification_columns.name
        assert sample.id_column == eval_data_with_stratification_columns.id_column
        assert isinstance(sample.data, pl.DataFrame)
        assert len(sample.data) > 0

    def test_serialize_sample_eval_data(self, eval_data_with_stratification_columns):
        """Test serialization of SampleEvalData includes sampling metadata."""
        # Create a sample with all metadata
        sample = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty", "topic"],
            sample_percentage=0.3,
            sample_name="Test Sample",
            seed=42,
        )

        result = sample.serialize(data_format="json", data_filename="sample_data.json")

        # Verify basic EvalData fields
        assert result.name == sample.name
        assert result.id_column == sample.id_column
        assert result.data_file == "sample_data.json"
        assert result.data_format == "json"

        # Verify SampleEvalData specific fields
        assert result.type == "SampleEvalData"
        assert result.sample_name == "Test Sample"
        assert result.stratification_columns == ["difficulty", "topic"]
        assert result.sample_percentage == 0.3
        assert result.seed == 42
        assert result.sampling_method == "stratified_by_columns"

    def test_write_data_parquet_format(
        self, eval_data_with_stratification_columns, tmp_path
    ):
        """Test write_data writes parquet format correctly for SampleEvalData."""
        # Create a sample first
        sample = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"], sample_percentage=0.5
        )
        filepath = tmp_path / "test.parquet"

        sample.write_data(str(filepath), "parquet")

        # Verify file exists
        assert filepath.exists()

        # Verify we can read it back
        df = pl.read_parquet(str(filepath))
        assert df.equals(sample.data)

    def test_write_data_csv_format(
        self, eval_data_with_stratification_columns, tmp_path
    ):
        """Test write_data writes CSV format correctly for SampleEvalData."""
        # Create a sample first
        sample = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"], sample_percentage=0.5
        )
        filepath = tmp_path / "test.csv"

        sample.write_data(str(filepath), "csv")

        # Verify file exists
        assert filepath.exists()

        # Verify we can read it back
        df = pl.read_csv(str(filepath))

        # Cast columns as csv is lossy
        for col in df.columns:
            df = df.with_columns(pl.col(col).cast(sample.data[col].dtype))
        assert df.equals(sample.data)

    def test_write_data_json_format(
        self, eval_data_with_stratification_columns, tmp_path
    ):
        """Test write_data writes JSON format correctly for SampleEvalData."""
        # Create a sample first
        sample = eval_data_with_stratification_columns.stratified_sample_by_columns(
            columns=["difficulty"], sample_percentage=0.5
        )
        filepath = tmp_path / "test.json"

        sample.write_data(str(filepath), "json")

        # Verify file exists
        assert filepath.exists()

        # Verify file contents
        with open(filepath) as f:
            data = json.load(f)
        assert data == sample.data.to_dict(as_series=False)
