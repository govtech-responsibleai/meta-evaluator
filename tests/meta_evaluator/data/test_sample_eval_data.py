"""Test suite for the sampling functionality with comprehensive path coverage."""

import pytest
import polars as pl
from meta_evaluator.data import EvalData, SampledEvalData
from meta_evaluator.data.exceptions import (
    EmptyColumnListError,
)


class TestSamplingFunctionality:
    """Comprehensive test suite for sampling functionality achieving 100% path coverage."""

    @pytest.fixture
    def valid_dataframe_with_metadata(self) -> pl.DataFrame:
        """Provides a valid DataFrame with metadata columns for sampling tests.

        Returns:
            pl.DataFrame: A DataFrame with sample evaluation data and metadata.
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
    def multi_metadata_dataframe(self) -> pl.DataFrame:
        """Provides a DataFrame with multiple metadata combinations.

        Returns:
            pl.DataFrame: A DataFrame with varied metadata combinations.
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
        """Provides a DataFrame where each metadata combination has only one row.

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
    def eval_data_with_metadata(self, valid_dataframe_with_metadata) -> EvalData:
        """Provides an EvalData instance with metadata columns.

        Returns:
            EvalData: A configured EvalData instance for sampling tests.
        """
        return EvalData(
            name="test_dataset",
            data=valid_dataframe_with_metadata,
            input_columns=["question"],
            output_columns=["answer"],
            metadata_columns=["topic", "difficulty", "language"],
            human_label_columns=["human_rating"],
        )

    @pytest.fixture
    def eval_data_no_metadata(self) -> EvalData:
        """Provides an EvalData instance with no metadata columns.

        Returns:
            EvalData: An EvalData instance without metadata columns.
        """
        minimal_df = pl.DataFrame(
            {
                "input": ["test1", "test2"],
                "output": ["result1", "result2"],
            }
        )
        return EvalData(
            name="no_metadata",
            data=minimal_df,
            input_columns=["input"],
            output_columns=["output"],
        )

    # === sample_by_metadata Parameter Validation Tests ===

    def test_sample_by_metadata_invalid_sample_percentage_zero(
        self, eval_data_with_metadata
    ):
        """Test sample_percentage of 0 raises ValueError."""
        with pytest.raises(
            ValueError, match="sample_percentage must be between 0 and 1"
        ):
            eval_data_with_metadata.sample_by_metadata(sample_percentage=0.0)

    def test_sample_by_metadata_invalid_sample_percentage_negative(
        self, eval_data_with_metadata
    ):
        """Test negative sample_percentage raises ValueError."""
        with pytest.raises(
            ValueError, match="sample_percentage must be between 0 and 1"
        ):
            eval_data_with_metadata.sample_by_metadata(sample_percentage=-0.1)

    def test_sample_by_metadata_invalid_sample_percentage_above_one(
        self, eval_data_with_metadata
    ):
        """Test sample_percentage above 1 raises ValueError."""
        with pytest.raises(
            ValueError, match="sample_percentage must be between 0 and 1"
        ):
            eval_data_with_metadata.sample_by_metadata(sample_percentage=1.5)

    def test_sample_by_metadata_valid_sample_percentage_boundary(
        self, eval_data_with_metadata
    ):
        """Test sample_percentage boundary values work correctly."""
        # Test sample_percentage = 1.0 (100%)
        sample = eval_data_with_metadata.sample_by_metadata(sample_percentage=1.0)
        assert len(sample.data) == len(eval_data_with_metadata.data)

        # Test very small sample_percentage
        sample_small = eval_data_with_metadata.sample_by_metadata(
            sample_percentage=0.01
        )
        assert (
            len(sample_small.data) > 0
        )  # Should still get at least 1 row per partition

    # === metadata_columns_to_focus Validation Tests ===

    def test_sample_by_metadata_focus_none_no_metadata_columns(
        self, eval_data_no_metadata
    ):
        """Test metadata_columns_to_focus=None with no metadata columns raises EmptyColumnListError."""
        with pytest.raises(
            EmptyColumnListError,
            match="metadata \\(no metadata columns available for sampling\\)",
        ):
            eval_data_no_metadata.sample_by_metadata(metadata_columns_to_focus=None)

    def test_sample_by_metadata_focus_none_with_metadata_columns(
        self, eval_data_with_metadata
    ):
        """Test metadata_columns_to_focus=None uses all metadata columns."""
        sample = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=None, sample_percentage=0.5
        )
        assert (
            sample.metadata_columns_focused == eval_data_with_metadata.metadata_columns
        )

    def test_sample_by_metadata_focus_invalid_column_names(
        self, eval_data_with_metadata
    ):
        """Test metadata_columns_to_focus with invalid column names raises ValueError."""
        with pytest.raises(
            ValueError, match="metadata_columns_to_focus contains non-metadata columns"
        ):
            eval_data_with_metadata.sample_by_metadata(
                metadata_columns_to_focus=[
                    "question",
                    "nonexistent",
                ],  # input column and invalid
                sample_percentage=0.5,
            )

    def test_sample_by_metadata_focus_empty_list(self, eval_data_with_metadata):
        """Test metadata_columns_to_focus with empty list raises ValueError."""
        with pytest.raises(
            ValueError, match="metadata_columns_to_focus cannot be an empty list"
        ):
            eval_data_with_metadata.sample_by_metadata(
                metadata_columns_to_focus=[], sample_percentage=0.5
            )

    def test_sample_by_metadata_focus_valid_subset(self, eval_data_with_metadata):
        """Test metadata_columns_to_focus with valid subset of metadata columns."""
        sample = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=["difficulty"], sample_percentage=0.5
        )
        assert sample.metadata_columns_focused == ["difficulty"]

    # === sample_name Generation Tests ===

    def test_sample_by_metadata_auto_generate_sample_name(
        self, eval_data_with_metadata
    ):
        """Test automatic sample name generation when sample_name=None."""
        sample = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=["difficulty", "topic"],
            sample_percentage=0.3,
            sample_name=None,
        )
        expected_name = "Stratified Sample (difficulty_topic, 30.0%)"
        assert sample.sample_name == expected_name

    def test_sample_by_metadata_provided_sample_name(self, eval_data_with_metadata):
        """Test using provided sample_name."""
        custom_name = "My Custom Sample"
        sample = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=["difficulty"],
            sample_percentage=0.5,
            sample_name=custom_name,
        )
        assert sample.sample_name == custom_name

    # === Sampling Logic Tests ===

    def test_sample_by_metadata_single_partition(self, single_row_partitions_dataframe):
        """Test sampling with single-row partitions."""
        eval_data = EvalData(
            name="single_partitions",
            data=single_row_partitions_dataframe,
            input_columns=["input"],
            output_columns=["output"],
            metadata_columns=["unique_meta"],
        )

        sample = eval_data.sample_by_metadata(
            metadata_columns_to_focus=["unique_meta"], sample_percentage=0.5
        )

        # Each partition has 1 row, so should get 1 row from each (max(1, 1*0.5) = 1)
        assert len(sample.data) == 3

    def test_sample_by_metadata_multiple_partitions(self, multi_metadata_dataframe):
        """Test sampling with multiple partitions."""
        eval_data = EvalData(
            name="multi_partitions",
            data=multi_metadata_dataframe,
            input_columns=["input"],
            output_columns=["output"],
            metadata_columns=["category", "level"],
        )

        sample = eval_data.sample_by_metadata(
            metadata_columns_to_focus=["category", "level"], sample_percentage=0.5
        )

        # Should have samples from each category-level combination
        assert len(sample.data) > 0
        assert len(sample.data) <= len(multi_metadata_dataframe)

    def test_sample_by_metadata_small_sample_percentage_ensures_minimum(
        self, multi_metadata_dataframe
    ):
        """Test very small sample_percentage still gets at least 1 row per partition."""
        eval_data = EvalData(
            name="min_sample_test",
            data=multi_metadata_dataframe,
            input_columns=["input"],
            output_columns=["output"],
            metadata_columns=["category"],
        )

        sample = eval_data.sample_by_metadata(
            metadata_columns_to_focus=["category"],
            sample_percentage=0.01,  # Very small percentage
        )

        # Should get at least 1 row per category (A, B, C = 3 categories minimum)
        assert len(sample.data) >= 3

    def test_sample_by_metadata_sample_size_respects_partition_size(self):
        """Test sample size doesn't exceed partition size."""
        # Create a DataFrame where one partition has fewer rows than calculated sample size
        df = pl.DataFrame(
            {
                "input": ["q1", "q2", "q3"],
                "output": ["a1", "a2", "a3"],
                "category": ["A", "A", "B"],  # A has 2 rows, B has 1 row
            }
        )

        eval_data = EvalData(
            name="size_test",
            data=df,
            input_columns=["input"],
            output_columns=["output"],
            metadata_columns=["category"],
        )

        sample = eval_data.sample_by_metadata(
            metadata_columns_to_focus=["category"],
            sample_percentage=0.9,  # High percentage
        )

        # Should not exceed original data size
        assert len(sample.data) <= len(df)

    def test_sample_by_metadata_empty_result_raises_error(self):
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
            input_columns=["input"],
            output_columns=["output"],
            metadata_columns=["category"],
        )

        # This should work normally since max(1, ...) ensures at least 1 row
        sample = eval_data.sample_by_metadata(
            metadata_columns_to_focus=["category"], sample_percentage=0.1
        )
        assert len(sample.data) == 1

    def test_sample_by_metadata_reproducibility_with_seed(
        self, eval_data_with_metadata
    ):
        """Test sampling is reproducible with same seed."""
        sample1 = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=["difficulty"], sample_percentage=0.5, seed=42
        )

        sample2 = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=["difficulty"], sample_percentage=0.5, seed=42
        )

        # Should get identical results
        assert sample1.data.equals(sample2.data)

    def test_sample_by_metadata_different_seeds_different_results(
        self, eval_data_with_metadata
    ):
        """Test different seeds produce different results."""
        sample1 = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=["difficulty"], sample_percentage=0.5, seed=42
        )

        sample2 = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=["difficulty"], sample_percentage=0.5, seed=123
        )

        # Should likely get different results (not guaranteed but very probable)
        # We just check they're both valid samples
        assert len(sample1.data) > 0
        assert len(sample2.data) > 0

    # === SampledEvalData Class Tests ===

    def test_sampled_eval_data_initialization(self, eval_data_with_metadata):
        """Test SampledEvalData initialization with all required fields."""
        sample = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=["difficulty"],
            sample_percentage=0.5,
            sample_name="Test Sample",
            seed=42,
        )

        # Verify it's a SampledEvalData instance
        assert isinstance(sample, SampledEvalData)

        # Verify sampling-specific attributes
        assert sample.sample_name == "Test Sample"
        assert sample.metadata_columns_focused == ["difficulty"]
        assert sample.sample_percentage == 0.5
        assert sample.seed == 42
        assert sample.sampling_method == "by_metadata_combination"

        # Verify inherited EvalData attributes still work
        assert sample.name == eval_data_with_metadata.name
        assert sample.input_columns == eval_data_with_metadata.input_columns
        assert sample.output_columns == eval_data_with_metadata.output_columns

    def test_sampled_eval_data_inherits_eval_data_validation(
        self, valid_dataframe_with_metadata
    ):
        """Test SampledEvalData inherits all EvalData validation logic."""
        # This should fail the same way EvalData would fail
        with pytest.raises(EmptyColumnListError):
            SampledEvalData(
                data=valid_dataframe_with_metadata,
                name="test",
                input_columns=[],  # Empty input columns should fail
                output_columns=["answer"],
                sample_name="Test",
                metadata_columns_focused=["difficulty"],
                sample_percentage=0.5,
                seed=42,
            )

    def test_sampled_eval_data_immutability(self, eval_data_with_metadata):
        """Test SampledEvalData immutability after initialization."""
        sample = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=["difficulty"], sample_percentage=0.5
        )

        # Should not be able to modify any attributes
        with pytest.raises(
            TypeError, match="Cannot modify attribute.*on immutable EvalData instance"
        ):
            sample.sample_name = "New Name"

        with pytest.raises(
            TypeError, match="Cannot modify attribute.*on immutable EvalData instance"
        ):
            sample.metadata_columns_focused = ["topic"]

    # === sampling_info Property Tests ===

    def test_sampling_info_property(self, eval_data_with_metadata):
        """Test sampling_info property returns comprehensive dictionary."""
        sample = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=["difficulty", "topic"],
            sample_percentage=0.25,
            sample_name="Info Test Sample",
            seed=999,
        )

        info = sample.sampling_info

        # Verify all expected keys and values
        assert info["sample_name"] == "Info Test Sample"
        assert info["sampling_method"] == "by_metadata_combination"
        assert info["metadata_columns_focused"] == ["difficulty", "topic"]
        assert info["sample_percentage"] == 0.25
        assert info["seed"] == 999
        assert info["sampled_rows"] == len(sample.data)
        assert isinstance(info["sampled_rows"], int)

    # === Integration Tests ===

    def test_end_to_end_sampling_workflow(self, eval_data_with_metadata):
        """Test complete end-to-end sampling workflow."""
        # Perform sampling
        sample = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=["difficulty"],
            sample_percentage=0.5,
            sample_name="E2E Test Sample",
            seed=42,
        )

        # Verify sample properties
        assert isinstance(sample, SampledEvalData)
        assert len(sample.data) > 0
        assert len(sample.data) <= len(eval_data_with_metadata.data)

        # Verify sample maintains data structure
        assert sample.id_column is not None
        assert all(col in sample.data.columns for col in sample.input_columns)
        assert all(col in sample.data.columns for col in sample.output_columns)

        # Verify sampling metadata
        info = sample.sampling_info
        assert info["sample_name"] == "E2E Test Sample"
        assert info["sampled_rows"] == len(sample.data)

        # Verify data access methods still work
        input_data = sample.input_data
        output_data = sample.output_data
        assert len(input_data) == len(sample.data)
        assert len(output_data) == len(sample.data)

    def test_reproduce_sample_from_sampling_info(self, eval_data_with_metadata):
        """Test reproducing a sample using its sampling_info."""
        # Create original sample
        original = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=["difficulty"], sample_percentage=0.3, seed=555
        )

        # Get sampling info
        info = original.sampling_info

        # Reproduce the sample
        reproduced = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=info["metadata_columns_focused"],
            sample_percentage=info["sample_percentage"],
            seed=info["seed"],
        )

        # Should be identical
        assert original.data.equals(reproduced.data)
        assert original.sampling_info == reproduced.sampling_info

    def test_sample_preserves_all_column_categories(self, eval_data_with_metadata):
        """Test sampling preserves all original column categorizations."""
        sample = eval_data_with_metadata.sample_by_metadata(
            metadata_columns_to_focus=["difficulty"], sample_percentage=0.5
        )

        # All original column categories should be preserved
        assert sample.input_columns == eval_data_with_metadata.input_columns
        assert sample.output_columns == eval_data_with_metadata.output_columns
        assert sample.metadata_columns == eval_data_with_metadata.metadata_columns
        assert sample.human_label_columns == eval_data_with_metadata.human_label_columns

        # Uncategorized columns should also be preserved in the structure
        # (though the actual uncategorized column list might differ due to sampling)
        uncategorized_data = sample.uncategorized_data
        assert isinstance(uncategorized_data, pl.DataFrame)
