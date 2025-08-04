"""Tests for text similarity metrics."""

import pytest
import numpy as np
import polars as pl
from meta_evaluator.scores import TextSimilarityScorer


@pytest.fixture
def sample_judge_df():
    """Reusable sample judge DataFrame for text testing.

    Returns:
        pl.DataFrame: A sample judge DataFrame.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3"],
            "judge_id": ["judge_1", "judge_1", "judge_1"],
            "task_name": ["task1", "task1", "task1"],
            "task_value": ["hello world", "test case", "example text"],
        }
    )


@pytest.fixture
def sample_human_df():
    """Reusable sample human DataFrame for text testing.

    Returns:
        pl.DataFrame: A sample human DataFrame.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3"],
            "annotator_id": ["human_1", "human_1", "human_1"],
            "task_name": ["task1", "task1", "task1"],
            "task_value": ["hello world", "test case", "example text"],
        }
    )


class TestTextSimilarityScorer:
    """Test TextSimilarityScorer functionality."""

    @pytest.fixture
    def text_similarity_scorer(self):
        """Create a TextSimilarityScorer instance.

        Returns:
            TextSimilarityScorer: A TextSimilarityScorer instance.
        """
        return TextSimilarityScorer()

    def test_can_score_task_free_form(self, text_similarity_scorer):
        """Test that TextSimilarityScorer can handle free-form text tasks and rejects classification tasks."""
        # Should accept free-form text tasks
        assert text_similarity_scorer.can_score_task(None) is True

    def test_cannot_score_classification(self, text_similarity_scorer):
        """Test that TextSimilarityScorer cannot handle classification tasks."""
        # Should reject classification tasks
        assert text_similarity_scorer.can_score_task(["A", "B", "C"]) is False
        assert text_similarity_scorer.can_score_task(["positive", "negative"]) is False
        assert (
            text_similarity_scorer.can_score_task(["class1"]) is False
        )  # Single class

    def test_compute_text_similarity_identical(self, text_similarity_scorer):
        """Test text similarity computation with identical texts."""
        similarity = text_similarity_scorer._compute_text_similarity(
            "hello world", "hello world"
        )
        assert similarity == 1.0

    def test_compute_text_similarity_different(self, text_similarity_scorer):
        """Test text similarity computation with completely different texts."""
        similarity = text_similarity_scorer._compute_text_similarity("hello", "goodbye")
        assert 0.0 <= similarity < 1.0

    def test_compute_text_similarity_case_insensitive(self, text_similarity_scorer):
        """Test that text similarity is case insensitive."""
        similarity = text_similarity_scorer._compute_text_similarity(
            "Hello World", "hello world"
        )
        assert similarity == 1.0

    def test_compute_text_similarity_whitespace_normalized(
        self, text_similarity_scorer
    ):
        """Test that whitespace is normalized."""
        similarity = text_similarity_scorer._compute_text_similarity(
            "  hello world  ", "hello world"
        )
        assert similarity == 1.0

    def test_compute_single_task_similarity_perfect(
        self, text_similarity_scorer, sample_judge_df, sample_human_df
    ):
        """Test similarity computation with perfect match."""
        similarity = text_similarity_scorer._compute_single_judge_task_similarity(
            sample_judge_df, sample_human_df, "task1"
        )
        assert similarity == 1.0

    def test_compute_single_task_similarity_no_match(self, text_similarity_scorer):
        """Test similarity computation with no match."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "judge_id": ["judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": ["hello", "world", "test"],
            }
        )

        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "annotator_id": ["human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": ["goodbye", "universe", "exam"],
            }
        )

        similarity = text_similarity_scorer._compute_single_judge_task_similarity(
            judge_df, human_df, "task1"
        )
        assert 0.0 <= similarity < 1.0

    def test_compute_single_task_similarity_partial_match(self, text_similarity_scorer):
        """Test similarity computation with partial match."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2"],
                "judge_id": ["judge_1", "judge_1"],
                "task_name": ["task1", "task1"],
                "task_value": ["hello world", "completely different"],
            }
        )

        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2"],
                "annotator_id": ["human_1", "human_1"],
                "task_name": ["task1", "task1"],
                "task_value": [
                    "hello world",
                    "hello world",
                ],  # First perfect match, second partial
            }
        )

        similarity = text_similarity_scorer._compute_single_judge_task_similarity(
            judge_df, human_df, "task1"
        )
        # Should be average of 1.0 (perfect match) and some lower value (partial match)
        assert 0.0 < similarity < 1.0

    def test_compute_single_task_similarity_multiple_human_annotations(
        self, text_similarity_scorer
    ):
        """Test similarity computation with multiple human annotations."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1"],
                "judge_id": ["judge_1"],
                "task_name": ["task1"],
                "task_value": ["hello world"],
            }
        )

        # Multiple human annotations for the same original_id
        human_df = pl.DataFrame(
            {
                "original_id": ["1", "1", "1"],
                "annotator_id": ["human_1", "human_2", "human_3"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": ["hello world", "hello there", "goodbye world"],
            }
        )

        similarity = text_similarity_scorer._compute_single_judge_task_similarity(
            judge_df, human_df, "task1"
        )
        # Should compute average similarity across all human texts
        # "hello world" vs ["hello world", "hello there", "goodbye world"]
        # Expected: (1.0 + ~0.8 + ~0.4) / 3 = ~0.73
        assert 0.6 < similarity < 0.9

    def test_compute_multi_task_similarity(self, text_similarity_scorer):
        """Test multi-task similarity computation."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "1", "2"],
                "judge_id": ["judge_1", "judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task2", "task2"],
                "task_value": ["hello", "world", "test", "example"],
            }
        )

        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "1", "2"],
                "annotator_id": ["human_1", "human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task2", "task2"],
                "task_value": ["hello", "world", "test", "sample"],
            }
        )

        similarity = text_similarity_scorer._compute_single_judge_multi_task_similarity(
            judge_df, human_df, ["task1", "task2"]
        )

        # Task1: perfect match (1.0), Task2: partial match -> average should be > 0.5
        assert 0.5 < similarity <= 1.0

    def test_compute_score_single_task(
        self, text_similarity_scorer, sample_judge_df, sample_human_df
    ):
        """Test compute_score method for single task."""
        task_schemas = {"task1": None}
        result = text_similarity_scorer.compute_score(
            "judge_1", sample_judge_df, sample_human_df, ["task1"], task_schemas
        )

        assert result.scorer_name == "text_similarity"
        assert result.task_name == "task1"
        assert result.judge_id == "judge_1"
        assert result.score == 1.0
        assert result.metadata["ground_truth_method"] == "average_similarity"
        assert result.metadata["scoring_method"] == "single_task"

    def test_compute_score_multi_task(self, text_similarity_scorer):
        """Test compute_score method for multiple tasks."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "1", "2"],
                "judge_id": ["judge_1", "judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task2", "task2"],
                "task_value": ["hello", "world", "test", "example"],
            }
        )

        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "1", "2"],
                "annotator_id": ["human_1", "human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task2", "task2"],
                "task_value": ["hello", "world", "test", "sample"],
            }
        )

        task_schemas = {"task1": None, "task2": None}
        result = text_similarity_scorer.compute_score(
            "judge_1", judge_df, human_df, ["task1", "task2"], task_schemas
        )

        assert result.scorer_name == "text_similarity"
        assert result.task_name == "2_tasks_avg"
        assert result.judge_id == "judge_1"
        assert 0.5 < result.score <= 1.0
        assert result.metadata["ground_truth_method"] == "average_similarity"
        assert result.metadata["scoring_method"] == "average_across_tasks"

    def test_compute_score_with_none_values(self, text_similarity_scorer):
        """Test compute_score with None values in data."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2"],
                "judge_id": ["judge_1", "judge_1"],
                "task_name": ["task1", "task1"],
                "task_value": [None, "hello"],
            }
        )

        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2"],
                "annotator_id": ["human_1", "human_1"],
                "task_name": ["task1", "task1"],
                "task_value": [None, "hello"],
            }
        )

        task_schemas = {"task1": None}
        result = text_similarity_scorer.compute_score(
            "judge_1", judge_df, human_df, ["task1"], task_schemas
        )

        assert result.scorer_name == "text_similarity"
        assert result.judge_id == "judge_1"
        # Should handle None values gracefully
        assert 0.0 <= result.score <= 1.0

    def test_all_null_task_values(self, text_similarity_scorer):
        """Test when all judge/human values are None for a task."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "judge_id": ["judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": [None, None, None],
            }
        )

        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "annotator_id": ["human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": [None, None, None],
            }
        )

        # Should return np.nan when all values are null
        similarity = text_similarity_scorer._compute_single_judge_task_similarity(
            judge_df, human_df, "task1"
        )
        assert np.isnan(similarity)

    def test_mixed_null_and_valid_data(self, text_similarity_scorer):
        """Test some valid, some null values in same task."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "judge_id": ["judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": [None, "test case", "example text"],
            }
        )

        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "annotator_id": ["human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": ["hello", None, "example text"],
            }
        )

        # Should compute similarity only on valid pairs (should be only 1 valid pair: "example text"=="example text")
        similarity = text_similarity_scorer._compute_single_judge_task_similarity(
            judge_df, human_df, "task1"
        )
        # Should be 1.0 since the only valid comparison is identical
        assert similarity == 1.0

    def test_aggregate_results_saves_individual_results(
        self, text_similarity_scorer, tmp_path
    ):
        """Test that aggregate_results saves individual results as JSON files."""
        from meta_evaluator.scores.base_scoring_result import BaseScoringResult

        # Create mock results for 2 judges
        results = [
            BaseScoringResult(
                scorer_name="text_similarity",
                task_name="summary",
                judge_id="judge_1",
                score=0.85,
                metadata={
                    "ground_truth_method": "average_similarity",
                    "task_names": ["summary"],
                },
            ),
            BaseScoringResult(
                scorer_name="text_similarity",
                task_name="summary",
                judge_id="judge_2",
                score=0.72,
                metadata={
                    "ground_truth_method": "average_similarity",
                    "task_names": ["summary"],
                },
            ),
        ]

        scores_dir = str(tmp_path)

        # Call aggregate_results
        text_similarity_scorer.aggregate_results(results, scores_dir)

        # Verify text_similarity directory was created
        text_similarity_dir = tmp_path / "text_similarity"
        assert text_similarity_dir.exists()

        # Verify individual result files were saved
        result_files = list(text_similarity_dir.glob("*_result.json"))
        assert len(result_files) == 2

        # Verify file naming convention
        expected_files = {"judge_1_summary_result.json", "judge_2_summary_result.json"}
        actual_files = {f.name for f in result_files}
        assert actual_files == expected_files

        # Verify we can load back the results
        for result_file in result_files:
            loaded_result = BaseScoringResult.load_state(str(result_file))
            assert loaded_result.scorer_name == "text_similarity"
            assert loaded_result.task_name == "summary"
            assert loaded_result.judge_id in ["judge_1", "judge_2"]
            assert isinstance(loaded_result.score, float)

    def test_aggregate_results_handles_empty_list(
        self, text_similarity_scorer, tmp_path
    ):
        """Test that aggregate_results handles empty results list gracefully."""
        scores_dir = str(tmp_path)

        # Should not crash with empty list
        text_similarity_scorer.aggregate_results([], scores_dir)

        # Should not create directory
        text_similarity_dir = tmp_path / "text_similarity"
        assert not text_similarity_dir.exists()
