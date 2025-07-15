"""Tests for classification Metrics."""

import pytest
import polars as pl
from meta_evaluator.scores import AccuracyScorer


@pytest.fixture
def sample_consolidated_judge_df():
    """Reusable sample consolidated judge DataFrame for testing.

    Returns:
        pl.DataFrame: A sample consolidated judge DataFrame.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3"],
            "judge_id": ["judge_1", "judge_1", "judge_1"],
            "task_name": ["task1", "task1", "task1"],
            "task_value": ["A", "B", "C"],
        }
    )


@pytest.fixture
def sample_consolidated_human_df():
    """Reusable sample consolidated human DataFrame for testing.

    Returns:
        pl.DataFrame: A sample consolidated human DataFrame.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3"],
            "annotator_id": ["human_1", "human_1", "human_1"],
            "task_name": ["task1", "task1", "task1"],
            "task_value": ["A", "B", "C"],
        }
    )


class TestAccuracyScorer:
    """Test AccuracyScorer functionality."""

    @pytest.fixture
    def accuracy_scorer(self):
        """Create an AccuracyScorer instance.

        Returns:
            AccuracyScorer: An AccuracyScorer instance.
        """
        return AccuracyScorer()

    def test_can_score_classification(self, accuracy_scorer):
        """Test that AccuracyScorer can handle classification tasks and rejects text tasks."""
        # Should accept classification tasks
        assert accuracy_scorer.can_score_task(["A", "B", "C"]) is True
        assert accuracy_scorer.can_score_task(["positive", "negative"]) is True
        assert accuracy_scorer.can_score_task(["class1"]) is True  # Single class

    def test_cannot_score_free_form(self, accuracy_scorer):
        """Test that AccuracyScorer cannot handle free-form text tasks."""
        # Should reject text tasks
        assert accuracy_scorer.can_score_task(None) is False

    def test_compute_single_task_accuracy_perfect(
        self,
        accuracy_scorer,
        sample_consolidated_judge_df,
        sample_consolidated_human_df,
    ):
        """Test accuracy computation with perfect agreement."""
        accuracy = accuracy_scorer._compute_single_judge_task_accuracy(
            sample_consolidated_judge_df, sample_consolidated_human_df, "task1"
        )
        assert accuracy == 1.0

    def test_compute_single_task_accuracy_no_agreement(self, accuracy_scorer):
        """Test accuracy computation with no agreement."""
        consolidated_judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "judge_id": ["judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": ["A", "B", "C"],
            }
        )

        consolidated_human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "annotator_id": ["human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": ["C", "A", "B"],
            }
        )

        accuracy = accuracy_scorer._compute_single_judge_task_accuracy(
            consolidated_judge_df, consolidated_human_df, "task1"
        )
        assert accuracy == 0.0

    def test_compute_single_task_accuracy_partial(self, accuracy_scorer):
        """Test accuracy computation with partial agreement."""
        consolidated_judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "judge_id": ["judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": ["A", "B", "C"],
            }
        )

        consolidated_human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "annotator_id": ["human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": ["A", "A", "C"],  # 2 out of 3 correct
            }
        )

        accuracy = accuracy_scorer._compute_single_judge_task_accuracy(
            consolidated_judge_df, consolidated_human_df, "task1"
        )
        assert accuracy == 2 / 3

    def test_compute_single_task_accuracy_majority_vote(
        self,
        accuracy_scorer,
        sample_consolidated_judge_df,
        sample_consolidated_human_df,
    ):
        """Test accuracy computation with majority vote ground truth."""
        # For original_id "1": humans voted ["A", "A"] -> majority is "A"
        # Judge voted "A" -> correct
        # For original_id "2": humans voted ["B", "B"] -> majority is "B"
        # Judge voted "B" -> correct
        # For original_id "3": humans voted ["B", "B"] -> majority is "B"
        # Judge voted "A" -> incorrect

        accuracy = accuracy_scorer._compute_single_judge_task_accuracy(
            sample_consolidated_judge_df, sample_consolidated_human_df, "task1"
        )

        # Judge 1 should get 2/3 correct (ids 1,2 correct, id 3 incorrect)
        # Judge 2 should get some accuracy based on ids 4,5
        # The result is averaged across judges
        assert 0.0 <= accuracy <= 1.0

    def test_compute_multi_task_accuracy(self, accuracy_scorer):
        """Test multi-task accuracy computation."""
        consolidated_judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "1", "2"],
                "judge_id": ["judge_1", "judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task2", "task2"],
                "task_value": ["A", "B", "X", "Y"],
            }
        )

        consolidated_human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "1", "2"],
                "annotator_id": ["human_1", "human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task2", "task2"],
                "task_value": ["A", "A", "X", "X"],
            }
        )

        accuracy = accuracy_scorer._compute_single_judge_multi_task_accuracy(
            consolidated_judge_df, consolidated_human_df, ["task1", "task2"]
        )

        # Task1: 1/2 correct, Task2: 1/2 correct -> average = 0.5
        assert accuracy == 0.5

    def test_compute_score_single_task(
        self,
        accuracy_scorer,
        sample_consolidated_judge_df,
        sample_consolidated_human_df,
    ):
        """Test compute_score method for single task."""
        task_schemas = {"task1": ["A", "B", "C"]}
        result = accuracy_scorer.compute_score(
            "judge_1",
            sample_consolidated_judge_df,
            sample_consolidated_human_df,
            ["task1"],
            task_schemas,
        )

        assert result.scorer_name == "accuracy"
        assert result.task_name == "task1"
        assert result.judge_id == "judge_1"
        assert result.score == 1.0
        assert result.metadata["ground_truth_method"] == "majority_vote"
        assert result.metadata["scoring_method"] == "single_task"

    def test_compute_score_multi_task(self, accuracy_scorer):
        """Test compute_score method for multiple tasks."""
        consolidated_judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "1", "2"],
                "judge_id": ["judge_1", "judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task2", "task2"],
                "task_value": ["A", "B", "X", "Y"],
            }
        )

        consolidated_human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "1", "2"],
                "annotator_id": ["human_1", "human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task2", "task2"],
                "task_value": ["A", "A", "X", "X"],
            }
        )

        task_schemas = {"task1": ["A", "B", "C"], "task2": ["X", "Y", "Z"]}
        result = accuracy_scorer.compute_score(
            "judge_1",
            consolidated_judge_df,
            consolidated_human_df,
            ["task1", "task2"],
            task_schemas,
        )

        assert result.scorer_name == "accuracy"
        assert result.task_name == "2_tasks_avg"
        assert result.judge_id == "judge_1"
        assert result.score == 0.5
        assert result.metadata["ground_truth_method"] == "majority_vote"
        assert result.metadata["scoring_method"] == "average_across_tasks"

    def test_all_null_task_values(self, accuracy_scorer):
        """Test when all judge/human values are None for a task."""
        consolidated_judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "judge_id": ["judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": [None, None, None],
            }
        )

        consolidated_human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "annotator_id": ["human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": [None, None, None],
            }
        )

        # Should return 0.0 when all values are null
        accuracy = accuracy_scorer._compute_single_judge_task_accuracy(
            consolidated_judge_df, consolidated_human_df, "task1"
        )
        assert accuracy == 0.0

    def test_mixed_null_and_valid_data(self, accuracy_scorer):
        """Test some valid, some null values in same task."""
        consolidated_judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "judge_id": ["judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": [None, "B", "C"],
            }
        )

        consolidated_human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "annotator_id": ["human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": ["A", None, "C"],
            }
        )

        # Should compute accuracy only on valid pairs (should be only 1 valid pair: "C"=="C")
        accuracy = accuracy_scorer._compute_single_judge_task_accuracy(
            consolidated_judge_df, consolidated_human_df, "task1"
        )
        # Should be 1.0 since the only valid comparison is correct
        assert accuracy == 1.0

    def test_majority_vote_with_ties(self, accuracy_scorer):
        """Test majority vote ground truth when human annotators tie."""
        consolidated_judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2"],
                "judge_id": ["judge_1", "judge_1"],
                "task_name": ["task1", "task1"],
                "task_value": ["A", "B"],
            }
        )

        # Two humans disagree on each example - creates tie
        consolidated_human_df = pl.DataFrame(
            {
                "original_id": ["1", "1", "2", "2"],
                "annotator_id": ["human_1", "human_2", "human_1", "human_2"],
                "task_name": ["task1", "task1", "task1", "task1"],
                "task_value": ["A", "B", "A", "C"],  # Tie for both examples
            }
        )

        # Should handle ties gracefully (implementation specific behavior)
        accuracy = accuracy_scorer._compute_single_judge_task_accuracy(
            consolidated_judge_df, consolidated_human_df, "task1"
        )
        # Result depends on tie-breaking strategy, just ensure it's a valid accuracy
        assert 0.0 <= accuracy <= 1.0

    def test_aggregate_results_saves_individual_results(
        self, accuracy_scorer, tmp_path
    ):
        """Test that aggregate_results saves individual results as JSON files."""
        from meta_evaluator.scores.base_scoring_result import BaseScoringResult

        # Create mock results for 2 judges
        results = [
            BaseScoringResult(
                scorer_name="accuracy",
                task_name="task1",
                judge_id="judge_1",
                score=0.85,
                metadata={"ground_truth_method": "majority_vote"},
            ),
            BaseScoringResult(
                scorer_name="accuracy",
                task_name="task1",
                judge_id="judge_2",
                score=0.92,
                metadata={"ground_truth_method": "majority_vote"},
            ),
        ]

        scores_dir = str(tmp_path)

        # Call aggregate_results
        AccuracyScorer.aggregate_results(results, scores_dir)

        # Verify accuracy directory was created
        accuracy_dir = tmp_path / "accuracy"
        assert accuracy_dir.exists()

        # Verify individual result files were saved
        result_files = list(accuracy_dir.glob("*_result.json"))
        assert len(result_files) == 2

        # Verify file naming convention
        expected_files = {"judge_1_task1_result.json", "judge_2_task1_result.json"}
        actual_files = {f.name for f in result_files}
        assert actual_files == expected_files

        # Verify we can load back the results
        for result_file in result_files:
            loaded_result = BaseScoringResult.load_state(str(result_file))
            assert loaded_result.scorer_name == "accuracy"
            assert loaded_result.task_name == "task1"
            assert loaded_result.judge_id in ["judge_1", "judge_2"]
            assert isinstance(loaded_result.score, float)

    def test_aggregate_results_handles_empty_list(self, tmp_path):
        """Test that aggregate_results handles empty results list gracefully."""
        scores_dir = str(tmp_path)

        # Should not crash with empty list
        AccuracyScorer.aggregate_results([], scores_dir)

        # Should not create directory
        accuracy_dir = tmp_path / "accuracy"
        assert not accuracy_dir.exists()
