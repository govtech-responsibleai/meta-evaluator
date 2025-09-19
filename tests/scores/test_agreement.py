"""Tests for agreement metrics."""

import polars as pl
import pytest

from meta_evaluator.scores.base_scoring_result import BaseScoringResult
from meta_evaluator.scores.enums import TaskAggregationMode

# Basic fixtures are now provided by conftest.py


class TestCohensKappaScorer:
    """Test CohensKappaScorer functionality."""

    def test_can_score_classification(self, cohens_kappa_scorer):
        """Test that CohensKappaScorer can handle classification tasks."""
        # Should accept classification tasks
        assert cohens_kappa_scorer.can_score_task("A") is True
        assert cohens_kappa_scorer.can_score_task(["A", "B", "C"]) is True
        assert cohens_kappa_scorer.can_score_task(1) is True

        # Should reject float tasks
        assert cohens_kappa_scorer.can_score_task(3.14) is False

    async def test_compute_score_async(self, cohens_kappa_scorer):
        """Test compute_score_async method."""
        # Create test data with perfect agreement
        judge_data = pl.DataFrame(
            {"original_id": ["1", "2", "3"], "label": ["A", "B", "A"]}
        )

        human_data = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "human_id": ["human_1", "human_1", "human_1"],
                "label": ["A", "B", "A"],
            }
        )

        result = await cohens_kappa_scorer.compute_score_async(
            judge_data, human_data, "task1", "judge_1", None
        )

        assert result.scorer_name == "cohens_kappa"
        assert result.task_name == "task1"
        assert result.judge_id == "judge_1"
        assert "kappa" in result.scores
        assert isinstance(result.scores["kappa"], float)
        assert result.scores["kappa"] == 1.0  # Perfect agreement

    @pytest.mark.asyncio
    async def test_majority_vote_warning(self, cohens_kappa_scorer, caplog):
        """Test that CohensKappaScorer logs warning when majority_vote aggregation is used."""
        # Create dummy data
        judge_data = pl.DataFrame({"original_id": [1, 2, 3], "label": ["A", "B", "A"]})

        human_data = pl.DataFrame(
            {
                "original_id": [1, 2, 3, 1, 2, 3],
                "human_id": [
                    "human1",
                    "human1",
                    "human1",
                    "human2",
                    "human2",
                    "human2",
                ],
                "label": ["A", "B", "A", "A", "A", "B"],
            }
        )

        # Test with majority_vote - should log warning
        with caplog.at_level("WARNING"):
            await cohens_kappa_scorer.compute_score_async(
                judge_data,
                human_data,
                "test",
                "judge1",
                TaskAggregationMode.SINGLE,
                "majority_vote",
            )

        # Check that warning was logged
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any(
            "CohensKappaScorer does not support majority_vote aggregation" in msg
            for msg in warning_messages
        )

        # Test with individual_average - should not log warning
        caplog.clear()
        with caplog.at_level("WARNING"):
            await cohens_kappa_scorer.compute_score_async(
                judge_data,
                human_data,
                "test",
                "judge1",
                TaskAggregationMode.SINGLE,
                "individual_average",
            )

        # Check that no warning was logged
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert not any(
            "CohensKappaScorer does not support majority_vote aggregation" in msg
            for msg in warning_messages
        )


class TestAltTestScorer:
    """Test AltTestScorer functionality."""

    def test_can_score_any_task(self, alt_test_scorer):
        """Test that AltTestScorer can handle any task type."""
        assert alt_test_scorer.can_score_task(["A", "B", "C"]) is True  # Classification
        assert alt_test_scorer.can_score_task("text response") is True  # Text
        assert alt_test_scorer.can_score_task(1) is True  # Numeric

    def test_get_scoring_function_calls_right_function(self, alt_test_scorer):
        """Test that _get_scoring_function calls the right scoring function."""
        # Single classification label -> accuracy
        assert alt_test_scorer._determine_scoring_function("SAFE") == "accuracy"

        # Multi-label classification (list of labels) -> jaccard_similarity
        assert (
            alt_test_scorer._determine_scoring_function(["SAFE", "NON_TOXIC"])
            == "jaccard_similarity"
        )

        # Text label -> accuracy
        assert (
            alt_test_scorer._determine_scoring_function("This is a text response")
            == "accuracy"
        )

    @pytest.mark.asyncio
    async def test_compute_score_async_classification(self, alt_test_scorer):
        """Test compute_score_async with classification data."""
        # Create test data for classification
        judge_data = pl.DataFrame(
            {"original_id": ["1", "2", "3"], "label": ["SAFE", "UNSAFE", "SAFE"]}
        )

        human_data = pl.DataFrame(
            {
                "original_id": ["1", "2", "3", "1", "2", "3"],
                "human_id": [
                    "human_1",
                    "human_1",
                    "human_1",
                    "human_2",
                    "human_2",
                    "human_2",
                ],
                "label": ["SAFE", "UNSAFE", "SAFE", "SAFE", "SAFE", "UNSAFE"],
            }
        )

        result = await alt_test_scorer.compute_score_async(
            judge_data, human_data, "task1", "judge_1", TaskAggregationMode.SINGLE
        )

        assert result.scorer_name == "alt_test"
        assert result.task_name == "task1"
        assert result.judge_id == "judge_1"
        assert "winning_rate" in result.scores
        assert "advantage_probability" in result.scores
        assert isinstance(result.scores["advantage_probability"], float)

    @pytest.mark.asyncio
    async def test_compute_score_async_text(self, alt_test_scorer):
        """Test compute_score_async with text data."""
        # Create test data for text
        judge_data = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "label": ["Good summary", "Bad summary", "OK summary"],
            }
        )

        human_data = pl.DataFrame(
            {
                "original_id": ["1", "2", "3", "1", "2", "3"],
                "human_id": [
                    "human_1",
                    "human_1",
                    "human_1",
                    "human_2",
                    "human_2",
                    "human_2",
                ],
                "label": [
                    "Good summary",
                    "Bad summary",
                    "OK summary",
                    "Great summary",
                    "Poor summary",
                    "OK summary",
                ],
            }
        )

        result = await alt_test_scorer.compute_score_async(
            judge_data, human_data, "task1", "judge_1", TaskAggregationMode.SINGLE
        )

        assert result.scorer_name == "alt_test"
        assert result.task_name == "task1"
        assert result.judge_id == "judge_1"
        assert "winning_rate" in result.scores
        assert "advantage_probability" in result.scores
        assert isinstance(result.scores["advantage_probability"], float)

    def test_aggregate_results_generates_3_plots(self, alt_test_scorer, tmp_path):
        """Test that aggregate_results generates 3 plots."""
        # Create mock results
        results = [
            BaseScoringResult(
                scorer_name="alt_test",
                task_name="safety",
                judge_id="judge_1",
                scores={
                    "winning_rate": {"0.00": 0.8, "0.01": 0.75, "0.02": 0.7},
                    "advantage_probability": 0.7,
                },
                metadata={
                    "scoring_function": "accuracy",
                    "advantage_probability": 0.7,
                    "human_advantage_probabilities": {
                        "human_1": (0.6, 0.4),
                        "human_2": (0.8, 0.2),
                    },
                },
                aggregation_mode=TaskAggregationMode.SINGLE,
                num_comparisons=10,
            ),
            BaseScoringResult(
                scorer_name="alt_test",
                task_name="safety",
                judge_id="judge_2",
                scores={
                    "winning_rate": {"0.00": 0.9, "0.01": 0.85, "0.02": 0.8},
                    "advantage_probability": 0.8,
                },
                metadata={
                    "scoring_function": "accuracy",
                    "advantage_probability": 0.8,
                    "human_advantage_probabilities": {
                        "human_1": (0.7, 0.3),
                        "human_2": (0.9, 0.1),
                    },
                },
                aggregation_mode=TaskAggregationMode.SINGLE,
                num_comparisons=12,
            ),
        ]

        scores_dir = str(tmp_path)

        # Call aggregate_results
        alt_test_scorer.aggregate_results(results, scores_dir)

        # Check that alt_test directory was created
        alt_test_dir = tmp_path / "alt_test"
        assert alt_test_dir.exists()

        # Check that 3 aggregate plots were generated
        expected_plots = [
            "aggregate_winning_rates.png",
            "aggregate_advantage_probabilities.png",
            "aggregate_human_vs_llm_advantage.png",
        ]

        for plot_name in expected_plots:
            plot_path = alt_test_dir / plot_name
            assert plot_path.exists(), f"Missing plot: {plot_name}"

    @pytest.mark.asyncio
    async def test_majority_vote_warning(self, alt_test_scorer, caplog):
        """Test that AltTestScorer logs warning when majority_vote aggregation is used."""
        # Create dummy data
        judge_data = pl.DataFrame({"original_id": [1, 2, 3], "label": ["A", "B", "A"]})

        human_data = pl.DataFrame(
            {
                "original_id": [1, 2, 3, 1, 2, 3],
                "human_id": [
                    "human1",
                    "human1",
                    "human1",
                    "human2",
                    "human2",
                    "human2",
                ],
                "label": ["A", "B", "A", "A", "A", "B"],
            }
        )

        # Test with majority_vote - should log warning
        with caplog.at_level("WARNING"):
            await alt_test_scorer.compute_score_async(
                judge_data,
                human_data,
                "test",
                "judge1",
                TaskAggregationMode.SINGLE,
                "majority_vote",
            )

        # Check that warning was logged
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any(
            "AltTestScorer does not support majority_vote aggregation" in msg
            for msg in warning_messages
        )

        # Test with individual_average - should not log warning
        caplog.clear()
        with caplog.at_level("WARNING"):
            await alt_test_scorer.compute_score_async(
                judge_data,
                human_data,
                "test",
                "judge1",
                TaskAggregationMode.SINGLE,
                "individual_average",
            )

        # Check that no warning was logged
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert not any(
            "AltTestScorer does not support majority_vote aggregation" in msg
            for msg in warning_messages
        )
