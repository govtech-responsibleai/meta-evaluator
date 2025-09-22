"""Tests for classification Metrics."""

import polars as pl
import pytest

from meta_evaluator.scores.enums import TaskAggregationMode


class TestAccuracyScorer:
    """Test AccuracyScorer functionality."""

    def test_can_score_classification(self, accuracy_scorer):
        """Test that AccuracyScorer can handle classification tasks."""
        # Should accept classification tasks
        assert accuracy_scorer.can_score_task("A") is True
        assert accuracy_scorer.can_score_task(["A", "B", "C"]) is True
        assert accuracy_scorer.can_score_task(1) is True

        # Should reject float tasks
        assert accuracy_scorer.can_score_task(3.14) is False

    async def test_compute_score_async_perfect_match(self, accuracy_scorer):
        """Test compute_score_async with 100% match."""
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

        result = await accuracy_scorer.compute_score_async(
            judge_data, human_data, "task1", "judge_1", None
        )

        assert result.scorer_name == "accuracy"
        assert result.task_name == "task1"
        assert result.judge_id == "judge_1"
        assert "accuracy" in result.scores
        assert isinstance(result.scores["accuracy"], float)
        assert result.scores["accuracy"] == 1.0  # Perfect match

    async def test_compute_score_async_little_match(self, accuracy_scorer):
        """Test compute_score_async with little match."""
        # Create test data with poor agreement
        judge_data = pl.DataFrame(
            {"original_id": ["1", "2", "3"], "label": ["A", "B", "C"]}
        )

        human_data = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "human_id": ["human_1", "human_1", "human_1"],
                "label": ["C", "A", "B"],  # All different
            }
        )

        result = await accuracy_scorer.compute_score_async(
            judge_data, human_data, "task1", "judge_1", None
        )

        assert result.scorer_name == "accuracy"
        assert result.task_name == "task1"
        assert result.judge_id == "judge_1"
        assert "accuracy" in result.scores
        assert isinstance(result.scores["accuracy"], float)
        assert result.scores["accuracy"] == 0.0  # No matches

    @pytest.mark.asyncio
    async def test_compute_score_async_majority_vote(self, accuracy_scorer):
        """Test compute_score_async with majority vote."""
        # Judge gets 1 out of 2 correct vs majority vote
        judge_data = pl.DataFrame({"original_id": [1, 2], "label": ["A", "B"]})

        human_data = pl.DataFrame(
            {
                "original_id": [1, 1, 1, 2, 2, 2],
                "human_id": ["h1", "h2", "h3", "h1", "h2", "h3"],
                "label": ["A", "A", "C", "X", "X", "B"],  # Majorities: "A", "X"
            }
        )

        result = await accuracy_scorer.compute_score_async(
            judge_data,
            human_data,
            "test",
            "judge1",
            TaskAggregationMode.SINGLE,
            "majority_vote",
        )

        assert result.scores["accuracy"] == 0.5  # Judge matches 1/2 majorities
