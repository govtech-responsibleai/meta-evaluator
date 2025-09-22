"""Tests for text similarity metrics."""

import polars as pl
import pytest

from meta_evaluator.scores.enums import TaskAggregationMode


class TestTextSimilarityScorer:
    """Test TextSimilarityScorer functionality."""

    def test_can_score_task_free_form(self, text_similarity_scorer):
        """Test that TextSimilarityScorer can handle text tasks."""
        # Should accept free-form text tasks
        assert text_similarity_scorer.can_score_task("My explanation is...") is True
        assert (
            text_similarity_scorer.can_score_task(
                ["Explanation 1...", "Explanation 2..."]
            )
            is True
        )

    async def test_compute_score_async_perfect_match(self, text_similarity_scorer):
        """Test compute_score_async with 100% match."""
        # Create test data with perfect agreement
        judge_data = pl.DataFrame(
            {"original_id": ["1", "2", "3"], "label": ["hello", "world", "test"]}
        )

        human_data = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "human_id": ["human_1", "human_1", "human_1"],
                "label": ["hello", "world", "test"],
            }
        )

        result = await text_similarity_scorer.compute_score_async(
            judge_data, human_data, "task1", "judge_1", None
        )

        assert result.scorer_name == "text_similarity"
        assert result.task_name == "task1"
        assert result.judge_id == "judge_1"
        assert "text_similarity" in result.scores
        assert isinstance(result.scores["text_similarity"], float)
        assert result.scores["text_similarity"] == 1.0  # Perfect match

    async def test_compute_score_async_little_match(self, text_similarity_scorer):
        """Test compute_score_async with little match."""
        # Create test data with perfect agreement
        judge_data = pl.DataFrame(
            {"original_id": ["1", "2", "3"], "label": ["hello", "world", "test"]}
        )

        human_data = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "human_id": ["human_1", "human_1", "human_1"],
                "label": ["bye", "test", "example"],
            }
        )

        result = await text_similarity_scorer.compute_score_async(
            judge_data, human_data, "task1", "judge_1", None
        )

        assert result.scorer_name == "text_similarity"
        assert result.task_name == "task1"
        assert result.judge_id == "judge_1"
        assert "text_similarity" in result.scores
        assert isinstance(result.scores["text_similarity"], float)
        assert result.scores["text_similarity"] < 0.5  # Little match

    @pytest.mark.asyncio
    async def test_compute_score_async_majority_vote_best_match(
        self, text_similarity_scorer
    ):
        """Test majority vote using best match approach."""
        # Judge gets perfect match with one human, partial with others
        judge_data = pl.DataFrame(
            {"original_id": [1, 2], "label": ["hello world", "test case"]}
        )

        human_data = pl.DataFrame(
            {
                "original_id": [1, 1, 1, 2, 2, 2],
                "human_id": ["h1", "h2", "h3", "h1", "h2", "h3"],
                "label": [
                    "hello world",  # Perfect match
                    "goodbye world",  # Partial match
                    "hi there",  # Low match
                    "test case",  # Perfect match
                    "exam question",  # Partial match
                    "different text",  # Low match
                ],
            }
        )

        result = await text_similarity_scorer.compute_score_async(
            judge_data,
            human_data,
            "test",
            "judge1",
            TaskAggregationMode.SINGLE,
            "majority_vote",
        )

        # Should get close to 1.0 since best matches are perfect
        assert result.scores["similarity"] > 0.9
