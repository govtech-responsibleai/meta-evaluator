"""Tests for custom scorer."""

import asyncio
from typing import List

import polars as pl
import pytest

from meta_evaluator.scores import (
    BaseScorer,
    BaseScoringResult,
)
from meta_evaluator.scores.enums import TaskAggregationMode


class CustomScorer(BaseScorer):
    """Custom scorer for testing integration."""

    def __init__(self, name="custom_scorer"):
        """Initialize custom scorer."""
        super().__init__(name)

    @property
    def min_human_annotators(self) -> int:
        """Minimum number of human annotators required for custom scoring.

        Returns:
            int: 1 human annotator minimum
        """
        return 1

    def can_score_task(self, sample_label: str | int | float | List[str | int | float]):
        """This custom scorer can only handle single column text tasks.

        Returns:
            bool: True, since this custom scorer can handle any task.
        """
        if isinstance(sample_label, str):
            return True
        else:
            return False

    async def compute_score_async(
        self,
        judge_data: pl.DataFrame,
        human_data: pl.DataFrame,
        task_name: str,
        judge_id: str,
        aggregation_mode,
        annotator_aggregation: str = "individual_average",
    ):
        """Simple custom scoring logic - count number of times judge has more letter "A"s than human, 0 otherwise.

        Returns:
            BaseScoringResult: A BaseScoringResult instance.
        """
        # Join judge and human data on original_id (approach 1: join first)
        comparison_df = judge_data.join(human_data, on="original_id", how="inner")

        # Simple scoring: return 1 if judge and human have the same text, 0 otherwise
        if comparison_df.is_empty():
            score = float("nan")
            num_comparisons = 0
            failed_comparisons = 1
        else:
            judge_wins = []
            num_comparisons = 0
            failed_comparisons = 0

            humans = comparison_df["human_id"].unique()
            for human_id in humans:
                try:
                    comparison_subset = comparison_df.filter(
                        pl.col("human_id") == human_id
                    )
                    judge_texts = comparison_subset["label"].to_list()
                    human_texts = comparison_subset["label_right"].to_list()

                    # Count number of times judge has more "A"s than human
                    judge_more_A = 0
                    human_more_A = 0
                    for judge_text, human_text in zip(judge_texts, human_texts):
                        judge_count = str(judge_text).count("A")
                        human_count = str(human_text).count("A")
                        if judge_count > human_count:
                            judge_more_A += 1
                        else:
                            human_more_A += 1

                    # If judge has more "A"s than human, count as a win
                    if judge_more_A > human_more_A:
                        judge_wins.append(1)
                    else:
                        judge_wins.append(0)

                    num_comparisons += 1

                except Exception as e:
                    self.logger.error(f"Error computing score: {e}")
                    failed_comparisons += 1
                    continue

            # Calculate win rate
            score = sum(judge_wins) / len(judge_wins) if len(judge_wins) > 0 else 0.0

            # Calculate number of comparisons and failed comparisons
            num_comparisons = len(comparison_df)
            failed_comparisons = 0

        return BaseScoringResult(
            scorer_name=self.scorer_name,
            task_name=task_name,
            judge_id=judge_id,
            scores={"win_rate": score},
            metadata={},
            aggregation_mode=aggregation_mode,
            num_comparisons=num_comparisons,
            failed_comparisons=failed_comparisons,
        )


class TestCustomScorer:
    """Test custom scorer functionality with mock data."""

    @pytest.fixture
    def custom_scorer(self):
        """Create a custom scorer.

        Returns:
            CustomScorer: A CustomScorer instance.
        """
        return CustomScorer()

    def test_custom_scorer_can_only_score_text_tasks(self, custom_scorer):
        """Test that custom scorer can only score text tasks."""
        assert custom_scorer.can_score_task("A") is True
        assert custom_scorer.can_score_task(None) is False
        assert custom_scorer.can_score_task([]) is False
        assert custom_scorer.can_score_task(["A", "B", "C"]) is False
        assert custom_scorer.can_score_task(1) is False
        assert custom_scorer.can_score_task(1.0) is False

    def test_custom_scorer_compute_score_single_task(self, custom_scorer):
        """Test custom scorer with single task."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "label": ["AAA", "AAB", "CCC"],
            }
        )

        human_df = pl.DataFrame(
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
                "label": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            }
        )

        result = asyncio.run(
            custom_scorer.compute_score_async(
                judge_df, human_df, "task1", "judge_1", TaskAggregationMode.SINGLE
            )
        )

        assert result.scorer_name == "custom_scorer"
        assert result.task_name == "task1"
        assert result.judge_id == "judge_1"
        assert result.scores["win_rate"] == 0.5  # Won against human 1, lost to human 2
        assert result.num_comparisons == 6
        assert result.failed_comparisons == 0
