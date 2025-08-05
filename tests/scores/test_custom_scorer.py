"""Tests for custom scorer."""

import pytest
import polars as pl

from meta_evaluator.scores import (
    BaseScorer,
    BaseScoringResult,
)


class CustomScorer(BaseScorer):
    """Custom scorer for testing integration."""

    def __init__(self, name="custom_scorer"):
        """Initialize custom scorer."""
        super().__init__(name)

    def can_score_task(self, task_schema):
        """This custom scorer can handle any task.

        Returns:
            bool: True, since this custom scorer can handle any task.
        """
        return True

    def compute_score(
        self,
        judge_id,
        consolidated_judge_df,
        consolidated_human_df,
        task_names,
        task_schemas,
    ):
        """Simple custom scoring logic - return average of task counts.

        Returns:
            BaseScoringResult: A BaseScoringResult instance.
        """
        judge_count = len(consolidated_judge_df)
        human_count = len(consolidated_human_df)

        # Simple scoring: return normalized count ratio
        if human_count == 0:
            score = 0.0
        else:
            score = min(judge_count / human_count, 1.0)

        task_display = (
            task_names[0] if len(task_names) == 1 else f"{len(task_names)}_tasks_avg"
        )

        return BaseScoringResult(
            scorer_name=self.scorer_name,
            task_name=task_display,
            judge_id=judge_id,
            score=score,
            metadata={
                "judge_count": judge_count,
                "human_count": human_count,
                "task_names": task_names,
                "task_schemas": task_schemas,
                "scoring_method": "custom_count_ratio",
            },
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

    def test_custom_scorer_can_score_any_task(self, custom_scorer):
        """Test that custom scorer can handle any task type."""
        assert custom_scorer.can_score_task(["A", "B", "C"]) is True
        assert custom_scorer.can_score_task(None) is True
        assert custom_scorer.can_score_task([]) is True

    def test_custom_scorer_compute_score_single_task(self, custom_scorer):
        """Test custom scorer with single task."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "judge_id": ["judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task1"],
                "task1": ["A", "B", "C"],
            }
        )

        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "annotator_id": ["human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task1"],
                "task1": ["A", "B", "C"],
            }
        )

        task_schemas = {"task1": ["A", "B", "C"]}
        result = custom_scorer.compute_score(
            "judge_1", judge_df, human_df, ["task1"], task_schemas
        )

        assert result.scorer_name == "custom_scorer"
        assert result.task_name == "task1"
        assert result.judge_id == "judge_1"
        assert result.score == 1.0  # 3/3 = 1.0
        assert result.metadata["judge_count"] == 3
        assert result.metadata["human_count"] == 3
        assert result.metadata["scoring_method"] == "custom_count_ratio"

    def test_custom_scorer_compute_score_multi_task(self, custom_scorer):
        """Test custom scorer with multiple tasks."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "1", "2"],
                "judge_id": ["judge_1", "judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task2", "task2"],
                "task1": ["A", "B", None, None],
                "task2": [None, None, "hello", "world"],
            }
        )

        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "1", "2"],
                "annotator_id": ["human_1", "human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task2", "task2"],
                "task1": ["A", "B", None, None],
                "task2": [None, None, "hello", "world"],
            }
        )

        task_schemas = {"task1": ["A", "B", "C"], "task2": None}
        result = custom_scorer.compute_score(
            "judge_1", judge_df, human_df, ["task1", "task2"], task_schemas
        )

        assert result.scorer_name == "custom_scorer"
        assert result.task_name == "2_tasks_avg"
        assert result.judge_id == "judge_1"
        assert result.score == 1.0  # 4/4 = 1.0
        assert result.metadata["judge_count"] == 4
        assert result.metadata["human_count"] == 4
