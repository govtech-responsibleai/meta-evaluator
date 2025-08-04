"""Accuracy scorer for classification tasks."""

from typing import List, Optional
import os

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score

from meta_evaluator.scores.base_scorer import BaseScorer
from meta_evaluator.scores.base_scoring_result import BaseScoringResult


class AccuracyScorer(BaseScorer):
    """Scorer for classification tasks using accuracy metric with majority vote."""

    def __init__(self):
        """Initialize accuracy scorer."""
        super().__init__("accuracy")

    def can_score_task(self, task_schema: Optional[List[str]]) -> bool:
        """Accuracy scorer only works with classification tasks (predefined outcomes).

        Returns:
            bool: True if the task is a classification task, False otherwise.
        """
        return task_schema is not None

    def compute_score(
        self,
        judge_id: str,
        consolidated_judge_df: pl.DataFrame,
        consolidated_human_df: pl.DataFrame,
        task_names: List[str],
        task_schemas: dict,
    ) -> BaseScoringResult:
        """Compute accuracy score for a single judge vs many humans.

        Returns:
            BaseScoringResult: A BaseScoringResult instance.
        """
        if len(task_names) == 1:
            # Single task
            task_name = task_names[0]
            score = self._compute_single_judge_task_accuracy(
                consolidated_judge_df, consolidated_human_df, task_name
            )
            task_display = task_name
        else:
            # Multi-task - average accuracy across all tasks
            score = self._compute_single_judge_multi_task_accuracy(
                consolidated_judge_df, consolidated_human_df, task_names
            )
            task_display = f"{len(task_names)}_tasks_avg"

        return BaseScoringResult(
            scorer_name=self.scorer_name,
            task_name=task_display,
            judge_id=judge_id,
            score=score,
            metadata={
                "ground_truth_method": "majority_vote",
                "task_names": task_names,
                "task_schemas": task_schemas,
                "scoring_method": (
                    "single_task" if len(task_names) == 1 else "average_across_tasks"
                ),
            },
        )

    def _compute_single_judge_task_accuracy(
        self,
        consolidated_judge_df: pl.DataFrame,
        consolidated_human_df: pl.DataFrame,
        task_name: str,
    ) -> float:
        """Compute accuracy for a single judge on a single task.

        Returns:
            float: The accuracy score.
        """
        # Filter data for this specific task
        task_judge_df = consolidated_judge_df.filter(pl.col("task_name") == task_name)
        task_human_df = consolidated_human_df.filter(pl.col("task_name") == task_name)

        if task_judge_df.is_empty() or task_human_df.is_empty():
            return np.nan

        # Compute majority vote for each original_id
        task_human_df = (
            task_human_df.filter(
                pl.col("task_value").is_not_null()
            )  # Remove null human annotations
            .group_by("original_id")
            .agg(
                pl.col("task_value")
                .mode()
                .first()
                .alias("majority_vote")  # Get most common value
            )
        )

        # Join judge predictions with majority vote
        comparison_df = (
            task_judge_df.filter(
                pl.col("task_value").is_not_null()
            )  # Remove null judge predictions
            .join(task_human_df, on="original_id", how="inner")
            .select("original_id", "task_value", "majority_vote")
        )

        # Extract judge and human predictions
        judge_predictions = comparison_df["task_value"].to_list()
        human_predictions = comparison_df["majority_vote"].to_list()

        # Compute accuracy using sklearn
        accuracy = accuracy_score(y_true=human_predictions, y_pred=judge_predictions)

        # Handle case where sklearn returns nan (no valid comparisons)
        if np.isnan(accuracy):
            return np.nan
        else:
            return accuracy

    def _compute_single_judge_multi_task_accuracy(
        self,
        consolidated_judge_df: pl.DataFrame,
        consolidated_human_df: pl.DataFrame,
        task_names: List[str],
    ) -> float:
        """Compute average accuracy across multiple tasks for a single judge.

        Returns:
            float: The average accuracy score.
        """
        task_accuracies = []
        for task_name in task_names:
            task_accuracy = self._compute_single_judge_task_accuracy(
                consolidated_judge_df, consolidated_human_df, task_name
            )
            task_accuracies.append(task_accuracy)

        if task_accuracies:
            return float(np.nanmean(task_accuracies))
        else:
            return np.nan

    @classmethod
    def aggregate_results(
        cls, results: List[BaseScoringResult], scores_dir: str
    ) -> None:
        """Generate aggregate plots and save individual results for accuracy scorer.

        Args:
            results: List of accuracy scoring results
            scores_dir: Directory to save results and plots
        """
        if not results:
            print("No accuracy results to aggregate")
            return

        # Create accuracy directory for results and plots
        accuracy_dir = os.path.join(scores_dir, "accuracy")
        os.makedirs(accuracy_dir, exist_ok=True)

        # Save individual results
        cls._save_results(results, accuracy_dir)

        print(
            f"Generated accuracy results for {len(results)} judge(s) in {accuracy_dir}"
        )

    @classmethod
    def _save_results(cls, results: List[BaseScoringResult], accuracy_dir: str) -> None:
        """Save individual ScoringResult objects as JSON files."""
        for result in results:
            # Create filename: judge_id_task_name_result.json
            filename = f"{result.judge_id}_{result.task_name}_result.json"
            file_path = os.path.join(accuracy_dir, filename)

            result.save_state(file_path)
            print(f"Saved individual result to {file_path}")
