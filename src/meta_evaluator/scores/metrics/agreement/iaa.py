"""Cohen's kappa scorer for inter-annotator agreement."""

from typing import List, Optional
import os

import numpy as np
import polars as pl
from sklearn.metrics import cohen_kappa_score

from meta_evaluator.scores.base_scorer import BaseScorer
from meta_evaluator.scores.base_scoring_result import BaseScoringResult


class CohensKappaScorer(BaseScorer):
    """Scorer for inter-annotator agreement using Cohen's kappa."""

    def __init__(self):
        """Initialize Cohen's kappa scorer."""
        super().__init__("cohens_kappa")

    def can_score_task(self, task_schema: Optional[List[str]]) -> bool:
        """Cohen's kappa only works with classification tasks (categorical data).

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
        """Compute Cohen's kappa for a single judge vs many humans.

        Returns:
            BaseScoringResult: A BaseScoringResult instance.
        """
        if len(task_names) == 1:
            # Single task
            task_name = task_names[0]
            task_schema = task_schemas[task_name]
            score = self._compute_single_judge_task_kappa(
                consolidated_judge_df, consolidated_human_df, task_name, task_schema
            )
            task_display = task_name
        else:
            # Multi-task - average kappa across all tasks
            score = self._compute_single_judge_multi_task_kappa(
                consolidated_judge_df, consolidated_human_df, task_names, task_schemas
            )
            task_display = f"{len(task_names)}_tasks_avg"

        return BaseScoringResult(
            scorer_name=self.scorer_name,
            task_name=task_display,
            judge_id=judge_id,
            score=score,
            metadata={
                "task_names": task_names,
                "task_schemas": task_schemas,
                "scoring_method": (
                    "single_task" if len(task_names) == 1 else "average_across_tasks"
                ),
            },
        )

    def _compute_single_judge_task_kappa(
        self,
        consolidated_judge_df: pl.DataFrame,
        consolidated_human_df: pl.DataFrame,
        task_name: str,
        task_schema: Optional[List[str]],
    ) -> float:
        """Compute Cohen's kappa for a single judge on a single task.

        Returns:
            float: The Cohen's kappa score.
        """
        # Filter data for this specific task
        task_judge_df = consolidated_judge_df.filter(pl.col("task_name") == task_name)
        task_human_df = consolidated_human_df.filter(pl.col("task_name") == task_name)

        if task_judge_df.is_empty() or task_human_df.is_empty():
            return np.nan

        # Cohen's kappa only works with classification tasks
        return self._compute_classification_kappa(
            task_judge_df, task_human_df, task_name
        )

    def _compute_single_judge_multi_task_kappa(
        self,
        consolidated_judge_df: pl.DataFrame,
        consolidated_human_df: pl.DataFrame,
        task_names: List[str],
        task_schemas: dict,
    ) -> float:
        """Compute average Cohen's kappa across multiple tasks for a single judge.

        Returns:
            float: The average Cohen's kappa score.
        """
        task_kappas = []
        for task_name in task_names:
            task_schema = task_schemas[task_name]
            task_kappa = self._compute_single_judge_task_kappa(
                consolidated_judge_df, consolidated_human_df, task_name, task_schema
            )
            task_kappas.append(task_kappa)

        if task_kappas:
            return float(np.mean(task_kappas))
        else:
            return np.nan

    def _compute_classification_kappa(
        self,
        consolidated_judge_df: pl.DataFrame,
        consolidated_human_df: pl.DataFrame,
        task_name: str,
    ) -> float:
        """Compute Cohen's kappa for classification tasks.

        Returns:
            float: The Cohen's kappa score.
        """
        # Sort both DataFrames by original_id to ensure proper alignment
        judge_df_sorted = consolidated_judge_df.sort("original_id")
        human_df_sorted = consolidated_human_df.sort("original_id")

        # Handle multiple human annotators per original_id by using majority vote
        # Group human annotations by original_id and compute majority vote
        human_aggregated = (
            human_df_sorted.group_by("original_id")
            .agg(
                [
                    pl.col("task_value")
                    .mode()
                    .first()
                    .alias("task_value")  # Mode (majority vote)
                ]
            )
            .sort("original_id")
        )

        # Judge data should have one entry per original_id, but group just in case
        judge_aggregated = (
            judge_df_sorted.group_by("original_id")
            .agg(
                [
                    pl.col("task_value").first().alias("task_value")  # Take first entry
                ]
            )
            .sort("original_id")
        )

        # Find common original_ids between judge and human data
        judge_ids = set(judge_aggregated["original_id"].to_list())
        human_ids = set(human_aggregated["original_id"].to_list())
        common_ids = judge_ids & human_ids

        if len(common_ids) < 2:
            return np.nan

        # Filter to common IDs and ensure same order
        judge_final = judge_aggregated.filter(
            pl.col("original_id").is_in(list(common_ids))
        ).sort("original_id")
        human_final = human_aggregated.filter(
            pl.col("original_id").is_in(list(common_ids))
        ).sort("original_id")

        # Extract labels
        judge_labels = judge_final["task_value"].to_list()
        human_labels = human_final["task_value"].to_list()

        # Filter out None values while maintaining alignment
        valid_pairs = [
            (j, h)
            for j, h in zip(judge_labels, human_labels)
            if j is not None and h is not None
        ]

        if len(valid_pairs) < 2:
            return np.nan

        judge_labels, human_labels = zip(*valid_pairs)

        # Compute Cohen's kappa
        return cohen_kappa_score(human_labels, judge_labels)

    @classmethod
    def aggregate_results(
        cls, results: List[BaseScoringResult], scores_dir: str
    ) -> None:
        """Generate aggregate plots and save individual results for Cohen's kappa scorer.

        Args:
            results: List of Cohen's kappa scoring results
            scores_dir: Directory to save results and plots
        """
        if not results:
            print("No Cohen's kappa results to aggregate")
            return

        # Create cohens_kappa directory for results and plots
        cohens_kappa_dir = os.path.join(scores_dir, "cohens_kappa")
        os.makedirs(cohens_kappa_dir, exist_ok=True)

        # Save individual results
        cls._save_results(results, cohens_kappa_dir)

        print(
            f"Generated Cohen's kappa results for {len(results)} judge(s) in {cohens_kappa_dir}"
        )

    @classmethod
    def _save_results(
        cls, results: List[BaseScoringResult], cohens_kappa_dir: str
    ) -> None:
        """Save individual ScoringResult objects as JSON files."""
        for result in results:
            # Create filename: judge_id_task_name_result.json
            filename = f"{result.judge_id}_{result.task_name}_result.json"
            file_path = os.path.join(cohens_kappa_dir, filename)

            result.save_state(file_path)
            print(f"Saved individual result to {file_path}")
