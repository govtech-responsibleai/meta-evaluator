"""Cohen's kappa scorer for inter-annotator agreement."""

from typing import List, Optional
import statistics
import os

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
            return 0.0

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

        return statistics.mean(task_kappas) if task_kappas else 0.0

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
        # Data is already aligned by original_id, so we can directly extract labels
        # Use task_value column instead of task_name column
        judge_labels = consolidated_judge_df["task_value"].to_list()
        human_labels = consolidated_human_df["task_value"].to_list()

        # Filter out None values
        valid_pairs = [
            (j, h)
            for j, h in zip(judge_labels, human_labels)
            if j is not None and h is not None
        ]

        if len(valid_pairs) < 2:
            return 0.0

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
