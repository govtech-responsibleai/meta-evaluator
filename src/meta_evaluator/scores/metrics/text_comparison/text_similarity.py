"""Text similarity scorer for free-form text tasks."""

from typing import List, Optional
import os

import numpy as np
import polars as pl
from difflib import SequenceMatcher

from meta_evaluator.scores.base_scorer import BaseScorer
from meta_evaluator.scores.base_scoring_result import BaseScoringResult


class TextSimilarityScorer(BaseScorer):
    """Scorer for free-form text tasks using text similarity with best match."""

    def __init__(self):
        """Initialize text similarity scorer."""
        super().__init__("text_similarity")

    def can_score_task(self, task_schema: Optional[List[str]]) -> bool:
        """Text similarity scorer only works with free-form text tasks.

        Text similarity measures semantic similarity between text responses.
        It requires free-form text content, not discrete categorical choices.

        Args:
            task_schema: List of allowed categorical outcomes, or None for free-form text tasks

        Returns:
            bool: True if task_schema is None (free-form text task), False otherwise
        """
        return task_schema is None

    def compute_score(
        self,
        judge_id: str,
        consolidated_judge_df: pl.DataFrame,
        consolidated_human_df: pl.DataFrame,
        task_names: List[str],
        task_schemas: dict,
    ) -> BaseScoringResult:
        """Compute text similarity score for a single judge vs many humans.

        Returns:
            BaseScoringResult: A BaseScoringResult instance.
        """
        if len(task_names) == 1:
            # Single task
            task_name = task_names[0]
            score = self._compute_single_judge_task_similarity(
                consolidated_judge_df, consolidated_human_df, task_name
            )
            task_display = task_name
        else:
            # Multi-task - average similarity across all tasks
            score = self._compute_single_judge_multi_task_similarity(
                consolidated_judge_df, consolidated_human_df, task_names
            )
            task_display = f"{len(task_names)}_tasks_avg"

        return BaseScoringResult(
            scorer_name=self.scorer_name,
            task_name=task_display,
            judge_id=judge_id,
            score=score,
            metadata={
                "ground_truth_method": "average_similarity",
                "task_names": task_names,
                "task_schemas": task_schemas,
                "scoring_method": (
                    "single_task" if len(task_names) == 1 else "average_across_tasks"
                ),
            },
        )

    def _compute_single_judge_task_similarity(
        self,
        consolidated_judge_df: pl.DataFrame,
        consolidated_human_df: pl.DataFrame,
        task_name: str,
    ) -> float:
        """Compute text similarity for a single judge on a single task.

        Returns:
            float: The text similarity score.
        """
        # Filter judge data
        task_judge_df = (
            consolidated_judge_df.filter(pl.col("task_name") == task_name)
            .filter(pl.col("task_value").is_not_null())
            .with_columns(pl.col("task_value").cast(pl.Utf8).alias("judge_text"))
        )

        # Filter human data
        task_human_df = (
            consolidated_human_df.filter(pl.col("task_name") == task_name)
            .filter(pl.col("task_value").is_not_null())
            .with_columns(pl.col("task_value").cast(pl.Utf8).alias("human_text"))
        )

        # Check if there are any valid predictions
        if task_judge_df.is_empty() or task_human_df.is_empty():
            return np.nan

        # Group human texts by original_id for each judge prediction
        judge_similarities = []

        for judge_row in task_judge_df.iter_rows(named=True):
            original_id = judge_row["original_id"]
            judge_text = judge_row["judge_text"].lower().strip()

            # Get all human texts for this original_id
            human_texts = task_human_df.filter(pl.col("original_id") == original_id)[
                "human_text"
            ].to_list()

            if not human_texts:
                continue

            # Compute average similarity with each human text
            text_similarities = []
            for human_text in human_texts:
                human_text_clean = human_text.lower().strip()
                similarity = self._compute_text_similarity(judge_text, human_text_clean)
                text_similarities.append(similarity)
            avg_similarity = float(np.mean(text_similarities))
            judge_similarities.append(avg_similarity)

        # Return average across all judge predictions
        if judge_similarities:
            return float(np.mean(judge_similarities))
        else:
            return np.nan

    def _compute_single_judge_multi_task_similarity(
        self,
        consolidated_judge_df: pl.DataFrame,
        consolidated_human_df: pl.DataFrame,
        task_names: List[str],
    ) -> float:
        """Compute average text similarity across multiple tasks for a single judge.

        Returns:
            float: The average text similarity score.
        """
        task_similarities = []
        for task_name in task_names:
            task_similarity = self._compute_single_judge_task_similarity(
                consolidated_judge_df, consolidated_human_df, task_name
            )
            task_similarities.append(task_similarity)

        if task_similarities:
            return float(np.mean(task_similarities))
        else:
            return np.nan

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two text strings using SequenceMatcher.

        Returns:
            float: The similarity score.
        """
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        # Use SequenceMatcher for similarity
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()

    def aggregate_results(
        self, results: List[BaseScoringResult], scores_dir: str
    ) -> None:
        """Generate aggregate plots and save individual results for text similarity scorer.

        Args:
            results: List of text similarity scoring results
            scores_dir: Directory to save results and plots
        """
        if not results:
            print("No text similarity results to aggregate")
            return

        # Create text_similarity directory for results and plots
        text_similarity_dir = os.path.join(scores_dir, "text_similarity")
        os.makedirs(text_similarity_dir, exist_ok=True)

        # Save individual results
        self.save_results(results, text_similarity_dir)

        print(
            f"Generated text similarity results for {len(results)} judge(s) in {text_similarity_dir}"
        )
