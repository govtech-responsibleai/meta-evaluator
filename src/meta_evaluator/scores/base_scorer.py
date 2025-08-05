"""Base classes for scoring functionality."""

import os
from abc import ABC, abstractmethod
from typing import List, Optional

import polars as pl

from .base_scoring_result import BaseScoringResult


class BaseScorer(ABC):
    """Base class for all scorers."""

    def __init__(self, scorer_name: str):
        """Initialize the scorer.

        Args:
            scorer_name: Name identifier for this scorer
        """
        self.scorer_name = scorer_name

    @abstractmethod
    def can_score_task(self, task_schema: Optional[List[str]]) -> bool:
        """Determine if this scorer is compatible with the given task type.

        Task compatibility is determined by the task schema:
        - task_schema=None: Free-form text task (no predefined categories)
        - task_schema=[...]: Classification task with predefined categories

        Args:
            task_schema: List of allowed categorical outcomes, or None for free-form text tasks

        Returns:
            bool: True if this scorer can compute meaningful metrics for this task type
        """
        pass

    @abstractmethod
    def compute_score(
        self,
        judge_id: str,
        consolidated_judge_df: pl.DataFrame,
        consolidated_human_df: pl.DataFrame,
        task_names: List[str],
        task_schemas: dict,
    ) -> BaseScoringResult:
        """Compute the score for a single judge vs many humans.

        Args:
            judge_id: ID of the judge being scored
            consolidated_judge_df: DataFrame with single judge outcomes (columns: original_id, judge_id, task_name, task_value)
            consolidated_human_df: DataFrame with human outcomes (columns: original_id, annotator_id, task_name, task_value)
            task_names: List of task names to score
            task_schemas: Dictionary mapping task names to their schemas

        Returns:
            BaseScoringResult: The scoring result for this judge
        """
        pass

    def aggregate_results(
        self, results: List[BaseScoringResult], scores_dir: str
    ) -> None:
        """Generate aggregate plots from scoring results.

        Args:
            results: List of scoring results
            scores_dir: Directory to save aggregate plots
        """
        pass

    def save_results(self, results: List[BaseScoringResult], output_dir: str) -> None:
        """Save individual ScoringResult objects as JSON files.

        This method provides common functionality for saving scoring results
        to disk. Each result is saved as a separate JSON file with the naming
        pattern: {judge_id}_{task_name}_result.json

        Args:
            results: List of scoring results to save
            output_dir: Directory where the result files will be saved
        """
        for result in results:
            # Create filename: judge_id_task_name_result.json
            filename = f"{result.judge_id}_{result.task_name}_result.json"
            file_path = os.path.join(output_dir, filename)

            result.save_state(file_path)
            print(f"Saved individual result to {file_path}")
