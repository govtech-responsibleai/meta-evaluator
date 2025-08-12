"""Base classes for scoring functionality."""

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import polars as pl

from .base_scoring_result import BaseScoringResult


class BaseScorer(ABC):
    """Abstract base class defining the settings and interface for all scoring implementations.

    This class establishes the framework for computing alignment scores between judge
    evaluations and human annotations. Scorers implement different metrics (accuracy,
    Cohen's kappa, etc.) and must implement methods to:
    1. Determine compatibility with task schemas via `can_score_task()`
    2. Compute alignment scores between judge and human results via `compute_score()`
    3. Optionally override post-processing methods for visualization and aggregation

    Attributes:
        scorer_name (str): Name identifier for this scorer instance.
        logger (logging.Logger): Logger instance for this scorer.

    Methods:
        can_score_task: Determine if scorer is compatible with given task schema.
        compute_score: Compute alignment scores for judge vs human evaluations.
        aggregate_results: Post-processing method to generate aggregate plots and
            visualizations from multiple scoring results. Default implementation
            does nothing - override to provide custom visualization.

    Examples:
        >>> class AccuracyScorer(BaseScorer):
        ...     def can_score_task(self, task_schema):
        ...         return task_schema is not None  # Only classification tasks
        ...     def compute_score(self, judge_id, judge_df, human_df, task_names, task_schemas):
        ...         # Core scoring logic
        ...         return AccuracyScoringResult(...)
        ...     def aggregate_results(self, results, scores_dir):
        ...         # Optional: generate accuracy plots
        ...         self._create_accuracy_plots(results, scores_dir)
    """

    def __init__(self, scorer_name: str):
        """Initialize the scorer.

        Args:
            scorer_name: Name identifier for this scorer
        """
        self.scorer_name = scorer_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

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
            self.logger.info(f"Saved individual result to {file_path}")
