"""Base classes for scoring functionality."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List

import polars as pl

from .base_scoring_result import BaseScoringResult
from .enums import TaskAggregationMode


class BaseScorer(ABC):
    """Abstract base class defining the settings and interface for all scoring implementations.

    This class establishes the framework for computing alignment scores between judge
    evaluations and human annotations. Scorers implement different metrics (accuracy,
    Cohen's kappa, etc.) and must implement methods to:
    1. Determine compatibility with task schemas via `can_score_task()`
    2. Compute alignment scores between judge and human results via `compute_score_async()`
    3. Define minimum human annotator requirements via `min_human_annotators`
    4. Optionally override post-processing methods for visualization and aggregation

    Attributes:
        scorer_name (str): Name identifier for this scorer instance.
        logger (logging.Logger): Logger instance for this scorer.

    Properties:
        min_human_annotators: Minimum number of human annotators required for this scorer.

    Methods:
        can_score_task: Determine if scorer is compatible with given task schema.
        compute_score: Compute alignment scores for judge vs human evaluations.
        aggregate_results: Post-processing method to generate aggregate plots and
            visualizations from multiple scoring results. Default implementation
            does nothing - override to provide custom visualization.

    Examples:
        >>> class AccuracyScorer(BaseScorer):
        ...     @property
        ...     def min_human_annotators(self) -> int:
        ...         return 1  # Accuracy works with just 1 human
        ...     def can_score_task(self, task_schema):
        ...         return task_schema is not None  # Only classification tasks
        ...     def compute_score_async(self, judge_data, human_data, human_df, task_name, judge_id, aggregation_mode):
        ...         # Core scoring logic
        ...         return AccuracyScoringResult(...) # TODO: Implement ScoringResult for each Scorer. Currently using BaseScoringResult.
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

    @property
    @abstractmethod
    def min_human_annotators(self) -> int:
        """Minimum number of human annotators required for this scorer.

        Returns:
            int: Minimum number of human annotators needed for meaningful results
        """
        pass

    @abstractmethod
    def can_score_task(self, sample_label: Any) -> bool:
        """Determine if this scorer is compatible with the given data format.

        Args:
            sample_label: Sample of the actual data that will be scored.
                        Can be a single value, list, or any data type.

        Returns:
            bool: True if this scorer can compute meaningful metrics for this data type
        """
        pass

    @abstractmethod
    async def compute_score_async(
        self,
        judge_data: pl.DataFrame,
        human_data: pl.DataFrame,
        task_name: str,
        judge_id: str,
        aggregation_mode: TaskAggregationMode,
    ) -> BaseScoringResult:
        """Compute the score for a single judge vs many humans (async).

        Args:
            judge_data: DataFrame with judge outcomes (columns: original_id, label)
            human_data: DataFrame with human outcomes (columns: original_id, human_id, label)
            task_name: Name of the task(s) being scored
            judge_id: ID of the judge being scored
            aggregation_mode: How the tasks were aggregated for this result

        Returns:
            BaseScoringResult: The scoring result for this judge
        """
        pass

    def aggregate_results(
        self,
        results: List[BaseScoringResult],
        scores_dir: str,
        unique_name: str = "",
    ) -> None:
        """Generate aggregate plots from scoring results.

        Args:
            results: List of scoring results
            scores_dir: Directory to save aggregate plots
            unique_name: Optional unique identifier for this metric configuration
        """
        pass

    def save_results(
        self,
        results: List[BaseScoringResult],
        scores_dir: str,
        unique_name: str = "",
    ) -> None:
        """Save individual ScoringResult objects as JSON files.

        This method provides common functionality for saving scoring results
        to disk. Each result is saved as a separate JSON file with the naming
        pattern: {unique_name}_{judge_id}_result.json or {judge_id}_{task_name}_result.json

        Args:
            results: List of scoring results to save
            scores_dir: Directory where the scoring result files will be saved
            unique_name: Unique identifier for this metric configuration
        """
        # Create scorer-specific subdirectory
        alt_test_dir = os.path.join(scores_dir, self.scorer_name, unique_name)
        os.makedirs(alt_test_dir, exist_ok=True)

        for result in results:
            # Create filename with unique name: unique_name_judge_id_result.json
            filename = f"{result.judge_id}_result.json"
            file_path = os.path.join(alt_test_dir, filename)

            result.save_state(file_path)
            self.logger.info(f"Saved individual result to {file_path}")
