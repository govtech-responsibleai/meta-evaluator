"""Accuracy scorer for classification tasks."""

import os
from typing import List

import numpy as np
import polars as pl

from meta_evaluator.scores.base_scorer import BaseScorer
from meta_evaluator.scores.base_scoring_result import BaseScoringResult
from meta_evaluator.scores.utils import generate_simple_bar_plot


class AccuracyScorer(BaseScorer):
    """Scorer for classification tasks using accuracy metric with majority vote."""

    def __init__(self):
        """Initialize accuracy scorer."""
        super().__init__("accuracy")

    def can_score_task(
        self, sample_label: str | int | float | List[str | int | float]
    ) -> bool:
        """Accuracy scorer works with categorical data (int, str, or list of int/str).

        Args:
            sample_label: Sample of the actual label data to validate

        Returns:
            bool: True if data contains categorical values (int/str or lists of int/str)
        """
        # Accept int, str, or list of int/str
        if isinstance(sample_label, (int, str)):
            return True
        elif isinstance(sample_label, list):
            # Check if list contains int/str
            if len(sample_label) > 0:
                return isinstance(sample_label[0], (int, str))
            return True  # Empty list is acceptable
        else:
            return False

    async def compute_score_async(
        self,
        judge_data: pl.DataFrame,
        human_data: pl.DataFrame,
        task_name: str,
        judge_id: str,
        aggregation_mode,
    ) -> BaseScoringResult:
        """Compute accuracy score for a single judge vs many humans (async).

        Args:
            judge_data: DataFrame with judge outcomes (columns: original_id, label)
            human_data: DataFrame with human outcomes (columns: original_id, human_id, label)
            task_name: Name of the task(s) being scored
            judge_id: ID of the judge being scored
            aggregation_mode: How the tasks were aggregated for this result

        Returns:
            BaseScoringResult: The scoring result for this judge
        """
        # Join judge and human data on original_id (approach 1: join first)
        comparison_df = judge_data.join(human_data, on="original_id", how="inner")

        if comparison_df.is_empty():
            accuracy_score = float("nan")
            num_comparisons = 0
            failed_comparisons = 1
        else:
            # For each human, compute accuracy between judge and that human
            human_accuracies = []
            humans = comparison_df["human_id"].unique()

            for human_id in humans:
                human_comparisons = comparison_df.filter(pl.col("human_id") == human_id)
                judge_labels = human_comparisons["label"].to_list()
                human_labels = human_comparisons["label_right"].to_list()  # From join

                # Compute accuracy for this judge-human pair
                correct = sum(j == h for j, h in zip(judge_labels, human_labels))
                total = len(judge_labels)
                if total > 0:
                    human_accuracies.append(correct / total)

            # Average across all humans
            accuracy_score = (
                float(np.mean(human_accuracies)) if human_accuracies else float("nan")
            )
            num_comparisons = len(comparison_df)
            failed_comparisons = 0

        return BaseScoringResult(
            scorer_name=self.scorer_name,
            task_name=task_name,
            judge_id=judge_id,
            scores={"accuracy": accuracy_score},
            metadata={
                "ground_truth_method": "individual_human_comparison",
                "scoring_method": "average_across_humans",
            },
            aggregation_mode=aggregation_mode,
            num_comparisons=num_comparisons,
            failed_comparisons=failed_comparisons,
        )

    def aggregate_results(
        self, results: List[BaseScoringResult], scores_dir: str, unique_name: str = ""
    ) -> None:
        """Generate aggregate plots and save individual results for accuracy scorer.

        Args:
            results: List of accuracy scoring results
            scores_dir: Directory to save results and plots
            unique_name: Unique identifier for this metric configuration
        """
        if not results:
            self.logger.info("No accuracy results to aggregate")
            return

        # Create accuracy directory for results and plots, using unique_name to avoid overwrites
        accuracy_dir = os.path.join(scores_dir, self.scorer_name, unique_name)
        os.makedirs(accuracy_dir, exist_ok=True)

        # Generate simple bar plot (placeholder for other plots)
        generate_simple_bar_plot(
            results=results,
            score_key="accuracy",
            output_dir=accuracy_dir,
            plot_filename="accuracy_scores.png",
            title="Accuracy Scores by Judge",
            ylabel="Accuracy Score",
            unique_name=unique_name,
            logger=self.logger,
        )

        self.logger.info(
            f"Generated accuracy results for {len(results)} judge(s) in {accuracy_dir}"
        )
