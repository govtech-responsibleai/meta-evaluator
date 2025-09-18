"""Cohen's kappa scorer for inter-annotator agreement."""

import os
from typing import List

import numpy as np
import polars as pl
from sklearn.metrics import cohen_kappa_score

from meta_evaluator.scores.base_scorer import BaseScorer
from meta_evaluator.scores.base_scoring_result import BaseScoringResult
from meta_evaluator.scores.utils import generate_simple_bar_plot


class CohensKappaScorer(BaseScorer):
    """Scorer for inter-annotator agreement using Cohen's kappa."""

    def __init__(self):
        """Initialize Cohen's kappa scorer."""
        super().__init__("cohens_kappa")

    @property
    def min_human_annotators(self) -> int:
        """Minimum number of human annotators required for Cohen's kappa.

        Returns:
            int: 2 human annotators minimum for inter-annotator agreement
        """
        return 2

    def can_score_task(
        self, sample_label: str | int | float | List[str | int | float]
    ) -> bool:
        """Cohen's kappa works with categorical data (int, str, or list of int/str).

        Args:
            sample_label: Sample of the actual label data to validate

        Returns:
            bool: True if data contains categorical values (int/str or lists of int/str)
        """
        # Accept int, str, or list of int/str (same as AccuracyScorer)
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
        annotator_aggregation: str = "individual_average",
    ) -> BaseScoringResult:
        """Compute Cohen's kappa score for a single judge vs many humans (async).

        Args:
            judge_data: DataFrame with judge outcomes (columns: original_id, label)
            human_data: DataFrame with human outcomes (columns: original_id, human_id, label)
            task_name: Name of the task(s) being scored
            judge_id: ID of the judge being scored
            aggregation_mode: How the tasks were aggregated for this result
            annotator_aggregation: How to aggregate multiple human annotators

        Returns:
            BaseScoringResult: The scoring result for this judge
        """
        # Join judge and human data on original_id
        comparison_df = judge_data.join(human_data, on="original_id", how="inner")

        if comparison_df.is_empty():
            kappa_score = float("nan")
            num_comparisons = 0
            failed_comparisons = 1
        else:
            # For each human, compute Cohen's kappa between judge and that human
            human_kappas = []
            humans = comparison_df["human_id"].unique()

            for human_id in humans:
                human_comparisons = comparison_df.filter(pl.col("human_id") == human_id)
                judge_labels = human_comparisons["label"].to_list()
                human_labels = human_comparisons["label_right"].to_list()  # From join

                # Filter out None values while maintaining alignment
                valid_pairs = [
                    (j, h)
                    for j, h in zip(judge_labels, human_labels)
                    if j is not None and h is not None
                ]

                if len(valid_pairs) >= 2:  # Cohen's kappa needs at least 2 samples
                    j_labels, h_labels = zip(*valid_pairs)
                    try:
                        kappa = cohen_kappa_score(h_labels, j_labels)
                        human_kappas.append(kappa)
                    except ValueError:
                        # Handle cases where labels are all the same (perfect agreement but undefined kappa)
                        continue

            # Average across all humans
            kappa_score = float(np.mean(human_kappas)) if human_kappas else float("nan")
            num_comparisons = len(comparison_df)
            failed_comparisons = 0

        return BaseScoringResult(
            scorer_name=self.scorer_name,
            task_name=task_name,
            judge_id=judge_id,
            scores={"kappa": kappa_score},
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
        """Generate aggregate plots and save individual results for Cohen's kappa scorer.

        Args:
            results: List of Cohen's kappa scoring results
            scores_dir: Directory to save results and plots
            unique_name: Optional unique identifier for this metric configuration
        """
        if not results:
            self.logger.info("No Cohen's kappa results to aggregate")
            return

        # Create cohens_kappa directory for results and plots, using unique_name to avoid overwrites
        cohens_kappa_dir = os.path.join(scores_dir, self.scorer_name, unique_name)
        os.makedirs(cohens_kappa_dir, exist_ok=True)

        # Generate simple bar plot (placeholder for other plots)
        generate_simple_bar_plot(
            results=results,
            score_key="kappa",
            output_dir=cohens_kappa_dir,
            plot_filename="cohens_kappa_scores.png",
            title="Cohen's Kappa Scores by Judge",
            ylabel="Cohen's Kappa Score",
            unique_name=unique_name,
            logger=self.logger,
        )

        self.logger.info(
            f"Generated Cohen's kappa results for {len(results)} judge(s) in {cohens_kappa_dir}"
        )
