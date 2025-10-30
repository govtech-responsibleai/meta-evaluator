"""Classification metrics scorer for classification tasks."""

import os
from typing import List, Literal

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from meta_evaluator.scores.base_scorer import BaseScorer
from meta_evaluator.scores.base_scoring_result import BaseScoringResult
from meta_evaluator.scores.utils import generate_simple_bar_plot


class ClassificationScorer(BaseScorer):
    """Scorer for classification tasks with selectable metrics.

    Supports accuracy, F1, precision, and recall metrics. User selects which
    metric to compute at initialization.
    """

    def __init__(
        self,
        metric: Literal["accuracy", "f1", "precision", "recall"] = "accuracy",
        pos_label: int | float | bool | str = 1,
        average: Literal["binary", "macro", "micro", "samples", "weighted"]
        | None = "binary",
    ):
        """Initialize classification scorer.

        Args:
            metric: Which classification metric to compute:
                - 'accuracy': Overall accuracy
                - 'f1': F1 score (harmonic mean of precision and recall)
                - 'precision': Precision score
                - 'recall': Recall score
            pos_label: The label to consider as positive class for binary classification. See sklearn.metrics documentation.
            average: The averaging strategy for multi-class classification. See sklearn.metrics documentation.
        """
        super().__init__(f"classification_{metric}")
        self.metric = metric
        self.pos_label = pos_label
        self.average = average

    @property
    def min_human_annotators(self) -> int:
        """Minimum number of human annotators required for classification scoring.

        Returns:
            int: 1 human annotator minimum
        """
        return 1

    def can_score_task(
        self, sample_label: str | int | float | List[str | int | float]
    ) -> bool:
        """Classification scorer works with categorical data (int, str, or list of int/str).

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

    def _compute_metric(self, human_labels: list, judge_labels: list) -> float:
        """Compute the selected metric for given labels.

        Args:
            human_labels: Ground truth labels
            judge_labels: Predicted labels from judge

        Returns:
            float: The computed metric value

        Raises:
            ValueError: If the selected metric is unknown
        """
        if self.metric == "accuracy":
            return float(accuracy_score(human_labels, judge_labels))

        # Compute the requested metric
        if self.metric == "f1":
            metric_fn = f1_score
        elif self.metric == "precision":
            metric_fn = precision_score
        elif self.metric == "recall":
            metric_fn = recall_score
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return float(
            metric_fn(
                human_labels,
                judge_labels,
                pos_label=self.pos_label,  # type: ignore
                average=self.average,  # type: ignore
            )
        )

    async def compute_score_async(
        self,
        judge_data: pl.DataFrame,
        human_data: pl.DataFrame,
        task_name: str,
        judge_id: str,
        aggregation_mode,
        annotator_aggregation: str = "individual_average",
    ) -> BaseScoringResult:
        """Compute classification metric for a single judge vs many humans (async).

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
            metric_value = float("nan")
            num_comparisons = 0
            failed_comparisons = 1
        else:
            if annotator_aggregation == "majority_vote":
                # Majority vote approach: aggregate humans first, then compare with judge
                from collections import Counter

                judge_labels_list = []
                human_labels_list = []

                for original_id in comparison_df["original_id"].unique():
                    sample_data = comparison_df.filter(
                        pl.col("original_id") == original_id
                    )
                    judge_label = sample_data["label"].to_list()[0]
                    human_labels = sample_data["label_right"].to_list()

                    if isinstance(human_labels[0], list):
                        # Multilabel: per-position majority (same length assumed)
                        consensus = []
                        for pos in range(len(human_labels[0])):
                            position_votes = [labels[pos] for labels in human_labels]
                            majority = max(
                                Counter(position_votes).items(),
                                key=lambda x: (x[1], x[0]),
                            )[0]
                            consensus.append(majority)
                        majority_label = consensus
                    else:
                        # Single label: simple majority
                        majority_label = max(
                            Counter(human_labels).items(), key=lambda x: (x[1], x[0])
                        )[0]

                    judge_labels_list.append(judge_label)
                    human_labels_list.append(majority_label)

                # Compute the selected metric
                try:
                    metric_value = self._compute_metric(
                        human_labels_list, judge_labels_list
                    )
                except (ValueError, ZeroDivisionError) as e:
                    self.logger.warning(
                        f"Error computing {self.metric} for judge {judge_id}: {e}"
                    )
                    metric_value = float("nan")

            else:
                # Individual average approach
                human_metric_values = []
                humans = comparison_df["human_id"].unique()

                for human_id in humans:
                    human_comparisons = comparison_df.filter(
                        pl.col("human_id") == human_id
                    )
                    judge_labels = human_comparisons["label"].to_list()
                    human_labels = human_comparisons["label_right"].to_list()

                    # Compute metric for this judge-human pair
                    try:
                        human_metric = self._compute_metric(human_labels, judge_labels)
                        human_metric_values.append(human_metric)
                    except (ValueError, ZeroDivisionError) as e:
                        self.logger.warning(
                            f"Error computing {self.metric} for judge {judge_id} vs human {human_id}: {e}"
                        )
                        continue

                # Average across all humans
                metric_value = (
                    float(np.mean(human_metric_values))
                    if human_metric_values
                    else float("nan")
                )

            num_comparisons = len(comparison_df)
            failed_comparisons = 0

        # Build metadata
        metadata = {
            "ground_truth_method": "individual_human_comparison",
            "scoring_method": (
                "majority_vote"
                if annotator_aggregation == "majority_vote"
                else "average_across_humans"
            ),
        }

        # Add metric-specific metadata
        if self.metric != "accuracy":
            metadata["pos_label"] = str(self.pos_label)
            metadata["average"] = str(self.average)

        return BaseScoringResult(
            scorer_name=self.scorer_name,
            task_name=task_name,
            judge_id=judge_id,
            scores={self.metric: metric_value},
            metadata=metadata,
            aggregation_mode=aggregation_mode,
            num_comparisons=num_comparisons,
            failed_comparisons=failed_comparisons,
        )

    def aggregate_results(
        self, results: List[BaseScoringResult], scores_dir: str, unique_name: str = ""
    ) -> None:
        """Generate aggregate plot and save individual results for classification scorer.

        Args:
            results: List of classification scoring results
            scores_dir: Directory to save results and plots
            unique_name: Unique identifier for this metric configuration
        """
        if not results:
            self.logger.info(f"No {self.metric} results to aggregate")
            return

        # Create directory for results and plots
        metric_dir = os.path.join(scores_dir, self.scorer_name, unique_name)
        os.makedirs(metric_dir, exist_ok=True)

        # Generate plot for the selected metric only
        generate_simple_bar_plot(
            results=results,
            score_key=self.metric,
            output_dir=metric_dir,
            plot_filename=f"{self.metric}_scores.png",
            title=f"{self.metric.replace('_', ' ').title()} Scores by Judge",
            ylabel=f"{self.metric.replace('_', ' ').title()} Score",
            unique_name=unique_name,
            logger=self.logger,
        )

        self.logger.info(
            f"Generated {self.metric} results for {len(results)} judge(s) in {metric_dir}"
        )
