"""Alt-Test scorer for evaluating LLM judges against human annotators.

Based on: https://arxiv.org/abs/2501.10970
Code adapted from: https://github.com/nitaytech/AltTest/
"""

import asyncio
import os
from typing import Any, Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import ttest_1samp
from sklearn.metrics import jaccard_score

from meta_evaluator.scores.base_scorer import BaseScorer
from meta_evaluator.scores.base_scoring_result import BaseScoringResult
from meta_evaluator.scores.exceptions import (
    AltTestInsufficientAnnotationsError,
    AltTestInvalidScoringFunctionError,
)
from meta_evaluator.scores.utils import save_plot


class AltTestScorer(BaseScorer):
    """Alt-Test scorer for evaluating LLM judges vs human annotators."""

    def __init__(
        self,
        epsilon: float = 0.2,
        multiplicative_epsilon: bool = False,
        q_fdr: float = 0.05,
        min_instances_per_human: int = 30,
    ):
        """Initialize Alt-Test scorer.

        Args:
            epsilon: Threshold parameter for alt-test
            multiplicative_epsilon: Whether to use multiplicative epsilon
            q_fdr: False discovery rate for multiple testing correction
            min_instances_per_human: Minimum instances required per human
        """
        super().__init__("alt_test")
        self.epsilon = epsilon
        self.multiplicative_epsilon = multiplicative_epsilon
        self.q_fdr = q_fdr
        self.min_instances_per_human = min_instances_per_human

        # Set up matplotlib config
        plt.rcParams["font.size"] = 12
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Verdana"]

    @property
    def min_human_annotators(self) -> int:
        """Minimum number of human annotators required for Alt-Test.

        Returns:
            int: 3 human annotators minimum for statistical significance
        """
        return 3

    def can_score_task(
        self, sample_label: str | int | float | List[str | int | float]
    ) -> bool:
        """Alt-Test works with both categorical and text data.

        Args:
            sample_label: Sample of the actual label data to validate

        Returns:
            bool: True if data contains categorical values (int/str/list) or text (str)
        """
        # Accept int, str, list of int/str (classification), or str (text)
        if isinstance(sample_label, (int, str, float)):
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
        """Compute Alt-Test score for a single judge vs many humans (async).

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
        # Check annotator aggregation strategy and warn if majority_vote is used
        if annotator_aggregation == "majority_vote":
            self.logger.warning(
                "AltTestScorer does not support majority_vote aggregation. Using individual_average instead."
            )

        # Join judge and human data on original_id
        comparison_df = judge_data.join(human_data, on="original_id", how="inner")

        # Set defaults
        num_comparisons = 0
        failed_comparisons = 1
        epsilons = np.arange(0.0, 0.31, 0.05)
        winning_rates = {
            f"{eps:.2f}": float("nan") for eps in np.arange(0.0, 0.31, 0.05)
        }
        advantage_prob = float("nan")
        human_advantage_probs = {}
        scoring_function_name = "accuracy"

        if not comparison_df.is_empty():
            # Automatically determine scoring function based on data
            first_judge_label = judge_data["label"].drop_nulls().first()
            scoring_function_name = self._determine_scoring_function(first_judge_label)

            # Run alt-test with multiple epsilon values in parallel
            epsilons = np.arange(0.0, 0.31, 0.05)
            winning_rates = {}

            try:
                # Create tasks for all epsilon calculations
                tasks = [
                    self.alt_test(
                        judge_data,
                        human_data,
                        scoring_function_name,
                        float(epsilon),
                    )
                    for epsilon in epsilons
                ]

                # Run all epsilon calculations in parallel
                results = await asyncio.gather(*tasks)

                # Process results
                for epsilon, (
                    winning_rate,
                    ep_advantage_prob,
                    ep_human_advantage_probs,
                ) in zip(epsilons, results):
                    winning_rates[f"{epsilon:.2f}"] = winning_rate

                    # Store advantage prob and human advantage probs from default epsilon
                    if abs(epsilon - self.epsilon) < 0.01:
                        advantage_prob = ep_advantage_prob
                        human_advantage_probs = ep_human_advantage_probs

                num_comparisons = len(comparison_df)
                failed_comparisons = 0

            except Exception:
                # Handle alt-test failures
                winning_rates = {f"{eps:.2f}": float("nan") for eps in epsilons}
                advantage_prob = float("nan")
                human_advantage_probs = {}
                num_comparisons = len(comparison_df)
                failed_comparisons = 1

        return BaseScoringResult(
            scorer_name=self.scorer_name,
            task_name=task_name,
            judge_id=judge_id,
            scores={
                "winning_rate": winning_rates,
                "advantage_probability": advantage_prob,
            },
            metadata={
                "human_advantage_probabilities": human_advantage_probs,
                "scoring_function": scoring_function_name,
                "epsilon": self.epsilon,
                "multiplicative_epsilon": self.multiplicative_epsilon,
                "q_fdr": self.q_fdr,
                "min_human_annotators": self.min_human_annotators,
                "min_instances_per_human": self.min_instances_per_human,
                "ground_truth_method": "alt_test_procedure",
                "scoring_method": "leave_one_out_cross_validation",
            },
            aggregation_mode=aggregation_mode,
            num_comparisons=num_comparisons,
            failed_comparisons=failed_comparisons,
        )

    ###########################
    # Alt-test core functions #
    ###########################
    def _determine_scoring_function(self, sample_label) -> str:
        """Automatically determine the best scoring function based on sample label.

        Returns:
            str: The name of the best scoring function.

        Raises:
            ValueError: If the sample_label type is not supported.
        """
        # Accept int, str, list of int/str (classification), or str (text)
        if isinstance(sample_label, str):
            return "accuracy"
        elif isinstance(sample_label, (int, float)):
            return "neg_rmse"
        elif isinstance(sample_label, list):
            return "jaccard_similarity"
        else:
            raise ValueError(f"Unknown sample label type: {type(sample_label)}.")

    def _accuracy(self, pred: Any, annotations: List[Any]) -> float:
        """Accuracy scoring function using exact match.

        Returns:
            float: The accuracy score.
        """
        return float(np.mean([pred == ann for ann in annotations]))

    def _neg_rmse(
        self, pred: Union[int, float], annotations: List[Union[int, float]]
    ) -> float:
        """Negative RMSE scoring function.

        Returns:
            float: The negative RMSE score.
        """
        return -1 * float(np.sqrt(np.mean([(pred - ann) ** 2 for ann in annotations])))

    def _jaccard_similarity(
        self, pred: List[str], annotations: List[List[str]]
    ) -> float:
        """Macro-averaged jaccard similarity.

        Returns:
            float: The macro-averaged jaccard similarity score.
        """
        jaccard_scores = []
        for ann in annotations:
            jaccard_scores.append(
                jaccard_score(y_true=ann, y_pred=pred, average="macro")
            )
        return float(np.mean(jaccard_scores))

    def _get_scoring_function(self, scoring_function_name: str) -> Callable:
        """Get the scoring function based on string name.

        Returns:
            Callable: The scoring function.

        Raises:
            AltTestInvalidScoringFunctionError: If the scoring function is unknown.
        """
        if scoring_function_name == "accuracy":
            return self._accuracy
        elif scoring_function_name == "neg_rmse":
            return self._neg_rmse
        elif scoring_function_name == "jaccard_similarity":
            return self._jaccard_similarity
        else:
            raise AltTestInvalidScoringFunctionError(
                f"Unknown scoring function: {scoring_function_name}"
            )

    def _by_procedure(self, p_values: List[float], q: float) -> List[Any]:
        """Benjamini-Yekutieli procedure for multiple testing correction.

        Returns:
            List[Any]: A list of indices of rejected hypotheses.
        """
        p_values_array = np.array(p_values, dtype=float)
        m = len(p_values_array)
        sorted_indices = np.argsort(p_values_array)
        sorted_pvals = p_values_array[sorted_indices]
        # Compute the harmonic sum H_m = 1 + 1/2 + ... + 1/m
        H_m = np.sum(1.0 / np.arange(1, m + 1))
        # Compute the BY thresholds for each rank i
        by_thresholds = (np.arange(1, m + 1) / m) * (q / H_m)
        max_i = -1
        for i in range(m):
            if sorted_pvals[i] <= by_thresholds[i]:
                max_i = i
        if max_i == -1:
            return []
        rejected_sorted_indices = sorted_indices[: max_i + 1]
        return list(rejected_sorted_indices)

    def _ttest(self, indicators, epsilon: float) -> float:
        """One-sample t-test.

        Returns:
            float: The p-value of the t-test.
        """
        return ttest_1samp(indicators, epsilon, alternative="less").pvalue  # type: ignore

    def _alt_test_core(
        self,
        llm_annotations: Dict[Union[int, str], Any],
        humans_annotations: Dict[Union[int, str], Dict[Union[int, str], Any]],
        scoring_function_name: str,
        epsilon: float = 0.0,
    ) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
        """Run the Alt-Test procedure.

        Returns:
            Tuple[float, float, Dict[str, Tuple[float, float]]]: A tuple containing the winning rate, advantage probability, and human advantage probabilities.

        Raises:
            AltTestInsufficientAnnotationsError: If no annotators meet the minimum threshold of instances per human.
        """
        # Get scoring function from name
        scoring_function = self._get_scoring_function(scoring_function_name)

        # prepare sets - i_set has humans as keys, h_set has instances as keys
        i_set, h_set = {}, {}
        for h, anns in sorted(humans_annotations.items()):
            i_set[h] = sorted(list(anns.keys()))
            for i, ann in sorted(anns.items()):
                if i not in h_set:
                    h_set[i] = []
                h_set[i].append(h)

        # remove instances with less than min_human_annotators
        instances_to_keep = {
            i
            for i in sorted(h_set.keys())
            if len(h_set[i]) >= self.min_human_annotators and i in llm_annotations
        }
        i_set = {
            h: sorted([i for i in i_set[h] if i in instances_to_keep])
            for h in sorted(i_set.keys())
        }
        h_set = {
            i: sorted(h_set[i]) for i in sorted(h_set.keys()) if i in instances_to_keep
        }

        # Save llm vs human advantage probability
        human_advantage_probs = {}

        p_values, advantage_probs, humans = [], [], []
        for excluded_h in sorted(humans_annotations.keys()):
            llm_indicators = []
            excluded_indicators = []
            instances = sorted([i for i in i_set[excluded_h] if i in llm_annotations])
            if len(instances) < self.min_instances_per_human:
                self.logger.info(
                    f"Skipping annotator {excluded_h} with only {len(instances)} instances < {self.min_instances_per_human}."
                )
                continue

            for i in instances:
                human_ann = humans_annotations[excluded_h][i]
                llm_ann = llm_annotations[i]
                remaining_anns = [
                    humans_annotations[h][i] for h in h_set[i] if h != excluded_h
                ]
                human_score = scoring_function(human_ann, remaining_anns)
                llm_score = scoring_function(llm_ann, remaining_anns)
                llm_indicators.append(1 if llm_score >= human_score else 0)
                excluded_indicators.append(1 if human_score >= llm_score else 0)

            # Calculate p-value based on epsilon type
            if self.multiplicative_epsilon:
                diff_indicators = [
                    exc_ind - (llm_ind / (1 - epsilon))
                    for exc_ind, llm_ind in zip(excluded_indicators, llm_indicators)
                ]
                p = self._ttest(diff_indicators, 0)
            else:
                diff_indicators = [
                    exc_ind - llm_ind
                    for exc_ind, llm_ind in zip(excluded_indicators, llm_indicators)
                ]
                p = self._ttest(diff_indicators, epsilon)

            p_values.append(p)
            advantage_probs.append(float(np.mean(llm_indicators)))
            humans.append(excluded_h)

            # Save human advantage probability
            human_advantage_probs[excluded_h] = (
                float(np.mean(llm_indicators)),
                float(np.mean(excluded_indicators)),
            )

        # Check if we have any valid humans to analyze
        if not humans:
            raise AltTestInsufficientAnnotationsError(
                f"No annotators meet the minimum threshold of {self.min_instances_per_human} instances per human"
            )

        rejected_indices = self._by_procedure(p_values, self.q_fdr)
        advantage_prob = float(np.mean(advantage_probs))
        winning_rate = len(rejected_indices) / len(humans) if humans else np.nan

        return winning_rate, advantage_prob, human_advantage_probs

    async def alt_test(
        self,
        judge_data: pl.DataFrame,
        human_data: pl.DataFrame,
        scoring_function_name: str,
        epsilon: float = 0.0,
    ) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
        """Format the dataframes into annotations to run the Alt-Test procedure.

        Instead of running the Alt-Test procedure directly on DataFrames, we convert
        our DataFrame format to the original alt-test annotation format to minimize
        edits to the core algorithm for maintainability and compatibility with the
        original research implementation.

        Args:
            judge_data: DataFrame with judge outcomes (columns: original_id, label)
            human_data: DataFrame with human outcomes (columns: original_id, human_id, label)
            scoring_function_name: Name of the scoring function to use
            epsilon: Epsilon threshold for the test

        Returns:
            Tuple[float, float, Dict[str, Tuple[float, float]]]: A tuple containing the winning rate, advantage probability, and human advantage probabilities.
        """
        # Convert to alt-test format
        judge_annotations = {}
        human_annotations = {}

        # Build judge annotations: {original_id: label}
        for row in judge_data.iter_rows(named=True):
            if row["label"] is not None:
                judge_annotations[row["original_id"]] = row["label"]

        # Build human annotations: {human_id: {original_id: label}}
        for row in human_data.iter_rows(named=True):
            if row["label"] is not None:
                human_id = row["human_id"]
                if human_id not in human_annotations:
                    human_annotations[human_id] = {}
                human_annotations[human_id][row["original_id"]] = row["label"]

        # Use the existing _alt_test implementation
        return self._alt_test_core(
            judge_annotations, human_annotations, scoring_function_name, epsilon
        )

    ######################
    # Plotting functions #
    ######################
    def _get_plot_colors(self, n: int) -> np.ndarray:
        """Get n distinct colors by concatenating tab20 colormaps.

        Concatenates tab20, tab20b, and tab20c to provide up to 60 distinct colors.
        If more than 60 colors are needed, colors will repeat.

        Args:
            n: Number of colors needed

        Returns:
            np.ndarray: Array of n RGBA color tuples
        """
        # Concatenate tab20, tab20b, tab20c (60 colors total)
        tab20 = plt.cm.tab20(np.linspace(0, 1, 20))  # type: ignore
        tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))  # type: ignore
        tab20c = plt.cm.tab20c(np.linspace(0, 1, 20))  # type: ignore
        all_colors = np.vstack([tab20, tab20b, tab20c])

        # Return first n colors (will repeat if n > 60)
        return all_colors[:n]

    def generate_aggregate_winning_rates_plot(
        self,
        results: List[BaseScoringResult],
        alt_test_dir: str,
        scoring_function: str,
        unique_name: str = "",
    ) -> None:
        """Generate aggregate winning rates plot as line chart across epsilon values."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Generate colors for each judge
        judge_ids = [result.judge_id for result in results]
        colors = self._get_plot_colors(len(judge_ids))

        # Extract epsilon values from the first result's winning_rate dict keys
        winning_rates_dict = results[0].scores["winning_rate"]
        epsilons = [float(eps) for eps in sorted(winning_rates_dict.keys())]

        # Highlight pass/fail regions
        ax.axhline(
            y=0.5, color="gray", linestyle="--", alpha=0.7, label="Pass Threshold"
        )
        ax.fill_between(
            epsilons, 0.5, 1.0, color="lightgreen", alpha=0.15, label="PASS"
        )
        ax.fill_between(epsilons, 0, 0.5, color="mistyrose", alpha=0.15, label="FAIL")

        # Plot each judge's winning rate across epsilon values
        for i, result in enumerate(results):
            if not result.scores:
                self.logger.info(f"Skipping judge {result.judge_id} with no scores")
                continue
            winning_rates_dict = result.scores["winning_rate"]
            winning_rates = [winning_rates_dict[f"{eps:.2f}"] for eps in epsilons]
            advantage_prob = result.scores[
                "advantage_probability"
            ]  # From default epsilon

            ax.plot(
                epsilons,
                winning_rates,
                marker="o",
                color=colors[i],
                label=f"{result.judge_id} (AP={advantage_prob:.2f})",
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Winning Rate")
        ax.set_title(
            f"Alt-Test Winning Rates: {scoring_function.replace('_', ' ').title()} ({unique_name})"
        )
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Save plot
        save_path = os.path.join(alt_test_dir, "aggregate_winning_rates.png")
        save_plot(fig, save_path, self.logger)
        plt.close(fig)

    def generate_aggregate_advantage_probabilities_plot(
        self,
        results: List[BaseScoringResult],
        alt_test_dir: str,
        scoring_function: str,
        unique_name: str = "",
    ) -> None:
        """Generate aggregate advantage probabilities plot for all judges."""
        fig, ax = plt.subplots(figsize=(12, 8))

        judge_ids = []
        advantage_probs = []

        for result in results:
            if not result.scores:
                self.logger.info(f"Skipping judge {result.judge_id} with no scores")
                continue
            judge_ids.append(result.judge_id)
            advantage_probs.append(result.scores["advantage_probability"])

        # Create bar plot
        bars = ax.bar(
            judge_ids,
            advantage_probs,
            color="steelblue",
            edgecolor="black",
            linewidth=1,
        )

        # Add value labels on bars
        for bar, ap in zip(bars, advantage_probs):
            height = bar.get_height()
            ax.annotate(
                f"{ap:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        ax.set_ylabel("Advantage Probability")
        ax.set_title(
            f"LLM Advantage Probabilities: {scoring_function.replace('_', ' ').title()} ({unique_name})"
        )
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save plot
        save_path = os.path.join(alt_test_dir, "aggregate_advantage_probabilities.png")
        save_plot(fig, save_path, self.logger)
        plt.close(fig)

    def generate_aggregate_human_vs_llm_plot(
        self, results: List[BaseScoringResult], alt_test_dir: str, unique_name: str = ""
    ) -> None:
        """Generate human vs LLM advantage probabilities plot - one chart per LLM."""
        # Filter out results with no valid human advantage probabilities
        valid_results = [
            result
            for result in results
            if result.metadata.get("human_advantage_probabilities")
        ]

        if not valid_results:
            self.logger.warning(
                "No valid results with human advantage probabilities for human vs LLM plot"
            )
            return

        num_llms = len(valid_results)
        fig, axes = plt.subplots(num_llms, 1, figsize=(8, 3 * num_llms))

        # Ensure axes is always iterable
        if num_llms == 1:
            axes = [axes]

        # Get all unique human annotators from the first valid result
        first_result = valid_results[0]
        human_annotators = list(
            first_result.metadata["human_advantage_probabilities"].keys()
        )

        if not human_annotators:
            self.logger.warning("No human annotators found in results")
            return

        human_colors = plt.cm.jet(np.linspace(0, 1, len(human_annotators)))  # type: ignore

        for idx, result in enumerate(valid_results):
            ax = axes[idx]
            judge_id = result.judge_id
            human_advantage_probs = result.metadata["human_advantage_probabilities"]

            # Add shaded regions for better/worse/equal performance
            ax.fill_between(
                [0, 1],
                [0, 1],
                [1, 1],
                alpha=0.2,
                color="royalblue",
                label="Human Better Agreement",
            )
            ax.fill_between(
                [0, 1],
                [0, 0],
                [0, 1],
                alpha=0.2,
                color="crimson",
                label="LLM Better Agreement",
            )
            ax.plot(
                [0, 1], [0, 1], ":", color="black", alpha=0.5, label="Equal Agreement"
            )

            # Plot each human's point
            for i, (human_id, (llm_adv, human_adv)) in enumerate(
                human_advantage_probs.items()
            ):
                ax.scatter(
                    llm_adv,
                    human_adv,
                    label=f"{human_id} vs LLM",
                    color=human_colors[i],
                    s=120,
                    edgecolor="white",
                    linewidth=2,
                    alpha=0.8,
                )

            ax.set_title(f"{judge_id}", fontsize=12, pad=15, loc="left")
            ax.set_xlabel("LLM Advantage Probability", fontsize=10)
            ax.set_ylabel("Human Advantage Probability", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="both", which="major", labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")

        # Add overall title and legend
        fig.suptitle(
            f"Human vs LLM Advantage Probabilities ({unique_name})", fontsize=14
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        # Save plot
        save_path = os.path.join(alt_test_dir, "aggregate_human_vs_llm_advantage.png")
        save_plot(fig, save_path, self.logger)
        plt.close(fig)

    def aggregate_results(
        self, results: List[BaseScoringResult], scores_dir: str, unique_name: str = ""
    ) -> None:
        """Generate aggregate plots from alt-test results.

        Args:
            results: List of alt-test scoring results (pre-filtered by MetaEvaluator)
            scores_dir: Directory to save aggregate plots
            unique_name: Unique identifier for this metric configuration
        """
        if len(results) == 0:
            self.logger.info(
                "Alt-test aggregation skipped: no alt-test results provided"
            )
            return

        # Create alt_test directory for aggregate plots, using unique_name to avoid overwrites
        alt_test_dir = os.path.join(scores_dir, self.scorer_name, unique_name)
        os.makedirs(alt_test_dir, exist_ok=True)

        # Extract the scoring function from the first result
        scoring_function = results[0].metadata["scoring_function"]

        # Generate the 3 aggregate plots using stored detailed results
        self.generate_aggregate_winning_rates_plot(
            results, alt_test_dir, scoring_function, unique_name
        )
        self.generate_aggregate_advantage_probabilities_plot(
            results, alt_test_dir, scoring_function, unique_name
        )
        self.generate_aggregate_human_vs_llm_plot(results, alt_test_dir, unique_name)

        judge_count = len(results)
        self.logger.info(
            f"Generated alt-test aggregate plots for {judge_count} judge(s) in {alt_test_dir}"
        )
