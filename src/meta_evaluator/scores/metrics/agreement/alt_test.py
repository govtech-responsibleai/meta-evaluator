"""Alt-Test scorer for evaluating LLM judges against human annotators.

Based on: https://arxiv.org/abs/2501.10970
Code adapted from: https://github.com/nitaytech/AltTest/
"""

import os
from typing import List, Optional, Dict, Any, Callable, Union, Tuple

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from sklearn.metrics import jaccard_score

from meta_evaluator.scores.base_scorer import BaseScorer
from meta_evaluator.scores.base_scoring_result import BaseScoringResult


class AltTestScorer(BaseScorer):
    """Alt-Test scorer for evaluating LLM judges vs human annotators."""

    def __init__(
        self,
        epsilon: float = 0.2,
        multiplicative_epsilon: bool = False,
        q_fdr: float = 0.05,
        min_humans_per_instance: int = 2,
        min_instances_per_human: int = 30,
    ):
        """Initialize Alt-Test scorer.

        Args:
            epsilon: Threshold parameter for alt-test
            multiplicative_epsilon: Whether to use multiplicative epsilon
            q_fdr: False discovery rate for multiple testing correction
            min_humans_per_instance: Minimum humans required per instance
            min_instances_per_human: Minimum instances required per human
        """
        super().__init__("alt_test")
        self.epsilon = epsilon
        self.multiplicative_epsilon = multiplicative_epsilon
        self.q_fdr = q_fdr
        self.min_humans_per_instance = min_humans_per_instance
        self.min_instances_per_human = min_instances_per_human

        # Set up matplotlib config
        plt.rcParams["font.size"] = 12
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Verdana"]

    def can_score_task(self, task_schema: Optional[List[str]]) -> bool:
        """Alt-Test can work with both classification and free-form text tasks.

        Alt-Test evaluates human-vs-LLM performance differences for supported task types.
        For classification tasks, it compares categorical predictions.
        For text tasks, it can compare text similarity or semantic equivalence.

        Args:
            task_schema: List of allowed categorical outcomes, or None for free-form text tasks

        Returns:
            bool: True for classification tasks (task_schema is not None) and free-form text tasks (task_schema is None)
        """
        # Alt-Test supports both classification and free-form text tasks
        # For classification: task_schema is a list of categories
        # For free-form text: task_schema is None
        return task_schema is None or isinstance(task_schema, list)

    def compute_score(
        self,
        judge_id: str,
        consolidated_judge_df: pl.DataFrame,
        consolidated_human_df: pl.DataFrame,
        task_names: List[str],
        task_schemas: dict,
        scores_dir: Optional[str] = None,
    ) -> BaseScoringResult:
        """Compute Alt-Test score for a single judge vs many humans.

        Returns:
            BaseScoringResult: A BaseScoringResult instance.
        """
        # Determine appropriate scoring function based on task characteristics
        scoring_function = self._determine_scoring_function(task_names, task_schemas)

        # Convert to format expected by alt-test functions
        judge_annotations = self._convert_judge_to_alttest_format(
            consolidated_judge_df, task_names, task_schemas
        )
        human_annotations = self._convert_humans_to_alttest_format(
            consolidated_human_df, task_names, task_schemas
        )

        # Run alt-test with multiple epsilon values for detailed analysis
        epsilons = np.arange(0.0, 0.31, 0.05)
        detailed_results = []

        # Run alt-test for each epsilon value
        for epsilon in epsilons:
            winning_rate, advantage_prob, human_advantage_probs = self._alt_test(
                judge_annotations,
                human_annotations,
                scoring_function,
                epsilon=float(epsilon),
            )
            detailed_results.append(
                {
                    "epsilon": epsilon,
                    "winning_rate": winning_rate,
                    "advantage_probability": advantage_prob,
                    "human_advantage_probabilities": human_advantage_probs,
                }
            )

        # Use the default epsilon (0.2) for the main score
        default_result = next(
            (r for r in detailed_results if r["epsilon"] == self.epsilon),
            detailed_results[4],
        )
        main_winning_rate = default_result["winning_rate"]
        main_advantage_prob = default_result["advantage_probability"]
        main_human_advantage_probs = default_result["human_advantage_probabilities"]

        # Set task display name
        task_display = (
            task_names[0]
            if len(task_names) == 1
            else f"{len(task_names)}_tasks_combined"
        )

        return BaseScoringResult(
            scorer_name=self.scorer_name,
            task_name=task_display,
            judge_id=judge_id,
            score=main_winning_rate,
            metadata={
                "advantage_probability": main_advantage_prob,
                "human_advantage_probabilities": main_human_advantage_probs,
                "scoring_function": scoring_function,
                "epsilon": self.epsilon,
                "multiplicative_epsilon": self.multiplicative_epsilon,
                "q_fdr": self.q_fdr,
                "min_humans_per_instance": self.min_humans_per_instance,
                "min_instances_per_human": self.min_instances_per_human,
                "task_names": task_names,
                "task_schemas": task_schemas,
                "detailed_results": detailed_results,  # Store all epsilon results
            },
        )

    def _convert_to_alttest_format(
        self, df: pl.DataFrame, task_names: List[str], task_schemas: dict
    ) -> Dict[Union[int, str], Any]:
        """Convert dataframe rows to alt-test format: {original_id: labels}.

        For single-label classification: {original_id1: "SAFE", original_id2: "UNSAFE", ...}
        For multi-label classification: {original_id1: ["TOXIC", "HATEFUL"], original_id2: ["SAFE"], ...}
        For text tasks: {original_id1: "hello", original_id2: "world", ...}

        Returns:
            Dict[Union[int, str], Any]: A dictionary of annotations.
        """
        annotations = {}

        for row in df.iter_rows(named=True):
            original_id = row["original_id"]
            task_name = row["task_name"]

            if task_name in task_names:
                value = row["task_value"]
                if value is not None:
                    if original_id not in annotations:
                        # Determine the container type based on task characteristics
                        is_multilabel = self._is_multilabel_task(
                            task_names, task_schemas
                        )
                        annotations[original_id] = [] if is_multilabel else None

                    # Handle different task types
                    if task_schemas.get(task_name) is not None:
                        # Classification task - use value as-is
                        if self._is_multilabel_task(task_names, task_schemas):
                            # Multi-label: append to list
                            annotations[original_id].append(value)
                        else:
                            # Single-label: overwrite (should be one value per original_id)
                            annotations[original_id] = value
                    else:
                        # Text task: store as is
                        annotations[original_id] = value

        return annotations

    def _is_multilabel_task(self, task_names: List[str], task_schemas: dict) -> bool:
        """Check if this is a multilabel task based on having multiple classification tasks.

        Returns:
            bool: True if the task is a multilabel task, False otherwise.
        """
        classification_count = sum(
            1 for task_name in task_names if task_schemas.get(task_name) is not None
        )
        return classification_count > 1

    def _convert_judge_to_alttest_format(
        self,
        consolidated_judge_df: pl.DataFrame,
        task_names: List[str],
        task_schemas: dict,
    ) -> Dict[Union[int, str], Any]:
        """Convert judge dataframe to alt-test format.

        Outputs {original_id1: label, original_id2: label, ...}.

        Returns:
            Dict[Union[int, str], Any]: A dictionary of annotations.
        """
        return self._convert_to_alttest_format(
            consolidated_judge_df, task_names, task_schemas
        )

    def _convert_humans_to_alttest_format(
        self,
        consolidated_human_df: pl.DataFrame,
        task_names: List[str],
        task_schemas: dict,
    ) -> Dict[Union[int, str], Dict[Union[int, str], Any]]:
        """Convert human dataframe to alt-test format.

        Outputs {annotator1: {original_id1: label, original_id2: label, ...}, annotator2: {...}, ...}

        Returns:
            Dict[Union[int, str], Dict[Union[int, str], Any]]: A dictionary of annotations.
        """
        annotations = {}

        # Group by annotator_id
        for annotator_id in consolidated_human_df["annotator_id"].unique():
            annotator_df = consolidated_human_df.filter(
                pl.col("annotator_id") == annotator_id
            )
            annotations[annotator_id] = self._convert_to_alttest_format(
                annotator_df, task_names, task_schemas
            )

        return annotations

    def _determine_scoring_function(
        self, task_names: List[str], task_schemas: dict
    ) -> str:
        """Automatically determine the best scoring function based on task characteristics.

        Returns:
            str: The name of the best scoring function.
        """
        # Analyze task types
        classification_tasks = 0
        text_tasks = 0

        for task_name in task_names:
            schema = task_schemas.get(task_name)
            if schema is None:
                # Free-form text task
                text_tasks += 1
            else:
                # Classification task
                classification_tasks += 1

        # Determine best scoring function
        if text_tasks > 0:
            # For text tasks, default to accuracy
            # TODO: Add a text similarity function
            return "accuracy"
        elif classification_tasks > 1:
            # For multilabel classification (multiple classification tasks), use macro jaccard
            return "jaccard_similarity"
        else:
            # For single classification tasks, use accuracy
            return "accuracy"

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
            ValueError: If the scoring function is unknown.
        """
        if scoring_function_name == "accuracy":
            return self._accuracy
        elif scoring_function_name == "neg_rmse":
            return self._neg_rmse
        elif scoring_function_name == "jaccard_similarity":
            return self._jaccard_similarity
        else:
            raise ValueError(f"Unknown scoring function: {scoring_function_name}")

    # Alt-test core functions
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

    def _alt_test(
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
            ValueError: If no annotators meet the minimum threshold of instances per human.
        """
        # Use provided epsilon or fall back to default epsilon == 0.0
        scoring_function = self._get_scoring_function(scoring_function_name)

        # prepare sets - i_set has humans as keys, h_set has instances as keys
        i_set, h_set = {}, {}
        for h, anns in sorted(humans_annotations.items()):
            i_set[h] = sorted(list(anns.keys()))
            for i, ann in sorted(anns.items()):
                if i not in h_set:
                    h_set[i] = []
                h_set[i].append(h)

        # remove instances with less than min_humans_per_instance
        instances_to_keep = {
            i
            for i in sorted(h_set.keys())
            if len(h_set[i]) >= self.min_humans_per_instance and i in llm_annotations
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
                print(
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
            raise ValueError(
                f"No annotators meet the minimum threshold of {self.min_instances_per_human} instances per human"
            )

        rejected_indices = self._by_procedure(p_values, self.q_fdr)
        advantage_prob = float(np.mean(advantage_probs))
        winning_rate = len(rejected_indices) / len(humans) if humans else np.nan

        return winning_rate, advantage_prob, human_advantage_probs

    def aggregate_results(
        self, results: List[BaseScoringResult], scores_dir: str
    ) -> None:
        """Generate aggregate plots from alt-test results.

        Args:
            results: List of alt-test scoring results (pre-filtered by MetaEvaluator)
            scores_dir: Directory to save aggregate plots
        """
        if len(results) == 0:
            print("Alt-test aggregation skipped: no alt-test results provided")
            return

        # Create alt_test directory for aggregate plots
        alt_test_dir = os.path.join(scores_dir, "alt_test")
        os.makedirs(alt_test_dir, exist_ok=True)

        # Extract the scoring function from the first result
        scoring_function = results[0].metadata["scoring_function"]

        # Save individual ScoringResult objects as JSON files
        self.save_results(results, alt_test_dir)
        print(
            f"Generated alt-test results for {len(results)} judge(s) in {alt_test_dir}"
        )

        # Generate the 3 aggregate plots using stored detailed results
        self.generate_aggregate_winning_rates_plot(
            results, alt_test_dir, scoring_function
        )
        self.generate_aggregate_advantage_probabilities_plot(
            results, alt_test_dir, scoring_function
        )
        self.generate_aggregate_human_vs_llm_plot(results, alt_test_dir)

        judge_count = len(results)
        print(
            f"Generated alt-test aggregate plots for {judge_count} judge(s) in {alt_test_dir}"
        )

    def generate_aggregate_winning_rates_plot(
        self, results: List[BaseScoringResult], alt_test_dir: str, scoring_function: str
    ) -> None:
        """Generate aggregate winning rates plot as line chart across epsilon values."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Generate colors for each judge
        judge_ids = [result.judge_id for result in results]
        colors = plt.cm.Set1(np.linspace(0, 1, len(judge_ids)))  # type: ignore

        # Extract epsilon values from the first result
        epsilons = [dr["epsilon"] for dr in results[0].metadata["detailed_results"]]

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
            detailed_results = result.metadata["detailed_results"]
            winning_rates = [dr["winning_rate"] for dr in detailed_results]
            advantage_prob = result.metadata[
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
            f"Alt-Test Winning Rates: {scoring_function.replace('_', ' ').title()}"
        )
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Save plot
        save_path = os.path.join(alt_test_dir, "aggregate_winning_rates.png")
        self.save_plot(fig, save_path)
        plt.close(fig)

    def generate_aggregate_advantage_probabilities_plot(
        self, results: List[BaseScoringResult], alt_test_dir: str, scoring_function: str
    ) -> None:
        """Generate aggregate advantage probabilities plot for all judges."""
        fig, ax = plt.subplots(figsize=(12, 8))

        judge_ids = []
        advantage_probs = []

        for result in results:
            judge_ids.append(result.judge_id)
            advantage_probs.append(result.metadata["advantage_probability"])

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
            f"LLM Advantage Probabilities: {scoring_function.replace('_', ' ').title()}"
        )
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save plot
        save_path = os.path.join(alt_test_dir, "aggregate_advantage_probabilities.png")
        self.save_plot(fig, save_path)
        plt.close(fig)

    def generate_aggregate_human_vs_llm_plot(
        self, results: List[BaseScoringResult], alt_test_dir: str
    ) -> None:
        """Generate human vs LLM advantage probabilities plot - one chart per LLM."""
        num_llms = len(results)
        fig, axes = plt.subplots(num_llms, 1, figsize=(8, 3 * num_llms))

        # Ensure axes is always iterable
        if num_llms == 1:
            axes = [axes]

        # Get all unique human annotators from the first result
        first_result = results[0]
        human_annotators = list(
            first_result.metadata["human_advantage_probabilities"].keys()
        )
        human_colors = plt.cm.jet(np.linspace(0, 1, len(human_annotators)))  # type: ignore

        for idx, result in enumerate(results):
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

        # Add a single legend for all plots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        # Save plot
        save_path = os.path.join(alt_test_dir, "aggregate_human_vs_llm_advantage.png")
        self.save_plot(fig, save_path)
        plt.close(fig)

    @staticmethod
    def save_plot(fig, save_path: str):
        """Save a matplotlib figure."""
        fig.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Saved plot to {save_path}")
