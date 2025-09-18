"""Text similarity scorer for free-form text tasks."""

import os
from difflib import SequenceMatcher
from typing import List

import numpy as np
import polars as pl

from meta_evaluator.scores.base_scorer import BaseScorer
from meta_evaluator.scores.base_scoring_result import BaseScoringResult
from meta_evaluator.scores.utils import generate_simple_bar_plot


class TextSimilarityScorer(BaseScorer):
    """Scorer for free-form text tasks using text similarity with best match."""

    def __init__(self):
        """Initialize text similarity scorer."""
        super().__init__("text_similarity")

    @property
    def min_human_annotators(self) -> int:
        """Minimum number of human annotators required for text similarity.

        Returns:
            int: 1 human annotator minimum
        """
        return 1

    def can_score_task(
        self, sample_label: str | int | float | List[str | int | float]
    ) -> bool:
        """Text similarity scorer works with string data or lists of strings.

        Args:
            sample_label: Sample of the actual label data to validate

        Returns:
            bool: True if data contains string values (str or list of str)
        """
        # Accept str or list of str
        if isinstance(sample_label, str):
            return True
        elif isinstance(sample_label, list):
            # Check if list contains str
            if len(sample_label) > 0:
                return isinstance(sample_label[0], str)
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
        """Compute text similarity score for a single judge vs many humans (async).

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
            similarity_score = float("nan")
            num_comparisons = 0
            failed_comparisons = 1
        else:
            # For each human, compute text similarity between judge and that human
            human_similarities = []
            humans = comparison_df["human_id"].unique()

            for human_id in humans:
                human_comparisons = comparison_df.filter(pl.col("human_id") == human_id)
                judge_texts = human_comparisons["label"].to_list()
                human_texts = human_comparisons["label_right"].to_list()  # From join

                # Compute similarity for this judge-human pair
                similarities = []
                for judge_text, human_text in zip(judge_texts, human_texts):
                    if judge_text is not None and human_text is not None:
                        # Normalize texts and compute similarity
                        text1 = str(judge_text).lower().strip()
                        text2 = str(human_text).lower().strip()
                        matcher = SequenceMatcher(None, text1, text2)
                        similarity = matcher.ratio()
                        similarities.append(similarity)

                if similarities:
                    human_similarities.append(float(np.mean(similarities)))

            # Average across all humans
            similarity_score = (
                float(np.mean(human_similarities))
                if human_similarities
                else float("nan")
            )
            num_comparisons = len(comparison_df)
            failed_comparisons = 0

        return BaseScoringResult(
            scorer_name=self.scorer_name,
            task_name=task_name,
            judge_id=judge_id,
            scores={"similarity": similarity_score},
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
        """Generate aggregate plots and save individual results for text similarity scorer.

        Args:
            results: List of text similarity scoring results
            scores_dir: Directory to save results and plots
            unique_name: Unique identifier for this metric configuration
        """
        if not results:
            self.logger.info("No text similarity results to aggregate")
            return

        # Create text_similarity directory for results and plots, using unique_name to avoid overwrites
        text_similarity_dir = os.path.join(scores_dir, self.scorer_name, unique_name)
        os.makedirs(text_similarity_dir, exist_ok=True)

        # Generate simple bar plot (placeholder for other plots)
        generate_simple_bar_plot(
            results=results,
            score_key="similarity",
            output_dir=text_similarity_dir,
            plot_filename="text_similarity_scores.png",
            title="Text Similarity Scores by Judge",
            ylabel="Similarity Score",
            unique_name=unique_name,
            logger=self.logger,
        )

        self.logger.info(
            f"Generated text similarity results for {len(results)} judge(s) in {text_similarity_dir}"
        )
