"""Semantic similarity scorer using OpenAI embeddings and cosine similarity."""

import asyncio
import os
from typing import List

import numpy as np
import polars as pl
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity

from meta_evaluator.scores.base_scorer import BaseScorer
from meta_evaluator.scores.base_scoring_result import BaseScoringResult
from meta_evaluator.scores.utils import generate_simple_bar_plot


class SemanticSimilarityScorer(BaseScorer):
    """Scorer for text tasks using OpenAI embeddings and cosine similarity."""

    def __init__(self, model: str = "text-embedding-3-large", batch_size: int = 100):
        """Initialize semantic similarity scorer.

        Args:
            model: OpenAI embedding model to use
            batch_size: Number of texts to process in each batch
        """
        super().__init__("semantic_similarity")
        self.model = model
        self.batch_size = batch_size
        self.client = AsyncOpenAI()

    def can_score_task(
        self, sample_label: str | int | float | List[str | int | float]
    ) -> bool:
        """Semantic similarity scorer works with string data or lists of strings.

        Args:
            sample_label: Sample of the actual label data to validate

        Returns:
            bool: True if data contains string values (str or list of str)
        """
        # Accept str or list of str (same as TextSimilarityScorer)
        if isinstance(sample_label, str):
            return True
        elif isinstance(sample_label, list):
            # Check if list contains str
            if len(sample_label) > 0:
                return isinstance(sample_label[0], str)
            return True  # Empty list is acceptable
        else:
            return False

    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts using OpenAI API.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not texts:
            return []

        # Filter out None/empty texts and track indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text is not None and str(text).strip():
                valid_texts.append(str(text).strip())
                valid_indices.append(i)

        if not valid_texts:
            # Return zero embeddings for all texts
            return [[0.0] * 3072] * len(
                texts
            )  # text-embedding-3-large has 3072 dimensions

        try:
            response = await self.client.embeddings.create(
                model=self.model, input=valid_texts
            )

            # Create result array with proper dimensions
            embeddings = [[0.0] * 3072] * len(texts)

            # Fill in embeddings for valid texts
            for i, embedding_data in enumerate(response.data):
                original_index = valid_indices[i]
                embeddings[original_index] = embedding_data.embedding

            return embeddings

        except Exception as e:
            self.logger.error(f"Failed to get embeddings: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * 3072] * len(texts)

    async def _get_all_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for all texts using batch processing.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await self._get_embeddings_batch(batch)
            all_embeddings.extend(batch_embeddings)

            # Add small delay between batches to avoid rate limiting
            if i + self.batch_size < len(texts):
                await asyncio.sleep(0.1)

        return all_embeddings

    def _compute_cosine_similarities(
        self, judge_embeddings: List[List[float]], human_embeddings: List[List[float]]
    ) -> List[float]:
        """Compute cosine similarities between judge and human embeddings.

        Args:
            judge_embeddings: List of judge text embeddings
            human_embeddings: List of human text embeddings

        Returns:
            List[float]: Cosine similarity scores
        """
        similarities = []

        for judge_emb, human_emb in zip(judge_embeddings, human_embeddings):
            if judge_emb and human_emb:
                # Convert to numpy arrays and reshape for sklearn
                judge_array = np.array(judge_emb).reshape(1, -1)
                human_array = np.array(human_emb).reshape(1, -1)

                # Compute cosine similarity
                similarity = cosine_similarity(judge_array, human_array)[0, 0]
                similarities.append(float(similarity))
            else:
                similarities.append(0.0)

        return similarities

    async def compute_score_async(
        self,
        judge_data: pl.DataFrame,
        human_data: pl.DataFrame,
        task_name: str,
        judge_id: str,
        aggregation_mode,
    ) -> BaseScoringResult:
        """Compute semantic similarity score for a single judge vs many humans (async).

        Args:
            judge_data: DataFrame with judge outcomes (columns: original_id, label)
            human_data: DataFrame with human outcomes (columns: original_id, human_id, label)
            task_name: Name of the task(s) being scored
            judge_id: ID of the judge being scored
            aggregation_mode: How the tasks were aggregated for this result

        Returns:
            BaseScoringResult: The scoring result for this judge
        """
        # Join judge and human data on original_id
        comparison_df = judge_data.join(human_data, on="original_id", how="inner")

        if comparison_df.is_empty():
            semantic_similarity_score = float("nan")
            num_comparisons = 0
            failed_comparisons = 1
        else:
            # Collect all unique texts for batch embedding
            judge_texts = comparison_df["label"].to_list()
            human_texts = comparison_df["label_right"].to_list()

            # Get unique texts to minimize embedding calls
            all_texts = list(
                set([str(t) for t in judge_texts + human_texts if t is not None])
            )

            self.logger.info(f"Getting embeddings for {len(all_texts)} unique texts")

            # Get embeddings for all unique texts
            all_embeddings = await self._get_all_embeddings(all_texts)

            # Create mapping from text to embedding
            text_to_embedding = dict(zip(all_texts, all_embeddings))

            # For each human, compute semantic similarity between judge and that human
            human_similarities = []
            humans = comparison_df["human_id"].unique()

            for human_id in humans:
                human_comparisons = comparison_df.filter(pl.col("human_id") == human_id)
                judge_texts_human = human_comparisons["label"].to_list()
                human_texts_human = human_comparisons["label_right"].to_list()

                # Get embeddings for this human's comparisons
                judge_embeddings = [
                    text_to_embedding.get(str(t), [0.0] * 3072)
                    for t in judge_texts_human
                    if t is not None
                ]
                human_embeddings = [
                    text_to_embedding.get(str(t), [0.0] * 3072)
                    for t in human_texts_human
                    if t is not None
                ]

                # Compute similarities for this judge-human pair
                if judge_embeddings and human_embeddings:
                    similarities = self._compute_cosine_similarities(
                        judge_embeddings, human_embeddings
                    )
                    if similarities:
                        human_similarities.append(float(np.mean(similarities)))

            # Average across all humans
            semantic_similarity_score = (
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
            scores={"semantic_similarity": semantic_similarity_score},
            metadata={
                "embedding_model": self.model,
                "batch_size": self.batch_size,
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
        """Generate aggregate plots and save individual results for semantic similarity scorer.

        Args:
            results: List of semantic similarity scoring results
            scores_dir: Directory to save results and plots
            unique_name: Unique identifier for this metric configuration
        """
        if not results:
            self.logger.info("No semantic similarity results to aggregate")
            return

        # Create semantic_similarity directory for results and plots
        semantic_similarity_dir = os.path.join(
            scores_dir, self.scorer_name, unique_name
        )
        os.makedirs(semantic_similarity_dir, exist_ok=True)

        # Generate simple bar plot
        generate_simple_bar_plot(
            results=results,
            score_key="semantic_similarity",
            output_dir=semantic_similarity_dir,
            plot_filename="semantic_similarity_scores.png",
            title="Semantic Similarity Scores by Judge",
            ylabel="Semantic Similarity Score",
            unique_name=unique_name,
            logger=self.logger,
        )

        self.logger.info(
            f"Generated semantic similarity results for {len(results)} judge(s) in {semantic_similarity_dir}"
        )
