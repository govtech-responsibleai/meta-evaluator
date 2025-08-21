"""Shared plotting utilities for scoring results."""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .base_scoring_result import BaseScoringResult


# Plotting utils
def save_plot(fig, save_path: str, logger=None):
    """Save a matplotlib figure."""
    fig.savefig(
        save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    if logger:
        logger.info(f"Saved plot to {save_path}")


def generate_simple_bar_plot(
    results: List[BaseScoringResult],
    score_key: str,
    output_dir: str,
    plot_filename: str,
    title: str,
    ylabel: str,
    unique_name: str = "",
    logger=None,
):
    """Generate a simple bar plot for a single metric across all judges.

    Args:
        results: List of scoring results
        score_key: Key in the scores dict to plot (e.g., "accuracy", "similarity", "kappa")
        output_dir: Directory to save the plot
        plot_filename: Name of the plot file (e.g., "accuracy_scores.png")
        title: Plot title
        ylabel: Y-axis label
        unique_name: Unique identifier for this metric configuration for display purposes
        logger: Optional logger for info messages
    """
    if not results:
        if logger:
            logger.info(f"No results to plot for {score_key}")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    judge_ids = []
    scores = []

    for result in results:
        judge_ids.append(result.judge_id)
        score_value = result.scores.get(score_key, float("nan"))
        scores.append(score_value)

    # Create bar plot
    bars = ax.bar(
        judge_ids,
        scores,
        color="steelblue",
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        if not np.isnan(score):
            height = bar.get_height()
            ax.annotate(
                f"{score:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    ax.set_ylabel(ylabel)
    full_title = f"{title} ({unique_name})" if unique_name else title
    ax.set_title(full_title)
    ax.set_ylim(
        0, max(1.0, max(s for s in scores if not np.isnan(s)) * 1.1) if scores else 1.0
    )
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(output_dir, plot_filename)
    save_plot(fig, save_path, logger)
    plt.close(fig)
