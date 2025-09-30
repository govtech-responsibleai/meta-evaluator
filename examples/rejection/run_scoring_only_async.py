#!/usr/bin/env python3
"""Run scoring-only evaluation for rejection task using external judge/human results.

This example demonstrates how to:
1. Load the original rejection evaluation data
2. Create mock judge and human annotation results
3. Load external results directly into MetaEvaluator
4. Run scoring metrics to compare judge vs human performance

This is useful when you already have judge/human results from previous runs
or external sources and only want to compute alignment metrics.
"""

import logging
import random
import sys
import tempfile
from pathlib import Path

import dotenv
import polars as pl

from meta_evaluator.data import DataLoader, EvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.meta_evaluator import MetaEvaluator
from meta_evaluator.scores import MetricConfig, MetricsConfig
from meta_evaluator.scores.metrics import (
    AccuracyScorer,
    AltTestScorer,
    CohensKappaScorer,
    SemanticSimilarityScorer,
    TextSimilarityScorer,
)

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            filename="logs/rejection_scoring_only.log",
            mode="w",
            encoding="utf-8",
        ),
    ],
)


def create_mock_judge_results(eval_data: pl.DataFrame) -> str:
    """Create mock judge results for a single judge.

    The CSV format only requires user-provided columns:
    - original_id: Unique identifier for each sample
    - rejection: Task result ("rejection" or "not rejection")
    - explanation: Free-form explanation text

    System columns (sample_example_id, run_id, judge_id) are auto-generated.

    Args:
        eval_data: The original evaluation data

    Returns:
        Path to the created judge results CSV file
    """
    # Extract original_id values (assuming they exist or create them)
    original_ids = [f"sample_{i}" for i in range(1, len(eval_data) + 1)]

    # Define judge classification pattern for readability
    judge_labels = ["rejection", "not rejection", "rejection"] * (
        len(eval_data) // 3 + 1
    )
    judge_labels = judge_labels[: len(eval_data)]  # Trim to exact length

    # Create judge results with rejection classification and explanations
    judge_results = []
    for i, (original_id, rejection) in enumerate(zip(original_ids, judge_labels)):
        judge_results.append(
            {
                "original_id": original_id,
                "rejection": rejection,
                "explanation": f"Judge analysis for {original_id}: This response shows clear signs of {'harmful content detection' if rejection == 'rejection' else 'appropriate content handling'}.",
            }
        )

    judge_df = pl.DataFrame(judge_results)

    # Save to temporary file
    temp_dir = tempfile.mkdtemp()
    judge_path = Path(temp_dir) / "judge_results.csv"
    judge_df.write_csv(judge_path)

    return str(judge_path)


def create_mock_human_results(eval_data: pl.DataFrame, annotator_id: str) -> str:
    """Create randomised mock human annotation results for a single annotator.

    The CSV format only requires user-provided columns:
    - original_id: Unique identifier for each sample
    - rejection: Task result ("rejection" or "not rejection")
    - explanation: Free-form explanation text

    System columns (sample_example_id, run_id, annotator_id) are auto-generated.

    Args:
        eval_data: The original evaluation data
        annotator_id: Unique identifier for this specific annotator

    Returns:
        Path to the created human results CSV file for this annotator
    """
    original_ids = [f"sample_{i}" for i in range(1, len(eval_data) + 1)]

    # Define valid label options
    label_options = ["rejection", "not rejection"]

    # Create human results with random labels
    human_results = []

    for original_id in original_ids:
        # Randomly select a label
        rejection = random.choice(label_options)

        # Generate explanation based on the random label
        explanation = f"Human {annotator_id} analysis for {original_id}: This content appears to be {'problematic' if rejection == 'rejection' else 'acceptable'} based on safety guidelines."

        human_results.append(
            {
                "original_id": original_id,
                "rejection": rejection,
                "explanation": explanation,
            }
        )

    human_df = pl.DataFrame(human_results)

    # Save to temporary file
    temp_dir = tempfile.mkdtemp()
    human_path = Path(temp_dir) / f"human_results_{annotator_id}.csv"
    human_df.write_csv(human_path)

    return str(human_path)


def rejection_task() -> EvalTask:
    """Define the evaluation task for rejection classification.

    Returns:
        EvalTask: The evaluation task configured for rejection classification
    """
    task = EvalTask(
        task_schemas={
            "rejection": ["rejection", "not rejection"],  # Classification field
            "explanation": None,  # Free-form text field for explanations
        },
        prompt_columns=["prompt"],  # Column name of the prompt to evaluate
        response_columns=["llm_response"],  # Column name of the response to evaluate
        answering_method="structured",  # Output method for the judge
        structured_outputs_fallback=True,  # Enable fallback to other methods if structured is not supported
    )
    return task


def rejection_data() -> EvalData:
    """Load the original rejection evaluation data.

    Returns:
        EvalData: The evaluation data loaded from CSV
    """
    eval_data = DataLoader.load_csv(
        name="rejection_data",
        file_path="data/sample_rejection.csv",
    )
    return eval_data


def main():
    """Main function to run scoring-only evaluation with external results."""
    print("=== Rejection Scoring-Only Evaluation ===")

    # Set random seed at the start for reproducibility
    random.seed(42)

    try:
        # Initialize MetaEvaluator
        evaluator = MetaEvaluator(project_dir="test_project", load=False)

        # Add eval task and data
        eval_task = rejection_task()
        eval_data = rejection_data()
        evaluator.add_eval_task(eval_task)
        evaluator.add_data(eval_data)

        # Create mock judge and human results
        judge_results_path = create_mock_judge_results(eval_data.data)
        human1_results_path = create_mock_human_results(eval_data.data, "annotator_1")
        human2_results_path = create_mock_human_results(eval_data.data, "annotator_2")
        human3_results_path = create_mock_human_results(eval_data.data, "annotator_3")

        print("\n=== Loading External Results ===")
        # Load external judge results (can be called multiple times for multiple judges)
        evaluator.add_external_judge_results(
            file_path=judge_results_path,
            judge_id="rejection_judge",
            llm_client="mock_client",
            model_used="mock_model",
            run_id="judge_run_1",
        )

        # Load external human results (can be called multiple times for multiple annotators)
        evaluator.add_external_annotation_results(
            file_path=human1_results_path,
            annotator_id="annotator_1",
            run_id="human_run_1",
        )
        evaluator.add_external_annotation_results(
            file_path=human2_results_path,
            annotator_id="annotator_2",
            run_id="human_run_2",
        )
        evaluator.add_external_annotation_results(
            file_path=human3_results_path,
            annotator_id="annotator_3",
            run_id="human_run_3",
        )

        # Create scorers
        accuracy_scorer = AccuracyScorer()
        alt_test_scorer = AltTestScorer()
        cohens_kappa_scorer = CohensKappaScorer()
        text_similarity_scorer = TextSimilarityScorer()
        semantic_similarity_scorer = SemanticSimilarityScorer()

        # Create metrics config
        config = MetricsConfig(
            metrics=[
                MetricConfig(
                    scorer=accuracy_scorer,
                    task_names=["rejection"],
                    task_strategy="single",
                ),
                MetricConfig(
                    scorer=cohens_kappa_scorer,
                    task_names=["rejection"],
                    task_strategy="single",
                ),
                MetricConfig(
                    scorer=alt_test_scorer,
                    task_names=["rejection"],
                    task_strategy="single",
                ),
                MetricConfig(
                    scorer=text_similarity_scorer,
                    task_names=["explanation"],
                    task_strategy="single",
                ),
                MetricConfig(
                    scorer=semantic_similarity_scorer,
                    task_names=["explanation"],
                    task_strategy="single",
                ),
            ]
        )

        print("\n=== Running Scoring ===")
        # Add metrics configuration and run comparison
        evaluator.add_metrics_config(config)
        evaluator.compare_async()

        print(f"\nResults saved in: {evaluator.paths.scores}")

        # Generate score report
        evaluator.score_report.save("score_report.html", format="html")
        evaluator.score_report.save("score_report.csv", format="csv")
        evaluator.score_report.print()  # Print to console

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
