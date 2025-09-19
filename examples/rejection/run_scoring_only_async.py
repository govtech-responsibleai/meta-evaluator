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
    """Create mock judge results for the rejection task.

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


def create_mock_human_results(eval_data: pl.DataFrame) -> str:
    """Create mock human annotation results for the rejection task.

    These results will have ~80% agreement with judge results to test scoring.

    The CSV format only requires user-provided columns:
    - original_id: Unique identifier for each sample
    - rejection: Task result ("rejection" or "not rejection")
    - explanation: Free-form explanation text

    System columns (sample_example_id, run_id, annotator_id) are auto-generated.

    Args:
        eval_data: The original evaluation data

    Returns:
        Path to the created human results CSV file
    """
    original_ids = [f"sample_{i}" for i in range(1, len(eval_data) + 1)]

    # Define judge labels (same pattern as in create_mock_judge_results)
    judge_labels = ["rejection", "not rejection", "rejection"] * (
        len(eval_data) // 3 + 1
    )
    judge_labels = judge_labels[: len(eval_data)]

    # Create human labels with 80% agreement (flip 2 out of 10 samples)
    disagreement_indices = [2, 7]  # Make these indices different for 80% agreement
    human_labels = []
    for i, judge_label in enumerate(judge_labels):
        if i in disagreement_indices:
            # Flip the judge's decision
            human_label = "not rejection" if judge_label == "rejection" else "rejection"
        else:
            # Match the judge's decision
            human_label = judge_label
        human_labels.append(human_label)

    # Create human results with 2 annotators each
    human_results = []

    for annotator_id in ["annotator_1", "annotator_2"]:
        for i, (original_id, rejection) in enumerate(zip(original_ids, human_labels)):
            # Create somewhat different explanations to test text similarity
            if i in disagreement_indices:
                explanation = f"Human {annotator_id} analysis for {original_id}: This content appears to be {'acceptable' if rejection == 'not rejection' else 'problematic'} based on safety guidelines."
            else:
                explanation = f"Human {annotator_id} analysis for {original_id}: This response demonstrates {'concerning patterns' if rejection == 'rejection' else 'proper safety measures'}."

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
    human_path = Path(temp_dir) / "human_results.csv"
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

    # Create temporary project directory
    project_dir = tempfile.mkdtemp()

    try:
        # Initialize MetaEvaluator
        evaluator = MetaEvaluator(project_dir=project_dir, load=False)

        # Add eval task and data
        eval_task = rejection_task()
        eval_data = rejection_data()
        evaluator.add_eval_task(eval_task)
        evaluator.add_data(eval_data)

        # Create mock judge and human results
        judge_results_path = create_mock_judge_results(eval_data.data)
        human_results_path = create_mock_human_results(eval_data.data)

        print("\n=== Loading External Results ===")
        # Load external judge results
        evaluator.add_external_judge_results(
            file_path=judge_results_path,
            judge_id="rejection_judge",
            llm_client="mock_client",
            model_used="mock_model",
            run_id="judge_run_1",
        )

        # Load external human results
        evaluator.add_external_annotation_results(
            file_path=human_results_path,
            annotator_id="human_annotators",
            run_id="human_run_1",
        )

        # Create scorers
        accuracy_scorer = AccuracyScorer()
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

        # Load judge and human results
        judge_results = evaluator.load_all_judge_results()
        human_results = evaluator.load_all_human_results()

        print("\n=== Running Scoring ===")
        # Add metrics configuration and run comparison
        evaluator.add_metrics_config(config)
        results = evaluator.compare_async(
            judge_results=judge_results,
            human_results=human_results,
        )

        print("\n=== Results ===")
        # Display results
        for unique_name, (metric_config, scoring_results) in results.items():
            print(f"\n--- {unique_name} ---")

            for result in scoring_results:
                print(f"Scorer: {result.scorer_name}")
                print(f"Task: {result.task_name}")
                print(f"Judge: {result.judge_id}")
                print(f"Scores: {result.scores}")
                print(f"Comparisons: {result.num_comparisons}")
                print(f"Failed: {result.failed_comparisons}")

        print(f"\nResults saved in: {evaluator.paths.scores}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up temporary files
        import shutil

        if Path(project_dir).exists():
            shutil.rmtree(project_dir)


if __name__ == "__main__":
    main()
