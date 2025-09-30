#!/usr/bin/env python3
"""Run evaluation with judge/human loading and metric comparison."""

import logging
import sys
import time

import dotenv

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
        logging.StreamHandler(sys.stdout),  # Output to console
        logging.FileHandler(
            filename="logs/rejection_output.log",
            mode="w",  # 'a' for append, 'w' for overwrite
            encoding="utf-8",
        ),
    ],
)


def rejection_task() -> EvalTask:
    """Define the evaluation task.

    Returns:
        EvalTask: The evaluation task.
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
    """Load the evaluation data.

    Returns:
        EvalData: The evaluation data.
    """
    eval_data = DataLoader.load_csv(
        name="rejection_data",
        file_path="data/sample_rejection.csv",
    )
    return eval_data


def main():
    """Main function to run async evaluation."""
    # Set load = False to initialise new MetaEvaluator instance
    evaluator = MetaEvaluator(project_dir="project_dir_async", load=True)

    # Add eval task and eval data
    eval_task = rejection_task()
    eval_data = rejection_data()
    evaluator.add_eval_task(eval_task)
    evaluator.add_data(eval_data)

    # Add judges
    evaluator.load_judges_from_yaml(
        yaml_file="judges.yaml",
        on_duplicate="skip",  # Skip if judge_id already exists. Set to "overwrite" to replace existing judge.
        async_mode=True,  # Run judges asynchronously
    )

    # Save state (task, data, judges)
    evaluator.save_state(data_format="json")

    # Run judges
    print("Starting async judge evaluation with batching...")
    start_time = time.time()
    evaluator.run_judges_async(
        skip_duplicates=True  # Skip duplicates to avoid re-running judges
    )
    end_time = time.time()
    print(f"Async judge evaluation completed in {end_time - start_time:.2f} seconds")

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
                annotator_aggregation="majority_vote",  # Default is 'individual_average', set to 'majority_vote' for this example
            ),
            MetricConfig(
                scorer=alt_test_scorer,
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

    # Add metrics configuration and run comparison
    evaluator.add_metrics_config(config)
    evaluator.compare_async()

    # Generate score report
    evaluator.score_report.save("score_report.html", format="html")
    evaluator.score_report.save("score_report.csv", format="csv")
    evaluator.score_report.print()


if __name__ == "__main__":
    main()
