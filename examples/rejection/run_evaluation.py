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
        answering_method="structured",
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
    """Main function to run evaluation."""
    evaluator = MetaEvaluator(project_dir="project_dir")

    # Add eval task and eval data
    eval_task = rejection_task()
    eval_data = rejection_data()
    evaluator.add_eval_task(eval_task)
    evaluator.add_data(eval_data)

    # Add judges
    evaluator.load_judges_from_yaml(yaml_file="judges.yaml")

    # Run judges
    print("Starting sync judge evaluation without batching...")
    start_time = time.time()
    evaluator.run_judges()
    end_time = time.time()
    print(f"Sync judge evaluation completed in {end_time - start_time:.2f} seconds")

    # Load judge/human annotation results
    judge_results = evaluator.load_all_judge_results()
    human_results = evaluator.load_all_human_results()

    # Create scorers
    accuracy_scorer = AccuracyScorer()
    alt_test_scorer = AltTestScorer()
    alt_test_scorer.min_instances_per_human = (
        10  # Just for testing as this example only has 10 instances
    )
    cohens_kappa_scorer = CohensKappaScorer()
    text_similarity_scorer = TextSimilarityScorer()

    # Create metrics config
    config = MetricsConfig(
        metrics=[
            MetricConfig(scorer=accuracy_scorer, task_names=["rejection"]),
            MetricConfig(scorer=alt_test_scorer, task_names=["rejection"]),
            MetricConfig(scorer=cohens_kappa_scorer, task_names=["rejection"]),
            MetricConfig(scorer=text_similarity_scorer, task_names=["explanation"]),
        ]
    )

    # Run comparison and save results
    evaluator.compare(config, judge_results=judge_results, human_results=human_results)


if __name__ == "__main__":
    main()
