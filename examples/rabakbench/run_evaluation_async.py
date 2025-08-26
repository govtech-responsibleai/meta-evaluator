#!/usr/bin/env python3
"""Run alt-test evaluation with judge/human loading and metric comparison."""

import logging

import dotenv

from meta_evaluator.data import DataLoader, EvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.meta_evaluator import MetaEvaluator
from meta_evaluator.scores import MetricConfig, MetricsConfig
from meta_evaluator.scores.metrics import AltTestScorer, CohensKappaScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[
        # logging.StreamHandler(sys.stdout),  # Output to console
        logging.FileHandler(
            filename="rabakbench_output.log",
            mode="w",  # 'a' for append, 'w' for overwrite
            encoding="utf-8",
        ),
    ],
)

# Load environment variables
dotenv.load_dotenv()


# Helper function to define task
def rabakbench_task() -> EvalTask:
    """Define the evaluation task.

    Returns:
        EvalTask: The evaluation task.
    """
    task = EvalTask(
        task_schemas={
            "hateful": ["FALSE", "hateful"],
            "insults": ["FALSE", "insults"],
            "sexual": ["FALSE", "sexual"],
            "physical_violence": ["FALSE", "physical_violence"],
            "self_harm": ["FALSE", "self_harm"],
            "all_other_misconduct": ["FALSE", "all_other_misconduct"],
            "reasoning": None,  # free-form text
        },
        prompt_columns=None,
        response_columns=["text"],
        answering_method="structured",
        structured_outputs_fallback=True,
    )
    return task


# Helper function to define data
def rabakbench_data() -> EvalData:
    """Load the evaluation data.

    Returns:
        EvalData: The evaluation data.
    """
    eval_data = DataLoader.load_csv(
        id_column="prompt_id",
        name="rabakbench_sample",
        file_path="data/rabakbench_sample.csv",
    )
    return eval_data


def main():
    """Main function to run alt-test evaluation."""
    # Load judge results from converted data
    evaluator = MetaEvaluator(project_dir="project_dir", load=False)

    # Add eval task and eval data
    eval_task = rabakbench_task()
    eval_data = rabakbench_data()
    evaluator.add_eval_task(eval_task)
    evaluator.add_data(eval_data)

    # Add judges
    evaluator.load_judges_from_yaml(
        yaml_file="judges.yaml",
        on_duplicate="skip",
        async_mode=True,
    )

    evaluator.save_state(data_format="json")

    evaluator.run_judges_async(
        skip_duplicates=True  # Skip duplicates to avoid re-running judges
    )

    # Load judge/human annotation results
    judge_results = evaluator.load_all_judge_results()
    human_results = evaluator.load_all_human_results()

    # Create scorers
    alt_test_scorer = AltTestScorer(multiplicative_epsilon=True)
    alt_test_scorer.min_instances_per_human = (
        10  # Just for testing as this example does not have enough data
    )
    cohens_kappa_scorer = CohensKappaScorer()

    # Create metrics config
    config = MetricsConfig(
        metrics=[
            MetricConfig(
                scorer=alt_test_scorer,
                task_names=[
                    "hateful",
                    "insults",
                    "sexual",
                    "physical_violence",
                    "self_harm",
                    "all_other_misconduct",
                ],
                aggregation_name="multilabel",
            ),
            MetricConfig(
                scorer=cohens_kappa_scorer,
                task_names=[
                    "hateful",
                    "insults",
                    "sexual",
                    "physical_violence",
                    "self_harm",
                    "all_other_misconduct",
                ],
                aggregation_name="multitask",
            ),
            MetricConfig(
                scorer=cohens_kappa_scorer,
                task_names=["hateful"],
                aggregation_name="single",
            ),
        ]
    )

    # Run comparison and save results
    evaluator.compare_async(
        config, judge_results=judge_results, human_results=human_results
    )


if __name__ == "__main__":
    main()
