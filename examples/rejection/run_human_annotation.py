#!/usr/bin/env python3
"""Run alt-test dataset with human annotator app."""

from meta_evaluator.data import DataLoader, EvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.meta_evaluator import MetaEvaluator


def rejection_task() -> EvalTask:
    """Define the evaluation task.

    Returns:
        EvalTask: The evaluation task.
    """
    task = EvalTask(
        task_schemas={"rejection": ["rejection", "not rejection"]},
        required_tasks=["rejection"],
        prompt_columns=["prompt"],
        response_columns=["llm_response"],
        answering_method="structured",
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
    """Run human annotation on alt-test dataset."""
    evaluator = MetaEvaluator(project_dir="project_dir")

    # Add eval task and eval data
    eval_task = rejection_task()
    eval_data = rejection_data()
    evaluator.add_eval_task(eval_task)
    evaluator.add_data(eval_data)

    # Launch annotator
    evaluator.launch_annotator(port=8501)


if __name__ == "__main__":
    main()
