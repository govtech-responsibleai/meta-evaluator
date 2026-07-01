"""Run RabakBench scoring using the native multi-label task type.

This is the multi-label migration of ``examples/rabakbench``. The six harm types
are one native multi-label task (``harm_categories``) rather than six binary
tasks, so scoring uses ``task_strategy="single"`` on that one column instead of
the legacy ``task_strategy="multilabel"`` melt across six columns.

Scoring paths exercised here:

- ``ClassificationScorer(average="macro")`` — the whole-vector metric. The task's
  name-vector is positionally binarized (slot ``i`` -> 1 iff it holds outcome
  ``i``'s name, else 0) at the scorer call site, then macro-averaged per label.
  ``average="samples"`` (per-item overlap) is also valid for this pure
  multi-label configuration.
- ``AltTestScorer`` — scores the native *name* vector unchanged (it is NOT
  binarized; its jaccard similarity auto-routes on list-valued labels).

``CohensKappaScorer`` is intentionally not configured: kappa has no valid
averaging axis over a sparse multi-label vector and now raises a clear error for
multi-label tasks. Use ``ClassificationScorer`` / ``AltTestScorer`` instead.
"""

import logging

import dotenv

from meta_evaluator.data import DataLoader, EvalData
from meta_evaluator.eval_task import EvalTask, MultiLabelSchema
from meta_evaluator.meta_evaluator import MetaEvaluator
from meta_evaluator.scores import MetricConfig, MetricsConfig
from meta_evaluator.scores.metrics import AltTestScorer, ClassificationScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[
        logging.FileHandler(
            filename="rabakbench_output.log",
            mode="w",  # 'a' for append, 'w' for overwrite
            encoding="utf-8",
        ),
    ],
)

# Load environment variables
dotenv.load_dotenv()


def rabakbench_task() -> EvalTask:
    """Define the evaluation task.

    Returns:
        EvalTask: The evaluation task.
    """
    task = EvalTask(
        task_schemas={
            "harm_categories": MultiLabelSchema(
                outcomes=[
                    "hateful",
                    "insults",
                    "sexual",
                    "physical_violence",
                    "self_harm",
                    "all_other_misconduct",
                ]
            ),
            "reasoning": None,  # free-form text
        },
        prompt_columns=None,
        response_columns=["text"],
        answering_method="structured",
        structured_outputs_fallback=True,
    )
    return task


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
    """Score judges against humans on the native multi-label task."""
    evaluator = MetaEvaluator(project_dir="project_dir", load=True)

    alt_test_scorer = AltTestScorer(multiplicative_epsilon=True)
    alt_test_scorer.min_instances_per_human = (
        10  # Just for testing as this example does not have enough data
    )
    # Macro-average across the six labels (the recommended explicit setting for a
    # multi-label task). Use average="samples" for per-item set overlap instead.
    classification_scorer = ClassificationScorer(metric="f1", average="macro")

    config = MetricsConfig(
        metrics=[
            MetricConfig(
                scorer=classification_scorer,
                task_names=["harm_categories"],
                task_strategy="single",
            ),
            MetricConfig(
                scorer=alt_test_scorer,
                task_names=["harm_categories"],
                task_strategy="single",
            ),
        ]
    )

    evaluator.add_metrics_config(config)
    evaluator.compare_async()

    evaluator.score_report.save("score_report.html", format="html")  # Save HTML report
    evaluator.score_report.save("score_report.csv", format="csv")  # Save CSV report
    evaluator.score_report.print()  # Print to console


if __name__ == "__main__":
    main()
