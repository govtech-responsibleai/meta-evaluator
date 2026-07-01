"""Tests for multi-label scoring (Section 6).

Covers the binarize transform (positional, collapse, fail-loud), the average
validation, ClassificationScorer behaviour on indicator vectors, the Cohen's
kappa rejection, and AltTest keeping the name vector.
"""

import asyncio

import polars as pl
import pytest

from meta_evaluator.eval_task import EvalTask, MultiLabelSchema
from meta_evaluator.meta_evaluator import MetaEvaluator
from meta_evaluator.meta_evaluator.scoring import binarize_multilabel_vector
from meta_evaluator.scores.enums import TaskAggregationMode
from meta_evaluator.scores.exceptions import MultiLabelScoringError
from meta_evaluator.scores.metrics import (
    AltTestScorer,
    ClassificationScorer,
    CohensKappaScorer,
)

OUTCOMES = ["Clarify", "Refuse", "Support"]


class TestBinarizeFunction:
    """The positional binarize transform (assert at function level)."""

    def test_collapse_not_expand(self):
        """A name-vector collapses to ONE indicator vector, not one-hot expansion."""
        assert binarize_multilabel_vector(
            ["Clarify", "FALSE", "Support"], OUTCOMES
        ) == [1, 0, 1]

    def test_all_false_is_all_zero(self):
        """The all-FALSE vector binarizes to all-zero."""
        assert binarize_multilabel_vector(["FALSE", "FALSE", "FALSE"], OUTCOMES) == [
            0,
            0,
            0,
        ]

    def test_positional_foreign_value_fails_loud(self):
        """A foreign slot value raises (NOT silently counted as selected)."""
        with pytest.raises(MultiLabelScoringError, match="foreign"):
            binarize_multilabel_vector(["Clarify", "Nope", "FALSE"], OUTCOMES)

    def test_positional_wrong_slot_fails_loud(self):
        """A valid outcome name in the wrong slot raises (positional, not != FALSE)."""
        # "Clarify" is slot 0's name; in slot 1 it is foreign.
        with pytest.raises(MultiLabelScoringError):
            binarize_multilabel_vector(["FALSE", "Clarify", "FALSE"], OUTCOMES)

    def test_wrong_length_fails_loud(self):
        """A wrong-length vector raises."""
        with pytest.raises(MultiLabelScoringError, match="length"):
            binarize_multilabel_vector(["Clarify", "FALSE"], OUTCOMES)


class TestSklearnRequiresBinarize:
    """The name-vector cannot be scored directly; binarize is required."""

    def test_name_vector_raises_in_sklearn(self):
        """Un-binarized name-vectors fail sklearn (multiclass-multioutput)."""
        from sklearn.metrics import f1_score

        expert = [["Clarify", "FALSE", "Support"]]
        judge = [["Clarify", "FALSE", "FALSE"]]
        with pytest.raises(ValueError):
            f1_score(expert, judge, average="samples")

    def test_worked_example_values(self):
        """Binarized indicator vectors give the documented samples/macro values."""
        from sklearn.metrics import f1_score

        expert = [["Clarify", "FALSE", "Support"], ["FALSE", "Refuse", "FALSE"]]
        judge = [["Clarify", "FALSE", "FALSE"], ["FALSE", "Refuse", "FALSE"]]
        eb = [binarize_multilabel_vector(v, OUTCOMES) for v in expert]
        jb = [binarize_multilabel_vector(v, OUTCOMES) for v in judge]
        assert round(float(f1_score(eb, jb, average="samples")), 3) == 0.833
        assert round(float(f1_score(eb, jb, average="macro")), 3) == 0.667


def _build_evaluator(tmp_path, schema, scorer, task_strategy, task_names=None):
    """Build a MetaEvaluator with one multi-label task and a metrics config.

    Returns:
        MetaEvaluator: The configured evaluator.
    """
    from meta_evaluator.scores import MetricConfig, MetricsConfig

    evaluator = MetaEvaluator(project_dir=str(tmp_path), load=False)
    evaluator.add_eval_task(
        EvalTask(
            task_schemas=schema,
            prompt_columns=["prompt"],
            response_columns=["response"],
            answering_method="structured",
        )
    )
    evaluator.add_metrics_config(
        MetricsConfig(
            metrics=[
                MetricConfig(
                    scorer=scorer,
                    task_names=task_names or ["harm"],
                    task_strategy=task_strategy,
                )
            ]
        )
    )
    return evaluator


class TestAverageValidation:
    """Multi-label averaged metrics reject average='binary'; accuracy ignores it."""

    def test_accuracy_ignores_default_binary_average(self, tmp_path):
        """Accuracy accepts its default average because accuracy does not use it."""
        evaluator = _build_evaluator(
            tmp_path,
            {"harm": MultiLabelSchema(outcomes=["a", "b"])},
            ClassificationScorer(metric="accuracy"),
            "single",
        )
        evaluator._validate_multilabel_metrics()  # no raise

    def test_binary_average_rejected(self, tmp_path):
        """The default average='binary' is rejected up front for multi-label."""
        evaluator = _build_evaluator(
            tmp_path,
            {"harm": MultiLabelSchema(outcomes=["a", "b"])},
            ClassificationScorer(metric="f1", average="binary"),
            "single",
        )
        with pytest.raises(MultiLabelScoringError, match="binary"):
            evaluator._validate_multilabel_metrics()

    def test_macro_average_accepted(self, tmp_path):
        """average='macro' passes validation for multi-label."""
        evaluator = _build_evaluator(
            tmp_path,
            {"harm": MultiLabelSchema(outcomes=["a", "b"])},
            ClassificationScorer(metric="f1", average="macro"),
            "single",
        )
        evaluator._validate_multilabel_metrics()  # no raise

    def test_samples_average_accepted_for_pure_multilabel(self, tmp_path):
        """average='samples' is allowed when the only task is multi-label."""
        evaluator = _build_evaluator(
            tmp_path,
            {"harm": MultiLabelSchema(outcomes=["a", "b"])},
            ClassificationScorer(metric="f1", average="samples"),
            "single",
        )
        evaluator._validate_multilabel_metrics()  # no raise

    def test_samples_rejected_for_mixed_multitask(self, tmp_path):
        """A mixed scalar + multi-label multitask rejects average='samples'."""
        evaluator = _build_evaluator(
            tmp_path,
            {
                "harm": MultiLabelSchema(outcomes=["a", "b"]),
                "toxicity": ["toxic", "non_toxic"],
            },
            ClassificationScorer(metric="f1", average="samples"),
            "multitask",
            task_names=["toxicity", "harm"],
        )
        with pytest.raises(MultiLabelScoringError, match="macro"):
            evaluator._validate_multilabel_metrics()


class TestBinarizeLabelColumn:
    """The label-column binarize helper, gated to ClassificationScorer."""

    def test_binarize_label_column(self, tmp_path):
        """The label column is binarized positionally; nulls pass through."""
        evaluator = _build_evaluator(
            tmp_path,
            {"harm": MultiLabelSchema(outcomes=OUTCOMES)},
            ClassificationScorer(metric="f1", average="macro"),
            "single",
        )
        df = pl.DataFrame(
            {
                "original_id": ["s1", "s2"],
                "label": [["Clarify", "FALSE", "Support"], None],
            }
        )
        out = evaluator._binarize_label_column(df, OUTCOMES)
        labels = out["label"].to_list()
        assert labels[0] == [1, 0, 1]
        assert labels[1] is None


class TestClassificationScorerOnIndicators:
    """ClassificationScorer scores binarized indicator vectors."""

    def test_samples_score_on_indicators(self):
        """Per-item samples F1 over indicator vectors matches the worked example."""
        scorer = ClassificationScorer(metric="f1", average="samples")
        judge = pl.DataFrame(
            {
                "original_id": ["s1", "s2"],
                "label": [[1, 0, 0], [0, 1, 0]],
            }
        )
        human = pl.DataFrame(
            {
                "original_id": ["s1", "s2"],
                "human_id": ["h1", "h1"],
                "label": [[1, 0, 1], [0, 1, 0]],
            }
        )
        result = asyncio.run(
            scorer.compute_score_async(
                judge, human, "harm", "judge1", TaskAggregationMode.SINGLE
            )
        )
        assert round(result.scores["f1"], 3) == 0.833

    def test_all_zero_score_is_zero(self):
        """An all-zero expert and judge indicator scores 0 for f1."""
        scorer = ClassificationScorer(metric="f1", average="samples")
        judge = pl.DataFrame({"original_id": ["s1"], "label": [[0, 0, 0]]})
        human = pl.DataFrame(
            {"original_id": ["s1"], "human_id": ["h1"], "label": [[0, 0, 0]]}
        )
        result = asyncio.run(
            scorer.compute_score_async(
                judge, human, "harm", "judge1", TaskAggregationMode.SINGLE
            )
        )
        assert result.scores["f1"] == 0.0


class TestCohensKappaRejectsMultiLabel:
    """Cohen's kappa errors on multi-label but works on single-label."""

    def test_kappa_errors_on_multilabel(self):
        """A list-valued (vector) task raises a clear error."""
        scorer = CohensKappaScorer()
        judge = pl.DataFrame({"original_id": ["s1"], "label": [["Clarify", "FALSE"]]})
        human = pl.DataFrame(
            {"original_id": ["s1"], "human_id": ["h1"], "label": [["Clarify", "FALSE"]]}
        )
        with pytest.raises(MultiLabelScoringError, match="multi-label"):
            asyncio.run(
                scorer.compute_score_async(
                    judge, human, "harm", "judge1", TaskAggregationMode.SINGLE
                )
            )

    def test_kappa_works_on_single_label(self):
        """A scalar single-label task still computes kappa."""
        scorer = CohensKappaScorer()
        judge = pl.DataFrame(
            {"original_id": ["s1", "s2", "s3"], "label": ["a", "b", "a"]}
        )
        human = pl.DataFrame(
            {
                "original_id": ["s1", "s2", "s3"],
                "human_id": ["h1", "h1", "h1"],
                "label": ["a", "b", "b"],
            }
        )
        result = asyncio.run(
            scorer.compute_score_async(
                judge, human, "sentiment", "judge1", TaskAggregationMode.SINGLE
            )
        )
        assert "kappa" in result.scores


class TestAltTestKeepsNameVector:
    """AltTest routes the native name vector to jaccard (not binarized)."""

    def test_alt_test_routes_list_to_jaccard(self):
        """A list value is detected as jaccard scoring, on name vectors."""
        scorer = AltTestScorer()
        assert (
            scorer._determine_scoring_function(["Clarify", "FALSE", "Support"])
            == "jaccard_similarity"
        )


def _mock_results(cls, evaluator_id_attr, evaluator_id, df, has_human_id=False):
    """Build a Mock results object exposing get_successful_results().

    Returns:
        Mock: A mock results object with the given data.
    """
    from unittest.mock import Mock

    obj = Mock(spec=cls)
    setattr(obj, evaluator_id_attr, evaluator_id)
    obj.run_id = f"run_{evaluator_id}"
    obj.get_successful_results.return_value = df
    obj.results_data = df
    return obj


class TestEndToEndSingleMultiLabel:
    """A native multi-label task scored end-to-end through SINGLE binarizes."""

    def _evaluator_with_task(self, tmp_path, schema):
        from tests.meta_evaluator.conftest import MockMetaEvaluator

        evaluator = MockMetaEvaluator(scores_dir=str(tmp_path / "scores"))
        evaluator.eval_task = EvalTask(
            task_schemas=schema,
            prompt_columns=["prompt"],
            response_columns=["response"],
            answering_method="structured",
        )
        return evaluator

    def test_single_multilabel_binarizes_through_compare(self, tmp_path):
        """The native vector rides SINGLE, binarizes, and yields the worked score."""
        from meta_evaluator.results import HumanAnnotationResults, JudgeResults
        from meta_evaluator.scores import MetricConfig, MetricsConfig

        evaluator = self._evaluator_with_task(
            tmp_path, {"harm": MultiLabelSchema(outcomes=OUTCOMES)}
        )

        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2"],
                "harm": [["Clarify", "FALSE", "FALSE"], ["FALSE", "Refuse", "FALSE"]],
            }
        )
        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2"],
                "harm": [["Clarify", "FALSE", "Support"], ["FALSE", "Refuse", "FALSE"]],
            }
        )
        judge = _mock_results(JudgeResults, "judge_id", "judge_1", judge_df)
        human = _mock_results(
            HumanAnnotationResults, "annotator_id", "human_1", human_df
        )

        evaluator.metrics_config = MetricsConfig(
            metrics=[
                MetricConfig(
                    scorer=ClassificationScorer(metric="f1", average="samples"),
                    task_names=["harm"],
                    task_strategy="single",
                )
            ]
        )
        results = asyncio.run(
            evaluator._compare_async(
                judge_results={"run_1": judge}, human_results={"run_1": human}
            )
        )
        _config, scoring_results = list(results.values())[0]
        assert round(scoring_results[0].scores["f1"], 3) == 0.833

    def test_mixed_multitask_macro_averages(self, tmp_path):
        """A [single_class, multilabel] multitask with macro averages both tasks."""
        from meta_evaluator.results import HumanAnnotationResults, JudgeResults
        from meta_evaluator.scores import MetricConfig, MetricsConfig

        evaluator = self._evaluator_with_task(
            tmp_path,
            {
                "toxicity": ["toxic", "non_toxic"],
                "harm": MultiLabelSchema(outcomes=["a", "b"]),
            },
        )
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2"],
                "toxicity": ["toxic", "non_toxic"],
                "harm": [["a", "FALSE"], ["FALSE", "b"]],
            }
        )
        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2"],
                "toxicity": ["toxic", "non_toxic"],
                "harm": [["a", "FALSE"], ["FALSE", "b"]],
            }
        )
        judge = _mock_results(JudgeResults, "judge_id", "judge_1", judge_df)
        human = _mock_results(
            HumanAnnotationResults, "annotator_id", "human_1", human_df
        )

        evaluator.metrics_config = MetricsConfig(
            metrics=[
                MetricConfig(
                    scorer=ClassificationScorer(metric="f1", average="macro"),
                    task_names=["toxicity", "harm"],
                    task_strategy="multitask",
                )
            ]
        )
        results = asyncio.run(
            evaluator._compare_async(
                judge_results={"run_1": judge}, human_results={"run_1": human}
            )
        )
        _config, scoring_results = list(results.values())[0]
        # Perfect agreement on both tasks -> averaged score 1.0.
        assert round(scoring_results[0].scores["f1"], 3) == 1.0
