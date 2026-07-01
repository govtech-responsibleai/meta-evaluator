"""Tests for multi-label judge emission and consistency aggregation (Section 2)."""

import pytest

from meta_evaluator.eval_task import EvalTask, MultiLabelSchema
from meta_evaluator.eval_task.exceptions import TaskSchemaError


@pytest.fixture
def multilabel_eval_task() -> EvalTask:
    """A judge-style EvalTask with one multi-label task.

    Returns:
        EvalTask: The configured task.
    """
    return EvalTask(
        task_schemas={
            "response_category": MultiLabelSchema(
                outcomes=["Clarify", "Refuse", "Support"]
            )
        },
        prompt_columns=["prompt"],
        response_columns=["response"],
        answering_method="structured",
    )


class TestMultiLabelJudgeEmission:
    """The generated judge model carries the full ordered vector."""

    def test_structured_returns_full_vector(self, multilabel_eval_task):
        """A structured judge model returns a full ordered vector."""
        cls = multilabel_eval_task.create_task_class()
        record = cls(response_category=["FALSE", "Refuse", "FALSE"])
        assert record.model_dump()["response_category"] == ["FALSE", "Refuse", "FALSE"]

    def test_wrong_length_rejected_by_model(self, multilabel_eval_task):
        """A wrong-length vector is rejected by the judge model."""
        cls = multilabel_eval_task.create_task_class()
        with pytest.raises(Exception):
            cls(response_category=["Clarify"])

    def test_out_of_slot_rejected_by_model(self, multilabel_eval_task):
        """An out-of-slot value is rejected by the judge model."""
        cls = multilabel_eval_task.create_task_class()
        with pytest.raises(Exception):
            cls(response_category=["Refuse", "FALSE", "FALSE"])


class TestMultiLabelXmlExclusion:
    """XML is excluded for multi-label tasks (direct and fallback)."""

    def test_direct_xml_rejected(self):
        """Direct answering_method='xml' with a multi-label task is rejected."""
        with pytest.raises(TaskSchemaError, match="structured.*instructor|xml"):
            EvalTask(
                task_schemas={
                    "harm": MultiLabelSchema(outcomes=["hateful", "insults"])
                },
                response_columns=["response"],
                answering_method="xml",
            )

    def test_fallback_excludes_xml(self):
        """structured/instructor fallback never includes xml for multi-label."""
        for method in ("structured", "instructor"):
            task = EvalTask(
                task_schemas={
                    "harm": MultiLabelSchema(outcomes=["hateful", "insults"])
                },
                response_columns=["response"],
                answering_method=method,
                structured_outputs_fallback=True,
            )
            sequence = task.get_fallback_sequence()
            assert "xml" not in sequence
            assert set(sequence) <= {"structured", "instructor"}


class TestMultiLabelConsistencyAggregation:
    """Consistency runs vote per slot in canonical order."""

    def _evaluator(self, tmp_path, schema):
        from meta_evaluator import MetaEvaluator

        evaluator = MetaEvaluator(project_dir=str(tmp_path), load=False)
        evaluator.add_eval_task(
            EvalTask(
                task_schemas={"response_category": schema},
                prompt_columns=["prompt"],
                response_columns=["response"],
                answering_method="structured",
            )
        )
        return evaluator

    def test_per_slot_majority_vote(self, tmp_path):
        """Three runs aggregate per slot, in canonical schema order."""
        evaluator = self._evaluator(
            tmp_path, MultiLabelSchema(outcomes=["Clarify", "Refuse"])
        )
        outcomes_list = [
            {"response_category": ["Clarify", "FALSE"]},
            {"response_category": ["FALSE", "Refuse"]},
            {"response_category": ["Clarify", "Refuse"]},
        ]
        aggregated = evaluator._aggregate_outcomes(outcomes_list)
        # Slot 0: Clarify, FALSE, Clarify -> Clarify. Slot 1: FALSE, Refuse, Refuse -> Refuse.
        assert aggregated["response_category"] == ["Clarify", "Refuse"]

    def test_tie_breaks_to_first_occurrence(self, tmp_path):
        """A per-slot tie resolves to the first-occurring value."""
        evaluator = self._evaluator(
            tmp_path, MultiLabelSchema(outcomes=["Clarify", "Refuse"])
        )
        outcomes_list = [
            {"response_category": ["Clarify", "Refuse"]},
            {"response_category": ["FALSE", "FALSE"]},
        ]
        aggregated = evaluator._aggregate_outcomes(outcomes_list)
        # Each slot is a 1-1 tie; first occurrence wins both slots.
        assert aggregated["response_category"] == ["Clarify", "Refuse"]

    def test_single_select_unchanged(self, tmp_path):
        """Single-select aggregation is unchanged by the multi-label branch."""
        evaluator = self._evaluator(tmp_path, ["positive", "negative"])
        outcomes_list = [
            {"response_category": "positive"},
            {"response_category": "positive"},
            {"response_category": "negative"},
        ]
        aggregated = evaluator._aggregate_outcomes(outcomes_list)
        assert aggregated["response_category"] == "positive"
