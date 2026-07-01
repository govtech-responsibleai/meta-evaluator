"""Tests for the multi-label task type on EvalTask (Section 1)."""

import pytest
from pydantic import ValidationError

from meta_evaluator.eval_task import EvalTask
from meta_evaluator.eval_task.eval_task import MultiLabelSchema
from meta_evaluator.eval_task.exceptions import TaskSchemaError
from meta_evaluator.eval_task.serialization import EvalTaskState


class TestMultiLabelSchemaDeclaration:
    """Declaring and validating MultiLabelSchema."""

    def test_valid_multilabel_declaration(self):
        """A valid multi-label task is accepted and exposes its outcomes in order."""
        task = EvalTask(
            task_schemas={
                "response_category": MultiLabelSchema(
                    outcomes=["Clarify", "Refuse", "Support"]
                )
            },
            response_columns=["response"],
            answering_method="structured",
        )
        schema = task.task_schemas["response_category"]
        assert isinstance(schema, MultiLabelSchema)
        assert schema.outcomes == ["Clarify", "Refuse", "Support"]

    def test_bare_list_and_none_unchanged(self):
        """Bare list stays single-select and None stays free-form."""
        task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative"], "summary": None},
            response_columns=["response"],
            answering_method="structured",
        )
        assert task.task_schemas["sentiment"] == ["positive", "negative"]
        assert task.task_schemas["summary"] is None

    def test_too_short_wrapped_outcomes_rejected(self):
        """A MultiLabelSchema with fewer than 2 outcomes is rejected."""
        with pytest.raises((TaskSchemaError, ValidationError)):
            MultiLabelSchema(outcomes=["only_one"])

    def test_reserved_false_outcome_rejected(self):
        """Declaring the reserved 'FALSE' sentinel as an outcome is rejected."""
        with pytest.raises((TaskSchemaError, ValidationError), match="reserved"):
            MultiLabelSchema(outcomes=["Clarify", "FALSE"])

    def test_xml_with_multilabel_rejected(self):
        """answering_method='xml' with a multi-label task is rejected."""
        with pytest.raises(TaskSchemaError, match="xml"):
            EvalTask(
                task_schemas={
                    "harm": MultiLabelSchema(outcomes=["hateful", "insults"])
                },
                response_columns=["response"],
                answering_method="xml",
            )

    def test_get_all_outcomes_includes_multilabel(self):
        """get_all_outcomes flattens MultiLabelSchema outcomes."""
        task = EvalTask(
            task_schemas={
                "harm": MultiLabelSchema(outcomes=["hateful", "insults"]),
                "sentiment": ["positive", "negative"],
            },
            response_columns=["response"],
            answering_method="structured",
        )
        assert set(task.get_all_outcomes()) == {
            "hateful",
            "insults",
            "positive",
            "negative",
        }

    def test_multilabel_task_is_required_by_default(self):
        """A multi-label task is treated as required (non-None schema)."""
        task = EvalTask(
            task_schemas={"harm": MultiLabelSchema(outcomes=["hateful", "insults"])},
            response_columns=["response"],
            answering_method="structured",
        )
        assert task.get_required_tasks() == ["harm"]


class TestMultiLabelFallbackSequence:
    """XML is excluded from the fallback sequence for multi-label tasks."""

    def test_fallback_excludes_xml_for_multilabel(self):
        """Structured fallback for a multi-label task omits xml."""
        task = EvalTask(
            task_schemas={"harm": MultiLabelSchema(outcomes=["hateful", "insults"])},
            response_columns=["response"],
            answering_method="structured",
            structured_outputs_fallback=True,
        )
        sequence = task.get_fallback_sequence()
        assert "xml" not in sequence
        assert sequence == ["structured", "instructor"]

    def test_fallback_keeps_xml_for_single_select(self):
        """A single-select task keeps xml in its fallback sequence."""
        task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative"]},
            response_columns=["response"],
            answering_method="structured",
            structured_outputs_fallback=True,
        )
        assert task.get_fallback_sequence() == ["structured", "instructor", "xml"]


class TestMultiLabelTaskClass:
    """The generated task class types and validates the ordered vector."""

    def _task_class(self):
        task = EvalTask(
            task_schemas={
                "response_category": MultiLabelSchema(
                    outcomes=["Clarify", "Refuse", "Support"]
                )
            },
            response_columns=["response"],
            answering_method="structured",
        )
        return task.create_task_class()

    def test_full_vector_accepted(self):
        """A full ordered vector with selected and FALSE slots is accepted."""
        cls = self._task_class()
        record = cls(response_category=["FALSE", "Refuse", "FALSE"])
        assert record.model_dump()["response_category"] == ["FALSE", "Refuse", "FALSE"]

    def test_all_false_accepted(self):
        """The all-FALSE vector (nothing applies) is valid."""
        cls = self._task_class()
        record = cls(response_category=["FALSE", "FALSE", "FALSE"])
        assert record.model_dump()["response_category"] == [
            "FALSE",
            "FALSE",
            "FALSE",
        ]

    def test_wrong_length_rejected(self):
        """A vector of the wrong length is rejected."""
        cls = self._task_class()
        with pytest.raises(ValidationError):
            cls(response_category=["FALSE", "Refuse"])

    def test_out_of_slot_value_rejected(self):
        """A correct outcome name in the wrong slot is rejected (positional)."""
        cls = self._task_class()
        # "Clarify" belongs to slot 0, not slot 1.
        with pytest.raises(ValidationError):
            cls(response_category=["FALSE", "Clarify", "FALSE"])

    def test_foreign_value_rejected(self):
        """A value that is neither an outcome name nor FALSE is rejected."""
        cls = self._task_class()
        with pytest.raises(ValidationError):
            cls(response_category=["FALSE", "Nope", "FALSE"])

    def test_single_and_freeform_unchanged(self):
        """Single-select Literal and free-form str fields are unchanged."""
        task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative"], "summary": None},
            response_columns=["response"],
            answering_method="structured",
        )
        cls = task.create_task_class()
        record = cls(sentiment="positive", summary="free text")
        dumped = record.model_dump()
        assert dumped["sentiment"] == "positive"
        assert dumped["summary"] == "free text"
        with pytest.raises(ValidationError):
            cls(sentiment="invalid", summary="x")


class TestMultiLabelSerialization:
    """Round-trip serialization preserves the wrapper and slot order."""

    def test_serialize_roundtrip_preserves_wrapper_and_order(self):
        """serialize/deserialize keep MultiLabelSchema identity and slot order."""
        task = EvalTask(
            task_schemas={
                "response_category": MultiLabelSchema(
                    outcomes=["Clarify", "Refuse", "Support"]
                ),
                "sentiment": ["positive", "negative"],
                "summary": None,
            },
            response_columns=["response"],
            answering_method="structured",
        )
        state = task.serialize()
        restored = EvalTask.deserialize(state)

        schema = restored.task_schemas["response_category"]
        assert isinstance(schema, MultiLabelSchema)
        assert schema.outcomes == ["Clarify", "Refuse", "Support"]
        assert restored.task_schemas["sentiment"] == ["positive", "negative"]
        assert restored.task_schemas["summary"] is None

    def test_json_roundtrip_discriminates_union(self):
        """A JSON round-trip discriminates wrapper vs bare list vs None."""
        state = EvalTaskState(
            task_schemas={
                "multi": MultiLabelSchema(outcomes=["a", "b", "c"]),
                "single": ["x", "y"],
                "free": None,
            },
            prompt_columns=None,
            response_columns=["response"],
            answering_method="structured",
        )
        loaded = EvalTaskState.model_validate_json(state.model_dump_json())
        assert isinstance(loaded.task_schemas["multi"], MultiLabelSchema)
        assert loaded.task_schemas["multi"].outcomes == ["a", "b", "c"]
        assert loaded.task_schemas["single"] == ["x", "y"]
        assert loaded.task_schemas["free"] is None
