"""Tests for EvalTask state serialization."""

from meta_evaluator.eval_task import EvalTask


def test_roundtrip_preserves_explicit_required_tasks():
    """An explicit required-task subset survives serialization."""
    task = EvalTask(
        task_schemas={
            "sentiment": ["positive", "negative"],
            "safety": ["safe", "unsafe"],
        },
        required_tasks=["safety"],
        response_columns=["response"],
        answering_method="structured",
    )

    state = task.serialize()
    restored = EvalTask.deserialize(state)

    assert state.required_tasks == ["safety"]
    assert restored.required_tasks == ["safety"]
