# Preserve `required_tasks` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve an `EvalTask`'s explicitly configured `required_tasks` across serialization and deserialization.

**Architecture:** Keep `EvalTaskState` as the state boundary and populate its existing `required_tasks` field from `EvalTask.serialize()`. Restore that exact field in `EvalTask.deserialize()`; older states remain compatible through the state model's existing `None` default.

**Tech Stack:** Python 3.13, Pydantic 2, pytest, Ruff, Pyright

---

### Task 1: Add the round-trip regression and fix serialization

**Files:**
- Create: `tests/eval_task/test_eval_task_serialization.py`
- Modify: `src/meta_evaluator/eval_task/eval_task.py:354-392`

- [ ] **Step 1: Write the failing regression test**

```python
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
```

- [ ] **Step 2: Run the focused test and verify the expected failure**

Run:

```bash
uv run pytest tests/eval_task/test_eval_task_serialization.py -v
```

Expected: FAIL because `state.required_tasks` is `None`.

- [ ] **Step 3: Implement the minimal round-trip fix**

Add the existing field to `EvalTask.serialize()`:

```python
return EvalTaskState(
    task_schemas=self.task_schemas,
    required_tasks=self.required_tasks,
    prompt_columns=self.prompt_columns,
    response_columns=self.response_columns,
    answering_method=self.answering_method,
    structured_outputs_fallback=self.structured_outputs_fallback,
    annotation_prompt=self.annotation_prompt,
)
```

Restore it in `EvalTask.deserialize()`:

```python
return cls(
    task_schemas=state.task_schemas,
    required_tasks=state.required_tasks,
    prompt_columns=state.prompt_columns,
    response_columns=state.response_columns,
    answering_method=state.answering_method,
    structured_outputs_fallback=getattr(
        state, "structured_outputs_fallback", False
    ),
    annotation_prompt=getattr(
        state, "annotation_prompt", "Please evaluate the following response:"
    ),
)
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
uv run pytest tests/eval_task/test_eval_task_serialization.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Run repository verification in the required order**

```bash
uv tool run ruff check --preview --fix
uv tool run ruff format .
uv sync --extra all
uv run pyright
uv run pytest -m "not integration"
uv tool run ruff format .
```

Expected: Ruff reports no errors, Pyright reports zero errors, all selected tests pass, and the final formatter reports no remaining changes.

- [ ] **Step 6: Commit the implementation**

```bash
git add tests/eval_task/test_eval_task_serialization.py src/meta_evaluator/eval_task/eval_task.py
git commit -m "fix: preserve required tasks in saved state"
```
