# Preserve `required_tasks` Across State Round Trips

## Problem

`EvalTaskState` defines `required_tasks`, but `EvalTask.serialize()` does not
populate it and `EvalTask.deserialize()` does not restore it. Saving an
`EvalTask` with an explicit subset therefore writes `null`, and loading the
state loses the configured subset.

## Design

Preserve the exact configured value. `EvalTask.serialize()` will pass
`self.required_tasks` to `EvalTaskState`, and `EvalTask.deserialize()` will pass
`state.required_tasks` to the reconstructed `EvalTask`.

This preserves the distinction between an explicit list and `None`. Existing
state files that omit `required_tasks` remain compatible because
`EvalTaskState` defaults the field to `None`.

## Testing

Add a regression test that constructs an `EvalTask` with an explicit subset,
serializes and deserializes it, and asserts that the subset is unchanged. Run
the focused test through a red-green cycle, followed by the repository's full
lint, formatting, type-checking, and non-integration test workflow.
