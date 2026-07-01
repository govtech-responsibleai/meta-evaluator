## 1. Annotation guide

- [x] 1.1 In `docs/annotation_guide/annotation.md`, add a short "Multi-label tasks"
  note: renders as a checkbox group (one box per outcome), distinct from radio
  single-select; link to `guides/evaltask.md` for the schema detail.
- [x] 1.2 Document that the number-key shortcut toggles one slot (does not replace the
  selection) and that selecting nothing is stored as an all-`"FALSE"` vector.

## 2. Results guide

- [x] 2.1 In `docs/guides/results.md`, document that a multi-label outcome column holds
  a fixed-length ordered `list[str]` vector with slot order preserved.
- [x] 2.2 Document serialization: parquet/JSON store the list natively; CSV JSON-encodes
  each cell on write and decodes on load via the persisted schema.

## 3. Judge-run guide

- [x] 3.1 In `docs/guides/judges_run.md`, document that multi-label judges emit the full
  ordered vector and support only `structured`/`instructor` (XML rejected and excluded
  from fallback).
- [x] 3.2 Document that consistency runs aggregate independently per slot (most votes
  wins, ties by first occurrence).

## 4. Scoring guide: reframe task_strategy + deprecate legacy melt

- [x] 4.1 In `docs/guides/scoring.md` "Task Configuration Types", reframe `task_strategy`
  as an **aggregation** choice: `single` scores one task, `multitask` scores several
  tasks then averages — independent of a task's value shape. Adjust the intro/`!!!`
  boxes accordingly.
- [x] 4.2 Mark the "Multi-Label (Combined Classification)" tab and `task_strategy="multilabel"`
  as **deprecated** with a migration note: declare a native `MultiLabelSchema` task
  (scored with `task_strategy="single"`) instead of melting N single-select tasks. State
  it still works today and removal is planned for a future release (do not claim it is
  already removed).
- [x] 4.3 Confirm/complete the native scoring path text: `ClassificationScorer` with
  `average="macro"` (or `"samples"`), `"binary"` rejected, `AltTestScorer` on the native
  name vector, `CohensKappaScorer` raises for multi-label. Link to `evaltask.md`.
- [x] 4.4 Sweep `scoring.md` for the stray `task_strategy="multilabel"` examples (e.g. the
  `TextSimilarityScorer` "all tasks combined into one multilabel" snippet) and update or
  annotate them so they no longer present the melt as the recommended path.

## 5. Tutorial pointer

- [x] 5.1 In `docs/tutorial.md`, add a brief mention of the multi-label task type with a
  link to `guides/evaltask.md`.

## 6. Root README

- [x] 6.1 In `README.md` "Define Task" (~line 75), note the multi-label option via
  `MultiLabelSchema(outcomes=[...])` alongside single-select and free-form, and link to
  `docs/guides/evaltask.md`.
- [x] 6.2 In `README.md` "Available Metrics" (~line 244), note that Cohen's Kappa is
  unsupported for multi-label tasks and point to `ClassificationScorer`/`AltTestScorer`.

## 7. Verify

- [x] 7.1 Ensure all added examples use `MultiLabelSchema(outcomes=[...])`, the `"FALSE"`
  sentinel, and `average="macro"`, matching the shipped API.
- [x] 7.2 Build the docs (`uv run mkdocs build`) and confirm no errors and no broken
  internal links.
