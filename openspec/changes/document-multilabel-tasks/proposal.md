## Why

The native multi-label task type shipped in v0.3.0 (`MultiLabelSchema`, checkbox
annotation, positional binarize scoring). Its user-facing documentation is
uneven: `docs/guides/evaltask.md` and `docs/guides/scoring.md` cover it, but the
results, judge-running, annotation, and tutorial guides do not mention it at all —
so a reader following those guides hits list-valued outcome columns, a checkbox
widget, and `ClassificationScorer(average="macro")` with no explanation. The
`scoring.md` guide also overloads the word "multi-label": it still frames
`task_strategy="multilabel"` (the legacy N-column melt) as *the* multi-label story,
which now collides with the native task type and misleads readers.

## What Changes

- **Annotation guide** (`annotation_guide/annotation.md`): document the checkbox-group
  field for multi-label tasks — how it renders, that number-key shortcuts toggle a
  slot (rather than replace), and that "nothing applies" is an all-`"FALSE"` vector.
- **Results guide** (`docs/guides/results.md`): explain that a multi-label task's
  outcome column holds a fixed-length ordered vector (`list[str]`), how it survives
  parquet/JSON round-trips natively, and that CSV JSON-encodes each cell (decoded on
  load via the persisted schema).
- **Judge-run guide** (`docs/guides/judges_run.md`): note that multi-label tasks emit
  the full ordered vector and are limited to `structured`/`instructor` answering
  methods (XML rejected, excluded from fallback), and how consistency runs aggregate
  per slot.
- **Scoring guide** (`docs/guides/scoring.md`): reframe `task_strategy` as purely an
  **aggregation** choice — how a scorer's results are combined *across the task_names
  in one MetricConfig* (`single` = one task, `multitask` = several tasks scored then
  averaged) — decoupled from the *task's own value shape*. Mark the legacy
  `task_strategy="multilabel"` melt as **deprecated** and add a migration note steering
  readers to declare a native `MultiLabelSchema` task (scored with
  `task_strategy="single"`) instead of melting N single-select tasks. The strategy is
  not removed from code by this change; removal is deferred to a later, breaking change.
- **Tutorial** (`docs/tutorial.md`): add a short pointer to the multi-label task type
  so first-time readers know it exists and where the detail lives.
- **Root README** (`README.md`): the "Define Task" example shows only single-select and
  free-form schemas, and "Available Metrics" lists Cohen's Kappa without noting it is
  unsupported for multi-label. Add a brief multi-label mention (with the
  `MultiLabelSchema` option and a link to the guide) so the first-read entry point is
  not silent on the feature.
- No changes to the two guides that already cover it well
  (`evaltask.md`, and the `scoring.md` native section) beyond the disambiguation callout.

This is a **documentation-only** change: no source, API, or behavior changes.

## Capabilities

### New Capabilities
- `multilabel-docs`: user-facing documentation coverage for the native multi-label
  task type across the annotation, results, judge-run, scoring, and tutorial guides,
  including disambiguation from the legacy `task_strategy="multilabel"` melt.

### Modified Capabilities
<!-- No existing OpenSpec specs capture doc behavior; this is a new docs capability. -->

## Impact

- `docs/guides/results.md`, `docs/guides/judges_run.md`, `docs/guides/scoring.md`,
  `docs/tutorial.md`, `docs/annotation_guide/annotation.md`, `README.md`.
- No code, dependency, or public-API impact. mkdocs build must still succeed.
- Aligns with the already-merged `multilabel-task-type` change (source of truth for
  behavior); this change only closes documentation gaps it left.
