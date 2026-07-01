## Context

The native multi-label task type (`MultiLabelSchema`) shipped in v0.3.0. Behavior
is authoritative in the merged `multilabel-task-type` change and in the source. Two
guides already document it well (`docs/guides/evaltask.md`, and the native section of
`docs/guides/scoring.md`); the remaining guides do not mention it. This change is
purely additive documentation — no code moves.

The one non-trivial issue is a **terminology collision**: `scoring.md` predates the
native type and uses "multi-label" to mean `task_strategy="multilabel"` — the legacy
melt that combines *N separate single-select task columns* into one aligned vector at
scoring time. The native `MultiLabelSchema` is a *single* task/column scored via
`task_strategy="single"`. Both are legitimate and coexist in the current code; a reader
must not confuse them.

The intended long-term direction is that `task_strategy` means only **how a scorer's
results are aggregated across the `task_names` in one `MetricConfig`** — decoupled from
whether any individual task's *value* is multi-label. Under that framing the legacy
`"multilabel"` melt no longer has a place (a native `MultiLabelSchema` task carries its
own vector shape and is scored with `task_strategy="single"`). This change documents
that direction and **deprecates** the legacy melt in prose; it does **not** remove
`TaskAggregationMode.MULTILABEL` from code. Actual removal is a separate breaking change
(6 source + 3 test files) targeting a future major version.

## Goals / Non-Goals

**Goals:**
- Every guide a reader would follow for annotation, judging, results, and scoring
  mentions the native multi-label type where it is relevant, and links to the
  authoritative detail in `evaltask.md`.
- The legacy-melt vs. native-task distinction is stated explicitly wherever both
  could be read as "multi-label".
- Examples in docs match the shipped API (`MultiLabelSchema(outcomes=[...])`, the
  `"FALSE"` sentinel, `average="macro"`).

**Non-Goals:**
- No source, API, or behavior changes.
- No rewrite of the well-covered `evaltask.md` multi-label section (single source of
  truth; other guides link to it rather than duplicate).
- No new scorer, no change to the legacy `task_strategy="multilabel"` behavior — only
  clarifying prose.

## Decisions

- **`evaltask.md` stays the single source of truth for the task definition.** Other
  guides give a short, self-contained note for their own concern (a checkbox in
  annotation, a `list[str]` column in results, an ordered vector in judging) and link
  back, rather than re-explaining the sentinel/slot contract in five places.
  *Alternative considered:* duplicate the full explanation per guide — rejected as a
  drift hazard.
- **`task_strategy` is documented as an aggregation choice only.** Reframe the guide so
  `task_strategy` answers "how are this MetricConfig's task scores combined" (`single` =
  one task; `multitask` = several tasks scored then averaged), not "what shape is the
  task value". The native multi-label task's value shape lives in the `EvalTask` schema,
  and it is scored with `task_strategy="single"`. *Alternative:* keep the old framing and
  only add a disambiguation note — rejected; the framing itself is the root confusion.
- **Deprecate the legacy melt in prose, do not remove or rename it.** `scoring.md` keeps
  the `task_strategy="multilabel"` docs but marks them deprecated with a migration note
  (declare a `MultiLabelSchema` task instead). *Alternatives:* (a) remove the strategy
  now — rejected, that is a breaking code change out of scope here; (b) rename it —
  rejected, it would diverge from the shipped API. Removal is deferred to a separate
  breaking change, which is **not filed yet**, so the note names no target version.
- **Stop presenting the legacy melt as a recommended path.** Beyond the one deprecated
  reference tab, the native `MultiLabelSchema` task is the recommended way to score
  multi-label; incidental `task_strategy="multilabel"` examples elsewhere in the guide
  (e.g. the `TextSimilarityScorer` snippet) are rewritten off the melt or annotated as
  deprecated so a reader skimming for examples is not steered onto it.
- **Verify by mkdocs build**, since there is no test surface for docs. Broken internal
  links or malformed code fences are the realistic failure modes.

## Risks / Trade-offs

- [Docs drift from code as the feature evolves] → Keep detail centralized in
  `evaltask.md`; other guides link, minimizing places to update.
- [Terminology callout adds cognitive load in `scoring.md`] → Keep it a short admonition
  with one line each for legacy vs. native, plus a link.
- [Examples silently go stale] → Mirror the exact API already shown in `evaltask.md`
  and confirm imports resolve.

## Open Questions

- None blocking. A follow-up breaking change to remove `TaskAggregationMode.MULTILABEL`
  is anticipated but is deliberately **not filed yet**; the deprecation note therefore
  refers to "a future release" without naming a version.
