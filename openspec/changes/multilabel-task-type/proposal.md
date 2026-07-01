## Why

Some evaluation tasks require picking **several** labels at once for a single field
(e.g. a response is both "Clarify" and "Refuse"). meta-evaluator has no native
"pick several" field type — multi-label is faked as N independent binary tasks (the
RabakBench example uses six: `hateful`, `insults`, `sexual`, ...), which produces an
ugly N-widget annotation UI and loses the field boundary when more than one multi-label
field exists. The motivating use case has three multi-label fields, so the fake approach
is untenable.

## What Changes

- Add a native **multi-label task type**: a task whose value is a **fixed-length, ordered
  vector** with one slot per declared outcome. Each slot holds either the outcome's own
  name (selected) or the sentinel `"FALSE"` (not selected). This bundles the existing
  N-binary pattern into one task / one widget / one column, carried end to end
  (annotation UI → judge → storage → export → scoring) with the field boundary intact.
  The exact string `"FALSE"` is reserved and cannot be declared as an outcome of a
  multi-label task, because selected and unselected would otherwise be indistinguishable.
- Add an exported `MultiLabelSchema(outcomes: list[str])` wrapper. `EvalTask.task_schemas`
  widens to `dict[str, list[str] | MultiLabelSchema | None]`: existing `list[str]`
  remains single-select, `None` remains free-form, and the wrapper marks pick-several
  without a parallel side-set. A wrapped task's generated judge field becomes
  `list[Literal[*outcomes, "FALSE"]]`, constrained to the schema length with a validator
  that enforces slot `i` ∈ {outcome `i`, `"FALSE"`}.
- The judge emits the **full ordered vector** (every slot explicit); the annotation
  checkbox group produces the identical shape.
- Annotation frontend gains a checkbox-group field branch (new `ui/checkbox`).
- **BREAKING**: outcome value type widens from `str` to `str | list[str]` across the
  annotation API, results storage, and serialization.
- Score multi-label tasks by **reusing `ClassificationScorer`** via the existing
  `task_strategy="single"` path: binarize the vector **positionally** (slot `i` → 1 if it
  holds outcome `i`'s name, 0 if `"FALSE"`) into a per-slot indicator vector, then score
  with `average="macro"` (per-label, the recommended explicit setting) or `"samples"`
  (per-item overlap for a standalone multi-label task). The binarize collapses the
  length-N name-vector to ONE length-N indicator
  vector (it does **not** one-hot-expand each slot). Binarize is applied at the scorer call
  site, **gated to `ClassificationScorer`** (not in the scorer-agnostic preprocessing step,
  which would force every scorer onto the indicator form). The scorer's `average` is
  validated to be `samples`/`macro` for multi-label — the existing global default
  `binary` is rejected
  loudly (it would otherwise fail deep in sklearn and be swallowed to NaN). No new scorer;
  no Jaccard/empty-set machinery. `AltTest` scores the native name vector **unchanged** — it
  is NOT binarized (its `jaccard_similarity` auto-routes on list-valued labels).
- **`CohensKappaScorer` raises a clear error for multi-label (vector) tasks.** κ's
  chance-correction has no valid averaging axis over a sparse multi-label vector (flatten
  mixes incompatible per-label marginals; per-label collapses to `0`/`NaN` on absent
  labels; per-item is undefined for κ). Single-label κ is **untouched**. Krippendorff's α
  (the general N-rater alternative) is explicitly **out of scope** — see design Open
  Questions; it warrants its own change.
- **Scoring strategy semantics** (`task_strategy`, kept as-is — the legacy N-column melt
  needs it): a native multi-label task is **one column**, so it is scored only with
  `task_strategy="single"` (1 value, multi-label metric). The existing 1-vs-N task-count
  validator (`scores/metrics_config.py`) already rejects `multilabel`/`multitask` on a single
  task name, so a native task cannot accidentally route into the legacy melt. A native
  multi-label task may also participate in `multitask` **alongside** single-class tasks
  (e.g. `[SING, MULT]`, `multitask` → mean of each task's score); binarize fires per task
  keyed on whether `task_schemas[task]` is a `MultiLabelSchema`, so it engages for the
  MULT column and skips the SING column automatically.
  A mixed single-class + multi-label `multitask` configuration MUST use `average="macro"`,
  the averaging mode supported by both scalar multiclass and multi-label indicator input;
  `"samples"` remains valid only when every scored task is native multi-label.
- Multi-label is supported only by `structured`/`instructor` answering methods. Direct
  `answering_method="xml"` raises a clear error, and `structured_outputs_fallback` excludes
  `xml` for multi-label tasks so a failed JSON-based attempt cannot silently route the
  ordered vector through XML's scalar-only path. XML multi-label support is out of scope
  for this change.
- Judge consistency runs aggregate native multi-label results per slot. Each slot uses
  the existing consistency rule: most votes wins, with ties resolved by first occurrence.
- **Out of scope (possible fast-follow): slot/subset projection.** Selecting specific
  slots of the vector (e.g. score only `[hateful, insults]`, or a single-slot κ) is judged
  rare and deferred; native multi-label is scored as a whole vector. Adding a `labels`
  selector to `MetricConfig` later is additive and does not require redesign.

## Capabilities

### New Capabilities
- `multilabel-tasks`: defining, annotating, judging, storing, exporting, and scoring
  tasks whose value is a fixed-length ordered vector with one slot per declared outcome.

### Modified Capabilities
<!-- No existing OpenSpec specs in this repo; behavior is captured as a new capability. -->

## Impact

- `eval_task/`: new exported `MultiLabelSchema`, widened `task_schemas`, validation,
  `create_task_class` (typed vector + length/slot validator), serialize/deserialize, and
  **outcome-order stability** through the round-trip. Existing list/`None` inputs and
  their serialized shapes remain unchanged.
- `eval_task/eval_task.py` + `judge/judge.py`: JSON extraction passes the vector through;
  direct XML use is rejected and XML is removed from the fallback sequence for multi-label
  tasks.
- `annotator/api/` (`schemas.py`, `state.py`, `routes/task.py`): widened outcome type;
  task config serializes wrapped schemas as `{ "outcomes": [...] }`.
- `results/` (`human_results.py`, `judge_results.py`, `serialization.py`): widened schema
  metadata retains `MultiLabelSchema`, fixed-length list-valued label columns, and
  CSV/parquet/JSON round-trip. Standalone CSV loading identifies columns to decode from
  the persisted wrapped schema rather than requiring a separate `EvalTask`.
- `annotator/frontend/src/` (`TaskPanel.tsx`, new `ui/checkbox.tsx`, `lib/api.ts`):
  checkbox-group rendering, `string | string[]` state.
- `meta_evaluator/scoring.py` + `scores/`: a binarize step at the scorer call site, gated to
  `ClassificationScorer` (new transform; this exact path is not exercised today), plus
  `average ∈ {samples,macro}` validation for multi-label tasks. `AltTest` path untouched.
- Downstream: the platform repo consumes this via a separate, later change.
