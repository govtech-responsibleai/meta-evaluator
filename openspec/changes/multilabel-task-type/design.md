## Context

meta-evaluator models a task value as either a single outcome (`list[str]` schema) or
free-form text (`None` schema). Multi-label is currently faked as N independent binary
tasks (RabakBench: six harm-type columns) combined at scoring time by the `multilabel`
melt (`_create_multilabel_column`, `meta_evaluator/scoring.py:319`). This produces an ugly
N-widget annotation UI and loses the field boundary when more than one multi-label field
exists.

(All source paths below are relative to `src/meta_evaluator/`; line numbers are current as
of this change's authoring and may drift as the files are edited.)

Key existing-code facts that shape this design:
- The legacy `multilabel` melt builds a **fixed-length, position-aligned** vector (one
  slot per task, schema order). That alignment is why the existing per-position
  majority-vote in `ClassificationScorer`
  (`scores/metrics/classification/classification_scorer.py:155`) is valid.
- `task_strategy="single"` passes one task column through to the scorer untouched
  (`meta_evaluator/scoring.py:294`); `can_score_task` already accepts list values.
- Judge JSON extraction (`judge/judge.py:296`) passes values through unchanged — a list
  survives. XML extraction (`judge/judge.py:320`) hard-casts to `str`.
- RabakBench scores its faked multi-label only through `AltTest` / `CohensKappa`. It does
  **not** run `ClassificationScorer` with `average="samples"`/`"macro"`; that path is new.
- The annotation frontend is rebuildable; it has `radio-group` but no `checkbox`.

## Goals / Non-Goals

**Goals:**
- A native multi-label task whose value is a fixed-length ordered vector (one slot per
  declared outcome), carried end to end as ONE field / widget / column.
- Multiple independent multi-label fields without cross-field confusion.
- Scoring via the existing classification metrics, no new scorer.

**Non-Goals:**
- Per-persona / archetype scoring axis (separate future design).
- Structured severity (label+level pairs) — severity is flattened into outcome strings.
- Platform repo adaptation — a separate, later change.
- XML multi-cardinality support. Multi-label tasks use only `structured`/`instructor` in
  this change; direct XML use is rejected and XML fallback is disabled.
- A dedicated set-overlap (Jaccard) scorer — dropped; see Decisions.
- Cohen's κ for multi-label tasks — errored out; see Decisions.
- Krippendorff's α — the general N-rater agreement alternative. Out of scope; warrants its
  own change (the `krippendorff` lib handles the math, but fitting a *symmetric* rater
  reliability measure into this repo's *asymmetric* judge-vs-humans framing is a design
  question of its own, and α would also improve the single-label 2-rater κ path).
- Slot/subset projection (scoring only specific slots of the vector, e.g. `[hateful,
  insults]` only, or a single-slot κ). Judged rare; deferred as a possible fast-follow via
  a `MetricConfig.labels` selector (additive, no redesign).

## Decisions

**Represent the value as a fixed-length ordered vector, not a set.**
A multi-label task with outcomes `[A, B, C]` stores a length-3 list where slot `i` is
either outcome `i`'s name (selected) or the sentinel `"FALSE"` (not selected), e.g.
`["FALSE", "B", "FALSE"]`. Rationale: in the motivating domain, "not selected" is itself
meaningful information about each option — a set would erase it. This is precisely the
existing N-binary RabakBench pattern, bundled into one task/widget/column.
Consequence: there is no missing/undefined submission ("nothing applies" = a full
all-`FALSE` vector), and the alignment makes per-position majority-vote and classification
scoring valid without new math. After binarization, all-`FALSE` is the all-zero indicator;
for F1/precision/recall its score is explicitly `0`, including when both sides are all-zero.
**Outcome order becomes load-bearing** and must survive every
serialize/judge/storage round-trip (the schema list defines slot order).
Because `"FALSE"` is the not-selected sentinel, it is a reserved value for multi-label
schemas. Allowing an outcome literally named `"FALSE"` would encode both selected and
unselected as the same slot value, so `MultiLabelSchema` validation rejects it.

**Represent multi-label definitions with a dedicated wrapper, not a side-set.**
Add an exported Pydantic model:
```python
class MultiLabelSchema(BaseModel):
    outcomes: list[str]
```
and widen `EvalTask.task_schemas` to
`dict[str, list[str] | MultiLabelSchema | None]`. The existing meanings and input shapes
remain intact: a bare `list[str]` is single-select and `None` is free-form; only the new
wrapper is multi-label. Its serialized JSON shape is `{ "outcomes": [...] }`, which is
unambiguous from both an array and `null` while retaining outcome order.

Alternative considered: a parallel `multilabel_tasks: set[str]`. Rejected because it
duplicates type information outside the schema, permits the marker and schema to drift,
and does not naturally travel with standalone result metadata for CSV decoding. The
wrapper is still additive for existing callers but makes the schema self-describing at
every layer. Tuple was rejected because JSON serializes it as an indistinguishable array;
set/frozenset were rejected because outcome order is load-bearing.

**Judge emits the full ordered vector, typed as one list + validator.**
`create_task_class` types a multi-label field as
`list[Literal[*outcomes, "FALSE"]]`. A validator enforces (a) length == number of
outcomes and (b) slot `i` ∈ {outcome `i`, `"FALSE"`}. Alternative considered: per-slot
binary fields (`slot_0: Literal["A","FALSE"]`, ...), which makes misalignment
structurally impossible. Rejected: that is literally N binary fields again, nested under
one name, reintroducing the multi-field plumbing this change removes. One list + a
validator keeps a single clean column end-to-end and guarantees alignment by validation.

**Score by reusing `ClassificationScorer`, not a new scorer.**
Multi-label tasks score via `task_strategy="single"`. Because the value is already a
fixed-length aligned vector, scoring binarizes it (outcome name → 1, `"FALSE"` → 0) into
a per-slot indicator vector and feeds `ClassificationScorer` with `average="macro"`
(per-label, the recommended explicit setting) or `"samples"` (per-item set-overlap-like
for standalone multi-label scoring). This drops the previously planned
`SetOverlapScorer`, the empty-set convention question, and any new majority-vote logic —
the existing per-position vote already fits an aligned vector. The binarize transform is
the one genuinely new piece (this scoring path is not exercised by RabakBench today).

**Binarize is MANDATORY, POSITIONAL, and `ClassificationScorer`-ONLY (applied at the
scorer call site, NOT in shared preprocessing).**
Verified empirically against sklearn: feeding the name-vector
(`[["Clarify","FALSE","Support"], ...]`) to `f1_score(average="samples")` raises
`ValueError: multiclass-multioutput is not supported`; the binarized indicator matrix
(`[[1,0,1], ...]`, `type_of_target == "multilabel-indicator"`) scores cleanly
(`samples=0.833`, `macro=0.667` on the worked example). So binarize is not optional
plumbing — it is what makes the `ClassificationScorer` metric run at all.
- **Collapse, do not expand.** One length-N name-vector → one length-N indicator vector
  (`["Clarify","FALSE","Support"] → [1,0,1]`). Do NOT one-hot-expand a slot to
  `[1,0,0]/[0,1,0]/...`; that turns each slot into a separate multi-class problem and is
  wrong.
- **Positional, not `!= "FALSE"`.** Map slot `i` → 1 iff `vec[i] == outcomes[i]`, else 0.
  On validated judge output the `!= "FALSE"` shortcut coincides, but **imported external
  results** (`import_external_results`) need not have passed the judge validator; a typo'd
  or foreign slot value should fail loud (positional) rather than be silently counted as
  selected (`!= "FALSE"`).
- **Seam = the scorer call site, gated to `ClassificationScorer`, NOT the shared
  `_preprocess_task_data`.** `_preprocess_task_data` (`meta_evaluator/scoring.py:294`) is
  **scorer-agnostic** — it runs once (`scoring.py:592`) and its output feeds whichever
  scorer the config names. `self.eval_task` (hence
  `task_schemas[task].outcomes` → canonical slot order for a wrapped schema) IS reachable
  there, so the *data* binarize needs is in scope — but the *scorer
  identity* is not. Binarizing there would force **every** scorer onto the indicator form,
  which is wrong for `AltTest` (see next decision). So binarize is applied at the call site
  (`scoring.py:600`-ish, where `metric_config.scorer` is in scope), gated to
  `isinstance(scorer, ClassificationScorer)`, after `_preprocess_task_data` has produced the
  `label` column in canonical slot order. This keeps `ClassificationScorer` generic (it
  never learns the `"FALSE"` sentinel) and resolves an ordering hazard: the per-position
  majority vote (`scores/metrics/classification/classification_scorer.py:155`) and
  `_compute_metric` (`.../classification_scorer.py:76`, sklearn call at `:102`) both need the
  indicator form, so binarize must run before `compute_score_async` is called.
- **Fail-loud caveat.** The per-judge scoring path is wrapped in a broad
  `except Exception → NaN result` (`scoring.py:626`), and `_compute_metric` itself swallows
  `ValueError`/`ZeroDivisionError` to `nan` (`classification_scorer.py:180,202`). So a
  positional-binarize violation (foreign slot value) raised *inside* that region becomes a
  silent NaN, NOT a loud failure. The "fail loud" guarantee therefore requires the binarize
  validation to raise where it is visible — and the tests for it must assert at the
  binarize-function level, not end-to-end through `score()` (which would mask the raise as a
  per-judge NaN).

**`AltTest` keeps the NAME vector — it is NOT binarized.**
Verified empirically: `AltTest._jaccard_similarity` (`alt_test.py:239`) calls
`jaccard_score(y_true=ann, y_pred=pred, average="macro")` on one vector at a time. It runs
on BOTH the name vector (`type_of_target == "multiclass"`) and the binarized vector
(`type_of_target == "binary"`) without error, but the two produce **different scores in
general** (names macro-average over the full label set incl. `"FALSE"`; ints over `{0,1}`),
so binarizing is not a no-op. `AltTest` is *designed* to auto-detect a list value and route
to jaccard (`_determine_scoring_function`, `alt_test.py:204`), so it consumes the name
vector directly. Binarizing it would buy nothing and would break the "scores the vector
unchanged" contract. Decision: binarize is `ClassificationScorer`-only; `AltTest` scores the
native name vector unchanged.

**`ClassificationScorer.average` for a multi-label task MUST be `samples` or `macro`;
the existing global default `binary` is rejected loudly and `macro` is recommended.**
`ClassificationScorer.__init__` defaults `average="binary"`
(`classification_scorer.py:27`). `f1`/`precision`/`recall` with `average="binary"` on a
binarized indicator matrix raises in sklearn — and per the fail-loud caveat above that
raise would otherwise surface as a silent NaN. So a multi-label task scored with
`ClassificationScorer` SHALL be validated up front: `average ∈ {"samples", "macro"}`,
raising a clear error otherwise (naming the two valid choices), rather than letting the
default `binary` fail deep in the swallowed region. The scorer's global default is not
silently mutated based on task schema: callers choose explicitly. **The recommended
multi-label setting and the RabakBench reference example use `macro`**, because `macro`
works for both scalar multiclass and multi-label indicator input and therefore composes
with mixed `MULTITASK` configurations. `samples` remains available for standalone native
multi-label scoring. The two modes are not interchangeable: `samples` averages per ITEM
(each sample equal weight), while `macro` averages per LABEL/slot (each outcome equal
weight, so a rare harm type counts full weight). On the worked example they diverge
(`samples=0.833`, `macro=0.667`).

**Native multi-label rides `TaskAggregationMode.SINGLE`, NOT legacy `MULTILABEL`.**
Three similarly-named paths exist and must not be confused: `SINGLE` (one column, scalar —
the native vector rides here as one column whose *value* is a vector), `MULTILABEL` (the
legacy melt of N separate binary columns into one vector at scoring time — the fake
pattern this change supersedes), and `MULTITASK` (N columns scored separately then
averaged). Verified: a native vector column routed through `SINGLE` and N single-label
columns routed through legacy `MULTILABEL` produce the **identical** scorer `label` column
— *provided slot order matches* (native: schema order; legacy: `task_names` order). This
equivalence is the proof that reuse is sound, and it pinpoints the one thing that breaks
it: order drift. The existing 1-vs-N task-count validator (`scores/metrics_config.py:42-50`)
already forces a native task (one task name) into `single` and rejects `multilabel`/
`multitask` for it — so `task_strategy` stays in the API (legacy N-column paths need it)
but cannot be mis-set for a native task. A native MULT task may still appear in a
`[SING, MULT]` `multitask` config; each task is preprocessed independently through the
`SINGLE` path (binarize keyed on `isinstance(schema, MultiLabelSchema)`), then averaged.
Because one `ClassificationScorer` instance supplies one `average` value for every task
in the configuration, a mixed scalar + multi-label `MULTITASK` MUST use
`average="macro"`; sklearn does not accept `average="samples"` for scalar multiclass input.
`samples` is allowed only when every task in that configuration is native multi-label.

**Consistency runs vote per slot for native multi-label outcomes.**
The existing consistency aggregator counts complete scalar outcomes as dictionary keys;
a list-valued outcome is unhashable and cannot use that path. For `MultiLabelSchema`, each
slot is aggregated independently. The most frequent slot value wins, and a tie preserves
the existing consistency convention by choosing the first-occurring value. The aggregate
is reconstructed in canonical schema order and validated as a full vector.

**Cohen's κ is errored-out for multi-label vectors; single-label κ is untouched.**
`cohen_kappa_score` rejects both the name-vector (`multiclass-multioutput`) and the
binarized 2D matrix (`multilabel-indicator`) — verified. Beyond the array-shape issue, κ
has no statistically valid averaging axis for a sparse multi-label vector: FLATTEN (pool
all slot decisions, κ≈0.61) mixes incompatible per-label marginals into one chance
baseline; PER-LABEL (κ per slot then mean) collapses to `0.0` for a constant slot and
`NaN` for an all-`FALSE` slot (`nanmean` then silently drops labels); PER-ITEM is
undefined for κ (it needs a confusion matrix over a *set*). So `CohensKappaScorer`
SHALL raise a clear error when handed a multi-label (list-valued) task, naming
`ClassificationScorer`/`AltTest` as the supported scorers. A single-label (scalar) task
continues to score with κ exactly as today
(`scores/metrics/agreement/iaa.py:42,107` — scalar string labels).

**Exclude XML for multi-label tasks, including fallback.**
JSON-based methods (`structured`, `instructor`) carry lists natively. The current XML path
uses `cardinality="one"` and casts each parsed task value to `str`, so it cannot preserve
the ordered vector contract. An `EvalTask` that combines `answering_method="xml"` with
any `MultiLabelSchema` raises a clear configuration error. When
`structured_outputs_fallback=True`, a multi-label task's fallback sequence contains only
`structured` and `instructor`; it MUST NOT fall through to `xml`. If neither JSON-based
method is available or succeeds, evaluation fails clearly rather than converting the
vector through the scalar XML path. XML multi-cardinality support may be proposed as a
separate future change with an explicit ordered-list representation and validation rules.

**CSV multi-label cells are JSON-encoded; parquet/JSON store the list natively.**
Verified empirically: a `list[str]` (or tuple, which polars collapses to `List`, or even a
fixed-length `pl.Array`) column **cannot** be written to CSV — `write_csv`
(`results/base.py:329`) raises `ComputeError: CSV format does not support nested data` (the
`Array` variant says "Consider using JSON or a binary format"). This is intrinsic to CSV (a
cell is scalar), not a polars quirk. Parquet round-trips a `list[str]` column losslessly
(verified, dtype `List(String)` preserved); JSON arrays are native. So:
- **parquet / json**: store the ordered vector natively, no encoding.
- **csv**: on write, JSON-encode each multi-label cell (`json.dumps(vector)` → a string
  cell); on read, JSON-decode it back to the list. Chosen over delimiter-join because the
  vector is fixed-length and **positional** (slot order is load-bearing for binarize), and
  JSON preserves order and arbitrary label content with no delimiter-collision risk.
- **Decode is not automatic.** `pl.read_csv` returns a plain `String` column — nothing in the
  CSV marks it as a list. Results state therefore persists the widened `task_schemas`; the
  load path identifies each `MultiLabelSchema` and JSON-decodes its column. This works for
  standalone `HumanAnnotationResults.load_state` / `JudgeResults.load_state` without an
  `EvalTask` object and is the one non-obvious half of the round-trip.

(Aside, orthogonal to CSV: using `pl.Array(str, N)` as the in-memory dtype would make
"slot count == N" a structural dtype guarantee rather than a hand-written validator. Not
required; noted as a possible robustness refinement for the vector representation.)

## Risks / Trade-offs

- [Widening `dict[str,str]` → `dict[str, str | list[str]]` touches API, storage, and
  serialization — a breaking change] → Confine the union to one place per layer; cover
  with round-trip tests for parquet and CSV before merge.
- [Outcome-order drift would silently corrupt every score, since slot `i` is positional]
  → Treat schema order as canonical; add a round-trip test asserting the reconstructed
  vector aligns to the original schema order.
- [CSV cannot store a native list cell] → JSON-encode the list within the cell; document
  the encoding and verify recovery in a round-trip test.
- [The binarize → `ClassificationScorer` path is new code, not a reused path] → Cover it
  with explicit tests (recommended per-label `macro`, optional per-item `samples`, and the
  all-zero result) before merge.
- [Frontend state and keyboard shortcuts assume single-string values] → `outcomes`
  becomes `Record<string, string | string[]>`; number-key shortcut toggles slot
  membership instead of replacing; add component tests alongside `TaskPanel.test.tsx`.
- [Existing N-binary multi-label users unaffected] → the legacy melt path is untouched;
  native multi-label is purely additive.

## Migration Plan

Additive schema variant. Existing bare-list and `None` task schemas behave and serialize
identically. Existing saved states remain loadable because their array/`null` shapes are
still valid members of the widened union. The outcome value-type widening is backward
compatible at runtime (a plain string still validates). No data migration required.
Rollback = revert the change; only states that already use `MultiLabelSchema` depend on it.

The legacy `MULTILABEL` melt path is **kept** (additive, not deprecated): it remains the
adapter for data already shaped as N separate binary columns. Native multi-label is the
recommended first-class path for new tasks (one column, one widget, field boundary
intact). RabakBench is the sole current user of the faked N-binary pattern; **breaking it
is acceptable**, and it is migrated to the native form as the reference end-to-end example
(6 harm columns → one `MultiLabelSchema` task). The migration reproduces RabakBench's
existing whole-vector metric via `ClassificationScorer` + `task_strategy="single"`, and
serves as the integration test for the binarize → scorer path.

## Open Questions

- ~~CSV list encoding: JSON-encoded cell vs delimiter-joined string.~~ RESOLVED:
  JSON-encode the cell on write, JSON-decode on read (columns to decode driven by
  persisted `MultiLabelSchema` entries in results `task_schemas`). See Decisions.
- ~~Whether to expose a convenience knob for the `average` mode (`"samples"` vs `"macro"`).~~
  RESOLVED: no new knob — multi-label requires `average ∈ {"samples","macro"}`. Callers
  choose explicitly; `"macro"` is recommended and required for mixed scalar + multi-label
  `MULTITASK`, while the existing global default `"binary"` is rejected loudly. See Decisions.
- Whether to add a friendlier error message (in the scoring engine, where `eval_task` is
  visible) when a native multi-label task is given `task_strategy="multilabel"`/`"multitask"`.
  The existing count validator already blocks it with a generic message; a native-aware
  message is polish, not correctness.
- Krippendorff's α as a future change: how to express a symmetric N-rater reliability
  measure in the judge-vs-humans scorer interface (judge as one more rater, or α over
  humans reported alongside?).
