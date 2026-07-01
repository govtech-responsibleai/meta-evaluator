## 0. Approach (TDD-first)

- [x] 0.1 Write tests BEFORE implementation in each section below — author the failing
  test, watch it fail for the right reason, then implement. The scoring path in particular
  (binarize → `ClassificationScorer`) is new and assumption-heavy; tests pin the contract.
- [x] 0.2 Anchor the scoring tests on the three-mode distinction so they exercise the right
  machinery: SINGLE (native vector, binarize → multilabel metric), MULTILABEL (legacy
  N-column melt — must remain unchanged), MULTITASK (per-task then averaged, incl. a
  `[single_class, multilabel]` mix). A test that routes a native vector through MULTILABEL
  mode would "pass" while exercising the wrong path — assert SINGLE explicitly.

## 1. EvalTask multi-label declaration

- [x] 1.1 Add and export `MultiLabelSchema(outcomes: list[str])`; widen `EvalTask.task_schemas`
  and `EvalTaskState.task_schemas` to `dict[str, list[str] | MultiLabelSchema | None]`
- [x] 1.2 Validate wrapped outcomes contain ≥2 values and do not declare the reserved
  `"FALSE"` sentinel; keep existing bare-list and `None` behavior unchanged
- [x] 1.3 In `create_task_class`, emit `list[Literal[*outcomes, "FALSE"]]` for
  `MultiLabelSchema`, with a validator enforcing length == #outcomes and slot `i` ∈
  {outcome `i`, `"FALSE"`}; leave bare-list single-select and `None` free-form unchanged
- [x] 1.4 Persist the wrapper through `serialize`/`deserialize`, preserving its object shape
  and outcome (slot) order while retaining compatibility with existing array/`null` states
- [x] 1.5 Unit tests: valid wrapper declaration, bare-list/`None` compatibility, too-short
  wrapped outcomes rejected, reserved `"FALSE"` outcome rejected, serialize round-trip,
  wrapper identity and **slot order preserved through round-trip**

## 2. Judge emission

- [x] 2.1 Confirm JSON extraction (`judge.py:296`) passes the full ordered vector through for structured + instructor
- [x] 2.2 Exclude XML for multi-label tasks: reject direct
  `answering_method="xml"`, and when `structured_outputs_fallback=True`, restrict the
  fallback sequence to `structured`/`instructor` so it never attempts XML. If neither
  supported method succeeds, fail clearly.
- [x] 2.3 Tests: structured judge returns a full ordered vector; wrong-length / out-of-slot
  value rejected by validator; direct xml+multilabel raises; structured/instructor
  fallback for multi-label never includes XML and fails clearly when both supported
  methods fail
- [x] 2.4 Update consistency-run aggregation for `MultiLabelSchema` to vote independently
  per slot in canonical order, preserving the existing first-occurrence tie-break; test
  sync and async consistency paths, including a tie

## 3. Annotation API

- [x] 3.1 Widen `SubmitAnnotationRequest.outcomes` to `dict[str, str | list[str]]`
- [x] 3.2 Widen `TaskConfigResponse.task_schemas` and `routes/task.py` so wrapped schemas
  serialize as `{ "outcomes": [...] }`; do not add a parallel multi-label key list
- [x] 3.3 Widen `SampleResponse.previous_annotation` to allow list values
- [x] 3.4 Update `state.py` (`submit_annotation`, autosave) to accept and store the ordered vector
- [x] 3.5 Tests: submit + retrieve a vector-valued outcome through the API

## 4. Results storage and serialization

- [x] 4.1 Widen outcome type in `human_results.py` and `judge_results.py` to
  `dict[str, str | list[str]]`; widen results `task_schemas` and serialized state to retain
  `MultiLabelSchema`; label column holds a fixed-length vector for multi-label tasks
- [x] 4.2 Ensure parquet round-trip preserves `list[str]` columns (and slot order)
- [x] 4.3 Implement CSV list encoding: JSON-encode each multi-label cell on write
  (`write_csv` raises `ComputeError` on a native `list[str]` column, so this is mandatory),
  JSON-decode on read. Decode is NOT automatic — `pl.read_csv` returns a `String` column, so
  the load path must inspect persisted results `task_schemas` for `MultiLabelSchema` to
  know which columns to decode; standalone results loading has no `EvalTask` object.
  Parquet/JSON store the list natively (no encoding). Document the encoding.
- [x] 4.4 Tests: parquet and CSV round-trip for multi-label values, asserting slot order

## 5. Annotation frontend

- [x] 5.1 Add `ui/checkbox.tsx` component
- [x] 5.2 Change `TaskPanel` `outcomes` state to `Record<string, string | string[]>`
- [x] 5.3 Add third render branch: checkbox group when the task config schema is the wrapped
  `{ outcomes: string[] }` object; submit the full ordered vector (selected → name,
  unselected → `"FALSE"`)
- [x] 5.4 Required-task check: multi-label answered (the vector is always full-length; required means rendered/submitted)
- [x] 5.5 Number-key shortcut toggles a slot's selection instead of replacing
- [x] 5.6 Mirror `string | string[]` outcome values and
  `string[] | { outcomes: string[] } | null` task-schema values in `lib/api.ts`
- [x] 5.7 Component tests for the checkbox branch (select, deselect, slot toggle, full-vector submit)
- [x] 5.8 Rebuild the frontend bundle (`dist/`)

## 6. Scoring via ClassificationScorer

- [x] 6.1 Add a **positional** binarize transform applied at the **scorer call site**
  (`scoring.py:600`-ish, where `metric_config.scorer` is in scope), **gated to
  `ClassificationScorer`**, after `_preprocess_task_data` has produced the `label` column in
  canonical schema (slot) order: when `task_schemas[task]` is `MultiLabelSchema`, map slot
  `i` → 1 iff `vec[i] == schema.outcomes[i]`, else 0 (NOT `!= "FALSE"`), producing one length-N indicator vector
  per item. Do NOT put binarize in the scorer-agnostic `_preprocess_task_data` (that would
  force every scorer onto the indicator form). Keep `ClassificationScorer` unaware of
  `"FALSE"`. The foreign-slot-value check SHALL raise where visible (the per-judge path is
  wrapped in `except Exception → NaN` at `scoring.py:626`, and `_compute_metric` swallows
  `ValueError` to `nan`, so a raise inside that region becomes a silent NaN).
- [x] 6.2 Validate `ClassificationScorer.average ∈ {"samples","macro"}` for a multi-label
  task and reject the default `"binary"` with a clear error naming the two valid choices
  (raise up front, not deep in sklearn where it is swallowed to NaN). Recommend
  `average="macro"` (per-label) and confirm optional `average="samples"` (per-item overlap)
  also works when every configured task is native multi-label. Require `"macro"` for a
  mixed scalar + multi-label `MULTITASK` configuration.
- [x] 6.3 Confirm existing per-position majority-vote (`scores/metrics/classification/classification_scorer.py:155`) works unchanged on the aligned (binarized) vector
- [x] 6.4 Make `CohensKappaScorer` raise a clear error for a multi-label (list-valued) task, naming `ClassificationScorer`/`AltTestScorer`; leave single-label κ untouched
- [x] 6.5 Confirm `AltTestScorer` scores the native **name** vector unchanged (NOT binarized; jaccard auto-route on list values)
- [x] 6.6 Tests (TDD, write first):
  - binarize is positional and collapses (`["Clarify","FALSE","Support"]` → `[1,0,1]`, not one-hot expansion); a non-`FALSE` foreign slot value fails loud — assert at the binarize-function level, NOT end-to-end through `score()` (which masks the raise as a per-judge NaN)
  - un-binarized name-vector into `samples`/`macro` raises `multiclass-multioutput` (proves binarize is required)
  - multi-label `ClassificationScorer` with default `average="binary"` is rejected with a clear error (not a swallowed NaN)
  - known values on the worked example: `samples == 0.833…`, `macro == 0.667…`
  - F1/precision/recall on an all-zero expert and judge indicator vector returns `0`
  - `AltTestScorer` receives the name vector (not `[1,0,1]`) and routes to jaccard
  - multi-human `individual_average` and `majority_vote` on a multi-label task
  - **three-mode triad**: native vector via `SINGLE` (binarize → multilabel metric); legacy N-column `MULTILABEL` unchanged; `MULTITASK` over `[single_class, multilabel]` with `average="macro"` averages per-task scores and rejects `average="samples"`
  - `CohensKappaScorer` errors on a multi-label task; still works on a single-label task
  - lone multi-label task name with `task_strategy="multilabel"`/`"multitask"` is rejected by the count validator

## 7. Integration and docs

- [x] 7.1 Migrate RabakBench to the native form as the reference end-to-end example: 6 harm
  columns → one `MultiLabelSchema` task; annotate, run a judge, score with
  `ClassificationScorer` (`average="macro"`) via `task_strategy="single"`. Reproduces its
  whole-vector metric and serves as the integration test for the binarize → scorer path.
  (See `examples/rabakbench2/`.)
- [x] 7.2 Run quality gates (ruff, pyright, pytest, frontend tests)
- [x] 7.3 Update README/docs to describe the multi-label task type (ordered vector, `"FALSE"` sentinel, positional binarize, recommended `average="macro"`, optional `"samples"` for pure multi-label configurations, scoring via `ClassificationScorer`/`AltTest`, κ unsupported for multi-label, native rides `task_strategy="single"`, and answering methods are limited to `structured`/`instructor` with XML excluded from fallback)
