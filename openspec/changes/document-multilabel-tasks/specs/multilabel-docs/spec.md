## ADDED Requirements

### Requirement: Annotation guide documents the multi-label checkbox field

The annotation guide (`docs/annotation_guide/annotation.md`) SHALL describe how a
multi-label task appears and behaves in the annotation interface.

#### Scenario: Checkbox rendering is documented

- **WHEN** a reader consults the annotation guide about task field types
- **THEN** it states that a `MultiLabelSchema` task renders as a checkbox group (one
  checkbox per declared outcome), distinct from the single-select radio group

#### Scenario: Slot toggle and empty selection are documented

- **WHEN** a reader looks for multi-label interaction details
- **THEN** the guide states that the number-key shortcut toggles a single slot (rather
  than replacing the selection) and that selecting nothing is recorded as an
  all-`"FALSE"` vector, never an empty or missing value

### Requirement: Results guide documents multi-label outcome columns

The results guide (`docs/guides/results.md`) SHALL describe the storage shape and
round-trip behavior of multi-label outcome values.

#### Scenario: Vector-valued column is documented

- **WHEN** a reader consults the results guide about outcome columns
- **THEN** it states that a multi-label task's outcome column holds a fixed-length
  ordered vector (`list[str]`) with slot order preserved

#### Scenario: Serialization formats are documented

- **WHEN** a reader looks for how multi-label results persist
- **THEN** the guide states that parquet and JSON store the list natively while CSV
  JSON-encodes each cell on write and decodes it on load using the persisted schema

### Requirement: Judge-run guide documents multi-label emission and constraints

The judge-run guide (`docs/guides/judges_run.md`) SHALL describe how judges produce
multi-label outputs and the associated constraints.

#### Scenario: Answering-method restriction is documented

- **WHEN** a reader consults the judge-run guide for multi-label tasks
- **THEN** it states that judges emit the full ordered vector and that only
  `structured` and `instructor` answering methods are supported, with `xml` rejected
  and excluded from the fallback sequence

#### Scenario: Consistency aggregation is documented

- **WHEN** a reader looks for how consistency runs handle multi-label tasks
- **THEN** the guide states that aggregation happens independently per slot (most votes
  wins, ties by first occurrence)

### Requirement: Scoring guide frames task_strategy as an aggregation choice

The scoring guide (`docs/guides/scoring.md`) SHALL present `task_strategy` as a choice
about how a scorer's results are aggregated across the `task_names` in one
`MetricConfig`, decoupled from any individual task's value shape.

#### Scenario: Aggregation framing is explicit

- **WHEN** a reader consults the scoring guide about `task_strategy`
- **THEN** it states that `task_strategy="single"` scores one task and
  `task_strategy="multitask"` scores several tasks then averages their results, and that
  this is independent of whether a task's value is single-select, free-form, or
  multi-label

#### Scenario: Native multi-label uses single aggregation

- **WHEN** a reader wants to score a native `MultiLabelSchema` task
- **THEN** the guide states it is scored with `task_strategy="single"` (one task column)
  using `ClassificationScorer` with `average="macro"` (or `"samples"`), that the default
  `"binary"` is rejected, that `AltTestScorer` scores the native name vector unchanged,
  and that `CohensKappaScorer` raises for multi-label tasks

### Requirement: Scoring guide deprecates the legacy multilabel strategy

The scoring guide (`docs/guides/scoring.md`) SHALL mark `task_strategy="multilabel"` as
deprecated and steer readers to the native multi-label task type.

#### Scenario: Deprecation and migration are documented

- **WHEN** a reader encounters `task_strategy="multilabel"` in the scoring guide
- **THEN** it is marked deprecated with a migration note explaining that N single-select
  tasks combined by the melt should instead be declared as one `MultiLabelSchema` task
  scored with `task_strategy="single"`

#### Scenario: Deprecation does not claim removal

- **WHEN** the deprecation note is read
- **THEN** it makes clear the strategy still functions in the current version and that
  removal is planned for a future release, without asserting it is already removed or
  naming a specific target version

#### Scenario: Legacy melt is not presented as a recommended path

- **WHEN** a reader skims the scoring guide for multi-label examples
- **THEN** the native `MultiLabelSchema` task is the recommended approach, and any
  remaining `task_strategy="multilabel"` example outside the deprecated reference is
  rewritten off the melt or explicitly annotated as deprecated

### Requirement: Tutorial points to the multi-label task type

The tutorial (`docs/tutorial.md`) SHALL make first-time readers aware that the native
multi-label task type exists and where to find its detailed documentation.

#### Scenario: Discoverability pointer exists

- **WHEN** a first-time reader follows the tutorial
- **THEN** it mentions the multi-label task type and links to the task-definition guide
  for details

### Requirement: Root README mentions the multi-label task type

The root `README.md` SHALL make the multi-label task type visible at the primary
getting-started entry point.

#### Scenario: Define-task section mentions multi-label

- **WHEN** a reader reaches the "Define Task" section of the README
- **THEN** it notes that a task may be multi-label via `MultiLabelSchema(outcomes=[...])`
  (in addition to single-select and free-form) and links to `docs/guides/evaltask.md`

#### Scenario: Metrics section notes the kappa limitation

- **WHEN** a reader reads the README "Available Metrics" section
- **THEN** it notes that Cohen's Kappa is not supported for multi-label tasks and points
  to `ClassificationScorer`/`AltTestScorer` for them

### Requirement: Documentation examples match the shipped API

All documentation examples added by this change SHALL use the shipped public API and
build cleanly.

#### Scenario: Examples use current API and mkdocs builds

- **WHEN** the documentation is built
- **THEN** multi-label examples use `MultiLabelSchema(outcomes=[...])`, the reserved
  `"FALSE"` sentinel, and `average="macro"`, and the mkdocs build succeeds with no
  broken internal links
