# Annotation UI Migration: Streamlit → React + FastAPI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Streamlit annotation interface with React (shadcn/ui) + FastAPI backend while preserving identical functionality and public API.

**Architecture:** FastAPI backend wraps existing `HumanAnnotationResultsBuilder` and `EvalTask`/`EvalData` models. React SPA (Vite + shadcn/ui) communicates via REST API. Single process serves both API and static frontend build. Launcher starts uvicorn instead of Streamlit subprocess.

**Tech Stack:** Python 3.13, FastAPI, uvicorn, Polars, Pydantic v2, React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, Vitest, React Testing Library, pytest, httpx

## Global Constraints

- Python >=3.13 (matches pyproject.toml)
- `launch_annotator(port, use_ngrok, traffic_policy_file)` signature unchanged
- All data models in `src/meta_evaluator/results/`, `src/meta_evaluator/data/`, `src/meta_evaluator/eval_task/` are NOT modified
- Auto-save file format stays parquet with same naming pattern: `autosave_{run_id}_{annotator_id}_{name}_data.parquet`
- Final export format stays: `{run_id}_{annotator_id}_{name}_metadata.json` + `{run_id}_{annotator_id}_{name}_data.parquet`
- Existing custom exceptions in `src/meta_evaluator/annotator/exceptions.py` are reused (not rewritten)
- After every task: `uv tool run ruff check --preview --fix && uv tool run ruff format . && uv run pyright && uv run pytest -m "not integration"`
- No Co-Authored-By lines in commits

## File Structure

```
src/meta_evaluator/annotator/
├── __init__.py                          ← UPDATE: export AnnotationLauncher (replaces StreamlitAnnotator)
├── exceptions.py                        ← KEEP: reuse existing exceptions
├── api/
│   ├── __init__.py
│   ├── app.py                           ← CREATE: FastAPI app factory, mounts static files
│   ├── schemas.py                       ← CREATE: Pydantic request/response models
│   ├── state.py                         ← CREATE: SessionStore wrapping HumanAnnotationResultsBuilder
│   └── routes/
│       ├── __init__.py
│       ├── task.py                      ← CREATE: GET /api/task
│       ├── session.py                   ← CREATE: POST /api/session, GET /api/session/{run_id}
│       ├── samples.py                   ← CREATE: GET /api/samples/{index}
│       ├── annotations.py              ← CREATE: POST /api/annotations, GET /api/progress
│       └── export.py                    ← CREATE: POST /api/export, GET /api/export/download/{filename}
├── launcher/
│   ├── __init__.py                      ← UPDATE: export AnnotationLauncher
│   └── launcher.py                      ← CREATE: replaces streamlit_launcher.py
└── frontend/                            ← CREATE: entire React app
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── tailwind.config.ts
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── lib/
        │   └── api.ts                   ← fetch wrapper
        ├── hooks/
        │   └── useAnnotation.ts         ← main state hook
        └── components/
            ├── NameEntry.tsx
            ├── AnnotationView.tsx
            ├── SampleDisplay.tsx
            ├── TaskPanel.tsx
            ├── Navigation.tsx
            └── ExportDialog.tsx

Files to DELETE:
- src/meta_evaluator/annotator/interface/  (entire directory)
- src/meta_evaluator/annotator/launcher/streamlit_launcher.py
- src/meta_evaluator/annotator/launcher/entry_point.py
- tests/annotator/interface/  (entire directory)
- tests/annotator/test_annotator_integration.py

Files to MODIFY:
- src/meta_evaluator/meta_evaluator/base.py:9,671-712  ← import + launch_annotator body
- pyproject.toml:44-46  ← ui optional deps
- tests/annotator/conftest.py  ← remove Streamlit fixtures, keep data fixtures
```

---

### Task 1: Project Setup — Dependencies and Cleanup

**Files:**
- Modify: `pyproject.toml:39-49`
- Delete: `src/meta_evaluator/annotator/interface/` (entire directory)
- Delete: `src/meta_evaluator/annotator/launcher/streamlit_launcher.py`
- Delete: `src/meta_evaluator/annotator/launcher/entry_point.py`
- Delete: `tests/annotator/interface/` (entire directory)
- Delete: `tests/annotator/test_annotator_integration.py`
- Modify: `tests/annotator/conftest.py`
- Modify: `src/meta_evaluator/annotator/launcher/__init__.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: clean slate with new deps installed; test fixtures for `EvalTask`, `EvalData`, `HumanAnnotationResultsBuilder` still work

- [ ] **Step 1: Update pyproject.toml — swap streamlit for fastapi+uvicorn**

```toml
[project.optional-dependencies]
docs = [
    "mkdocs>=1.6.1",
    "mkdocs-material>=9.6.17",
]
ui = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
]
all = [
    "meta-evaluator[docs,ui]",
]
```

- [ ] **Step 2: Delete old Streamlit files**

```bash
rm -rf src/meta_evaluator/annotator/interface
rm src/meta_evaluator/annotator/launcher/streamlit_launcher.py
rm src/meta_evaluator/annotator/launcher/entry_point.py
```

- [ ] **Step 3: Delete old Streamlit tests**

```bash
rm -rf tests/annotator/interface
rm tests/annotator/test_annotator_integration.py
```

- [ ] **Step 4: Clean up conftest.py — remove Streamlit-specific fixtures**

Remove the `MockSessionState` class, `mock_streamlit_session_state` fixture, and `session_manager` fixture. Remove the `StreamlitSessionManager` import. Keep all other fixtures (`free_port`, `temp_annotations_dir`, `annotator_eval_task`, `annotator_eval_data`, `integration_eval_task`, `integration_eval_data`, `mock_eval_data`, `mock_eval_task`, `sample_dataframe`, `sample_task_schemas`, `sample_expected_ids`, `base_human_results_builder`, `single_task_human_results_builder`, `mock_results_builder`).

Updated `tests/annotator/conftest.py`:

```python
"""Fixtures for Annotator testing.

This conftest provides annotator-specific fixtures used across annotator test modules.
Common fixtures are inherited from the main conftest.py.
"""

import socket
from unittest.mock import Mock

import polars as pl
import pytest

from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.results import HumanAnnotationResultsBuilder

# ==== BASIC FIXTURES ====


@pytest.fixture
def free_port():
    """Find and return a free port for testing."""
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


@pytest.fixture
def temp_annotations_dir(tmp_path):
    """Create a temporary directory for storing annotation results."""
    return str(tmp_path / "annotations")


# ==== EVAL TASK FIXTURES ====


@pytest.fixture
def annotator_eval_task() -> EvalTask:
    """Create a test EvalTask for annotator testing."""
    return EvalTask(
        task_schemas={
            "sentiment": ["positive", "negative", "neutral"],
            "quality": ["high", "medium", "low"],
            "comments": None,
        },
        prompt_columns=["text", "question"],
        response_columns=["response", "answer"],
        answering_method="structured",
    )


@pytest.fixture
def integration_eval_task() -> EvalTask:
    """Create a test EvalTask for integration testing."""
    return EvalTask(
        task_schemas={
            "sentiment": ["positive", "negative", "neutral"],
            "quality": ["good", "bad"],
        },
        prompt_columns=["text"],
        response_columns=["response"],
        answering_method="structured",
    )


# ==== EVAL DATA FIXTURES ====


@pytest.fixture
def sample_dataframe():
    """Create a sample Polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3],
            "question": [
                "What is 2+2?",
                "What is the capital of France?",
                "Who wrote Hamlet?",
            ],
            "answer": ["4", "Paris", "Shakespeare"],
        }
    )


@pytest.fixture
def annotator_eval_data() -> EvalData:
    """Create test EvalData for annotator testing."""
    df = pl.DataFrame(
        {
            "id": ["1", "2"],
            "text": ["I love this!", "I hate this!"],
            "response": ["Great!", "Awful!"],
        }
    )
    return EvalData(
        name="test_data",
        data=df,
        id_column="id",
    )


@pytest.fixture
def integration_eval_data() -> EvalData:
    """Create test EvalData for integration testing."""
    df = pl.DataFrame(
        {
            "id": ["sample_1", "sample_2", "sample_3"],
            "text": [
                "I love this product!",
                "This is terrible!",
                "It's okay, nothing special",
            ],
            "response": [
                "Thank you for your feedback!",
                "We apologize for the inconvenience.",
                "We appreciate your honest review.",
            ],
        }
    )
    return EvalData(
        name="integration_test_data",
        data=df,
        id_column="id",
    )


# ==== MOCK FIXTURES ====


@pytest.fixture
def mock_eval_data(sample_dataframe):
    """Create a mock EvalData object for testing."""
    mock_data = Mock(spec=EvalData)
    mock_data.id_column = "id"
    mock_data.data = sample_dataframe
    return mock_data


@pytest.fixture
def mock_eval_task():
    """Create a mock EvalTask object for testing."""
    mock_task = Mock(spec=EvalTask)
    mock_task.task_schemas = {
        "sentiment": ["positive", "negative", "neutral"],
        "quality": ["high", "medium", "low"],
        "comments": None,
    }
    mock_task.prompt_columns = ["question"]
    mock_task.response_columns = ["answer"]
    mock_task.answering_method = "structured"
    mock_task.annotation_prompt = "Please evaluate the following response:"
    mock_task.get_required_tasks.return_value = ["sentiment", "quality"]
    return mock_task


# ==== TASK SCHEMA AND ID FIXTURES ====


@pytest.fixture
def sample_task_schemas():
    """Sample task schemas for testing."""
    return {
        "task1": ["yes", "no", "maybe"],
        "task2": ["good", "bad", "neutral"],
    }


@pytest.fixture
def sample_expected_ids():
    """Sample expected IDs for testing."""
    return ["id1", "id2", "id3"]


# ==== HUMAN ANNOTATION RESULTS FIXTURES ====


@pytest.fixture
def base_human_results_builder() -> HumanAnnotationResultsBuilder:
    """Provide a basic HumanAnnotationResultsBuilder instance."""
    return HumanAnnotationResultsBuilder(
        run_id="run_001",
        annotator_id="annotator_1",
        task_schemas={"task1": ["yes", "no"], "task2": ["good", "bad"]},
        expected_ids=["id1", "id2"],
        required_tasks=["task1", "task2"],
        is_sampled_run=False,
    )


@pytest.fixture
def single_task_human_results_builder() -> HumanAnnotationResultsBuilder:
    """Provides a builder with single task and id."""
    return HumanAnnotationResultsBuilder(
        run_id="single_task_run",
        annotator_id="annotator_1",
        task_schemas={"task1": ["yes", "no"]},
        expected_ids=["id1"],
        required_tasks=["task1"],
        is_sampled_run=False,
    )


@pytest.fixture
def mock_results_builder():
    """Mock HumanAnnotationResultsBuilder for testing."""
    mock_builder = Mock(spec=HumanAnnotationResultsBuilder)
    mock_builder.completed_count = 0
    mock_builder._results = {}
    return mock_builder
```

- [ ] **Step 5: Update launcher __init__.py to empty placeholder**

```python
"""Launcher for the annotation interface."""
```

- [ ] **Step 6: Install new deps and verify clean state**

```bash
uv sync --extra all
uv run pytest tests/annotator/test_human_results.py -v
```

Expected: all human_results tests pass (they don't depend on Streamlit).

- [ ] **Step 7: Run linting and type checking**

```bash
uv tool run ruff check --preview --fix
uv tool run ruff format .
uv run pyright
```

Note: pyright will report errors in `base.py` (import of `StreamlitLauncher` now broken). This is expected and fixed in Task 7.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: remove Streamlit annotation interface and update deps for FastAPI migration"
```

---

### Task 2: API Schemas and Session State Manager

**Files:**
- Create: `src/meta_evaluator/annotator/api/__init__.py`
- Create: `src/meta_evaluator/annotator/api/schemas.py`
- Create: `src/meta_evaluator/annotator/api/state.py`
- Create: `src/meta_evaluator/annotator/api/routes/__init__.py`
- Test: `tests/annotator/api/__init__.py`
- Test: `tests/annotator/api/test_state.py`

**Interfaces:**
- Consumes: `HumanAnnotationResultsBuilder` from `meta_evaluator.results`, `EvalTask` from `meta_evaluator.eval_task`, `EvalData` from `meta_evaluator.data`
- Produces:
  - `SessionStore` class with methods: `create_session(annotator_name, eval_task, eval_data, annotations_dir) -> SessionInfo`, `get_session(run_id) -> SessionInfo | None`, `submit_annotation(run_id, sample_index, outcomes) -> None`, `get_progress(run_id) -> ProgressInfo`, `export(run_id, annotator_name) -> ExportResult`
  - Pydantic schemas: `CreateSessionRequest`, `CreateSessionResponse`, `SampleResponse`, `SubmitAnnotationRequest`, `ProgressResponse`, `TaskConfigResponse`, `ExportResponse`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/meta_evaluator/annotator/api/routes
mkdir -p tests/annotator/api
touch src/meta_evaluator/annotator/api/__init__.py
touch src/meta_evaluator/annotator/api/routes/__init__.py
touch tests/annotator/api/__init__.py
```

- [ ] **Step 2: Write the schemas**

Create `src/meta_evaluator/annotator/api/schemas.py`:

```python
"""Pydantic request/response models for annotation API."""

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Request to create a new annotation session."""

    annotator_name: str = Field(..., min_length=1)


class CreateSessionResponse(BaseModel):
    """Response after creating a session."""

    run_id: str
    annotator_id: str
    total_samples: int
    resumed: bool = False
    annotated_count: int = 0


class TaskConfigResponse(BaseModel):
    """Response containing task configuration."""

    task_schemas: dict[str, list[str] | None]
    prompt_columns: list[str] | None
    response_columns: list[str]
    annotation_prompt: str
    required_tasks: list[str]


class SampleResponse(BaseModel):
    """Response containing a single sample's data."""

    index: int
    total: int
    sample_id: str
    prompt_data: dict[str, str] | None
    response_data: dict[str, str]
    previous_annotation: dict[str, str] | None = None


class SubmitAnnotationRequest(BaseModel):
    """Request to submit an annotation."""

    run_id: str
    sample_index: int = Field(..., ge=0)
    outcomes: dict[str, str]


class SubmitAnnotationResponse(BaseModel):
    """Response after submitting annotation."""

    success: bool
    annotated_count: int
    auto_saved: bool


class ProgressResponse(BaseModel):
    """Response containing annotation progress."""

    run_id: str
    annotated_count: int
    total_samples: int
    incomplete_indices: list[int]


class ExportRequest(BaseModel):
    """Request to export annotations."""

    run_id: str


class ExportResponse(BaseModel):
    """Response after export."""

    metadata_file: str
    data_file: str
    total_count: int
    succeeded_count: int
    error_count: int
```

- [ ] **Step 3: Write failing tests for SessionStore**

Create `tests/annotator/api/test_state.py`:

```python
"""Tests for annotation API session state manager."""

import os

import polars as pl
import pytest

from meta_evaluator.annotator.api.state import SessionStore
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask


@pytest.fixture
def eval_task() -> EvalTask:
    """Create test EvalTask."""
    return EvalTask(
        task_schemas={
            "sentiment": ["positive", "negative", "neutral"],
            "quality": ["high", "medium", "low"],
        },
        prompt_columns=["text"],
        response_columns=["response"],
        answering_method="structured",
    )


@pytest.fixture
def eval_data() -> EvalData:
    """Create test EvalData."""
    df = pl.DataFrame(
        {
            "id": ["s1", "s2", "s3"],
            "text": ["Hello", "World", "Test"],
            "response": ["Hi", "Earth", "Check"],
        }
    )
    return EvalData(name="test", data=df, id_column="id")


@pytest.fixture
def store(eval_task, eval_data, tmp_path) -> SessionStore:
    """Create a SessionStore instance."""
    return SessionStore(
        eval_task=eval_task,
        eval_data=eval_data,
        annotations_dir=str(tmp_path / "annotations"),
    )


class TestCreateSession:
    """Tests for session creation."""

    def test_create_session_returns_run_id(self, store):
        """Creating session returns valid run_id."""
        info = store.create_session("Alice")
        assert info.run_id.startswith("annotation_run_")
        assert info.annotator_id == "alice"
        assert info.total_samples == 3
        assert info.resumed is False

    def test_create_session_sanitizes_name(self, store):
        """Special characters in name are sanitized."""
        info = store.create_session("John Doe 123!")
        assert info.annotator_id == "john_doe_123_"

    def test_create_session_empty_name_raises(self, store):
        """Empty name raises ValueError."""
        with pytest.raises(ValueError):
            store.create_session("")

    def test_create_session_whitespace_name_raises(self, store):
        """Whitespace-only name raises ValueError."""
        with pytest.raises(ValueError):
            store.create_session("   ")


class TestGetSession:
    """Tests for session retrieval."""

    def test_get_existing_session(self, store):
        """Can retrieve session by run_id."""
        created = store.create_session("Bob")
        retrieved = store.get_session(created.run_id)
        assert retrieved is not None
        assert retrieved.run_id == created.run_id

    def test_get_nonexistent_session_returns_none(self, store):
        """Unknown run_id returns None."""
        result = store.get_session("nonexistent_run_id")
        assert result is None


class TestSubmitAnnotation:
    """Tests for annotation submission."""

    def test_submit_valid_annotation(self, store):
        """Submitting with all required fields succeeds."""
        session = store.create_session("Carol")
        result = store.submit_annotation(
            run_id=session.run_id,
            sample_index=0,
            outcomes={"sentiment": "positive", "quality": "high"},
        )
        assert result.success is True
        assert result.annotated_count == 1
        assert result.auto_saved is True

    def test_submit_updates_progress(self, store):
        """Annotation count increases after submit."""
        session = store.create_session("Dave")
        store.submit_annotation(
            run_id=session.run_id,
            sample_index=0,
            outcomes={"sentiment": "positive", "quality": "high"},
        )
        progress = store.get_progress(session.run_id)
        assert progress.annotated_count == 1
        assert 0 not in progress.incomplete_indices

    def test_submit_invalid_run_id_raises(self, store):
        """Submitting to unknown session raises KeyError."""
        with pytest.raises(KeyError):
            store.submit_annotation(
                run_id="bad_id",
                sample_index=0,
                outcomes={"sentiment": "positive", "quality": "high"},
            )

    def test_submit_out_of_bounds_index_raises(self, store):
        """Index beyond total samples raises IndexError."""
        session = store.create_session("Eve")
        with pytest.raises(IndexError):
            store.submit_annotation(
                run_id=session.run_id,
                sample_index=99,
                outcomes={"sentiment": "positive", "quality": "high"},
            )

    def test_resubmit_overwrites_previous(self, store):
        """Re-submitting same index overwrites previous annotation."""
        session = store.create_session("Frank")
        store.submit_annotation(
            run_id=session.run_id,
            sample_index=0,
            outcomes={"sentiment": "positive", "quality": "high"},
        )
        store.submit_annotation(
            run_id=session.run_id,
            sample_index=0,
            outcomes={"sentiment": "negative", "quality": "low"},
        )
        progress = store.get_progress(session.run_id)
        assert progress.annotated_count == 1


class TestAutoSave:
    """Tests for auto-save functionality."""

    def test_auto_save_creates_parquet(self, store, tmp_path):
        """Auto-save writes parquet file after annotation."""
        session = store.create_session("Grace")
        store.submit_annotation(
            run_id=session.run_id,
            sample_index=0,
            outcomes={"sentiment": "positive", "quality": "high"},
        )
        annotations_dir = tmp_path / "annotations"
        parquet_files = list(annotations_dir.glob("autosave_*.parquet"))
        assert len(parquet_files) == 1

    def test_resume_from_auto_save(self, store, eval_task, eval_data, tmp_path):
        """Creating session with same name resumes from auto-save."""
        session = store.create_session("Hank")
        store.submit_annotation(
            run_id=session.run_id,
            sample_index=0,
            outcomes={"sentiment": "positive", "quality": "high"},
        )
        # Create new store (simulates server restart)
        store2 = SessionStore(
            eval_task=eval_task,
            eval_data=eval_data,
            annotations_dir=str(tmp_path / "annotations"),
        )
        resumed = store2.create_session("Hank")
        assert resumed.resumed is True
        assert resumed.annotated_count == 1


class TestGetProgress:
    """Tests for progress tracking."""

    def test_initial_progress(self, store):
        """Fresh session shows zero progress."""
        session = store.create_session("Iris")
        progress = store.get_progress(session.run_id)
        assert progress.annotated_count == 0
        assert progress.total_samples == 3
        assert progress.incomplete_indices == [0, 1, 2]

    def test_progress_after_annotations(self, store):
        """Progress updates after annotations."""
        session = store.create_session("Jake")
        store.submit_annotation(
            run_id=session.run_id,
            sample_index=1,
            outcomes={"sentiment": "negative", "quality": "low"},
        )
        progress = store.get_progress(session.run_id)
        assert progress.annotated_count == 1
        assert 1 not in progress.incomplete_indices
        assert 0 in progress.incomplete_indices


class TestExport:
    """Tests for export functionality."""

    def test_export_creates_files(self, store, tmp_path):
        """Export produces metadata JSON and data parquet."""
        session = store.create_session("Kate")
        for i in range(3):
            store.submit_annotation(
                run_id=session.run_id,
                sample_index=i,
                outcomes={"sentiment": "positive", "quality": "high"},
            )
        result = store.export(session.run_id, "Kate")
        annotations_dir = tmp_path / "annotations"
        assert os.path.exists(str(annotations_dir / result.metadata_file))
        assert os.path.exists(str(annotations_dir / result.data_file))
        assert result.total_count == 3
        assert result.succeeded_count == 3

    def test_export_invalid_session_raises(self, store):
        """Export with unknown run_id raises KeyError."""
        with pytest.raises(KeyError):
            store.export("bad_id", "Nobody")


class TestGetSample:
    """Tests for sample retrieval."""

    def test_get_sample_valid_index(self, store):
        """Get sample at valid index returns correct data."""
        session = store.create_session("Leo")
        sample = store.get_sample(session.run_id, 0)
        assert sample.index == 0
        assert sample.total == 3
        assert sample.sample_id == "s1"
        assert sample.response_data == {"response": "Hi"}
        assert sample.prompt_data == {"text": "Hello"}

    def test_get_sample_out_of_bounds(self, store):
        """Get sample beyond total raises IndexError."""
        session = store.create_session("Mia")
        with pytest.raises(IndexError):
            store.get_sample(session.run_id, 99)

    def test_get_sample_negative_index(self, store):
        """Negative index raises IndexError."""
        session = store.create_session("Ned")
        with pytest.raises(IndexError):
            store.get_sample(session.run_id, -1)

    def test_get_sample_with_previous_annotation(self, store):
        """Sample returns previous annotation if exists."""
        session = store.create_session("Olivia")
        store.submit_annotation(
            run_id=session.run_id,
            sample_index=0,
            outcomes={"sentiment": "positive", "quality": "high"},
        )
        sample = store.get_sample(session.run_id, 0)
        assert sample.previous_annotation == {
            "sentiment": "positive",
            "quality": "high",
        }
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
uv run pytest tests/annotator/api/test_state.py -v
```

Expected: ImportError — `meta_evaluator.annotator.api.state` does not exist yet.

- [ ] **Step 5: Implement SessionStore**

Create `src/meta_evaluator/annotator/api/state.py`:

```python
"""In-memory session state manager for annotation API."""

import glob
import logging
import os
import re
import uuid
from datetime import datetime

import polars as pl

from meta_evaluator.annotator.api.schemas import (
    CreateSessionResponse,
    ExportResponse,
    ProgressResponse,
    SampleResponse,
    SubmitAnnotationResponse,
)
from meta_evaluator.annotator.exceptions import SaveError
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.results import HumanAnnotationResultsBuilder

logger = logging.getLogger(__name__)


def _generate_run_id() -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"annotation_run_{timestamp}_{unique_id}"


def _generate_annotator_id(annotator_name: str) -> str:
    """Generate sanitized annotator ID from name."""
    if not annotator_name or not annotator_name.strip():
        raise ValueError("Annotator name cannot be empty")
    annotator_id = annotator_name.lower().strip()
    annotator_id = re.sub(r"[^a-z0-9_-]", "_", annotator_id)
    annotator_id = re.sub(r"_+", "_", annotator_id)
    annotator_id = annotator_id.strip("_")
    if not annotator_id or not annotator_id[0].isalpha():
        annotator_id = f"user_{annotator_id}"
    return annotator_id


class _Session:
    """Internal session state."""

    def __init__(
        self,
        run_id: str,
        annotator_id: str,
        annotator_name: str,
        builder: HumanAnnotationResultsBuilder,
        sample_ids: list[str],
    ):
        self.run_id = run_id
        self.annotator_id = annotator_id
        self.annotator_name = annotator_name
        self.builder = builder
        self.sample_ids = sample_ids


class SessionStore:
    """Manages annotation sessions backed by HumanAnnotationResultsBuilder."""

    def __init__(
        self,
        eval_task: EvalTask,
        eval_data: EvalData,
        annotations_dir: str,
    ):
        self._eval_task = eval_task
        self._eval_data = eval_data
        self._annotations_dir = annotations_dir
        self._sessions: dict[str, _Session] = {}

        os.makedirs(self._annotations_dir, exist_ok=True)

    def _get_sample_ids(self) -> list[str]:
        """Get all sample IDs from eval_data."""
        id_col = self._eval_data.id_column
        return [str(v) for v in self._eval_data.data[id_col].to_list()]

    def _find_auto_save(self, annotator_name: str) -> str | None:
        """Find existing auto-save file for annotator."""
        pattern = os.path.join(
            self._annotations_dir,
            f"autosave_*_*_{annotator_name}_data.parquet",
        )
        matching = glob.glob(pattern)
        if matching:
            return max(matching, key=os.path.getmtime)
        return None

    def _auto_save(self, session: _Session) -> None:
        """Save current annotations to parquet."""
        results = session.builder._results.copy()
        if not results:
            return
        row_dicts = [row.model_dump() for row in results.values()]
        df = pl.DataFrame(row_dicts)
        filepath = os.path.join(
            self._annotations_dir,
            f"autosave_{session.run_id}_{session.annotator_id}_{session.annotator_name}_data.parquet",
        )
        df.write_parquet(filepath)

    def create_session(self, annotator_name: str) -> CreateSessionResponse:
        """Create or resume an annotation session."""
        if not annotator_name or not annotator_name.strip():
            raise ValueError("Annotator name cannot be empty")

        run_id = _generate_run_id()
        annotator_id = _generate_annotator_id(annotator_name)
        sample_ids = self._get_sample_ids()
        required_tasks = self._eval_task.get_required_tasks()

        builder = HumanAnnotationResultsBuilder(
            run_id=run_id,
            annotator_id=annotator_id,
            task_schemas=self._eval_task.task_schemas,
            expected_ids=sample_ids,
            required_tasks=required_tasks,
            is_sampled_run=False,
        )

        # Check for auto-save to resume
        resumed = False
        annotated_count = 0
        auto_save_file = self._find_auto_save(annotator_name)
        if auto_save_file:
            try:
                saved_df = pl.read_parquet(auto_save_file)
                for row in saved_df.iter_rows(named=True):
                    if row.get("status") == "SUCCESS":
                        outcomes = {
                            k: v
                            for k, v in row.items()
                            if k in self._eval_task.task_schemas and v is not None
                        }
                        original_id = str(row["original_id"])
                        if original_id in builder._results:
                            del builder._results[original_id]
                        builder.create_success_row(
                            sample_example_id=row["sample_example_id"],
                            original_id=original_id,
                            outcomes=outcomes,
                            annotation_timestamp=row.get("annotation_timestamp"),
                        )
                resumed = True
                annotated_count = builder.completed_count
            except Exception as e:
                logger.warning(f"Failed to load auto-save: {e}")

        session = _Session(
            run_id=run_id,
            annotator_id=annotator_id,
            annotator_name=annotator_name,
            builder=builder,
            sample_ids=sample_ids,
        )
        self._sessions[run_id] = session

        return CreateSessionResponse(
            run_id=run_id,
            annotator_id=annotator_id,
            total_samples=len(sample_ids),
            resumed=resumed,
            annotated_count=annotated_count,
        )

    def get_session(self, run_id: str) -> CreateSessionResponse | None:
        """Get session info by run_id."""
        session = self._sessions.get(run_id)
        if session is None:
            return None
        return CreateSessionResponse(
            run_id=session.run_id,
            annotator_id=session.annotator_id,
            total_samples=len(session.sample_ids),
            resumed=False,
            annotated_count=session.builder.completed_count,
        )

    def get_sample(self, run_id: str, index: int) -> SampleResponse:
        """Get sample data at index."""
        session = self._sessions.get(run_id)
        if session is None:
            raise KeyError(f"Session not found: {run_id}")
        if index < 0 or index >= len(session.sample_ids):
            raise IndexError(f"Sample index out of bounds: {index}")

        sample_id = session.sample_ids[index]
        df = self._eval_data.data
        id_col = self._eval_data.id_column
        row = df.filter(pl.col(id_col).cast(str) == sample_id)

        # Build prompt and response data
        prompt_data = None
        if self._eval_task.prompt_columns:
            prompt_data = {
                col: str(row[col][0]) for col in self._eval_task.prompt_columns if col in row.columns
            }
        response_data = {
            col: str(row[col][0]) for col in self._eval_task.response_columns if col in row.columns
        }

        # Check for previous annotation
        previous_annotation = None
        existing = session.builder._results.get(sample_id)
        if existing:
            previous_annotation = {
                k: getattr(existing, k)
                for k in self._eval_task.task_schemas
                if getattr(existing, k, None) is not None
            }

        return SampleResponse(
            index=index,
            total=len(session.sample_ids),
            sample_id=sample_id,
            prompt_data=prompt_data,
            response_data=response_data,
            previous_annotation=previous_annotation,
        )

    def submit_annotation(
        self, run_id: str, sample_index: int, outcomes: dict[str, str]
    ) -> SubmitAnnotationResponse:
        """Submit annotation for a sample."""
        session = self._sessions.get(run_id)
        if session is None:
            raise KeyError(f"Session not found: {run_id}")
        if sample_index < 0 or sample_index >= len(session.sample_ids):
            raise IndexError(f"Sample index out of bounds: {sample_index}")

        sample_id = session.sample_ids[sample_index]

        # Remove existing if re-submitting
        if sample_id in session.builder._results:
            del session.builder._results[sample_id]

        session.builder.create_success_row(
            sample_example_id=f"{session.run_id}_{sample_index}",
            original_id=sample_id,
            outcomes=outcomes,
            annotation_timestamp=datetime.now(),
        )

        # Auto-save
        auto_saved = False
        try:
            self._auto_save(session)
            auto_saved = True
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")

        return SubmitAnnotationResponse(
            success=True,
            annotated_count=session.builder.completed_count,
            auto_saved=auto_saved,
        )

    def get_progress(self, run_id: str) -> ProgressResponse:
        """Get annotation progress."""
        session = self._sessions.get(run_id)
        if session is None:
            raise KeyError(f"Session not found: {run_id}")

        annotated_ids = set(session.builder._results.keys())
        incomplete = [
            i
            for i, sid in enumerate(session.sample_ids)
            if sid not in annotated_ids
        ]

        return ProgressResponse(
            run_id=run_id,
            annotated_count=session.builder.completed_count,
            total_samples=len(session.sample_ids),
            incomplete_indices=incomplete,
        )

    def export(self, run_id: str, annotator_name: str) -> ExportResponse:
        """Export completed annotations."""
        session = self._sessions.get(run_id)
        if session is None:
            raise KeyError(f"Session not found: {run_id}")

        results = session.builder.complete()

        metadata_filename = (
            f"{session.run_id}_{session.annotator_id}_{annotator_name}_metadata.json"
        )
        data_filename = (
            f"{session.run_id}_{session.annotator_id}_{annotator_name}_data.parquet"
        )

        results.save_state(
            state_file=os.path.join(self._annotations_dir, metadata_filename),
            data_format="parquet",
            data_filename=data_filename,
            data_dir=self._annotations_dir,
        )

        return ExportResponse(
            metadata_file=metadata_filename,
            data_file=data_filename,
            total_count=results.total_count,
            succeeded_count=results.succeeded_count,
            error_count=results.error_count,
        )
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/annotator/api/test_state.py -v
```

Expected: all pass.

- [ ] **Step 7: Lint, typecheck, full test suite**

```bash
uv tool run ruff check --preview --fix
uv tool run ruff format .
uv run pyright
uv run pytest -m "not integration"
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(annotator): add API schemas and session state manager"
```

---

### Task 3: FastAPI Routes

**Files:**
- Create: `src/meta_evaluator/annotator/api/routes/task.py`
- Create: `src/meta_evaluator/annotator/api/routes/session.py`
- Create: `src/meta_evaluator/annotator/api/routes/samples.py`
- Create: `src/meta_evaluator/annotator/api/routes/annotations.py`
- Create: `src/meta_evaluator/annotator/api/routes/export.py`
- Create: `src/meta_evaluator/annotator/api/app.py`
- Test: `tests/annotator/api/test_routes.py`

**Interfaces:**
- Consumes: `SessionStore` from Task 2, all schemas from Task 2
- Produces: `create_app(eval_task, eval_data, annotations_dir, static_dir=None) -> FastAPI` factory function

- [ ] **Step 1: Write failing route tests**

Create `tests/annotator/api/test_routes.py`:

```python
"""Tests for annotation API routes."""

import os

import polars as pl
import pytest
from fastapi.testclient import TestClient

from meta_evaluator.annotator.api.app import create_app
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask


@pytest.fixture
def eval_task() -> EvalTask:
    """Test EvalTask."""
    return EvalTask(
        task_schemas={
            "sentiment": ["positive", "negative", "neutral"],
            "quality": ["high", "medium", "low"],
        },
        prompt_columns=["text"],
        response_columns=["response"],
        answering_method="structured",
    )


@pytest.fixture
def eval_data() -> EvalData:
    """Test EvalData."""
    df = pl.DataFrame(
        {
            "id": ["s1", "s2", "s3"],
            "text": ["Hello", "World", "Test"],
            "response": ["Hi", "Earth", "Check"],
        }
    )
    return EvalData(name="test", data=df, id_column="id")


@pytest.fixture
def client(eval_task, eval_data, tmp_path) -> TestClient:
    """Create test client."""
    app = create_app(
        eval_task=eval_task,
        eval_data=eval_data,
        annotations_dir=str(tmp_path / "annotations"),
    )
    return TestClient(app)


class TestTaskRoute:
    """Tests for GET /api/task."""

    def test_get_task_config(self, client):
        """Returns task configuration."""
        resp = client.get("/api/task")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_schemas"]["sentiment"] == ["positive", "negative", "neutral"]
        assert data["prompt_columns"] == ["text"]
        assert data["response_columns"] == ["response"]
        assert "sentiment" in data["required_tasks"]


class TestSessionRoutes:
    """Tests for session endpoints."""

    def test_create_session(self, client):
        """POST /api/session creates session."""
        resp = client.post("/api/session", json={"annotator_name": "Alice"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"].startswith("annotation_run_")
        assert data["total_samples"] == 3

    def test_create_session_empty_name(self, client):
        """Empty name returns 422."""
        resp = client.post("/api/session", json={"annotator_name": ""})
        assert resp.status_code == 422

    def test_get_session(self, client):
        """GET /api/session/{run_id} returns session info."""
        create_resp = client.post("/api/session", json={"annotator_name": "Bob"})
        run_id = create_resp.json()["run_id"]
        resp = client.get(f"/api/session/{run_id}")
        assert resp.status_code == 200
        assert resp.json()["run_id"] == run_id

    def test_get_session_not_found(self, client):
        """Unknown run_id returns 404."""
        resp = client.get("/api/session/nonexistent")
        assert resp.status_code == 404


class TestSampleRoutes:
    """Tests for sample endpoints."""

    def test_get_sample(self, client):
        """GET /api/samples/{index} returns sample data."""
        create_resp = client.post("/api/session", json={"annotator_name": "Carol"})
        run_id = create_resp.json()["run_id"]
        resp = client.get(f"/api/samples/0?run_id={run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sample_id"] == "s1"
        assert data["response_data"]["response"] == "Hi"
        assert data["prompt_data"]["text"] == "Hello"

    def test_get_sample_out_of_bounds(self, client):
        """Index beyond total returns 404."""
        create_resp = client.post("/api/session", json={"annotator_name": "Dave"})
        run_id = create_resp.json()["run_id"]
        resp = client.get(f"/api/samples/99?run_id={run_id}")
        assert resp.status_code == 404

    def test_get_sample_no_session(self, client):
        """Missing run_id returns 404."""
        resp = client.get("/api/samples/0?run_id=bad_id")
        assert resp.status_code == 404


class TestAnnotationRoutes:
    """Tests for annotation submission."""

    def test_submit_annotation(self, client):
        """POST /api/annotations submits successfully."""
        create_resp = client.post("/api/session", json={"annotator_name": "Eve"})
        run_id = create_resp.json()["run_id"]
        resp = client.post(
            "/api/annotations",
            json={
                "run_id": run_id,
                "sample_index": 0,
                "outcomes": {"sentiment": "positive", "quality": "high"},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert resp.json()["annotated_count"] == 1

    def test_submit_invalid_session(self, client):
        """Unknown run_id returns 404."""
        resp = client.post(
            "/api/annotations",
            json={
                "run_id": "bad",
                "sample_index": 0,
                "outcomes": {"sentiment": "positive", "quality": "high"},
            },
        )
        assert resp.status_code == 404

    def test_get_progress(self, client):
        """GET /api/progress returns correct counts."""
        create_resp = client.post("/api/session", json={"annotator_name": "Frank"})
        run_id = create_resp.json()["run_id"]
        client.post(
            "/api/annotations",
            json={
                "run_id": run_id,
                "sample_index": 0,
                "outcomes": {"sentiment": "positive", "quality": "high"},
            },
        )
        resp = client.get(f"/api/progress?run_id={run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["annotated_count"] == 1
        assert data["total_samples"] == 3


class TestExportRoutes:
    """Tests for export endpoints."""

    def test_export_all_annotated(self, client, tmp_path):
        """POST /api/export produces files."""
        create_resp = client.post("/api/session", json={"annotator_name": "Grace"})
        run_id = create_resp.json()["run_id"]
        for i in range(3):
            client.post(
                "/api/annotations",
                json={
                    "run_id": run_id,
                    "sample_index": i,
                    "outcomes": {"sentiment": "positive", "quality": "high"},
                },
            )
        resp = client.post("/api/export", json={"run_id": run_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["succeeded_count"] == 3
        # Verify files exist
        annotations_dir = tmp_path / "annotations"
        assert os.path.exists(str(annotations_dir / data["metadata_file"]))
        assert os.path.exists(str(annotations_dir / data["data_file"]))

    def test_export_invalid_session(self, client):
        """Export unknown session returns 404."""
        resp = client.post("/api/export", json={"run_id": "bad"})
        assert resp.status_code == 404

    def test_download_file(self, client, tmp_path):
        """GET /api/export/download/{filename} returns file."""
        create_resp = client.post("/api/session", json={"annotator_name": "Hank"})
        run_id = create_resp.json()["run_id"]
        for i in range(3):
            client.post(
                "/api/annotations",
                json={
                    "run_id": run_id,
                    "sample_index": i,
                    "outcomes": {"sentiment": "positive", "quality": "high"},
                },
            )
        export_resp = client.post("/api/export", json={"run_id": run_id})
        data_file = export_resp.json()["data_file"]
        resp = client.get(f"/api/export/download/{data_file}")
        assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/annotator/api/test_routes.py -v
```

Expected: ImportError — `create_app` does not exist.

- [ ] **Step 3: Implement route modules**

Create `src/meta_evaluator/annotator/api/routes/task.py`:

```python
"""Task configuration route."""

from fastapi import APIRouter, Request

from meta_evaluator.annotator.api.schemas import TaskConfigResponse

router = APIRouter()


@router.get("/task", response_model=TaskConfigResponse)
def get_task_config(request: Request) -> TaskConfigResponse:
    """Return task configuration."""
    store = request.app.state.store
    eval_task = store._eval_task
    return TaskConfigResponse(
        task_schemas=eval_task.task_schemas,
        prompt_columns=eval_task.prompt_columns,
        response_columns=eval_task.response_columns,
        annotation_prompt=eval_task.annotation_prompt,
        required_tasks=eval_task.get_required_tasks(),
    )
```

Create `src/meta_evaluator/annotator/api/routes/session.py`:

```python
"""Session management routes."""

from fastapi import APIRouter, HTTPException, Request

from meta_evaluator.annotator.api.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
)

router = APIRouter()


@router.post("/session", response_model=CreateSessionResponse)
def create_session(
    body: CreateSessionRequest, request: Request
) -> CreateSessionResponse:
    """Create or resume annotation session."""
    store = request.app.state.store
    try:
        return store.create_session(body.annotator_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/session/{run_id}", response_model=CreateSessionResponse)
def get_session(run_id: str, request: Request) -> CreateSessionResponse:
    """Get existing session info."""
    store = request.app.state.store
    result = store.get_session(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result
```

Create `src/meta_evaluator/annotator/api/routes/samples.py`:

```python
"""Sample data routes."""

from fastapi import APIRouter, HTTPException, Request

from meta_evaluator.annotator.api.schemas import SampleResponse

router = APIRouter()


@router.get("/samples/{index}", response_model=SampleResponse)
def get_sample(index: int, run_id: str, request: Request) -> SampleResponse:
    """Get sample data at index."""
    store = request.app.state.store
    try:
        return store.get_sample(run_id, index)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except IndexError:
        raise HTTPException(status_code=404, detail="Sample index out of bounds")
```

Create `src/meta_evaluator/annotator/api/routes/annotations.py`:

```python
"""Annotation submission and progress routes."""

from fastapi import APIRouter, HTTPException, Request

from meta_evaluator.annotator.api.schemas import (
    ProgressResponse,
    SubmitAnnotationRequest,
    SubmitAnnotationResponse,
)

router = APIRouter()


@router.post("/annotations", response_model=SubmitAnnotationResponse)
def submit_annotation(
    body: SubmitAnnotationRequest, request: Request
) -> SubmitAnnotationResponse:
    """Submit annotation for a sample."""
    store = request.app.state.store
    try:
        return store.submit_annotation(body.run_id, body.sample_index, body.outcomes)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except IndexError:
        raise HTTPException(status_code=404, detail="Sample index out of bounds")


@router.get("/progress", response_model=ProgressResponse)
def get_progress(run_id: str, request: Request) -> ProgressResponse:
    """Get annotation progress."""
    store = request.app.state.store
    try:
        return store.get_progress(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
```

Create `src/meta_evaluator/annotator/api/routes/export.py`:

```python
"""Export routes."""

import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from meta_evaluator.annotator.api.schemas import ExportRequest, ExportResponse

router = APIRouter()


@router.post("/export", response_model=ExportResponse)
def export_annotations(body: ExportRequest, request: Request) -> ExportResponse:
    """Export completed annotations."""
    store = request.app.state.store
    session = store._sessions.get(body.run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        return store.export(body.run_id, session.annotator_name)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/export/download/{filename}")
def download_file(filename: str, request: Request) -> FileResponse:
    """Download exported file."""
    store = request.app.state.store
    filepath = os.path.join(store._annotations_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, filename=filename)
```

Create `src/meta_evaluator/annotator/api/routes/__init__.py`:

```python
"""API route modules."""
```

- [ ] **Step 4: Implement app factory**

Create `src/meta_evaluator/annotator/api/app.py`:

```python
"""FastAPI application factory for annotation interface."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from meta_evaluator.annotator.api.routes import (
    annotations,
    export,
    samples,
    session,
    task,
)
from meta_evaluator.annotator.api.state import SessionStore
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask


def create_app(
    eval_task: EvalTask,
    eval_data: EvalData,
    annotations_dir: str,
    static_dir: str | None = None,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        eval_task: Task configuration.
        eval_data: Evaluation data.
        annotations_dir: Directory for saving annotations.
        static_dir: Optional path to frontend static build directory.

    Returns:
        FastAPI: Configured application instance.
    """
    app = FastAPI(title="Meta-Evaluator Annotation Interface")

    # Initialize session store
    store = SessionStore(
        eval_task=eval_task,
        eval_data=eval_data,
        annotations_dir=annotations_dir,
    )
    app.state.store = store

    # Mount API routes
    app.include_router(task.router, prefix="/api")
    app.include_router(session.router, prefix="/api")
    app.include_router(samples.router, prefix="/api")
    app.include_router(annotations.router, prefix="/api")
    app.include_router(export.router, prefix="/api")

    # Mount static frontend files if provided
    if static_dir:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app
```

Update `src/meta_evaluator/annotator/api/__init__.py`:

```python
"""Annotation API module."""

from .app import create_app

__all__ = ["create_app"]
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/annotator/api/ -v
```

Expected: all pass.

- [ ] **Step 6: Lint, typecheck, full test suite**

```bash
uv tool run ruff check --preview --fix
uv tool run ruff format .
uv run pyright
uv run pytest -m "not integration"
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(annotator): add FastAPI routes for annotation API"
```

---

### Task 4: Launcher and MetaEvaluator Integration

**Files:**
- Create: `src/meta_evaluator/annotator/launcher/launcher.py`
- Modify: `src/meta_evaluator/annotator/launcher/__init__.py`
- Modify: `src/meta_evaluator/meta_evaluator/base.py:1-12,671-712`
- Create: `src/meta_evaluator/annotator/__init__.py`
- Test: `tests/annotator/test_launcher.py`

**Interfaces:**
- Consumes: `create_app` from Task 3
- Produces: `AnnotationLauncher` class with `launch(use_ngrok, traffic_policy_file)` method; updated `launch_annotator()` in `MetaEvaluator`

- [ ] **Step 1: Write failing launcher tests**

Create `tests/annotator/test_launcher.py`:

```python
"""Tests for annotation launcher."""

import socket
from unittest.mock import patch

import polars as pl
import pytest

from meta_evaluator.annotator.launcher import AnnotationLauncher
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask


@pytest.fixture
def eval_task() -> EvalTask:
    """Test EvalTask."""
    return EvalTask(
        task_schemas={"sentiment": ["positive", "negative"]},
        prompt_columns=["text"],
        response_columns=["response"],
        answering_method="structured",
    )


@pytest.fixture
def eval_data() -> EvalData:
    """Test EvalData."""
    df = pl.DataFrame(
        {"id": ["1"], "text": ["Hi"], "response": ["Hello"]}
    )
    return EvalData(name="test", data=df, id_column="id")


class TestAnnotationLauncher:
    """Tests for AnnotationLauncher."""

    def test_init_creates_annotations_dir(self, eval_task, eval_data, tmp_path):
        """Launcher creates annotations directory if missing."""
        annotations_dir = str(tmp_path / "new_dir")
        AnnotationLauncher(
            eval_data=eval_data,
            eval_task=eval_task,
            annotations_dir=annotations_dir,
        )
        assert (tmp_path / "new_dir").exists()

    def test_port_occupied_raises(self, eval_task, eval_data, tmp_path):
        """Raises PortOccupiedError when port is in use."""
        from meta_evaluator.annotator.exceptions import PortOccupiedError

        # Occupy a port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            occupied_port = s.getsockname()[1]
            s.listen(1)

            launcher = AnnotationLauncher(
                eval_data=eval_data,
                eval_task=eval_task,
                annotations_dir=str(tmp_path),
                port=occupied_port,
            )
            with pytest.raises(PortOccupiedError):
                launcher.launch()

    def test_launch_calls_uvicorn(self, eval_task, eval_data, tmp_path):
        """Launch starts uvicorn with correct arguments."""
        launcher = AnnotationLauncher(
            eval_data=eval_data,
            eval_task=eval_task,
            annotations_dir=str(tmp_path),
            port=9999,
        )
        with patch("uvicorn.run") as mock_run:
            launcher.launch()
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["port"] == 9999
            assert call_kwargs["host"] == "0.0.0.0"

    def test_traffic_policy_without_ngrok_raises(self, eval_task, eval_data, tmp_path):
        """Traffic policy without ngrok raises ValueError."""
        launcher = AnnotationLauncher(
            eval_data=eval_data,
            eval_task=eval_task,
            annotations_dir=str(tmp_path),
        )
        with pytest.raises(ValueError):
            launcher.launch(traffic_policy_file="policy.yml")
```

- [ ] **Step 2: Run tests to verify failure**

```bash
uv run pytest tests/annotator/test_launcher.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement AnnotationLauncher**

Create `src/meta_evaluator/annotator/launcher/launcher.py`:

```python
"""Launcher for the annotation interface."""

import logging
import os
import socket
import subprocess
import time

import uvicorn

from meta_evaluator.annotator.api.app import create_app
from meta_evaluator.annotator.exceptions import PortOccupiedError
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8000


class AnnotationLauncher:
    """Launches the annotation interface (FastAPI + React)."""

    def __init__(
        self,
        eval_data: EvalData,
        eval_task: EvalTask,
        annotations_dir: str,
        port: int | None = None,
    ):
        self.eval_task = eval_task
        self.eval_data = eval_data
        self.annotations_dir = annotations_dir
        self.port = port or _DEFAULT_PORT

        os.makedirs(self.annotations_dir, exist_ok=True)

    def _is_port_occupied(self) -> bool:
        """Check if port is in use."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(("localhost", self.port))
                return result == 0
        except OSError:
            return False

    def _get_static_dir(self) -> str | None:
        """Find frontend build directory."""
        candidate = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "frontend", "dist"
        )
        if os.path.isdir(candidate):
            return candidate
        return None

    def launch(
        self,
        use_ngrok: bool = False,
        traffic_policy_file: str | None = None,
    ) -> None:
        """Launch the annotation interface.

        Args:
            use_ngrok: Whether to expose via ngrok tunnel.
            traffic_policy_file: Optional ngrok traffic policy file path.

        Raises:
            ValueError: If traffic_policy_file given without use_ngrok.
            PortOccupiedError: If port is already in use.
        """
        if traffic_policy_file and not use_ngrok:
            raise ValueError(
                "Traffic policy file provided but ngrok is not being used."
            )

        if self._is_port_occupied():
            raise PortOccupiedError(self.port)

        app = create_app(
            eval_task=self.eval_task,
            eval_data=self.eval_data,
            annotations_dir=self.annotations_dir,
            static_dir=self._get_static_dir(),
        )

        if use_ngrok:
            self._launch_with_ngrok(app, traffic_policy_file)
        else:
            logger.info(f"Starting annotation server on port {self.port}")
            uvicorn.run(app, host="0.0.0.0", port=self.port)

    def _launch_with_ngrok(self, app, traffic_policy_file: str | None) -> None:
        """Launch with ngrok tunnel."""
        import threading

        server_thread = threading.Thread(
            target=uvicorn.run,
            kwargs={"app": app, "host": "0.0.0.0", "port": self.port},
            daemon=True,
        )
        server_thread.start()
        time.sleep(2)

        ngrok_cmd = ["ngrok", "http", str(self.port)]
        if traffic_policy_file:
            ngrok_cmd.extend(["--traffic-policy-file", traffic_policy_file])

        try:
            ngrok_process = subprocess.Popen(ngrok_cmd)
            ngrok_process.wait()
        except KeyboardInterrupt:
            pass
```

- [ ] **Step 4: Update launcher __init__.py**

```python
"""Launcher for the annotation interface."""

from .launcher import AnnotationLauncher

__all__ = ["AnnotationLauncher"]
```

- [ ] **Step 5: Create annotator __init__.py**

Create `src/meta_evaluator/annotator/__init__.py`:

```python
"""Annotation interface module."""

from .launcher import AnnotationLauncher

__all__ = ["AnnotationLauncher"]
```

- [ ] **Step 6: Update MetaEvaluator base.py**

Change import at line 9 from:

```python
from ..annotator.launcher import StreamlitLauncher
```

to:

```python
from ..annotator.launcher import AnnotationLauncher
```

Change `launch_annotator` method (lines 671-712) to:

```python
    def launch_annotator(
        self,
        port: int | None = None,
        use_ngrok: bool = False,
        traffic_policy_file: str | None = None,
    ) -> None:
        """Launch the annotation interface.

        Args:
            port: Optional port number for the server. If None, uses default port (8000).
            use_ngrok: Whether to use ngrok to expose the interface to the internet.
            traffic_policy_file: Optional path to an ngrok traffic policy file.
                Only used when use_ngrok=True.

        Raises:
            EvalTaskNotFoundError: If the evaluation task is not set.
            EvalDataNotFoundError: If the evaluation data is not set.
        """
        if self.eval_task is None:
            raise EvalTaskNotFoundError(
                "eval_task must be set before launching annotator"
            )

        if self.data is None:
            raise EvalDataNotFoundError("data must be set before launching annotator")

        launcher = AnnotationLauncher(
            eval_data=self.data,
            eval_task=self.eval_task,
            annotations_dir=str(self.paths.annotations),
            port=port,
        )

        launcher.launch(use_ngrok=use_ngrok, traffic_policy_file=traffic_policy_file)
```

- [ ] **Step 7: Run tests**

```bash
uv run pytest tests/annotator/test_launcher.py -v
uv run pytest -m "not integration"
```

Expected: all pass.

- [ ] **Step 8: Lint, typecheck**

```bash
uv tool run ruff check --preview --fix
uv tool run ruff format .
uv run pyright
```

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat(annotator): add launcher and update MetaEvaluator integration"
```

---

### Task 5: React Frontend Setup

**Files:**
- Create: `src/meta_evaluator/annotator/frontend/` (entire directory)

**Interfaces:**
- Consumes: API endpoints from Task 3 (GET /api/task, POST /api/session, etc.)
- Produces: React SPA that builds to `frontend/dist/`, served by FastAPI

- [ ] **Step 1: Initialize Vite + React + TypeScript project**

```bash
cd src/meta_evaluator/annotator
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

- [ ] **Step 2: Install Tailwind CSS and dependencies**

```bash
cd src/meta_evaluator/annotator/frontend
npm install -D tailwindcss @tailwindcss/vite
```

Update `vite.config.ts`:

```typescript
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
```

Replace `src/index.css` with:

```css
@import "tailwindcss";
```

- [ ] **Step 3: Initialize shadcn/ui**

```bash
cd src/meta_evaluator/annotator/frontend
npx shadcn@latest init -d
```

Then add components:

```bash
npx shadcn@latest add button card input progress radio-group textarea dialog badge scroll-area
```

- [ ] **Step 4: Create API client**

Create `src/meta_evaluator/annotator/frontend/src/lib/api.ts`:

```typescript
const BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(detail.detail || resp.statusText);
  }
  return resp.json();
}

export interface TaskConfig {
  task_schemas: Record<string, string[] | null>;
  prompt_columns: string[] | null;
  response_columns: string[];
  annotation_prompt: string;
  required_tasks: string[];
}

export interface SessionInfo {
  run_id: string;
  annotator_id: string;
  total_samples: number;
  resumed: boolean;
  annotated_count: number;
}

export interface Sample {
  index: number;
  total: number;
  sample_id: string;
  prompt_data: Record<string, string> | null;
  response_data: Record<string, string>;
  previous_annotation: Record<string, string> | null;
}

export interface SubmitResult {
  success: boolean;
  annotated_count: number;
  auto_saved: boolean;
}

export interface Progress {
  run_id: string;
  annotated_count: number;
  total_samples: number;
  incomplete_indices: number[];
}

export interface ExportResult {
  metadata_file: string;
  data_file: string;
  total_count: number;
  succeeded_count: number;
  error_count: number;
}

export const api = {
  getTask: () => request<TaskConfig>("/task"),

  createSession: (annotator_name: string) =>
    request<SessionInfo>("/session", {
      method: "POST",
      body: JSON.stringify({ annotator_name }),
    }),

  getSession: (run_id: string) => request<SessionInfo>(`/session/${run_id}`),

  getSample: (run_id: string, index: number) =>
    request<Sample>(`/samples/${index}?run_id=${run_id}`),

  submitAnnotation: (run_id: string, sample_index: number, outcomes: Record<string, string>) =>
    request<SubmitResult>("/annotations", {
      method: "POST",
      body: JSON.stringify({ run_id, sample_index, outcomes }),
    }),

  getProgress: (run_id: string) => request<Progress>(`/progress?run_id=${run_id}`),

  exportAnnotations: (run_id: string) =>
    request<ExportResult>("/export", {
      method: "POST",
      body: JSON.stringify({ run_id }),
    }),

  getDownloadUrl: (filename: string) => `${BASE}/export/download/${filename}`,
};
```

- [ ] **Step 5: Create useAnnotation hook**

Create `src/meta_evaluator/annotator/frontend/src/hooks/useAnnotation.ts`:

```typescript
import { useCallback, useEffect, useState } from "react";
import { api, type ExportResult, type Progress, type Sample, type SessionInfo, type TaskConfig } from "@/lib/api";

export type AppState = "name_entry" | "annotating" | "exported";

export function useAnnotation() {
  const [appState, setAppState] = useState<AppState>("name_entry");
  const [taskConfig, setTaskConfig] = useState<TaskConfig | null>(null);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [currentSample, setCurrentSample] = useState<Sample | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [exportResult, setExportResult] = useState<ExportResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.getTask().then(setTaskConfig).catch((e) => setError(e.message));
  }, []);

  const startSession = useCallback(async (name: string) => {
    setError(null);
    setLoading(true);
    try {
      const sess = await api.createSession(name);
      setSession(sess);
      const sample = await api.getSample(sess.run_id, 0);
      setCurrentSample(sample);
      const prog = await api.getProgress(sess.run_id);
      setProgress(prog);
      setAppState("annotating");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadSample = useCallback(
    async (index: number) => {
      if (!session) return;
      setLoading(true);
      try {
        const sample = await api.getSample(session.run_id, index);
        setCurrentSample(sample);
        setError(null);
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    },
    [session]
  );

  const submitAnnotation = useCallback(
    async (outcomes: Record<string, string>) => {
      if (!session || !currentSample) return;
      setError(null);
      try {
        await api.submitAnnotation(session.run_id, currentSample.index, outcomes);
        const prog = await api.getProgress(session.run_id);
        setProgress(prog);
        // Auto-advance to next sample
        if (currentSample.index < currentSample.total - 1) {
          await loadSample(currentSample.index + 1);
        }
      } catch (e: any) {
        setError(e.message);
      }
    },
    [session, currentSample, loadSample]
  );

  const doExport = useCallback(async () => {
    if (!session) return;
    setError(null);
    setLoading(true);
    try {
      const result = await api.exportAnnotations(session.run_id);
      setExportResult(result);
      setAppState("exported");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [session]);

  return {
    appState,
    taskConfig,
    session,
    currentSample,
    progress,
    exportResult,
    error,
    loading,
    startSession,
    loadSample,
    submitAnnotation,
    doExport,
  };
}
```

- [ ] **Step 6: Create components**

Create `src/meta_evaluator/annotator/frontend/src/components/NameEntry.tsx`:

```tsx
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useState } from "react";

interface Props {
  onSubmit: (name: string) => void;
  loading: boolean;
  error: string | null;
}

export function NameEntry({ onSubmit, loading, error }: Props) {
  const [name, setName] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (name.trim()) onSubmit(name.trim());
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Annotation Session</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <Input
                placeholder="Enter your name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                autoFocus
              />
              {error && <p className="text-sm text-red-500 mt-1">{error}</p>}
            </div>
            <Button type="submit" className="w-full" disabled={!name.trim() || loading}>
              {loading ? "Starting..." : "Start Annotating"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
```

Create `src/meta_evaluator/annotator/frontend/src/components/SampleDisplay.tsx`:

```tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { Sample, TaskConfig } from "@/lib/api";

interface Props {
  sample: Sample;
  taskConfig: TaskConfig;
}

export function SampleDisplay({ sample, taskConfig }: Props) {
  return (
    <ScrollArea className="h-full">
      <div className="space-y-4 pr-4">
        {sample.prompt_data && taskConfig.prompt_columns && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Prompt
              </CardTitle>
            </CardHeader>
            <CardContent>
              {Object.entries(sample.prompt_data).map(([col, value]) => (
                <div key={col} className="mb-3">
                  <p className="text-xs font-medium text-muted-foreground mb-1">
                    {col}
                  </p>
                  <p className="text-sm whitespace-pre-wrap">{value}</p>
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Response
            </CardTitle>
          </CardHeader>
          <CardContent>
            {Object.entries(sample.response_data).map(([col, value]) => (
              <div key={col} className="mb-3">
                <p className="text-xs font-medium text-muted-foreground mb-1">
                  {col}
                </p>
                <p className="text-sm whitespace-pre-wrap">{value}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </ScrollArea>
  );
}
```

Create `src/meta_evaluator/annotator/frontend/src/components/TaskPanel.tsx`:

```tsx
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Textarea } from "@/components/ui/textarea";
import type { Sample, TaskConfig } from "@/lib/api";
import { useEffect, useState } from "react";

interface Props {
  taskConfig: TaskConfig;
  sample: Sample;
  onSubmit: (outcomes: Record<string, string>) => void;
}

export function TaskPanel({ taskConfig, sample, onSubmit }: Props) {
  const [outcomes, setOutcomes] = useState<Record<string, string>>({});
  const [attempted, setAttempted] = useState(false);

  useEffect(() => {
    setOutcomes(sample.previous_annotation || {});
    setAttempted(false);
  }, [sample.index, sample.previous_annotation]);

  const handleSubmit = () => {
    setAttempted(true);
    const missing = taskConfig.required_tasks.filter((t) => !outcomes[t]?.trim());
    if (missing.length > 0) return;
    onSubmit(outcomes);
  };

  const isFieldMissing = (task: string) =>
    attempted && taskConfig.required_tasks.includes(task) && !outcomes[task]?.trim();

  return (
    <div className="space-y-6">
      <p className="text-sm text-muted-foreground">{taskConfig.annotation_prompt}</p>

      {Object.entries(taskConfig.task_schemas).map(([taskName, options]) => (
        <div key={taskName} className="space-y-2">
          <div className="flex items-center gap-2">
            <Label className="font-medium">{taskName}</Label>
            {taskConfig.required_tasks.includes(taskName) && (
              <Badge variant="outline" className="text-xs">
                required
              </Badge>
            )}
            {outcomes[taskName] && (
              <Badge variant="secondary" className="text-xs">
                done
              </Badge>
            )}
          </div>

          {options ? (
            <RadioGroup
              value={outcomes[taskName] || ""}
              onValueChange={(v) => setOutcomes((prev) => ({ ...prev, [taskName]: v }))}
              className={isFieldMissing(taskName) ? "border border-red-300 rounded p-2" : ""}
            >
              {options.map((option) => (
                <div key={option} className="flex items-center space-x-2 min-h-[44px]">
                  <RadioGroupItem value={option} id={`${taskName}-${option}`} />
                  <Label htmlFor={`${taskName}-${option}`} className="cursor-pointer">
                    {option}
                  </Label>
                </div>
              ))}
            </RadioGroup>
          ) : (
            <Textarea
              value={outcomes[taskName] || ""}
              onChange={(e) =>
                setOutcomes((prev) => ({ ...prev, [taskName]: e.target.value }))
              }
              placeholder="Enter your response..."
              className={isFieldMissing(taskName) ? "border-red-300" : ""}
            />
          )}
        </div>
      ))}

      <Button onClick={handleSubmit} className="w-full">
        Submit
      </Button>
    </div>
  );
}
```

Create `src/meta_evaluator/annotator/frontend/src/components/Navigation.tsx`:

```tsx
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import type { Progress as ProgressType, Sample } from "@/lib/api";

interface Props {
  sample: Sample;
  progress: ProgressType | null;
  onPrevious: () => void;
  onNext: () => void;
}

export function Navigation({ sample, progress, onPrevious, onNext }: Props) {
  const percent = progress
    ? (progress.annotated_count / progress.total_samples) * 100
    : 0;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <Progress value={percent} className="flex-1" />
        <span className="text-sm text-muted-foreground whitespace-nowrap">
          {progress?.annotated_count ?? 0} / {sample.total}
        </span>
      </div>
      <div className="flex justify-between">
        <Button
          variant="outline"
          onClick={onPrevious}
          disabled={sample.index === 0}
        >
          Previous
        </Button>
        <span className="text-sm text-muted-foreground self-center">
          Sample {sample.index + 1} of {sample.total}
        </span>
        <Button
          variant="outline"
          onClick={onNext}
          disabled={sample.index >= sample.total - 1}
        >
          Next
        </Button>
      </div>
    </div>
  );
}
```

Create `src/meta_evaluator/annotator/frontend/src/components/ExportDialog.tsx`:

```tsx
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ExportResult } from "@/lib/api";
import { api } from "@/lib/api";

interface Props {
  result: ExportResult;
}

export function ExportDialog({ result }: Props) {
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-lg">
        <CardHeader>
          <CardTitle>Export Complete</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-2 text-sm">
            <span className="text-muted-foreground">Total samples:</span>
            <span>{result.total_count}</span>
            <span className="text-muted-foreground">Succeeded:</span>
            <span>{result.succeeded_count}</span>
            <span className="text-muted-foreground">Errors:</span>
            <span>{result.error_count}</span>
          </div>
          <div className="space-y-2">
            <a
              href={api.getDownloadUrl(result.data_file)}
              download
              className="block"
            >
              <Button variant="outline" className="w-full">
                Download Data (.parquet)
              </Button>
            </a>
            <a
              href={api.getDownloadUrl(result.metadata_file)}
              download
              className="block"
            >
              <Button variant="outline" className="w-full">
                Download Metadata (.json)
              </Button>
            </a>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
```

Create `src/meta_evaluator/annotator/frontend/src/components/AnnotationView.tsx`:

```tsx
import type { Progress, Sample, TaskConfig } from "@/lib/api";
import { Navigation } from "./Navigation";
import { SampleDisplay } from "./SampleDisplay";
import { TaskPanel } from "./TaskPanel";

interface Props {
  taskConfig: TaskConfig;
  sample: Sample;
  progress: Progress | null;
  onSubmit: (outcomes: Record<string, string>) => void;
  onNavigate: (index: number) => void;
  onExport: () => void;
}

export function AnnotationView({
  taskConfig,
  sample,
  progress,
  onSubmit,
  onNavigate,
  onExport,
}: Props) {
  const canExport =
    progress && progress.annotated_count === progress.total_samples;

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <div className="border-b p-4">
        <Navigation
          sample={sample}
          progress={progress}
          onPrevious={() => onNavigate(sample.index - 1)}
          onNext={() => onNavigate(sample.index + 1)}
        />
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* Left: Sample display */}
        <div className="flex-1 p-4 overflow-y-auto md:border-r">
          <SampleDisplay sample={sample} taskConfig={taskConfig} />
        </div>

        {/* Right: Annotation panel */}
        <div className="w-full md:w-96 p-4 overflow-y-auto md:sticky md:top-0 md:h-[calc(100vh-80px)]">
          <TaskPanel
            taskConfig={taskConfig}
            sample={sample}
            onSubmit={onSubmit}
          />
          {canExport && (
            <button
              onClick={onExport}
              className="mt-4 w-full bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700"
            >
              Export Results
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: Wire up App.tsx**

Replace `src/meta_evaluator/annotator/frontend/src/App.tsx`:

```tsx
import { AnnotationView } from "@/components/AnnotationView";
import { ExportDialog } from "@/components/ExportDialog";
import { NameEntry } from "@/components/NameEntry";
import { useAnnotation } from "@/hooks/useAnnotation";

function App() {
  const {
    appState,
    taskConfig,
    currentSample,
    progress,
    exportResult,
    error,
    loading,
    startSession,
    loadSample,
    submitAnnotation,
    doExport,
  } = useAnnotation();

  if (appState === "name_entry") {
    return <NameEntry onSubmit={startSession} loading={loading} error={error} />;
  }

  if (appState === "exported" && exportResult) {
    return <ExportDialog result={exportResult} />;
  }

  if (appState === "annotating" && taskConfig && currentSample) {
    return (
      <AnnotationView
        taskConfig={taskConfig}
        sample={currentSample}
        progress={progress}
        onSubmit={submitAnnotation}
        onNavigate={loadSample}
        onExport={doExport}
      />
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center">
      <p className="text-muted-foreground">Loading...</p>
    </div>
  );
}

export default App;
```

- [ ] **Step 8: Build and verify**

```bash
cd src/meta_evaluator/annotator/frontend
npm run build
```

Expected: `dist/` directory created with `index.html` and assets.

- [ ] **Step 9: Add frontend/dist to .gitignore**

Add to the project root `.gitignore`:

```
src/meta_evaluator/annotator/frontend/dist/
src/meta_evaluator/annotator/frontend/node_modules/
```

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat(annotator): add React frontend with shadcn/ui"
```

---

### Task 6: Frontend Tests

**Files:**
- Create: `src/meta_evaluator/annotator/frontend/src/__tests__/` (test files)
- Modify: `src/meta_evaluator/annotator/frontend/package.json` (add vitest)

**Interfaces:**
- Consumes: Components from Task 5
- Produces: Passing frontend test suite

- [ ] **Step 1: Install test dependencies**

```bash
cd src/meta_evaluator/annotator/frontend
npm install -D vitest @testing-library/react @testing-library/jest-dom @testing-library/user-event jsdom @types/testing-library__jest-dom
```

Add to `vite.config.ts`:

```typescript
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./src/__tests__/setup.ts",
  },
});
```

Create `src/meta_evaluator/annotator/frontend/src/__tests__/setup.ts`:

```typescript
import "@testing-library/jest-dom/vitest";
```

Add test script to `package.json`:

```json
"scripts": {
  "dev": "vite",
  "build": "tsc -b && vite build",
  "preview": "vite preview",
  "test": "vitest"
}
```

- [ ] **Step 2: Write NameEntry test**

Create `src/meta_evaluator/annotator/frontend/src/__tests__/NameEntry.test.tsx`:

```typescript
import { NameEntry } from "@/components/NameEntry";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

describe("NameEntry", () => {
  it("renders input and submit button", () => {
    render(<NameEntry onSubmit={vi.fn()} loading={false} error={null} />);
    expect(screen.getByPlaceholderText("Enter your name")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /start annotating/i })).toBeInTheDocument();
  });

  it("disables button when name is empty", () => {
    render(<NameEntry onSubmit={vi.fn()} loading={false} error={null} />);
    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("calls onSubmit with trimmed name", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<NameEntry onSubmit={onSubmit} loading={false} error={null} />);
    await user.type(screen.getByPlaceholderText("Enter your name"), "  Alice  ");
    await user.click(screen.getByRole("button"));
    expect(onSubmit).toHaveBeenCalledWith("Alice");
  });

  it("shows error message", () => {
    render(<NameEntry onSubmit={vi.fn()} loading={false} error="Name is required" />);
    expect(screen.getByText("Name is required")).toBeInTheDocument();
  });

  it("shows loading state", () => {
    render(<NameEntry onSubmit={vi.fn()} loading={true} error={null} />);
    expect(screen.getByRole("button", { name: /starting/i })).toBeDisabled();
  });
});
```

- [ ] **Step 3: Write TaskPanel test**

Create `src/meta_evaluator/annotator/frontend/src/__tests__/TaskPanel.test.tsx`:

```typescript
import { TaskPanel } from "@/components/TaskPanel";
import type { Sample, TaskConfig } from "@/lib/api";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

const taskConfig: TaskConfig = {
  task_schemas: {
    sentiment: ["positive", "negative", "neutral"],
    comments: null,
  },
  prompt_columns: ["text"],
  response_columns: ["response"],
  annotation_prompt: "Evaluate this:",
  required_tasks: ["sentiment"],
};

const sample: Sample = {
  index: 0,
  total: 3,
  sample_id: "s1",
  prompt_data: { text: "Hello" },
  response_data: { response: "Hi" },
  previous_annotation: null,
};

describe("TaskPanel", () => {
  it("renders radio options for classification tasks", () => {
    render(<TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={vi.fn()} />);
    expect(screen.getByLabelText("positive")).toBeInTheDocument();
    expect(screen.getByLabelText("negative")).toBeInTheDocument();
    expect(screen.getByLabelText("neutral")).toBeInTheDocument();
  });

  it("renders textarea for free-form tasks", () => {
    render(<TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={vi.fn()} />);
    expect(screen.getByPlaceholderText("Enter your response...")).toBeInTheDocument();
  });

  it("does not submit when required fields empty", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={onSubmit} />);
    await user.click(screen.getByRole("button", { name: /submit/i }));
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("submits when required fields filled", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={onSubmit} />);
    await user.click(screen.getByLabelText("positive"));
    await user.click(screen.getByRole("button", { name: /submit/i }));
    expect(onSubmit).toHaveBeenCalledWith({ sentiment: "positive" });
  });

  it("pre-fills from previous annotation", () => {
    const sampleWithPrev = { ...sample, previous_annotation: { sentiment: "negative" } };
    render(<TaskPanel taskConfig={taskConfig} sample={sampleWithPrev} onSubmit={vi.fn()} />);
    const radio = screen.getByLabelText("negative") as HTMLInputElement;
    expect(radio).toBeChecked();
  });
});
```

- [ ] **Step 4: Write Navigation test**

Create `src/meta_evaluator/annotator/frontend/src/__tests__/Navigation.test.tsx`:

```typescript
import { Navigation } from "@/components/Navigation";
import type { Progress, Sample } from "@/lib/api";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

const sample: Sample = {
  index: 1,
  total: 5,
  sample_id: "s2",
  prompt_data: null,
  response_data: { response: "test" },
  previous_annotation: null,
};

const progress: Progress = {
  run_id: "run_1",
  annotated_count: 2,
  total_samples: 5,
  incomplete_indices: [0, 2, 4],
};

describe("Navigation", () => {
  it("shows progress count", () => {
    render(
      <Navigation sample={sample} progress={progress} onPrevious={vi.fn()} onNext={vi.fn()} />
    );
    expect(screen.getByText("2 / 5")).toBeInTheDocument();
  });

  it("disables previous on first sample", () => {
    const firstSample = { ...sample, index: 0 };
    render(
      <Navigation sample={firstSample} progress={progress} onPrevious={vi.fn()} onNext={vi.fn()} />
    );
    expect(screen.getByRole("button", { name: /previous/i })).toBeDisabled();
  });

  it("disables next on last sample", () => {
    const lastSample = { ...sample, index: 4 };
    render(
      <Navigation sample={lastSample} progress={progress} onPrevious={vi.fn()} onNext={vi.fn()} />
    );
    expect(screen.getByRole("button", { name: /next/i })).toBeDisabled();
  });

  it("calls onNext when clicked", async () => {
    const user = userEvent.setup();
    const onNext = vi.fn();
    render(
      <Navigation sample={sample} progress={progress} onPrevious={vi.fn()} onNext={onNext} />
    );
    await user.click(screen.getByRole("button", { name: /next/i }));
    expect(onNext).toHaveBeenCalled();
  });
});
```

- [ ] **Step 5: Run frontend tests**

```bash
cd src/meta_evaluator/annotator/frontend
npm test -- --run
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "test(annotator): add frontend component tests"
```

---

### Task 7: End-to-End Verification and Cleanup

**Files:**
- Modify: `src/meta_evaluator/annotator/frontend/src/main.tsx` (ensure correct entry)
- Verify: all tests pass, pyright clean, ruff clean

**Interfaces:**
- Consumes: everything from Tasks 1-6
- Produces: working annotation system (build + serve in single process)

- [ ] **Step 1: Build frontend**

```bash
cd src/meta_evaluator/annotator/frontend
npm run build
```

- [ ] **Step 2: Run full Python test suite**

```bash
uv run pytest -m "not integration" -v
```

Expected: all pass.

- [ ] **Step 3: Run full linting and type checking**

```bash
uv tool run ruff check --preview --fix
uv tool run ruff format .
uv run pyright
```

Expected: clean (zero errors).

- [ ] **Step 4: Manual smoke test**

```bash
cd examples/rabakbench
uv run python run_human_annotation.py
```

Open browser to `http://localhost:8501` (or 8000). Verify:
1. Name entry screen appears
2. Enter name → annotation view loads
3. Radio buttons and text areas render
4. Submit works and advances to next sample
5. Progress bar updates
6. Navigation prev/next works
7. Export creates downloadable files

- [ ] **Step 5: Verify mobile layout**

Open browser dev tools → toggle device toolbar → check:
- Stacked layout at <768px
- Tap targets ≥44px
- Sticky navigation at bottom

- [ ] **Step 6: Final commit if any cleanup needed**

```bash
uv tool run ruff format .
git add -A
git commit -m "chore: final cleanup for annotation UI migration"
```

---
