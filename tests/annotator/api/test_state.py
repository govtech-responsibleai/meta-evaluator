"""Tests for annotation API session state manager."""

import os

import polars as pl
import pytest

from meta_evaluator.annotator.api.state import SessionStore
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask


@pytest.fixture
def eval_task() -> EvalTask:
    """Create test EvalTask.

    Returns:
        EvalTask: A configured evaluation task.
    """
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
    """Create test EvalData.

    Returns:
        EvalData: A test dataset.
    """
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
    """Create a SessionStore instance.

    Returns:
        SessionStore: A configured session store.
    """
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
