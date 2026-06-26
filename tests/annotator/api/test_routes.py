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
    """Test EvalTask.

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
    """Test EvalData.

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
def client(eval_task, eval_data, tmp_path) -> TestClient:
    """Create test client.

    Returns:
        TestClient: FastAPI test client.
    """
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
