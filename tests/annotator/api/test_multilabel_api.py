"""Tests for multi-label outcomes through the annotation API (Section 3)."""

import polars as pl
import pytest
from fastapi.testclient import TestClient

from meta_evaluator.annotator.api.app import create_app
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask, MultiLabelSchema


@pytest.fixture
def multilabel_eval_task() -> EvalTask:
    """EvalTask with a multi-label task plus a single-select task.

    Returns:
        EvalTask: The configured task.
    """
    return EvalTask(
        task_schemas={
            "harm": MultiLabelSchema(outcomes=["hateful", "insults", "sexual"]),
            "sentiment": ["positive", "negative"],
        },
        prompt_columns=["text"],
        response_columns=["response"],
        answering_method="structured",
    )


@pytest.fixture
def eval_data() -> EvalData:
    """A small test dataset.

    Returns:
        EvalData: The test dataset.
    """
    df = pl.DataFrame(
        {
            "id": ["s1", "s2"],
            "text": ["Hello", "World"],
            "response": ["Hi", "Earth"],
        }
    )
    return EvalData(name="test", data=df, id_column="id")


@pytest.fixture
def client(multilabel_eval_task, eval_data, tmp_path) -> TestClient:
    """FastAPI test client with a multi-label task.

    Returns:
        TestClient: The configured test client.
    """
    app = create_app(
        eval_task=multilabel_eval_task,
        eval_data=eval_data,
        annotations_dir=str(tmp_path / "annotations"),
    )
    return TestClient(app)


class TestMultiLabelTaskConfig:
    """The task config serializes a multi-label schema as {outcomes: [...]}."""

    def test_multilabel_schema_serialized_as_outcomes_object(self, client):
        """A MultiLabelSchema is serialized as {"outcomes": [...]}, not a bare list."""
        resp = client.get("/api/task")
        assert resp.status_code == 200
        schemas = resp.json()["task_schemas"]
        assert schemas["harm"] == {"outcomes": ["hateful", "insults", "sexual"]}
        # Single-select stays a bare list.
        assert schemas["sentiment"] == ["positive", "negative"]


class TestMultiLabelSubmitRetrieve:
    """Submitting and retrieving a vector-valued outcome through the API."""

    def test_submit_and_retrieve_vector(self, client):
        """A submitted ordered vector round-trips through previous_annotation."""
        run_id = client.post("/api/session", json={"annotator_name": "Alice"}).json()[
            "run_id"
        ]
        submit = client.post(
            "/api/annotations",
            json={
                "run_id": run_id,
                "sample_index": 0,
                "outcomes": {
                    "harm": ["hateful", "FALSE", "sexual"],
                    "sentiment": "positive",
                },
            },
        )
        assert submit.status_code == 200, submit.text
        assert submit.json()["success"] is True

        sample = client.get(f"/api/samples/0?run_id={run_id}").json()
        prev = sample["previous_annotation"]
        assert prev["harm"] == ["hateful", "FALSE", "sexual"]
        assert prev["sentiment"] == "positive"

    @pytest.mark.parametrize(
        "invalid_value",
        [
            "hateful",
            ["hateful"],
            ["FALSE", "hateful", "FALSE"],
            ["hateful", "unknown", "FALSE"],
        ],
    )
    def test_rejects_malformed_multilabel_outcome(self, client, invalid_value):
        """Malformed multi-label values are rejected before they are stored."""
        run_id = client.post("/api/session", json={"annotator_name": "Alice"}).json()[
            "run_id"
        ]

        response = client.post(
            "/api/annotations",
            json={
                "run_id": run_id,
                "sample_index": 0,
                "outcomes": {
                    "harm": invalid_value,
                    "sentiment": "positive",
                },
            },
        )

        assert response.status_code == 422
        progress = client.get(f"/api/progress?run_id={run_id}").json()
        assert progress["annotated_count"] == 0
