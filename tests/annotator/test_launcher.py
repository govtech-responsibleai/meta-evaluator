"""Tests for annotation launcher."""

from unittest.mock import patch

import polars as pl
import pytest

from meta_evaluator.annotator.exceptions import PortOccupiedError
from meta_evaluator.annotator.launcher import AnnotationLauncher
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask


@pytest.fixture
def eval_task() -> EvalTask:
    """Test EvalTask.

    Returns:
        EvalTask: A configured evaluation task.
    """
    return EvalTask(
        task_schemas={"sentiment": ["positive", "negative"]},
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
    df = pl.DataFrame({"id": ["1"], "text": ["Hi"], "response": ["Hello"]})
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
        launcher = AnnotationLauncher(
            eval_data=eval_data,
            eval_task=eval_task,
            annotations_dir=str(tmp_path),
            port=9999,
        )
        with patch.object(launcher, "_is_port_occupied", return_value=True):
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
        with patch(
            "meta_evaluator.annotator.launcher.launcher.uvicorn.run"
        ) as mock_run:
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
