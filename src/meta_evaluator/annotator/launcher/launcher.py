"""Launcher for the annotation interface."""

import logging
import os

from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask

logger = logging.getLogger(__name__)


class AnnotationLauncher:
    """Launches the annotation interface (FastAPI + React)."""

    def __init__(
        self,
        eval_data: EvalData,
        eval_task: EvalTask,
        annotations_dir: str,
        port: int | None = None,
    ):
        """Initialize the launcher.

        Args:
            eval_data: Evaluation data to annotate.
            eval_task: Task configuration.
            annotations_dir: Directory to save annotations.
            port: Optional port number for the server.
        """
        self.eval_task = eval_task
        self.eval_data = eval_data
        self.annotations_dir = annotations_dir
        self.port = port or 8000

        os.makedirs(self.annotations_dir, exist_ok=True)

    def launch(
        self,
        use_ngrok: bool = False,
        traffic_policy_file: str | None = None,
    ) -> None:
        """Launch the annotation interface."""
        raise NotImplementedError("Full implementation in Task 4")
