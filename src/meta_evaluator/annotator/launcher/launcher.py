"""Launcher for the annotation interface."""

import logging
import os
import socket
import subprocess
import time

import uvicorn

from meta_evaluator.annotator.api.app import create_app
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
        self.port = port or _DEFAULT_PORT

        os.makedirs(self.annotations_dir, exist_ok=True)

    def _is_port_occupied(self, port: int | None = None) -> bool:
        """Check if port is in use.

        Args:
            port: Port to check. Defaults to self.port.

        Returns:
            bool: True if port is occupied.
        """
        port = port or self.port
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(("localhost", port))
                return result == 0
        except OSError:
            return False

    def _find_available_port(self) -> int:
        """Find next available port starting from self.port + 1.

        Returns:
            int: An available port number.
        """
        port = self.port + 1
        while port < 65535:
            if not self._is_port_occupied(port):
                return port
            port += 1
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("localhost", 0))
            return sock.getsockname()[1]

    def _get_static_dir(self) -> str | None:
        """Find frontend build directory.

        Returns:
            str | None: Path to dist directory if it exists.
        """
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
        """
        if traffic_policy_file and not use_ngrok:
            raise ValueError(
                "Traffic policy file provided but ngrok is not being used."
            )

        if self._is_port_occupied():
            original_port = self.port
            self.port = self._find_available_port()
            logger.info(f"Port {original_port} in use, using port {self.port} instead")

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

    def _launch_with_ngrok(self, app, traffic_policy_file: str | None) -> None:  # type: ignore[type-arg]
        """Launch with ngrok tunnel.

        Args:
            app: FastAPI application instance.
            traffic_policy_file: Optional traffic policy file path.
        """
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
