"""Launcher for the Streamlit annotation interface."""

import logging
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from meta_evaluator.annotator.exceptions import PortOccupiedError
from meta_evaluator.data import EvalData
from meta_evaluator.data.serialization import DataMetadata
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.eval_task.serialization import EvalTaskState


class StreamlitLauncher:
    """Launcher for the Streamlit annotation interface.

    This class handles launching the Streamlit interface in a separate process,
    managing the serialization and deserialization of data between processes.
    """

    def __init__(
        self,
        eval_data: EvalData,
        eval_task: EvalTask,
        annotations_dir: str,
        port: Optional[int] = None,
    ):
        """Initialize the StreamlitLauncher.

        Args:
            eval_task: The evaluation task configuration
            eval_data: The evaluation data to annotate
            annotations_dir: Directory to save annotations. Is used as the parent directory for any tmp folders created.
            port: Optional port number for Streamlit server
        """
        self.eval_task = eval_task
        self.eval_data = eval_data
        self.annotations_dir = annotations_dir
        self.port = port
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Create the annotations directory if it doesn't exist
        os.makedirs(self.annotations_dir, exist_ok=True)

    def _is_port_occupied(self) -> bool:
        """Check if a port is occupied.

        Returns:
            bool: True if port is occupied, False otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(("localhost", self.port))
                return result == 0
        except (OSError, socket.error) as e:
            self.logger.debug(f"Port check failed for {self.port}: {e}")
            return False

    def _save_files_for_annotations(self, tmp_dir: str) -> None:
        """Save the evaluation task and data metadata to the temporary directory.

        Args:
            tmp_dir: The string path to the temporary directory
        """
        # Serialize evaluation task and data metadata
        evaluation_task_state = self.eval_task.serialize()
        data_metadata = self.eval_data.serialize_metadata(
            data_format="parquet", data_filename="eval_data.parquet"
        )

        # Define file paths
        tmp_path = Path(tmp_dir)
        task_filepath = tmp_path / "evaltask.json"
        metadata_filepath = tmp_path / "evaldata_metadata.json"
        data_filepath = tmp_path / "evaldata.parquet"

        # Save the evaluation task and data metadata
        with open(task_filepath, "w") as f:
            f.write(evaluation_task_state.model_dump_json(indent=2))

        with open(metadata_filepath, "w") as f:
            f.write(data_metadata.model_dump_json(indent=2))

        self.eval_data.write_data(
            filepath=str(data_filepath),
            data_format="parquet",
        )

    @staticmethod
    def load_files_for_annotations(tmp_dir: str) -> tuple[EvalTask, EvalData]:
        """Load EvalData and EvalTask from a config file.

        Args:
            tmp_dir: Path to the temporary directory containing the serialized data

        Returns:
            tuple[EvalData, EvalTask]: The loaded data and task
        """
        # Define file paths
        tmp_path = Path(tmp_dir)
        task_filepath = tmp_path / "evaltask.json"
        metadata_filepath = tmp_path / "evaldata_metadata.json"
        data_filepath = tmp_path / "evaldata.parquet"

        # Load the evaluation task and eval data
        with open(task_filepath, "r") as f:
            evaluation_task_state = EvalTaskState.model_validate_json(f.read())
        eval_task = EvalTask.deserialize(evaluation_task_state)

        with open(metadata_filepath, "r") as f:
            data_metadata = DataMetadata.model_validate_json(f.read())
        df = EvalData.load_data(filepath=str(data_filepath), data_format="parquet")
        eval_data = EvalData.deserialize(data=df, metadata=data_metadata)

        return eval_task, eval_data

    def _create_streamlit_command(self) -> list[str]:
        """Create the command to launch the Streamlit interface.

        Returns:
            list[str]: The command to launch the Streamlit interface
        """
        cmd = ["streamlit", "run"]
        cmd.extend(["--server.address", "0.0.0.0"])
        if self.port:
            cmd.extend(["--server.port", str(self.port)])
        return cmd

    def _create_ngrok_command(
        self, traffic_policy_file: Optional[str] = None
    ) -> list[str]:
        """Create the command to launch ngrok.

        Args:
            traffic_policy_file: The path to the traffic policy file for ngrok (See https://ngrok.com/docs/traffic-policy/).

        Returns:
            list[str]: The command to launch ngrok
        """
        cmd = ["ngrok", "http"]

        if self.port:
            cmd.extend([str(self.port)])

        if traffic_policy_file:
            cmd.extend(["--traffic-policy-file", traffic_policy_file])

        return cmd

    def _launch_streamlit_locally(self, streamlit_cmd: List[str]) -> None:
        """Launch Streamlit locally.

        Args:
            streamlit_cmd: The command to launch Streamlit
        """
        subprocess.run(streamlit_cmd)

    def _launch_streamlit_with_ngrok(
        self, streamlit_cmd: List[str], traffic_policy_file: Optional[str] = None
    ) -> None:
        """Launch Streamlit with ngrok.

        Args:
            streamlit_cmd: The command to launch Streamlit
            traffic_policy_file: Optional path to an ngrok traffic policy file for advanced
                configuration. See https://ngrok.com/docs/traffic-policy/ for details.
                Only used when use_ngrok=True.
        """
        # Launch Streamlit in background
        streamlit_process = subprocess.Popen(
            streamlit_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            # Wait a moment for Streamlit to start
            time.sleep(2)

            # Launch ngrok in foreground
            ngrok_cmd = self._create_ngrok_command(
                traffic_policy_file=traffic_policy_file
            )
            ngrok_process = subprocess.Popen(ngrok_cmd)

            # Wait for ngrok to finish (user closes it)
            ngrok_process.wait()

        finally:
            # Clean up Streamlit process
            streamlit_process.terminate()
            try:
                streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                streamlit_process.kill()

    def launch(
        self,
        use_ngrok: bool = False,
        traffic_policy_file: Optional[str] = None,
    ) -> None:
        """Launch the Streamlit interface in a separate process.

        Args:
            use_ngrok: Whether to use ngrok to expose the Streamlit interface to the internet. Defaults to False.
            traffic_policy_file: Optional path to an ngrok traffic policy file for advanced
                configuration. See https://ngrok.com/docs/traffic-policy/ for details.
                Only used when use_ngrok=True.

        Raises:
            ValueError: If a traffic policy file is provided but ngrok is not being used.
            PortOccupiedError: If the specified port is already in use.
        """
        if traffic_policy_file and not use_ngrok:
            raise ValueError(
                "Traffic policy file provided but ngrok is not being used."
            )

        self.logger.info("Launching Streamlit annotation interface...")
        # Create temporary folder to store files needed for the annotation interface
        with tempfile.TemporaryDirectory(
            dir=self.annotations_dir, prefix="tmp_"
        ) as tmp_dir:
            # Save files for annotations
            self._save_files_for_annotations(tmp_dir)

            # Build streamlit command
            streamlit_cmd = self._create_streamlit_command()
            if use_ngrok:
                streamlit_cmd.extend(["--server.headless", "true"])

            # Get the path to the launch script
            launch_script = Path(__file__).parent / "entry_point.py"
            streamlit_cmd.extend([str(launch_script), tmp_dir])

            if self.port and self._is_port_occupied():
                raise PortOccupiedError(
                    f"Port {self.port} is already in use. Please specify a different port."
                )

            if use_ngrok:
                self._launch_streamlit_with_ngrok(
                    streamlit_cmd, traffic_policy_file=traffic_policy_file
                )
            else:
                self._launch_streamlit_locally(streamlit_cmd)
