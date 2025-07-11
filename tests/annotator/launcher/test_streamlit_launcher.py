"""Tests for StreamlitLauncher.

Covers:
- File serialization (saving/loading task/data files, temp file management)
- Process launching (command construction, process management)
- Script entry (argument parsing, error handling, integration)
"""

import pytest

import os
import sys
import socket
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import your real EvalTask and EvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.data import EvalData

from meta_evaluator.annotator.launcher import StreamlitLauncher
from meta_evaluator.annotator.exceptions import PortOccupiedError

import polars as pl

# -------------------------
# Fixtures for Real Objects
# -------------------------


@pytest.fixture
def basic_eval_task():
    """Provides a basic evaluation task for testing.

    Returns:
        EvalTask: A basic evaluation task.
    """
    return EvalTask(
        task_schemas={"sentiment": ["positive", "negative", "neutral"]},
        prompt_columns=["text"],
        response_columns=["response"],
        answering_method="structured",
    )


@pytest.fixture
def basic_eval_data():
    """Provides a basic EvalData object for testing.

    Returns:
        EvalData: A basic EvalData object.
    """
    df = pl.DataFrame(
        {
            "id": ["1", "2"],
            "text": ["I love this!", "I hate this!"],
            "response": ["Great!", "Awful!"],
        }
    )
    # EvalData expects id_column and data
    return EvalData(
        name="test_data",
        data=df,
        id_column="id",
    )


# -------------------------
# File Serialization
# -------------------------


def test_launcher_creates_annotations_dir(basic_eval_task, basic_eval_data, tmp_path):
    """Test that the StreamlitLauncher creates the annotations directory if it doesn't exist."""
    # Create a path that does not exist
    non_existent_dir = tmp_path / "new_annotations_dir"
    assert not non_existent_dir.exists()

    # Instantiate the launcher (should create the directory)
    StreamlitLauncher(
        eval_data=basic_eval_data,
        eval_task=basic_eval_task,
        annotations_dir=str(non_existent_dir),
    )

    # The directory should now exist
    assert non_existent_dir.exists()
    assert non_existent_dir.is_dir()


def test_save_and_load_files_for_annotations(
    basic_eval_task, basic_eval_data, tmp_path
):
    """Test that the StreamlitLauncher can save and load the evaluation task and data."""
    # Setup
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    launcher = StreamlitLauncher(
        eval_data=basic_eval_data,
        eval_task=basic_eval_task,
        annotations_dir=str(annotations_dir),
    )

    # Create a test directory for saving files
    test_dir = annotations_dir / "test_save_load"
    test_dir.mkdir()

    # Save
    launcher._save_files_for_annotations(str(test_dir))

    # Check files exist
    assert os.path.exists(os.path.join(test_dir, "evaltask.json"))
    assert os.path.exists(os.path.join(test_dir, "evaldata_metadata.json"))
    assert os.path.exists(os.path.join(test_dir, "evaldata.parquet"))

    # Load
    loaded_task, loaded_data = StreamlitLauncher.load_files_for_annotations(
        str(test_dir)
    )

    # Verify the loaded objects match the original
    assert loaded_task == basic_eval_task
    assert loaded_data.name == basic_eval_data.name
    assert loaded_data.data.equals(basic_eval_data.data)


# -------------------------
# Process Launching
# -------------------------

### Command Construction


def test_streamlit_command_is_safe(basic_eval_task, basic_eval_data, tmp_path):
    """Test that the Streamlit command is safe."""
    launcher = StreamlitLauncher(
        eval_data=basic_eval_data,
        eval_task=basic_eval_task,
        annotations_dir=str(tmp_path),
        port=12345,
    )
    cmd = launcher._create_streamlit_command()
    cmd_str = " ".join(cmd)
    assert cmd_str.startswith("streamlit run")
    assert cmd_str.endswith(f"--server.port {launcher.port}")


def test_ngrok_command_is_safe(basic_eval_task, basic_eval_data, tmp_path):
    """Test that the ngrok command is safe."""
    launcher = StreamlitLauncher(
        eval_data=basic_eval_data,
        eval_task=basic_eval_task,
        annotations_dir=str(tmp_path),
        port=12345,
    )
    cmd = launcher._create_ngrok_command()
    cmd_str = " ".join(cmd)
    assert cmd_str.startswith("ngrok http")
    assert cmd_str.endswith(f"{launcher.port}")


### Process Management


def test_streamlit_locally_process_launches(basic_eval_task, basic_eval_data, tmp_path):
    """Test that the Streamlit app process launches."""
    launcher = StreamlitLauncher(
        eval_data=basic_eval_data,
        eval_task=basic_eval_task,
        annotations_dir=str(tmp_path),
        port=12345,
    )
    with patch("subprocess.run") as mock_run:
        launcher._launch_streamlit_locally(["streamlit", "run", "dummy.py"])
        mock_run.assert_called_once()


def test_streamlit_with_ngrok_process_launches(
    basic_eval_task, basic_eval_data, tmp_path
):
    """Test that the Streamlit app with ngrok process launches two processes."""
    launcher = StreamlitLauncher(
        eval_data=basic_eval_data,
        eval_task=basic_eval_task,
        annotations_dir=str(tmp_path),
        port=12345,
    )
    with patch("subprocess.Popen") as mock_popen, patch("time.sleep"):
        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc
        launcher._launch_streamlit_with_ngrok(["streamlit", "run", "dummy.py"])
        assert mock_popen.call_count >= 2  # Streamlit and ngrok


# -------------------------
# Script Entry
# -------------------------

### Test access to entry_point.py file


def test_entry_point_file_exists():
    """Test that the entry_point.py file exists and is accessible."""
    import meta_evaluator.annotator.launcher.streamlit_launcher

    launcher_file = meta_evaluator.annotator.launcher.streamlit_launcher.__file__
    entry_point = Path(launcher_file).parent / "entry_point.py"

    assert entry_point.exists(), f"Entry point file not found: {entry_point}"
    assert entry_point.is_file(), f"Entry point path is not a file: {entry_point}"


def test_entry_point_excess_args(tmp_path):
    """Test that the entry point raises a ValueError when given excess arguments."""
    import meta_evaluator.annotator.launcher.streamlit_launcher

    launcher_file = meta_evaluator.annotator.launcher.streamlit_launcher.__file__
    entry_point = str(Path(launcher_file).parent / "entry_point.py")

    # Create a dummy tmp_dir
    tmp_dir = tmp_path / "dummy"
    tmp_dir.mkdir()

    # Provide too many arguments: tmp_dir and an extra one
    with patch("subprocess.run") as mock_run:
        # Simulate the result of running the entry point with excess args
        mock_result = MagicMock()
        mock_result.stderr = (
            "ValueError: Usage: python launch_streamlit_app.py <config_file_path>"
        )
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        result = subprocess.run(
            [sys.executable, entry_point, str(tmp_dir), "extra_arg"],
            capture_output=True,
            text=True,
        )
        assert "ValueError: Usage" in result.stderr
        assert result.returncode != 0


### Port handling


def test_launcher_raises_on_occupied_port(basic_eval_task, basic_eval_data, tmp_path):
    """Test that the StreamlitLauncher raises a PortOccupiedError when the specified port is already in use."""
    # Occupy a port
    sock = socket.socket()
    sock.bind(("localhost", 0))  # Let OS choose an available port
    sock.listen(1)
    port = sock.getsockname()[1]

    launcher = StreamlitLauncher(
        eval_data=basic_eval_data,
        eval_task=basic_eval_task,
        annotations_dir=str(tmp_path),
        port=port,
    )

    with patch("subprocess.run"):
        # Should raise PortOccupiedError when trying to launch
        with pytest.raises(PortOccupiedError, match=f"Port {port} is already in use"):
            launcher.launch(use_ngrok=False)

    # Clean up
    sock.close()


def is_port_open(port):
    """Helper to check if a port is open (occupied).

    Returns:
        bool: True if the port is open, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("localhost", port)) == 0


def test_streamlit_occupies_port_on_launch(basic_eval_task, basic_eval_data, tmp_path):
    """Test that launching a Streamlit app with a given port occupies the given port."""
    # Pick a free port
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    launcher = StreamlitLauncher(
        eval_data=basic_eval_data,
        eval_task=basic_eval_task,
        annotations_dir=str(tmp_path),
        port=port,
    )

    # Patch subprocess.run to simulate a long-running Streamlit process
    with patch("subprocess.run") as mock_run:

        def occupy_port(*args, **kwargs):
            # While this function is running, the port should be open
            # Simulate a process that binds to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                sock.listen(1)
                # Now the port is occupied, check it
                assert is_port_open(port)
            return 0  # Simulate process exit code

        mock_run.side_effect = occupy_port

        launcher.launch(use_ngrok=False)

    # After launch, port should be free again
    assert not is_port_open(port)


# # -------------------------
# # Launch Behaviour
# # -------------------------


def test_tmp_directory_created_and_deleted(basic_eval_task, basic_eval_data, tmp_path):
    """Test that the temporary directory is created and deleted when the StreamlitLauncher is launched."""
    launcher = StreamlitLauncher(
        eval_data=basic_eval_data,
        eval_task=basic_eval_task,
        annotations_dir=str(tmp_path),
        port=12345,
    )

    # Store instantiated tmp_dir
    tmp_dir_holder = {}

    def fake_save_files_for_annotations(tmp_dir):
        # Check if created tmp_dir exists
        tmp_dir_holder["tmp_dir"] = tmp_dir
        assert os.path.exists(tmp_dir)

    with (
        patch.object(
            launcher,
            "_save_files_for_annotations",
            side_effect=fake_save_files_for_annotations,
        ),
        patch.object(launcher, "_launch_streamlit_locally") as mock_launch,
    ):
        # Upon calling launch, a tmp_dir is created and passed into _save_files_for_annotations.
        # Check that the tmp_dir is created and exists.
        launcher.launch(use_ngrok=False)

        # After launch, the temp dir should be deleted
        tmp_dir = tmp_dir_holder["tmp_dir"]
        assert not os.path.exists(tmp_dir)
        mock_launch.assert_called_once()


def test_launch_calls_correct_function(basic_eval_task, basic_eval_data, tmp_path):
    """Test that the correct launch function is called based on the use_ngrok argument."""
    launcher = StreamlitLauncher(
        eval_data=basic_eval_data,
        eval_task=basic_eval_task,
        annotations_dir=str(tmp_path),
        port=12345,
    )

    with (
        patch.object(launcher, "_launch_streamlit_locally") as mock_local,
        patch.object(launcher, "_launch_streamlit_with_ngrok") as mock_ngrok,
    ):
        # Test use_ngrok=False
        launcher.launch(use_ngrok=False)
        mock_local.assert_called_once()
        mock_ngrok.assert_not_called()
        mock_local.reset_mock()
        mock_ngrok.reset_mock()

        # Test use_ngrok=True
        launcher.launch(use_ngrok=True)
        mock_ngrok.assert_called_once()
        mock_local.assert_not_called()
