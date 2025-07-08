"""Integration tests for the annotator system.

This module provides comprehensive integration tests for the human annotation system,
testing the complete workflow from launcher initialization through annotation completion
and results export. These tests verify that all components work together correctly
in realistic scenarios.

Test Coverage:
- App startup and port management
- Results creation, storage, and export workflows
- Data lifecycle management and cleanup
- Resource management (ports, temp directories)

Key Integration Points Tested:
- Launcher ↔ Streamlit app communication
- Session management ↔ Results storage
- File system operations and cleanup
- Process management and termination
- Port allocation and release

The tests use HTTP requests to verify app accessibility without requiring
browser automation dependencies, making them suitable for CI/CD environments
while still testing the core integration functionality.
"""

import pytest
import time
import socket
import subprocess
import threading
import requests
from datetime import datetime

import polars as pl

from meta_evaluator.eval_task import EvalTask
from meta_evaluator.data import EvalData
from meta_evaluator.annotator.launcher import StreamlitLauncher
from meta_evaluator.annotator.results import HumanAnnotationResults


# -------------------------
# Test Setup and Fixtures
# -------------------------


@pytest.fixture
def test_eval_task():
    """Create a test EvalTask for integration testing.

    Returns:
        EvalTask: A configured evaluation task with:
            - Mixed task types (structured sentiment/quality, no free-form)
            - Prompt and response column definitions
            - Structured answering method for radio button interactions
    """
    return EvalTask(
        task_schemas={
            "sentiment": ["positive", "negative", "neutral"],
            "quality": ["good", "bad"],
        },
        prompt_columns=["text"],
        response_columns=["response"],
        answering_method="structured",
    )


@pytest.fixture
def test_eval_data():
    """Create test EvalData for integration testing.

    Returns:
        EvalData: A test dataset containing:
            - 3 sample text/response pairs for annotation
            - Varied content (positive, negative, neutral sentiment)
            - Properly structured with ID column for tracking
    """
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


@pytest.fixture
def free_port():
    """Find and return a free port for testing.

    Returns:
        int: An available port number that can be used for the Streamlit
             app during testing without conflicts.
    """
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


def check_streamlit_app_running(port, timeout=30):
    """Check if Streamlit app is running and accessible via HTTP.

    Args:
        port: Port number to check for Streamlit app
        timeout: Maximum time to wait for app to become accessible

    Returns:
        bool: True if app responds to HTTP requests, False otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return False


def is_port_available(port):
    """Check if a port is available for binding.

    Args:
        port: Port number to check

    Returns:
        bool: True if port is available, False if occupied
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("localhost", port)) != 0


def wait_for_port(port, timeout=30):
    """Wait for a port to become occupied by a service.

    Args:
        port: Port number to monitor
        timeout: Maximum time to wait in seconds

    Returns:
        bool: True if port becomes occupied within timeout, False otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not is_port_available(port):
            return True
        time.sleep(0.5)
    return False


def wait_for_port_free(port, timeout=30):
    """Wait for a port to become free.

    Args:
        port: Port number to monitor
        timeout: Maximum time to wait in seconds

    Returns:
        bool: True if port becomes free within timeout, False otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_available(port):
            return True
        time.sleep(0.5)
    return False


# -------------------------
# Integration Test Class
# -------------------------


class StreamlitLauncherRunner:
    """Helper class to run StreamlitLauncher in a separate thread for testing.

    This class provides a controlled way to run the StreamlitLauncher in a background
    thread, allowing the test to continue execution while monitoring the launcher's
    status and handling cleanup properly.

    Attributes:
        launcher: The StreamlitLauncher instance to run
        thread: The background thread running the launcher
        exception: Any exception that occurred during launcher execution
    """

    def __init__(self, launcher):
        """Initialize the runner with the given StreamlitLauncher instance.

        Args:
            launcher: The StreamlitLauncher instance to run
        """
        self.launcher = launcher
        self.thread = None
        self.exception = None

    def start(self):
        """Start the launcher in a separate thread."""
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def _run(self):
        """Run the launcher and capture any exceptions."""
        try:
            self.launcher.launch(use_ngrok=False)
        except Exception as e:
            self.exception = e

    def is_alive(self):
        """Check if the launcher thread is still running.

        Returns:
            bool: True if the launcher thread is still running, False otherwise
        """
        return self.thread and self.thread.is_alive()

    def join(self, timeout=None):
        """Wait for the launcher thread to complete."""
        if self.thread:
            self.thread.join(timeout)


# -------------------------
# End-to-End Integration Test
# -------------------------


@pytest.mark.integration
def test_streamlit_launcher_basic_integration(
    test_eval_task, test_eval_data, free_port, tmp_path
):
    """Basic integration test for the complete annotation launcher workflow.

    This test verifies the core integration between the StreamlitLauncher and
    the annotation system without requiring browser automation. It tests the
    essential functionality from app startup through results processing.

    Test Scenarios:
    1. Launcher successfully starts Streamlit app at specified port
    2. App becomes accessible via HTTP requests
    3. Results system can create and save annotation data
    4. Data persistence works correctly (save/load cycle)
    5. Process cleanup releases ports and removes temporary files
    6. Annotation directory persists with saved results

    Args:
        test_eval_task: Fixture providing test evaluation task configuration
        test_eval_data: Fixture providing test data for annotation
        free_port: Fixture providing an available port for testing
        tmp_path: pytest fixture providing temporary directory
    """
    # Setup annotations directory
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()

    # Create launcher
    launcher = StreamlitLauncher(
        eval_data=test_eval_data,
        eval_task=test_eval_task,
        annotations_dir=str(annotations_dir),
        port=free_port,
    )

    # Start launcher in separate thread
    runner = StreamlitLauncherRunner(launcher)

    try:
        # Step 1: Launch Streamlit app
        runner.start()

        # Wait for port to be occupied
        assert wait_for_port(free_port, timeout=60), (
            f"Port {free_port} never became occupied"
        )

        # Step 2: Verify app is accessible via HTTP
        assert check_streamlit_app_running(free_port, timeout=30), (
            f"Streamlit app not accessible on port {free_port}"
        )

        # Step 3: Verify temporary files are created in annotations directory
        # Check that a temp directory was created
        time.sleep(1)  # Give it a moment to create temp files
        temp_dirs = [d for d in annotations_dir.iterdir() if d.is_dir()]
        assert len(temp_dirs) > 0, "No temporary directories created"

        # Step 4: Simulate user interaction by creating mock results
        # This tests the results system without browser automation
        from meta_evaluator.annotator.results import (
            HumanAnnotationResultsBuilder,
            HumanAnnotationResultsConfig,
        )

        config = HumanAnnotationResultsConfig(
            run_id="test_run_001",
            annotator_id="test_annotator",
            task_schemas=test_eval_task.task_schemas,
            timestamp_local=datetime.now(),
            is_sampled_run=False,
            expected_ids=["sample_1", "sample_2", "sample_3"],
        )

        builder = HumanAnnotationResultsBuilder(config)

        # Add mock annotations
        for sample_id in ["sample_1", "sample_2", "sample_3"]:
            builder.create_success_row(
                sample_example_id=sample_id,
                original_id=sample_id,
                outcomes={"sentiment": "positive", "quality": "good"},
                annotation_timestamp=datetime.now(),
            )

        # Complete and save results
        results = builder.complete()
        metadata_file = annotations_dir / "test_results_metadata.json"
        results.save_state(str(metadata_file))

        # Step 5: Verify results can be read back
        loaded_results = HumanAnnotationResults.load_state(str(metadata_file))
        assert loaded_results.annotator_id == "test_annotator"
        assert loaded_results.succeeded_count == 3
        assert loaded_results.error_count == 0
        assert loaded_results.total_count == 3

        # Verify annotation data
        successful_results = loaded_results.get_successful_results()
        assert len(successful_results) == 3

        # Check that all samples have the expected annotations
        for row in successful_results.iter_rows(named=True):
            assert row["sentiment"] == "positive"
            assert row["quality"] == "good"
            assert row["original_id"] in ["sample_1", "sample_2", "sample_3"]

    finally:
        # Cleanup: Terminate the launcher process if still running
        if runner.is_alive():
            # Force terminate by sending SIGTERM to the process
            try:
                # Get all processes using the port
                result = subprocess.run(
                    ["lsof", "-ti", f":{free_port}"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    pids = result.stdout.strip().split("\n")
                    for pid in pids:
                        if pid:
                            subprocess.run(["kill", "-TERM", pid], capture_output=True)
            except (
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                FileNotFoundError,
                OSError,
            ):
                # Expected failures during cleanup:
                # - CalledProcessError: lsof/kill commands fail (process already dead, permission denied)
                # - TimeoutExpired: commands hang
                # - FileNotFoundError: lsof/kill commands don't exist on system
                # - OSError: general system call failures
                pass

            # Wait for thread to finish
            runner.join(timeout=10)

    # Step 6: Verify port is released and tmp dir is deleted
    assert wait_for_port_free(free_port, timeout=30), (
        f"Port {free_port} was not released"
    )

    # Check that annotations directory still exists with results
    assert annotations_dir.exists(), "Annotations directory was deleted"
    assert len(list(annotations_dir.glob("*.json"))) > 0, (
        "Annotation files were deleted"
    )

    # Verify no temporary directories remain (they should be cleaned up)
    time.sleep(1)  # Give cleanup time to complete
    temp_dirs = [d for d in annotations_dir.iterdir() if d.is_dir()]
    assert len(temp_dirs) == 0, f"Temporary directories not cleaned up: {temp_dirs}"
