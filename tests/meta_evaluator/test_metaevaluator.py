"""Test suite for the MetaEvaluator base functionality."""

import json
import re
from unittest.mock import patch

import pytest

from meta_evaluator.common.error_constants import INVALID_JSON_STRUCTURE_MSG
from meta_evaluator.data import EvalData, SampleEvalData
from meta_evaluator.meta_evaluator import MetaEvaluator
from meta_evaluator.meta_evaluator.exceptions import (
    DataAlreadyExistsError,
    DataFormatError,
    EvalDataNotFoundError,
    EvalTaskAlreadyExistsError,
    EvalTaskNotFoundError,
    InvalidFileError,
)


@pytest.mark.integration
class TestMetaEvaluatorBase:
    """Test suite for MetaEvaluator base functionality (initialization, data, tasks, serialization)."""

    # === MetaEvaluator Initialization Tests ===

    def test_initialization_state(self, meta_evaluator):
        """Test that MetaEvaluator initializes with proper initial state."""
        assert meta_evaluator.data is None
        assert meta_evaluator.eval_task is None
        assert meta_evaluator.judge_registry == {}

    # === add_data() Method Tests ===

    def test_add_data_first_time(self, meta_evaluator, sample_eval_data):
        """Test adding data when no data exists."""
        assert meta_evaluator.data is None

        meta_evaluator.add_data(sample_eval_data)

        assert meta_evaluator.data == sample_eval_data

    def test_add_data_already_exists_no_overwrite(
        self, meta_evaluator, sample_eval_data, another_eval_data
    ):
        """Test DataAlreadyExistsError when data exists and overwrite is False (default)."""
        # Add data first time
        meta_evaluator.add_data(sample_eval_data)
        assert meta_evaluator.data == sample_eval_data

        # Try to add again without overwrite
        with pytest.raises(DataAlreadyExistsError, match="Data already exists"):
            meta_evaluator.add_data(another_eval_data)

        # Verify original data is unchanged
        assert meta_evaluator.data == sample_eval_data

    def test_add_data_already_exists_with_overwrite(
        self, meta_evaluator, sample_eval_data, another_eval_data
    ):
        """Test successful replacement when data exists and overwrite is True."""
        # Add data first time
        meta_evaluator.add_data(sample_eval_data)
        original_data = meta_evaluator.data
        assert original_data == sample_eval_data

        # Overwrite with new data
        meta_evaluator.add_data(another_eval_data, overwrite=True)
        new_data = meta_evaluator.data

        assert new_data == another_eval_data
        assert new_data != original_data

    # === add_eval_task() Method Tests ===

    def test_add_eval_task_first_time(self, meta_evaluator, basic_eval_task):
        """Test adding evaluation task when no task exists."""
        assert meta_evaluator.eval_task is None

        meta_evaluator.add_eval_task(basic_eval_task)

        assert meta_evaluator.eval_task == basic_eval_task

    def test_add_eval_task_already_exists_no_overwrite(
        self, meta_evaluator, basic_eval_task, another_basic_eval_task
    ):
        """Test EvalTaskAlreadyExistsError when task exists and overwrite is False (default)."""
        # Add task first time
        meta_evaluator.add_eval_task(basic_eval_task)
        assert meta_evaluator.eval_task == basic_eval_task

        # Try to add again without overwrite
        with pytest.raises(
            EvalTaskAlreadyExistsError, match="Evaluation task already exists"
        ):
            meta_evaluator.add_eval_task(another_basic_eval_task)

        # Verify original task is unchanged
        assert meta_evaluator.eval_task == basic_eval_task

    def test_add_eval_task_already_exists_with_overwrite(
        self, meta_evaluator, basic_eval_task, another_basic_eval_task
    ):
        """Test successful replacement when task exists and overwrite is True."""
        # Add task first time
        meta_evaluator.add_eval_task(basic_eval_task)
        original_task = meta_evaluator.eval_task
        assert original_task == basic_eval_task

        # Overwrite with new task
        meta_evaluator.add_eval_task(another_basic_eval_task, overwrite=True)
        new_task = meta_evaluator.eval_task

        assert new_task == another_basic_eval_task
        assert new_task != original_task

    def test_add_eval_task_no_prompt_columns(
        self, meta_evaluator, basic_eval_task_no_prompt
    ):
        """Test adding evaluation task with no prompt columns."""
        # Add evaluation task
        meta_evaluator.add_eval_task(basic_eval_task_no_prompt)

        # Verify task was added correctly
        assert meta_evaluator.eval_task == basic_eval_task_no_prompt
        assert meta_evaluator.eval_task.prompt_columns is None
        assert meta_evaluator.eval_task.response_columns == ["response"]

    def test_add_eval_task_empty_prompt_columns(
        self, meta_evaluator, basic_eval_task_empty_prompt
    ):
        """Test adding evaluation task with empty prompt columns."""
        # Add evaluation task
        meta_evaluator.add_eval_task(basic_eval_task_empty_prompt)

        # Verify task was added correctly
        assert meta_evaluator.eval_task == basic_eval_task_empty_prompt
        assert meta_evaluator.eval_task.prompt_columns == []
        assert meta_evaluator.eval_task.response_columns == ["response"]

    # === save_state() Method Tests ===

    def test_save_state_invalid_file_extension(self, meta_evaluator):
        """Test InvalidFileError when state_filename doesn't end with .json."""
        with pytest.raises(
            InvalidFileError, match="state_filename must end with .json"
        ):
            meta_evaluator.save_state("invalid_file.txt")

    def test_save_state_include_data_true_but_no_format(self, meta_evaluator):
        """Test DataFormatError when include_data=True but data_format is None."""
        with pytest.raises(
            DataFormatError,
            match="data_format must be specified when include_data=True",
        ):
            meta_evaluator.save_state("test.json", include_data=True, data_format=None)

    def test_save_state_include_data_false(self, meta_evaluator, mock_openai_client):
        """Test saving state without data serialization."""
        # Save without data
        meta_evaluator.save_state("test_state.json", include_data=False)

        # Verify state file exists and data file doesn't exist in data directory
        state_file = meta_evaluator.project_dir / "test_state.json"
        data_dir = meta_evaluator.project_dir / "data"

        assert state_file.exists()
        assert not (data_dir / "test_state_data.json").exists()
        assert not (data_dir / "test_state_data.csv").exists()
        assert not (data_dir / "test_state_data.parquet").exists()

        # Verify state file contents
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["version"] == "1.0"
        assert state_data["data"] is None

    @pytest.mark.parametrize("data_format", ["parquet", "csv", "json"])
    def test_save_state_with_data_formats(
        self, meta_evaluator, sample_eval_data, data_format
    ):
        """Test saving state with different data formats."""
        meta_evaluator.add_data(sample_eval_data)
        data_filename = f"test_state_data.{data_format}"

        meta_evaluator.save_state(
            "test_state.json",
            include_data=True,
            data_format=data_format,
            data_filename=data_filename,
        )

        # Verify both files exist
        state_file = meta_evaluator.project_dir / "test_state.json"
        data_file = meta_evaluator.project_dir / "data" / data_filename
        assert state_file.exists()

        # Verify data file exists (for real data)
        if data_format == "json":
            assert data_file.exists()

        # Verify state file contents
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["version"] == "1.0"
        assert state_data["data"]["data_format"] == data_format
        assert state_data["data"]["data_file"] == data_filename

    def test_save_state_no_data_present(self, meta_evaluator):
        """Test saving when include_data=True but self.data is None."""
        # Don't add any data to meta_evaluator
        assert meta_evaluator.data is None

        meta_evaluator.save_state(
            "test_state.json", include_data=True, data_format="csv"
        )

        # Verify state file exists but no data file
        state_file = meta_evaluator.project_dir / "test_state.json"
        data_dir = meta_evaluator.project_dir / "data"
        assert state_file.exists()
        assert not (data_dir / "test_state_data.parquet").exists()

        # Verify state file contents
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["data"] is None

    def test_save_state_include_task_false(self, meta_evaluator, mock_openai_client):
        """Test saving state without evaluation task serialization."""
        # Save without data
        meta_evaluator.save_state("test_state.json", include_data=False)

        # Verify state file exists and data file doesn't
        state_file = meta_evaluator.project_dir / "test_state.json"
        data_dir = meta_evaluator.project_dir / "data"
        assert state_file.exists()
        assert not (data_dir / "test_state_data.json").exists()
        assert not (data_dir / "test_state_data.csv").exists()
        assert not (data_dir / "test_state_data.parquet").exists()

        # Verify state file contents
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["version"] == "1.0"
        assert state_data["eval_task"] is None

    def test_save_state_creates_directories(self, meta_evaluator, tmp_path):
        """Test that parent directories are created when they don't exist."""
        nested_dir = tmp_path / "deep" / "nested" / "path"
        state_file = nested_dir / "test_state.json"

        # Directory shouldn't exist initially
        assert not nested_dir.exists()

        meta_evaluator.save_state(str(state_file), include_data=False)

        # Directory should be created and file should exist
        assert nested_dir.exists()
        assert state_file.exists()

    @pytest.mark.parametrize(
        "data_format,custom_filename",
        [
            ("json", "my_custom_data.json"),
            ("csv", "my_custom_data.csv"),
            ("parquet", "my_custom_data.parquet"),
        ],
    )
    def test_save_state_with_custom_data_filename(
        self,
        meta_evaluator,
        sample_eval_data,
        data_format,
        custom_filename,
    ):
        """Test saving with custom data filename for different formats."""
        meta_evaluator.add_data(sample_eval_data)

        meta_evaluator.save_state(
            "test_state.json",
            include_data=True,
            data_format=data_format,
            data_filename=custom_filename,
        )

        # Verify data file exists for JSON format
        expected_data_path = meta_evaluator.project_dir / "data" / custom_filename
        if data_format == "json":
            assert expected_data_path.exists()

        # Verify state file references custom filename
        state_file = meta_evaluator.project_dir / "test_state.json"
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["version"] == "1.0"
        assert state_data["data"]["data_file"] == custom_filename
        assert state_data["data"]["data_format"] == data_format

    @pytest.mark.parametrize(
        "data_format,wrong_filename,expected_extension",
        [
            ("json", "wrong_extension.csv", ".json"),
            ("csv", "wrong_extension.json", ".csv"),
            ("parquet", "wrong_extension.csv", ".parquet"),
        ],
    )
    def test_save_state_data_filename_extension_mismatch(
        self, meta_evaluator, data_format, wrong_filename, expected_extension
    ):
        """Test DataFormatError for format mismatches."""
        with pytest.raises(
            DataFormatError,
            match=f"Data filename '{wrong_filename}' must have extension '{expected_extension}' when data_format is '{data_format}'",
        ):
            meta_evaluator.save_state(
                "test.json",
                include_data=True,
                data_format=data_format,
                data_filename=wrong_filename,
            )

    def test_save_state_data_filename_no_validation_when_include_data_false(
        self, meta_evaluator
    ):
        """Test that data_filename extension is not validated when include_data=False."""
        # This should not raise an exception even with wrong extension
        # because data_filename is ignored when include_data=False
        meta_evaluator.save_state(
            "test.json", include_data=False, data_filename="wrong_extension.csv"
        )

    def test_save_state_data_filename_no_validation_when_data_format_none(
        self, meta_evaluator
    ):
        """Test that data_filename extension is not validated when data_format=None."""
        # This should not raise an exception because data_format is None
        # (will raise DataFormatError for missing data_format instead)
        with pytest.raises(
            DataFormatError,
            match="data_format must be specified when include_data=True",
        ):
            meta_evaluator.save_state(
                "test.json",
                include_data=True,
                data_format=None,
                data_filename="any_name.csv",
            )

    def test_save_state_fallback_to_auto_generated_when_data_filename_none(
        self, meta_evaluator, sample_eval_data
    ):
        """Test that auto-generated filename is used when data_filename=None."""
        meta_evaluator.add_data(sample_eval_data)

        meta_evaluator.save_state(
            "test_state.json",
            include_data=True,
            data_format="json",
            data_filename=None,  # Explicitly set to None
        )

        # Verify auto-generated filename is used
        state_file = meta_evaluator.project_dir / "test_state.json"
        auto_generated_file = (
            meta_evaluator.project_dir / "data" / "test_state_data.json"
        )
        assert auto_generated_file.exists()

        # Verify state file references auto-generated filename
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["data"]["data_file"] == "test_state_data.json"

    def test_save_and_load_eval_task_no_prompt_columns(
        self, meta_evaluator, basic_eval_task_no_prompt
    ):
        """Test saving and loading MetaEvaluator with EvalTask that has no prompt columns."""
        # Add task to evaluator
        meta_evaluator.add_eval_task(basic_eval_task_no_prompt)

        # Save state
        meta_evaluator.save_state(
            "test_state.json",
            include_task=True,
            include_data=True,
            data_format="json",
            data_filename=None,
        )

        # Load state
        loaded_evaluator = MetaEvaluator.load_state(
            project_dir=str(meta_evaluator.project_dir),
            state_filename="test_state.json",
            load_task=True,
        )

        # Verify task was loaded correctly with no prompt columns
        assert loaded_evaluator.eval_task is not None
        assert loaded_evaluator.eval_task.prompt_columns is None
        assert loaded_evaluator.eval_task.response_columns == ["response"]
        assert loaded_evaluator.eval_task.task_schemas == {
            "sentiment": ["positive", "negative", "neutral"]
        }
        assert loaded_evaluator.eval_task.answering_method == "structured"

    def test_save_and_load_eval_task_empty_prompt_columns(
        self, meta_evaluator, basic_eval_task_empty_prompt
    ):
        """Test saving and loading MetaEvaluator with EvalTask that has empty prompt columns."""
        # Add task to evaluator
        meta_evaluator.add_eval_task(basic_eval_task_empty_prompt)

        # Save state
        meta_evaluator.save_state(
            "test_state.json",
            include_task=True,
            include_data=True,
            data_format="json",
            data_filename=None,
        )

        # Load state
        loaded_evaluator = MetaEvaluator.load_state(
            project_dir=str(meta_evaluator.project_dir),
            state_filename="test_state.json",
            load_task=True,
        )

        # Verify task was loaded correctly with empty prompt columns
        assert loaded_evaluator.eval_task is not None
        assert loaded_evaluator.eval_task.prompt_columns == []
        assert loaded_evaluator.eval_task.response_columns == ["response"]
        assert loaded_evaluator.eval_task.task_schemas == {
            "sentiment": ["positive", "negative", "neutral"]
        }
        assert loaded_evaluator.eval_task.answering_method == "structured"

    # === Private Serialization Method Tests ===

    def test_serialize_include_data_false(self, meta_evaluator):
        """Test _serialize when data should not be included."""
        state = meta_evaluator._serialize(
            include_task=True,
            include_data=False,
            data_format=None,
            data_filename=None,
        )

        assert state.version == "1.0"
        assert state.data is None
        assert state.eval_task is None

    def test_serialize_include_data_true(self, meta_evaluator, sample_eval_data):
        """Test _serialize when data should be included."""
        meta_evaluator.add_data(sample_eval_data)

        state = meta_evaluator._serialize(
            include_task=True,
            include_data=True,
            data_format="parquet",
            data_filename="test_data.parquet",
        )

        assert state.version == "1.0"
        assert state.data is not None
        assert state.data.data_format == "parquet"
        assert state.data.data_file == "test_data.parquet"
        assert state.eval_task is None

    def test_serialize_include_task_false(self, meta_evaluator, basic_eval_task):
        """Test _serialize when task should not be included."""
        state = meta_evaluator._serialize(
            include_task=False,
            include_data=False,
            data_format=None,
            data_filename=None,
        )

        assert state.version == "1.0"
        assert state.data is None
        assert state.eval_task is None

    def test_serialize_include_task_true(self, meta_evaluator, basic_eval_task):
        """Test _serialize when task should be included."""
        meta_evaluator.add_eval_task(basic_eval_task)

        state = meta_evaluator._serialize(
            include_task=True,
            include_data=False,
            data_format=None,
            data_filename=None,
        )

        assert state.version == "1.0"
        assert state.data is None
        assert state.eval_task is not None

    # === Integration Tests ===

    def test_save_and_verify_complete_file_contents(
        self,
        meta_evaluator,
        sample_eval_data,
        basic_eval_task,
    ):
        """Test complete save operation and verify actual file contents match expected structure."""
        # Set up complete MetaEvaluator state
        meta_evaluator.add_data(sample_eval_data)
        meta_evaluator.add_eval_task(basic_eval_task)

        # Add judges to the evaluator
        from meta_evaluator.common.models import Prompt

        prompt1 = Prompt(id="test_prompt_1", prompt="Evaluate sentiment")
        prompt2 = Prompt(id="test_prompt_2", prompt="Evaluate toxicity")

        meta_evaluator.add_judge("judge_1", "openai", "gpt-4", prompt1)
        meta_evaluator.add_judge("judge_2", "anthropic", "claude-3", prompt2)

        # Save state
        meta_evaluator.save_state(
            "integration_test.json", include_data=True, data_format="parquet"
        )

        # Verify files exist
        state_file = meta_evaluator.project_dir / "integration_test.json"
        assert state_file.exists()

        # Verify state file structure
        with open(state_file) as f:
            state_data = json.load(f)

        # Verify top-level structure
        assert state_data["version"] == "1.0"
        assert "data" in state_data
        assert "eval_task" in state_data
        assert "judge_registry" in state_data

        # Verify data structure
        data_config = state_data["data"]
        assert data_config["name"] == "sample_test_data"
        assert data_config["id_column"] == "sample_id"
        assert data_config["data_format"] == "parquet"
        assert data_config["data_file"] == "integration_test_data.parquet"
        assert data_config["type"] == "SampleEvalData"

        # Verify evaluation task structure
        eval_task_config = state_data["eval_task"]
        assert eval_task_config["task_schemas"] == {
            "sentiment": ["positive", "negative", "neutral"]
        }
        assert eval_task_config["prompt_columns"] == ["text"]
        assert eval_task_config["response_columns"] == ["response"]
        assert eval_task_config["answering_method"] == "structured"

        # Verify judge registry structure
        judge_registry = state_data["judge_registry"]
        assert len(judge_registry) == 2
        assert "judge_1" in judge_registry
        assert "judge_2" in judge_registry

        # Verify individual judge structure
        judge_1_data = judge_registry["judge_1"]
        assert judge_1_data["id"] == "judge_1"
        assert judge_1_data["llm_client"] == "openai"
        assert judge_1_data["model"] == "gpt-4"
        assert judge_1_data["prompt"]["id"] == "test_prompt_1"
        assert judge_1_data["prompt"]["prompt"] == "Evaluate sentiment"

        judge_2_data = judge_registry["judge_2"]
        assert judge_2_data["id"] == "judge_2"
        assert judge_2_data["llm_client"] == "anthropic"
        assert judge_2_data["model"] == "claude-3"
        assert judge_2_data["prompt"]["id"] == "test_prompt_2"
        assert judge_2_data["prompt"]["prompt"] == "Evaluate toxicity"

    def test_save_with_complex_file_paths(self, meta_evaluator, tmp_path):
        """Test saving with complex file paths including spaces and nested directories."""
        # Test with complex filename path (subdirectories within project_dir)
        complex_filename = "test dir/with spaces/and-dashes/complex file name.json"

        meta_evaluator.save_state(complex_filename, include_data=False)

        # Verify subdirectories were created and file exists within project directory
        state_file = meta_evaluator.project_dir / complex_filename
        assert state_file.exists()
        assert state_file.parent.exists()  # Verify subdirectories were created

        # Verify file is valid JSON
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["version"] == "1.0"

    # === load_state() Method Tests ===

    def test_load_state_with_eval_data_json_format(
        self, tmp_path, sample_eval_data, basic_eval_task, mock_openai_client
    ):
        """Test loading MetaEvaluator with EvalData in JSON format."""
        # Create and save evaluator with data and evaluation task
        original_evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        original_evaluator.add_data(sample_eval_data)
        original_evaluator.add_eval_task(basic_eval_task)

        original_evaluator.save_state(
            "test_state.json", include_data=True, data_format="json"
        )

        # Load from JSON
        loaded_evaluator = MetaEvaluator.load_state(
            project_dir=str(original_evaluator.project_dir),
            state_filename="test_state.json",
            load_data=True,
        )

        # Verify data was loaded
        assert loaded_evaluator.data is not None
        assert loaded_evaluator.data.name == sample_eval_data.name
        assert loaded_evaluator.data.id_column == sample_eval_data.id_column
        assert isinstance(loaded_evaluator.data, EvalData)

    def test_load_state_with_sample_eval_data_csv_format(
        self,
        tmp_path,
        sample_eval_data,
        basic_eval_task,
        mock_openai_client,
    ):
        """Test loading MetaEvaluator with SampleEvalData in CSV format."""
        # Create and save evaluator with sample data
        original_evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        original_evaluator.add_data(sample_eval_data)
        original_evaluator.add_eval_task(basic_eval_task)

        original_evaluator.save_state(
            "test_state.json", include_data=True, data_format="csv"
        )

        # Ensure the data file exists in the data directory
        data_file = original_evaluator.project_dir / "data" / "test_state_data.csv"
        assert data_file.exists()

        # Load from JSON
        loaded_evaluator = MetaEvaluator.load_state(
            project_dir=str(original_evaluator.project_dir),
            state_filename="test_state.json",
            load_data=True,
        )

        # Verify sample data was loaded with all metadata
        assert loaded_evaluator.data is not None
        assert isinstance(loaded_evaluator.data, SampleEvalData)
        assert loaded_evaluator.data.sample_name == sample_eval_data.sample_name
        assert (
            loaded_evaluator.data.stratification_columns
            == sample_eval_data.stratification_columns
        )
        assert (
            loaded_evaluator.data.sample_percentage
            == sample_eval_data.sample_percentage
        )
        assert loaded_evaluator.data.seed == sample_eval_data.seed
        assert loaded_evaluator.data.sampling_method == sample_eval_data.sampling_method

    def test_load_state_with_eval_task(
        self, tmp_path, sample_eval_data, basic_eval_task, mock_openai_client
    ):
        """Test loading MetaEvaluator with evaluation task from JSON."""
        # Create and save evaluator with data and evaluation task
        original_evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        original_evaluator.add_data(sample_eval_data)
        original_evaluator.add_eval_task(basic_eval_task)

        original_evaluator.save_state(
            "test_state.json", include_data=True, data_format="json"
        )

        # Load from JSON
        loaded_evaluator = MetaEvaluator.load_state(
            project_dir=str(original_evaluator.project_dir),
            state_filename="test_state.json",
            load_data=True,
        )

        # Verify evaluation task was loaded
        assert loaded_evaluator.eval_task is not None
        assert loaded_evaluator.eval_task.task_schemas == basic_eval_task.task_schemas
        assert (
            loaded_evaluator.eval_task.prompt_columns == basic_eval_task.prompt_columns
        )
        assert (
            loaded_evaluator.eval_task.response_columns
            == basic_eval_task.response_columns
        )
        assert (
            loaded_evaluator.eval_task.answering_method
            == basic_eval_task.answering_method
        )

    def test_load_state_with_judge_registry(
        self, tmp_path, sample_eval_data, basic_eval_task
    ):
        """Test loading MetaEvaluator with judge registry from JSON."""
        # Create and save evaluator with data, evaluation task, and judges
        original_evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        original_evaluator.add_data(sample_eval_data)
        original_evaluator.add_eval_task(basic_eval_task)

        # Add judges to the evaluator
        from meta_evaluator.common.models import Prompt

        prompt1 = Prompt(id="test_prompt_1", prompt="Evaluate sentiment")
        prompt2 = Prompt(id="test_prompt_2", prompt="Evaluate toxicity")

        original_evaluator.add_judge("judge_1", "openai", "gpt-4", prompt1)
        original_evaluator.add_judge("judge_2", "anthropic", "claude-3", prompt2)

        original_evaluator.save_state(
            "test_state_with_judges.json", include_data=True, data_format="json"
        )

        # Load from JSON
        loaded_evaluator = MetaEvaluator.load_state(
            project_dir=str(original_evaluator.project_dir),
            state_filename="test_state_with_judges.json",
            load_data=True,
        )

        # Verify judge registry was loaded correctly
        assert len(loaded_evaluator.judge_registry) == 2
        assert "judge_1" in loaded_evaluator.judge_registry
        assert "judge_2" in loaded_evaluator.judge_registry

        # Verify judges are actual Judge instances
        from meta_evaluator.judge import Judge

        assert isinstance(loaded_evaluator.judge_registry["judge_1"], Judge)
        assert isinstance(loaded_evaluator.judge_registry["judge_2"], Judge)

        # Verify judge configurations match original
        judge_1 = loaded_evaluator.judge_registry["judge_1"]
        assert judge_1.id == "judge_1"
        assert judge_1.llm_client == "openai"
        assert judge_1.model == "gpt-4"
        assert judge_1.prompt.id == "test_prompt_1"
        assert judge_1.prompt.prompt == "Evaluate sentiment"

        judge_2 = loaded_evaluator.judge_registry["judge_2"]
        assert judge_2.id == "judge_2"
        assert judge_2.llm_client == "anthropic"
        assert judge_2.model == "claude-3"
        assert judge_2.prompt.id == "test_prompt_2"
        assert judge_2.prompt.prompt == "Evaluate toxicity"

        # Verify judges are frozen (immutable)
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError, match="Instance is frozen"):
            judge_1.id = "new_id"

    def test_load_state_skip_data_and_eval_task_loading(
        self, tmp_path, sample_eval_data, basic_eval_task, mock_openai_client
    ):
        """Test loading MetaEvaluator while skipping data loading."""
        # Create and save evaluator with data and evaluation task
        original_evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        original_evaluator.add_data(sample_eval_data)
        original_evaluator.add_eval_task(basic_eval_task)

        original_evaluator.save_state(
            "test_state.json", include_data=True, data_format="json"
        )

        # Load from JSON without data and evaluation task
        loaded_evaluator = MetaEvaluator.load_state(
            project_dir=str(original_evaluator.project_dir),
            state_filename="test_state.json",
            load_data=False,
            load_task=False,
        )

        # Verifydata and evaluation task skipped
        assert loaded_evaluator.data is None
        assert loaded_evaluator.eval_task is None

    def test_load_state_invalid_file_extension(self, tmp_path):
        """Test InvalidFileError when state filename doesn't end with .json."""
        with pytest.raises(
            InvalidFileError, match="state_filename must end with .json"
        ):
            MetaEvaluator.load_state(str(tmp_path), "invalid_file.txt")

    def test_load_state_nonexistent_file(self, tmp_path):
        """Test InvalidFileError when state file doesn't exist."""
        with pytest.raises(InvalidFileError, match="State file not found"):
            MetaEvaluator.load_state(str(tmp_path), "nonexistent.json")

    def test_load_state_invalid_json(self, tmp_path):
        """Test InvalidFileError when state file contains invalid JSON."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        state_file = project_dir / "invalid.json"
        state_file.write_text("{ invalid json }")

        with pytest.raises(
            InvalidFileError,
            match=re.compile(rf"{re.escape(INVALID_JSON_STRUCTURE_MSG)}", re.DOTALL),
        ):
            MetaEvaluator.load_state(str(project_dir), "invalid.json")

    def test_load_state_missing_required_keys(self, tmp_path):
        """Test InvalidFileError when state file is missing required keys."""
        state_file = tmp_path / "incomplete.json"
        state_file.write_text('{"version": "1.0"}')

        with pytest.raises(
            InvalidFileError,
            match=re.compile(
                rf"{re.escape(INVALID_JSON_STRUCTURE_MSG)}.*Field required", re.DOTALL
            ),
        ):
            MetaEvaluator.load_state(str(tmp_path), "incomplete.json")

    def test_load_state_nonexistent_data_file(self, tmp_path, mock_openai_client):
        """Test FileNotFoundError when referenced data file doesn't exist."""
        # Create state file that references nonexistent data file
        state_data = {
            "version": "1.0",
            "data": {
                "name": "Test Data",
                "id_column": "message_id",
                "data_file": "nonexistent_data.json",
                "data_format": "json",
                "type": "EvalData",
            },
        }

        state_file = tmp_path / "test.json"
        with open(state_file, "w") as f:
            json.dump(state_data, f)

        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = mock_openai_client
            mock_client_class.return_value = mock_client

            with pytest.raises(FileNotFoundError, match="Data file not found"):
                MetaEvaluator.load_state(
                    project_dir=str(tmp_path),
                    state_filename="test.json",
                    load_data=True,
                )

    def test_load_state_unsupported_data_format(self, tmp_path, mock_openai_client):
        """Test ValueError when data format is unsupported."""
        # Create state file with unsupported data format
        state_data = {
            "version": "1.0",
            "data": {
                "name": "Test Data",
                "id_column": "message_id",
                "data_file": "test_data.xml",
                "data_format": "xml",  # Unsupported format
                "type": "EvalData",
            },
        }

        state_file = tmp_path / "test.json"
        with open(state_file, "w") as f:
            json.dump(state_data, f)

        # Create the referenced data file (even though format is unsupported)
        data_file = tmp_path / "test_data.xml"
        data_file.write_text("<data></data>")

        with patch(
            "meta_evaluator.meta_evaluator.clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = mock_openai_client
            mock_client_class.return_value = mock_client

            with pytest.raises(
                InvalidFileError,
                match=re.compile(
                    rf"{re.escape(INVALID_JSON_STRUCTURE_MSG)}.*Input should be 'json', 'csv' or 'parquet'",
                    re.DOTALL,
                ),
            ):
                MetaEvaluator.load_state(
                    project_dir=str(tmp_path),
                    state_filename="test.json",
                    load_data=True,
                )


class TestMetaEvaluatorAnnotator:
    """Test suite for MetaEvaluator annotator methods."""

    # === launch_annotator() Method Tests ===

    def test_launch_annotator_no_data(self, meta_evaluator, basic_eval_task):
        """Test that launch_annotator raises ValueError when no data is set."""
        meta_evaluator.add_eval_task(basic_eval_task)

        with pytest.raises(EvalDataNotFoundError):
            meta_evaluator.launch_annotator()

    def test_launch_annotator_no_eval_task(self, meta_evaluator, sample_eval_data):
        """Test that launch_annotator raises ValueError when no eval_task is set."""
        meta_evaluator.add_data(sample_eval_data)

        with pytest.raises(EvalTaskNotFoundError):
            meta_evaluator.launch_annotator()

    @patch("meta_evaluator.meta_evaluator.base.StreamlitLauncher")
    def test_launch_annotator_called(
        self, mock_launcher_class, meta_evaluator, sample_eval_data, basic_eval_task
    ):
        """Test that launch_annotator calls the StreamlitLauncher with the correct arguments."""
        # Setup
        meta_evaluator.add_data(sample_eval_data)
        meta_evaluator.add_eval_task(basic_eval_task)
        mock_launcher = mock_launcher_class.return_value

        # Execute
        meta_evaluator.launch_annotator()

        # Verify
        mock_launcher_class.assert_called_once_with(
            eval_data=sample_eval_data,
            eval_task=basic_eval_task,
            annotations_dir=str(meta_evaluator.paths.annotations),
            port=None,
        )
        mock_launcher.launch.assert_called_once_with(
            use_ngrok=False, traffic_policy_file=None
        )
