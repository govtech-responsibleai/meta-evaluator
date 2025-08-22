"""Test suite for the MetaEvaluator judge management."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import polars as pl
import pytest

from meta_evaluator.common.models import Prompt
from meta_evaluator.meta_evaluator.exceptions import (
    EvalDataNotFoundError,
    EvalTaskNotFoundError,
    InvalidYAMLStructureError,
    JudgeAlreadyExistsError,
    JudgeNotFoundError,
    PromptFileNotFoundError,
    ResultsSaveError,
)
from meta_evaluator.results import JudgeResults

from .conftest import create_mock_judge_state_file


class TestMetaEvaluatorJudges:
    """Test suite for MetaEvaluator judge management functionality."""

    # === Helper Methods ===

    def create_test_yaml_file(self, tmp_path, judges_config):
        """Create a test YAML file with judge configurations.

        Returns:
            Path to the created YAML file.
        """
        yaml_content = "judges:\n"
        for judge_config in judges_config:
            yaml_content += f"  - id: {judge_config['id']}\n"
            yaml_content += f"    llm_client: {judge_config['llm_client']}\n"
            yaml_content += f"    model: {judge_config['model']}\n"
            yaml_content += f"    prompt_file: {judge_config['prompt_file']}\n"

        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)
        return yaml_file

    def create_test_prompt_file(self, tmp_path, filename, content):
        """Create a test prompt file.

        Returns:
            Path to the created prompt file.
        """
        prompt_file = tmp_path / filename
        if "/" in filename:
            prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(content)
        return prompt_file

    def add_test_judge(
        self, meta_evaluator, sample_prompt, judge_id="test_judge", model="gpt-4"
    ):
        """Add a standard test judge to meta_evaluator.

        Args:
            meta_evaluator: The MetaEvaluator instance to add judge to.
            sample_prompt: The prompt to use for the judge.
            judge_id: The ID for the judge (default: "test_judge").
            model: The model to use for the judge (default: "gpt-4").

        Returns:
            The added judge instance.
        """
        meta_evaluator.add_judge(
            judge_id=judge_id,
            llm_client="openai",
            model=model,
            prompt=sample_prompt,
        )
        return meta_evaluator.judge_registry[judge_id]

    # === Judge Management Tests ===

    def test_add_judge_success(self, meta_evaluator_with_task, sample_prompt):
        """Test successfully adding a judge."""
        judge_id = "test_judge"

        judge = self.add_test_judge(meta_evaluator_with_task, sample_prompt, judge_id)

        # Verify judge was added
        assert judge_id in meta_evaluator_with_task.judge_registry
        assert judge.id == judge_id
        assert judge.llm_client == "openai"
        assert judge.model == "gpt-4"
        assert judge.prompt == sample_prompt

    def test_add_judge_without_eval_task(self, meta_evaluator, sample_prompt):
        """Test adding judge fails when no eval_task is set."""
        with pytest.raises(
            EvalTaskNotFoundError, match="eval_task must be set before adding judges"
        ):
            meta_evaluator.add_judge(
                judge_id="test_judge",
                llm_client="openai",
                model="gpt-4",
                prompt=sample_prompt,
            )

    def test_add_judge_already_exists(self, meta_evaluator_with_task, sample_prompt):
        """Test adding judge with same ID raises exception."""
        judge_id = "test_judge"

        # Add judge first time
        self.add_test_judge(meta_evaluator_with_task, sample_prompt, judge_id)

        # Try to add again without override
        with pytest.raises(JudgeAlreadyExistsError):
            self.add_test_judge(meta_evaluator_with_task, sample_prompt, judge_id)

    def test_add_judge_override(self, meta_evaluator_with_task, sample_prompt):
        """Test overriding existing judge."""
        judge_id = "test_judge"

        # Add judge first time
        self.add_test_judge(meta_evaluator_with_task, sample_prompt, judge_id)

        # Override with different model
        new_prompt = Prompt(id="new_prompt", prompt="New prompt")
        meta_evaluator_with_task.add_judge(
            judge_id=judge_id,
            llm_client="openai",
            model="gpt-3.5-turbo",
            prompt=new_prompt,
            on_duplicate="overwrite",
        )

        # Verify override worked
        judge = meta_evaluator_with_task.judge_registry[judge_id]
        assert judge.model == "gpt-3.5-turbo"
        assert judge.prompt == new_prompt

    def test_get_judge_success(self, meta_evaluator_with_task, sample_prompt):
        """Test successfully retrieving a judge."""
        judge_id = "test_judge"

        self.add_test_judge(meta_evaluator_with_task, sample_prompt, judge_id)

        retrieved_judge = meta_evaluator_with_task.get_judge(judge_id)
        assert retrieved_judge.id == judge_id
        assert retrieved_judge.model == "gpt-4"

    def test_get_judge_not_found(self, meta_evaluator_with_task):
        """Test retrieving non-existent judge raises exception."""
        with pytest.raises(JudgeNotFoundError):
            meta_evaluator_with_task.get_judge("non_existent_judge")

    def test_get_judge_list_empty(self, meta_evaluator_with_task):
        """Test getting judge list when empty."""
        judge_list = meta_evaluator_with_task.get_judge_list()
        assert judge_list == []

    def test_get_judge_list_with_judges(self, meta_evaluator_with_task, sample_prompt):
        """Test getting judge list with multiple judges."""
        judge_ids = ["judge1", "judge2", "judge3"]

        for judge_id in judge_ids:
            self.add_test_judge(meta_evaluator_with_task, sample_prompt, judge_id)

        judge_list = meta_evaluator_with_task.get_judge_list()
        assert len(judge_list) == 3
        retrieved_ids = [judge_id for judge_id, _ in judge_list]
        assert set(retrieved_ids) == set(judge_ids)

    def test_load_judges_from_yaml_success(self, meta_evaluator_with_task, tmp_path):
        """Test successfully loading judges from YAML file."""
        # Create prompt file
        prompt_file = self.create_test_prompt_file(
            tmp_path, "test_prompt.md", "You are a helpful evaluator."
        )

        # Create YAML file
        yaml_file = self.create_test_yaml_file(
            tmp_path,
            [
                {
                    "id": "test_judge",
                    "llm_client": "openai",
                    "model": "gpt-4",
                    "prompt_file": str(prompt_file),
                }
            ],
        )

        # Load judges
        meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

        # Verify judge was loaded
        assert "test_judge" in meta_evaluator_with_task.judge_registry
        judge = meta_evaluator_with_task.judge_registry["test_judge"]
        assert judge.id == "test_judge"
        assert judge.llm_client == "openai"
        assert judge.model == "gpt-4"
        assert judge.prompt.prompt == "You are a helpful evaluator."

    def test_load_judges_from_yaml_relative_path(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading judges with relative prompt paths."""
        # Create prompt file in subdirectory
        self.create_test_prompt_file(
            tmp_path, "prompts/test_prompt.md", "You are a helpful evaluator."
        )

        # Create YAML file with relative path
        yaml_file = self.create_test_yaml_file(
            tmp_path,
            [
                {
                    "id": "test_judge",
                    "llm_client": "openai",
                    "model": "gpt-4",
                    "prompt_file": "prompts/test_prompt.md",
                }
            ],
        )

        # Load judges
        meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

        # Verify judge was loaded
        assert "test_judge" in meta_evaluator_with_task.judge_registry
        judge = meta_evaluator_with_task.judge_registry["test_judge"]
        assert judge.prompt.prompt == "You are a helpful evaluator."

    def test_load_judges_from_yaml_without_eval_task(self, meta_evaluator, tmp_path):
        """Test loading judges fails when no eval_task is set."""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text("judges: []")

        with pytest.raises(
            EvalTaskNotFoundError,
            match="eval_task must be set before loading judges from YAML",
        ):
            meta_evaluator.load_judges_from_yaml(str(yaml_file))

    def test_load_judges_from_yaml_file_not_found(self, meta_evaluator_with_task):
        """Test loading judges from non-existent YAML file."""
        with pytest.raises(FileNotFoundError):
            meta_evaluator_with_task.load_judges_from_yaml("non_existent.yaml")

    def test_load_judges_from_yaml_invalid_yaml(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading judges from invalid YAML file."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: content: [")

        with pytest.raises(InvalidYAMLStructureError):
            meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

    def test_load_judges_from_yaml_invalid_structure(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading judges from YAML with invalid structure."""
        yaml_content = """judges:
  - id: test_judge
    # missing required fields
"""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(InvalidYAMLStructureError):
            meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

    def test_load_judges_from_yaml_prompt_file_not_found(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading judges with non-existent prompt file."""
        yaml_file = self.create_test_yaml_file(
            tmp_path,
            [
                {
                    "id": "test_judge",
                    "llm_client": "openai",
                    "model": "gpt-4",
                    "prompt_file": "non_existent_prompt.md",
                }
            ],
        )

        with pytest.raises(PromptFileNotFoundError):
            meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

    def test_load_judges_from_yaml_multiple_judges(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading multiple judges from YAML."""
        # Create prompt files
        prompt1 = self.create_test_prompt_file(tmp_path, "prompt1.md", "Prompt 1")
        prompt2 = self.create_test_prompt_file(tmp_path, "prompt2.md", "Prompt 2")

        yaml_file = self.create_test_yaml_file(
            tmp_path,
            [
                {
                    "id": "judge1",
                    "llm_client": "openai",
                    "model": "gpt-4",
                    "prompt_file": str(prompt1),
                },
                {
                    "id": "judge2",
                    "llm_client": "openai",
                    "model": "gpt-3.5-turbo",
                    "prompt_file": str(prompt2),
                },
            ],
        )

        meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

        # Verify both judges were loaded
        assert "judge1" in meta_evaluator_with_task.judge_registry
        assert "judge2" in meta_evaluator_with_task.judge_registry

        judge1 = meta_evaluator_with_task.judge_registry["judge1"]
        judge2 = meta_evaluator_with_task.judge_registry["judge2"]

        assert judge1.model == "gpt-4"
        assert judge2.model == "gpt-3.5-turbo"
        assert judge1.prompt.prompt == "Prompt 1"
        assert judge2.prompt.prompt == "Prompt 2"

    def test_load_judges_from_yaml_override(
        self, meta_evaluator_with_task, tmp_path, sample_prompt
    ):
        """Test overriding existing judge when loading from YAML."""
        # Add judge programmatically first
        self.add_test_judge(meta_evaluator_with_task, sample_prompt, "test_judge")

        # Create YAML with same judge ID
        prompt_file = self.create_test_prompt_file(
            tmp_path, "new_prompt.md", "New prompt content"
        )

        yaml_file = self.create_test_yaml_file(
            tmp_path,
            [
                {
                    "id": "test_judge",
                    "llm_client": "openai",
                    "model": "gpt-3.5-turbo",
                    "prompt_file": str(prompt_file),
                }
            ],
        )

        # Load with override
        meta_evaluator_with_task.load_judges_from_yaml(
            str(yaml_file), on_duplicate="overwrite"
        )

        # Verify judge was overridden
        judge = meta_evaluator_with_task.judge_registry["test_judge"]
        assert judge.model == "gpt-3.5-turbo"
        assert judge.prompt.prompt == "New prompt content"

    # === Judge Execution Tests ===

    def test_run_judges_no_eval_task(self, meta_evaluator):
        """Test run_judges raises exception when eval_task is not set."""
        with pytest.raises(EvalTaskNotFoundError):
            meta_evaluator.run_judges()

    def test_run_judges_no_eval_data(self, meta_evaluator_with_task):
        """Test run_judges raises exception when eval_data is not set."""
        with pytest.raises(EvalDataNotFoundError):
            meta_evaluator_with_task.run_judges()

    def test_run_judges_no_judges_available(
        self, meta_evaluator_with_task, sample_eval_data
    ):
        """Test run_judges raises exception when no judges are available."""
        meta_evaluator_with_task.add_data(sample_eval_data)

        with pytest.raises(JudgeNotFoundError):
            meta_evaluator_with_task.run_judges()

    def test_run_judges_judge_not_found(self, meta_evaluator_with_judges_and_data):
        """Test run_judges raises exception when specified judge doesn't exist."""
        with pytest.raises(JudgeNotFoundError):
            meta_evaluator_with_judges_and_data.run_judges(
                judge_ids="non_existent_judge"
            )

    def test_run_judges_multiple_judges_not_found(
        self, meta_evaluator_with_judges_and_data
    ):
        """Test run_judges raises exception when some specified judges don't exist."""
        with pytest.raises(JudgeNotFoundError):
            meta_evaluator_with_judges_and_data.run_judges(
                judge_ids=["judge1", "non_existent_judge", "another_missing"]
            )

    def test_run_judges_single_judge(self, meta_evaluator_with_judges_and_data):
        """Test running a single judge."""
        # Run single judge
        results = meta_evaluator_with_judges_and_data.run_judges(
            judge_ids="judge1", save_results=False
        )

        # Verify results
        assert len(results) == 1
        assert "judge1" in results

        # Get judge and verify it was called
        judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        judge1.evaluate_eval_data.assert_called_once()

        # Verify the returned result matches what the mock returned
        assert results["judge1"] == judge1.evaluate_eval_data.return_value

    def test_run_judges_all_judges(self, meta_evaluator_with_judges_and_data):
        """Test running all judges."""
        # Run all judges (judge_ids=None)
        results = meta_evaluator_with_judges_and_data.run_judges(
            judge_ids=None, save_results=False
        )

        # Verify results
        assert len(results) == 2
        assert "judge1" in results
        assert "judge2" in results

        # Get judges and verify results match what they returned
        judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        judge2 = meta_evaluator_with_judges_and_data.judge_registry["judge2"]
        assert results["judge1"] == judge1.evaluate_eval_data.return_value
        assert results["judge2"] == judge2.evaluate_eval_data.return_value

        # Verify both judges were called
        judge1.evaluate_eval_data.assert_called_once()
        judge2.evaluate_eval_data.assert_called_once()

    def test_run_judges_some_judges(self, meta_evaluator_with_judges_and_data):
        """Test running some judges by list."""
        # Add a third mock judge
        mock_judge3 = Mock()
        mock_judge3.llm_client = "openai"
        mock_judge3.evaluate_eval_data = Mock()
        meta_evaluator_with_judges_and_data.judge_registry["judge3"] = mock_judge3

        # Run specific judges
        results = meta_evaluator_with_judges_and_data.run_judges(
            judge_ids=["judge1", "judge2"], save_results=False
        )

        # Verify results
        assert len(results) == 2
        assert "judge1" in results
        assert "judge2" in results
        assert "judge3" not in results

        # Get judges and verify they were called
        judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        judge2 = meta_evaluator_with_judges_and_data.judge_registry["judge2"]
        judge3 = meta_evaluator_with_judges_and_data.judge_registry["judge3"]

        judge1.evaluate_eval_data.assert_called_once()
        judge2.evaluate_eval_data.assert_called_once()
        judge3.evaluate_eval_data.assert_not_called()

    def test_run_judges_custom_run_id(self, meta_evaluator_with_judges_and_data):
        """Test running judges with custom run_id."""
        custom_run_id = "custom_experiment_run"

        # Run judge with custom run_id
        meta_evaluator_with_judges_and_data.run_judges(
            judge_ids="judge1", run_id=custom_run_id, save_results=False
        )

        # Get the mock judge from fixture
        judge = meta_evaluator_with_judges_and_data.judge_registry["judge1"]

        # Verify judge.evaluate_eval_data was called with correct run_id
        call_args = judge.evaluate_eval_data.call_args
        run_id_arg = call_args[1]["run_id"]  # Keyword argument

        assert run_id_arg == f"{custom_run_id}_judge1"

    def test_run_judges_auto_generated_run_id(
        self, meta_evaluator_with_judges_and_data
    ):
        """Test running judges with auto-generated run_id."""
        # Run judge without specifying run_id
        meta_evaluator_with_judges_and_data.run_judges(
            judge_ids="judge1", save_results=False
        )

        # Get the mock judge from fixture
        judge = meta_evaluator_with_judges_and_data.judge_registry["judge1"]

        # Verify judge.evaluate_eval_data was called with auto-generated run_id
        call_args = judge.evaluate_eval_data.call_args
        run_id_arg = call_args[1]["run_id"]  # Keyword argument

        # Auto-generated run_id should contain timestamp and judge ID
        assert "judge1" in run_id_arg
        assert "run_" in run_id_arg

    @pytest.mark.parametrize("results_format", ["json", "csv", "parquet"])
    def test_run_judges_save_results_with_formats(
        self, meta_evaluator_with_judges_and_data, results_format, tmp_path
    ):
        """Test that judge results are saved to project_dir/results/ with different formats."""
        # Override the results path to use tmp_path for testing
        meta_evaluator_with_judges_and_data.paths.results = tmp_path / "results"
        meta_evaluator_with_judges_and_data.paths.results.mkdir(parents=True)

        # Create fake evaluation data with required columns
        fake_evaluations = pl.DataFrame(
            {
                "sample_example_id": ["test_run_judge1_1", "test_run_judge1_2"],
                "original_id": ["id-1", "id-2"],
                "run_id": ["test_run_judge1", "test_run_judge1"],
                "status": ["success", "success"],
                "error_message": [None, None],
                "error_details_json": [None, None],
                "judge_id": ["judge1", "judge1"],
                "llm_raw_response_content": [
                    '{"sentiment": "positive"}',
                    '{"sentiment": "negative"}',
                ],
                "llm_prompt_tokens": [100, 105],
                "llm_completion_tokens": [10, 12],
                "llm_total_tokens": [110, 117],
                "llm_call_duration_seconds": [1.5, 2.1],
                "sentiment": ["positive", "negative"],
            }
        )

        # Replace the mock judge's return value with real JudgeResults
        judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        real_results = JudgeResults(
            judge_id="judge1",
            run_id="test_run_judge1",
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            timestamp_local=datetime.now(),
            total_count=2,
            succeeded_count=2,
            is_sampled_run=False,
            results_data=fake_evaluations,
            llm_client="openai",
            model_used="gpt-4",
            skipped_count=0,
            partial_count=0,
            llm_error_count=0,
            parsing_error_count=0,
            other_error_count=0,
        )
        judge1.evaluate_eval_data.return_value = real_results

        # Run judge with save_results=True and specific format
        meta_evaluator_with_judges_and_data.run_judges(
            judge_ids="judge1", save_results=True, results_format=results_format
        )

        # Verify actual files were created
        results_dir = meta_evaluator_with_judges_and_data.paths.results

        # Find the created files (they have timestamps in names)
        state_files = list(results_dir.glob("*_state.json"))
        data_files = list(results_dir.glob(f"*_results.{results_format}"))

        assert len(state_files) == 1, f"Expected 1 state file, found {len(state_files)}"
        assert len(data_files) == 1, f"Expected 1 data file, found {len(data_files)}"

        # Verify state file contains expected data
        state_file = state_files[0]
        with open(state_file) as f:
            state_data = json.load(f)
            assert state_data["judge_id"] == "judge1"
            assert state_data["data_format"] == results_format
            assert state_data["data_file"] == data_files[0].name

        # Verify data file exists and has expected extension
        data_file = data_files[0]
        assert data_file.suffix == f".{results_format}"
        assert data_file.stat().st_size > 0  # File is not empty

    @pytest.mark.parametrize("results_format", ["json", "csv", "parquet"])
    def test_run_judges_results_can_be_loaded(
        self, meta_evaluator_with_judges_and_data, results_format
    ):
        """Test that saved judge results can be loaded from metaevaluator folder."""
        # Create a real temporary state file for testing
        results_dir = meta_evaluator_with_judges_and_data.paths.results
        state_file = results_dir / f"test_state_{results_format}.json"

        # Mock judge
        judge = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        mock_results = Mock(spec=JudgeResults)

        # Create mock state data that save_state would create
        def mock_save_state(state_file_path, data_format, data_filename):
            # Create a minimal state file for testing
            state_data = {
                "judge_id": "judge1",
                "data_format": data_format,
                "data_filename": data_filename,
            }
            with open(state_file_path, "w") as f:
                json.dump(state_data, f)

        mock_results.save_state = mock_save_state
        mock_results.succeeded_count = 2
        mock_results.total_count = 2
        judge.evaluate_eval_data = Mock(return_value=mock_results)

        # Run judge to save results
        meta_evaluator_with_judges_and_data.run_judges(
            judge_ids="judge1", save_results=True, results_format=results_format
        )

        # Mock JudgeResults.load_state to return a mock result
        mock_loaded_results = Mock(spec=JudgeResults)
        mock_loaded_results.judge_id = "judge1"

        # Test that load_state can be called (mock the class method)
        with patch.object(
            JudgeResults, "load_state", return_value=mock_loaded_results
        ) as mock_load:
            # Verify we can load the state file
            loaded_results = JudgeResults.load_state(str(state_file))

            # Verify load_state was called and returned expected result
            mock_load.assert_called_once_with(str(state_file))
            assert loaded_results.judge_id == "judge1"

    def test_run_judges_results_save_exception(
        self, meta_evaluator_with_judges_and_data
    ):
        """Test run_judges raises exception when saving results fails."""
        # Mock judge
        judge = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        mock_results = Mock(spec=JudgeResults)
        mock_results.save_state = Mock(side_effect=Exception("Save failed"))
        mock_results.succeeded_count = 2
        mock_results.total_count = 2
        judge.evaluate_eval_data = Mock(return_value=mock_results)

        # Run judge and expect ResultsSaveError
        with pytest.raises(ResultsSaveError):
            meta_evaluator_with_judges_and_data.run_judges(
                judge_ids="judge1", save_results=True
            )

    def test_run_judges_skip_duplicates_true_does_not_call_evaluate(
        self, meta_evaluator_with_judges_and_data, mock_judge_state_template
    ):
        """Test that when skip_duplicates=True, evaluate_eval_data is not called for duplicate judge_id."""
        # Create existing results for judge1
        results_dir = meta_evaluator_with_judges_and_data.paths.results
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create a mock state file for judge1
        create_mock_judge_state_file(
            results_dir, "judge1", "existing_run", mock_judge_state_template
        )

        # Mock JudgeResults.load_state to return a mock result
        with patch("meta_evaluator.results.JudgeResults.load_state") as mock_load:
            mock_judge_results = Mock()
            mock_judge_results.judge_id = "judge1"
            mock_load.return_value = mock_judge_results

            # Run judges with skip_duplicates=True
            results = meta_evaluator_with_judges_and_data.run_judges(
                skip_duplicates=True, save_results=False
            )

            # Should only run judge2, not judge1
            assert len(results) == 1
            assert "judge2" in results
            assert "judge1" not in results

            # Verify judge1.evaluate_eval_data was NOT called
            judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
            judge1.evaluate_eval_data.assert_not_called()

            # Verify judge2.evaluate_eval_data WAS called
            judge2 = meta_evaluator_with_judges_and_data.judge_registry["judge2"]
            judge2.evaluate_eval_data.assert_called_once()

    def test_run_judges_skip_duplicates_false_calls_evaluate(
        self, meta_evaluator_with_judges_and_data, mock_judge_state_template
    ):
        """Test that when skip_duplicates=False, evaluate_eval_data is called even for duplicate judge_id."""
        # Create existing results for judge1
        results_dir = meta_evaluator_with_judges_and_data.paths.results
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create a mock state file for judge1
        create_mock_judge_state_file(
            results_dir, "judge1", "existing_run", mock_judge_state_template
        )

        # Mock JudgeResults.load_state to return a mock result
        with patch("meta_evaluator.results.JudgeResults.load_state") as mock_load:
            mock_judge_results = Mock()
            mock_judge_results.judge_id = "judge1"
            mock_load.return_value = mock_judge_results

            # Run judges with skip_duplicates=False (default)
            results = meta_evaluator_with_judges_and_data.run_judges(
                skip_duplicates=False, save_results=False
            )

            # Should run both judges
            assert len(results) == 2
            assert "judge1" in results
            assert "judge2" in results

            # Verify both judges' evaluate_eval_data were called
            judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
            judge2 = meta_evaluator_with_judges_and_data.judge_registry["judge2"]
            judge1.evaluate_eval_data.assert_called_once()
            judge2.evaluate_eval_data.assert_called_once()

    # === Judge Registry Serialization Tests ===

    def test_load_judges_from_yaml_judge_registry_validation(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test that load_judges_from_yaml correctly populates judge_registry."""
        # Create test judges configuration
        judges_config = [
            {
                "id": "test_judge_1",
                "llm_client": "openai",
                "model": "gpt-4",
                "prompt_file": "prompt1.txt",
            },
            {
                "id": "test_judge_2",
                "llm_client": "anthropic",
                "model": "claude-3",
                "prompt_file": "prompt2.txt",
            },
        ]

        # Create YAML file and prompt files
        yaml_file = self.create_test_yaml_file(tmp_path, judges_config)
        self.create_test_prompt_file(tmp_path, "prompt1.txt", "Evaluate prompt 1")
        self.create_test_prompt_file(tmp_path, "prompt2.txt", "Evaluate prompt 2")

        # Verify judge registry is initially empty
        assert len(meta_evaluator_with_task.judge_registry) == 0

        # Load judges from YAML
        meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

        # Verify correct number of judges loaded
        assert len(meta_evaluator_with_task.judge_registry) == 2

        # Verify judges exist with correct IDs
        assert "test_judge_1" in meta_evaluator_with_task.judge_registry
        assert "test_judge_2" in meta_evaluator_with_task.judge_registry

        # Verify judge objects are actual Judge instances
        from meta_evaluator.judge import Judge

        assert isinstance(
            meta_evaluator_with_task.judge_registry["test_judge_1"], Judge
        )
        assert isinstance(
            meta_evaluator_with_task.judge_registry["test_judge_2"], Judge
        )

        # Verify judge configuration
        judge1 = meta_evaluator_with_task.judge_registry["test_judge_1"]
        assert judge1.id == "test_judge_1"
        assert judge1.llm_client == "openai"
        assert judge1.model == "gpt-4"

        judge2 = meta_evaluator_with_task.judge_registry["test_judge_2"]
        assert judge2.id == "test_judge_2"
        assert judge2.llm_client == "anthropic"
        assert judge2.model == "claude-3"

    # === EXISTING JUDGE RESULTS TESTS ===

    def test_get_existing_judge_ids_with_results_directory(
        self, meta_evaluator_with_judges_and_data, mock_judge_state_template
    ):
        """Test _get_existing_judge_ids returns correct number when results directory exists."""
        # Create results directory with some judge result files
        results_dir = meta_evaluator_with_judges_and_data.paths.results
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create mock state files
        judge_configs = [
            {"judge_id": "judge1", "run_id": "run_001"},
            {"judge_id": "judge2", "run_id": "run_002"},
            {"judge_id": "judge3", "run_id": "run_003"},
        ]

        for config in judge_configs:
            create_mock_judge_state_file(
                results_dir,
                config["judge_id"],
                config["run_id"],
                mock_judge_state_template,
            )

        # Mock JudgeResults.load_state to return appropriate mock results
        with patch("meta_evaluator.results.JudgeResults.load_state") as mock_load:

            def mock_load_side_effect(file_path):
                mock_result = Mock()
                for config in judge_configs:
                    if config["judge_id"] in str(file_path):
                        mock_result.judge_id = config["judge_id"]
                        break
                return mock_result

            mock_load.side_effect = mock_load_side_effect

            # Get existing judge IDs
            existing_ids = meta_evaluator_with_judges_and_data._get_existing_judge_ids()

            # Should return all 3 judge IDs
            assert len(existing_ids) == 3
            assert "judge1" in existing_ids
            assert "judge2" in existing_ids
            assert "judge3" in existing_ids

    def test_get_existing_judge_ids_no_results_directory(
        self, meta_evaluator_with_judges_and_data
    ):
        """Test _get_existing_judge_ids returns 0 when results directory doesn't exist."""
        # Ensure results directory doesn't exist
        results_dir = meta_evaluator_with_judges_and_data.paths.results
        if results_dir.exists():
            import shutil

            shutil.rmtree(results_dir)

        # Get existing judge IDs
        existing_ids = meta_evaluator_with_judges_and_data._get_existing_judge_ids()

        # Should return empty set
        assert len(existing_ids) == 0
        assert existing_ids == set()

    def test_run_judges_no_results_directory_runs_all(
        self, meta_evaluator_with_judges_and_data
    ):
        """Test that run_judges runs all judges when results directory doesn't exist."""
        # Ensure results directory doesn't exist
        results_dir = meta_evaluator_with_judges_and_data.paths.results
        if results_dir.exists():
            import shutil

            shutil.rmtree(results_dir)

        # Run judges with skip_duplicates=False (default)
        results = meta_evaluator_with_judges_and_data.run_judges(
            skip_duplicates=False, save_results=False
        )

        # Should run both judges
        assert len(results) == 2
        assert "judge1" in results
        assert "judge2" in results

        # Verify both judges' evaluate_eval_data were called
        judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        judge2 = meta_evaluator_with_judges_and_data.judge_registry["judge2"]
        judge1.evaluate_eval_data.assert_called_once()
        judge2.evaluate_eval_data.assert_called_once()


# === ASYNC JUDGE TESTS ===


@pytest.mark.asyncio
class TestAsyncMetaEvaluatorJudges:
    """Test async judge functionality in MetaEvaluator."""

    async def test__run_judges_async_single_judge(
        self, meta_evaluator_with_judges_and_data
    ):
        """Test running a single judge asynchronously."""
        # Mock the async evaluation method for the judge from the fixture
        mock_results = Mock(spec=JudgeResults)
        mock_results.succeeded_count = 2
        mock_results.total_count = 3
        mock_results.save_state = Mock()

        # Get the judge from the fixture and mock its async evaluation method
        judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        judge1.evaluate_eval_data_async = AsyncMock(return_value=mock_results)

        # Run single judge
        results = await meta_evaluator_with_judges_and_data._run_judges_async(
            judge_ids="judge1", save_results=False
        )

        # Verify results
        assert len(results) == 1
        assert "judge1" in results
        assert results["judge1"] == mock_results

        # Verify the judge's async method was called
        judge1.evaluate_eval_data_async.assert_called_once()

    async def test__run_judges_async_multiple_judges(
        self, meta_evaluator_with_judges_and_data
    ):
        """Test running multiple judges concurrently."""
        # Mock results for both judges
        mock_results1 = Mock(spec=JudgeResults)
        mock_results1.succeeded_count = 2
        mock_results1.total_count = 3
        mock_results1.save_state = Mock()

        mock_results2 = Mock(spec=JudgeResults)
        mock_results2.succeeded_count = 3
        mock_results2.total_count = 3
        mock_results2.save_state = Mock()

        # Get judges from fixture and mock their async evaluation methods
        judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        judge2 = meta_evaluator_with_judges_and_data.judge_registry["judge2"]
        judge1.evaluate_eval_data_async = AsyncMock(return_value=mock_results1)
        judge2.evaluate_eval_data_async = AsyncMock(return_value=mock_results2)

        results = await meta_evaluator_with_judges_and_data._run_judges_async(
            judge_ids=["judge1", "judge2"]
        )

        assert len(results) == 2
        assert "judge1" in results
        assert "judge2" in results
        assert results["judge1"] == mock_results1
        assert results["judge2"] == mock_results2
        judge1.evaluate_eval_data_async.assert_called_once()
        judge2.evaluate_eval_data_async.assert_called_once()

    async def test__run_judges_async_with_save_results(
        self, meta_evaluator_with_judges_and_data
    ):
        """Test async execution with result saving."""
        # Mock the async evaluation method
        mock_results = Mock(spec=JudgeResults)
        mock_results.succeeded_count = 2
        mock_results.total_count = 3
        mock_results.save_state = Mock()

        # Get judge from fixture and mock its async evaluation method
        judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        judge1.evaluate_eval_data_async = AsyncMock(return_value=mock_results)

        results = await meta_evaluator_with_judges_and_data._run_judges_async(
            judge_ids="judge1", save_results=True
        )

        assert len(results) == 1
        assert results["judge1"] == mock_results
        mock_results.save_state.assert_called_once()
        judge1.evaluate_eval_data_async.assert_called_once()

    @pytest.mark.parametrize("results_format", ["json", "csv", "parquet"])
    async def test__run_judges_async_save_results_with_formats(
        self,
        meta_evaluator_with_judges_and_data,
        results_format,
        tmp_path,
        completed_judge_results,
    ):
        """Test that async judge results are saved with different formats."""
        # Override the results path to use tmp_path for testing
        meta_evaluator_with_judges_and_data.paths.results = tmp_path / "results"
        meta_evaluator_with_judges_and_data.paths.results.mkdir(parents=True)

        # Use the fixture-provided JudgeResults instead of fake data
        judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        judge1.evaluate_eval_data_async = AsyncMock(
            return_value=completed_judge_results
        )

        # Run judge with save_results=True and specific format
        await meta_evaluator_with_judges_and_data._run_judges_async(
            judge_ids="judge1", save_results=True, results_format=results_format
        )

        # Verify actual files were created
        results_dir = meta_evaluator_with_judges_and_data.paths.results

        # Find the created files (they have timestamps in names)
        state_files = list(results_dir.glob("*_state.json"))
        data_files = list(results_dir.glob(f"*_results.{results_format}"))

        assert len(state_files) == 1, f"Expected 1 state file, found {len(state_files)}"
        assert len(data_files) == 1, f"Expected 1 data file, found {len(data_files)}"

        # Verify state file contains expected data
        state_file = state_files[0]
        with open(state_file) as f:
            state_data = json.load(f)
            assert (
                state_data["judge_id"] == "test_judge"
            )  # From completed_judge_results fixture
            assert state_data["data_format"] == results_format
            assert state_data["data_file"] == data_files[0].name

        # Verify data file exists and has expected extension
        data_file = data_files[0]
        assert data_file.suffix == f".{results_format}"
        assert data_file.stat().st_size > 0  # File is not empty

    async def test__run_judges_async_results_save_exception(
        self, meta_evaluator_with_judges_and_data
    ):
        """Test async run_judges raises exception when saving results fails."""
        # Mock judge
        judge = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        mock_results = Mock(spec=JudgeResults)
        mock_results.save_state = Mock(side_effect=Exception("Async save failed"))
        mock_results.succeeded_count = 2
        mock_results.total_count = 2
        judge.evaluate_eval_data_async = AsyncMock(return_value=mock_results)

        # Run judge and expect ResultsSaveError
        with pytest.raises(ResultsSaveError):
            await meta_evaluator_with_judges_and_data._run_judges_async(
                judge_ids="judge1", save_results=True
            )

    async def test__run_judges_async_all_judges(
        self, meta_evaluator_with_judges_and_data
    ):
        """Test running all judges asynchronously (judge_ids=None)."""
        # Mock results for both judges
        mock_results1 = Mock(spec=JudgeResults)
        mock_results1.succeeded_count = 2
        mock_results1.total_count = 3
        mock_results1.save_state = Mock()

        mock_results2 = Mock(spec=JudgeResults)
        mock_results2.succeeded_count = 3
        mock_results2.total_count = 3
        mock_results2.save_state = Mock()

        # Get judges from fixture and mock their async evaluation methods
        judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        judge2 = meta_evaluator_with_judges_and_data.judge_registry["judge2"]
        judge1.evaluate_eval_data_async = AsyncMock(return_value=mock_results1)
        judge2.evaluate_eval_data_async = AsyncMock(return_value=mock_results2)

        # Run all judges (judge_ids=None)
        results = await meta_evaluator_with_judges_and_data._run_judges_async(
            judge_ids=None, save_results=False
        )

        # Verify results
        assert len(results) == 2
        assert "judge1" in results
        assert "judge2" in results
        assert results["judge1"] == mock_results1
        assert results["judge2"] == mock_results2

        # Verify both judges were called
        judge1.evaluate_eval_data_async.assert_called_once()
        judge2.evaluate_eval_data_async.assert_called_once()

    async def test__run_judges_async_multiple_judges_not_found(
        self, meta_evaluator_with_judges_and_data
    ):
        """Test async run_judges raises exception when some specified judges don't exist."""
        with pytest.raises(JudgeNotFoundError):
            await meta_evaluator_with_judges_and_data._run_judges_async(
                judge_ids=["judge1", "non_existent_judge", "another_missing"]
            )

    async def test__run_judges_async_no_eval_task(self, meta_evaluator):
        """Test async run_judges raises exception when no eval_task is set."""
        with pytest.raises(
            EvalTaskNotFoundError, match="eval_task must be set before running judges"
        ):
            await meta_evaluator._run_judges_async()

    async def test__run_judges_async_no_eval_data(self, meta_evaluator_with_task):
        """Test async run_judges raises exception when no eval_data is set."""
        with pytest.raises(
            EvalDataNotFoundError, match="data must be set before running judges"
        ):
            await meta_evaluator_with_task._run_judges_async()

    async def test__run_judges_async_skip_duplicates_true_does_not_call_evaluate(
        self, meta_evaluator_with_judges_and_data, mock_judge_state_template
    ):
        """Test that async version with skip_duplicates=True doesn't call evaluate for duplicates."""
        # Create existing results for judge1
        results_dir = meta_evaluator_with_judges_and_data.paths.results
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create a mock state file for judge1
        create_mock_judge_state_file(
            results_dir, "judge1", "existing_run", mock_judge_state_template
        )

        # Mock async evaluation methods
        mock_result = Mock()
        mock_result.succeeded_count = 5
        mock_result.total_count = 10

        judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        judge2 = meta_evaluator_with_judges_and_data.judge_registry["judge2"]
        judge1.evaluate_eval_data_async = AsyncMock(return_value=mock_result)
        judge2.evaluate_eval_data_async = AsyncMock(return_value=mock_result)

        # Mock JudgeResults.load_state to return a mock result
        with patch("meta_evaluator.results.JudgeResults.load_state") as mock_load:
            mock_judge_results = Mock()
            mock_judge_results.judge_id = "judge1"
            mock_load.return_value = mock_judge_results

            # Run async judges with skip_duplicates=True
            results = await meta_evaluator_with_judges_and_data._run_judges_async(
                skip_duplicates=True, save_results=False
            )

            # Should only run judge2, not judge1
            assert len(results) == 1
            assert "judge2" in results
            assert "judge1" not in results

            # Verify judge1.evaluate_eval_data_async was NOT called
            judge1.evaluate_eval_data_async.assert_not_called()

            # Verify judge2.evaluate_eval_data_async WAS called
            judge2.evaluate_eval_data_async.assert_called_once()

    async def test__run_judges_async_skip_duplicates_false_calls_evaluate(
        self, meta_evaluator_with_judges_and_data, mock_judge_state_template
    ):
        """Test that async version with skip_duplicates=False calls evaluate even for duplicates."""
        # Create existing results for judge1
        results_dir = meta_evaluator_with_judges_and_data.paths.results
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create a mock state file for judge1
        create_mock_judge_state_file(
            results_dir, "judge1", "existing_run", mock_judge_state_template
        )

        # Mock async evaluation methods
        mock_result = Mock()
        mock_result.succeeded_count = 5
        mock_result.total_count = 10

        judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        judge2 = meta_evaluator_with_judges_and_data.judge_registry["judge2"]
        judge1.evaluate_eval_data_async = AsyncMock(return_value=mock_result)
        judge2.evaluate_eval_data_async = AsyncMock(return_value=mock_result)

        # Mock JudgeResults.load_state to return a mock result
        with patch("meta_evaluator.results.JudgeResults.load_state") as mock_load:
            mock_judge_results = Mock()
            mock_judge_results.judge_id = "judge1"
            mock_load.return_value = mock_judge_results

            # Run async judges with skip_duplicates=False (default)
            results = await meta_evaluator_with_judges_and_data._run_judges_async(
                skip_duplicates=False, save_results=False
            )

            # Should run both judges
            assert len(results) == 2
            assert "judge1" in results
            assert "judge2" in results

            # Verify both judges' evaluate_eval_data_async were called
            judge1.evaluate_eval_data_async.assert_called_once()
            judge2.evaluate_eval_data_async.assert_called_once()

    def test_run_judges_async_sync_wrapper(self, meta_evaluator_with_judges_and_data):
        """Test that run_judges_async sync wrapper works without async/await."""
        # Create mock results for judges
        judge1_results = Mock(spec=JudgeResults)
        judge1_results.judge_id = "judge1"
        judge1_results.succeeded_count = 5
        judge1_results.total_count = 5

        judge2_results = Mock(spec=JudgeResults)
        judge2_results.judge_id = "judge2"
        judge2_results.succeeded_count = 3
        judge2_results.total_count = 5

        # Mock the judges to return results
        judge1 = meta_evaluator_with_judges_and_data.get_judge("judge1")
        judge2 = meta_evaluator_with_judges_and_data.get_judge("judge2")

        judge1.evaluate_eval_data_async = AsyncMock(return_value=judge1_results)
        judge2.evaluate_eval_data_async = AsyncMock(return_value=judge2_results)

        # Call sync wrapper (no await needed)
        results = meta_evaluator_with_judges_and_data.run_judges_async()

        # Verify results structure
        assert len(results) == 2
        assert "judge1" in results
        assert "judge2" in results

        # Verify both judges' evaluate_eval_data_async were called
        judge1.evaluate_eval_data_async.assert_called_once()
        judge2.evaluate_eval_data_async.assert_called_once()
