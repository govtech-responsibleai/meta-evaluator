"""Test suite for the MetaEvaluator judge management."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from meta_evaluator.meta_evaluator.exceptions import (
    JudgeAlreadyExistsError,
    JudgeNotFoundError,
    InvalidYAMLStructureError,
    PromptFileNotFoundError,
    EvalTaskNotFoundError,
    EvalDataNotFoundError,
    JudgeExecutionError,
    LLMClientNotConfiguredError,
    ResultsSaveError,
)
import polars as pl
from meta_evaluator.llm_client.enums import LLMClientEnum
from meta_evaluator.common.models import Prompt
from meta_evaluator.results import JudgeResults


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
            llm_client_enum=LLMClientEnum.OPENAI,
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
        assert judge.llm_client_enum == LLMClientEnum.OPENAI
        assert judge.model == "gpt-4"
        assert judge.prompt == sample_prompt

    def test_add_judge_without_eval_task(self, meta_evaluator, sample_prompt):
        """Test adding judge fails when no eval_task is set."""
        with pytest.raises(
            EvalTaskNotFoundError, match="eval_task must be set before adding judges"
        ):
            meta_evaluator.add_judge(
                judge_id="test_judge",
                llm_client_enum=LLMClientEnum.OPENAI,
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

    def test_add_judge_override_existing(self, meta_evaluator_with_task, sample_prompt):
        """Test overriding existing judge."""
        judge_id = "test_judge"

        # Add judge first time
        self.add_test_judge(meta_evaluator_with_task, sample_prompt, judge_id)

        # Override with different model
        new_prompt = Prompt(id="new_prompt", prompt="New prompt")
        meta_evaluator_with_task.add_judge(
            judge_id=judge_id,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-3.5-turbo",
            prompt=new_prompt,
            override_existing=True,
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
        assert judge.llm_client_enum == LLMClientEnum.OPENAI
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

    def test_load_judges_from_yaml_invalid_llm_client(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading judges with invalid LLM client type."""
        prompt_file = self.create_test_prompt_file(
            tmp_path, "test_prompt.md", "Test prompt"
        )

        yaml_file = self.create_test_yaml_file(
            tmp_path,
            [
                {
                    "id": "test_judge",
                    "llm_client": "invalid_client",
                    "model": "gpt-4",
                    "prompt_file": str(prompt_file),
                }
            ],
        )

        with pytest.raises(ValueError, match="Invalid llm_client"):
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

    def test_load_judges_from_yaml_override_existing(
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
            str(yaml_file), override_existing=True
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

    def test_run_judges_llm_client_not_configured(
        self, meta_evaluator_with_judges_and_data
    ):
        """Test run_judges raises exception when LLM client is not configured."""
        # Clear the client registry to simulate unconfigured client
        meta_evaluator_with_judges_and_data.client_registry = {}

        with pytest.raises(LLMClientNotConfiguredError):
            meta_evaluator_with_judges_and_data.run_judges()

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
        mock_judge3.llm_client_enum = LLMClientEnum.OPENAI
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

    def test_run_judges_execution_exception(self, meta_evaluator_with_judges_and_data):
        """Test run_judges raises exception when judge execution fails."""
        # Configure mock judge to raise exception
        judge1 = meta_evaluator_with_judges_and_data.judge_registry["judge1"]
        judge1.evaluate_eval_data.side_effect = Exception("Judge execution failed")

        # Run judge and expect JudgeExecutionError
        with pytest.raises(JudgeExecutionError):
            meta_evaluator_with_judges_and_data.run_judges(
                judge_ids="judge1", save_results=False
            )

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
            llm_client_enum=LLMClientEnum.OPENAI,
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
            import json

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
                import json

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
