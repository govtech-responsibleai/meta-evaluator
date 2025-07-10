"""Test suite for the MetaEvaluator judge management."""

import pytest
from meta_evaluator.meta_evaluator.exceptions import (
    JudgeAlreadyExistsException,
    JudgeNotFoundException,
    InvalidYAMLStructureException,
    PromptFileNotFoundException,
)
from meta_evaluator.llm_client.models import LLMClientEnum
from meta_evaluator.common.models import Prompt


class TestMetaEvaluatorJudges:
    """Test suite for MetaEvaluator judge management functionality."""

    # === Judge-specific Fixtures ===

    @pytest.fixture
    def meta_evaluator_with_task(self, meta_evaluator, basic_eval_task):
        """Provides a MetaEvaluator with eval_task set.

        Args:
            meta_evaluator: The MetaEvaluator instance to modify.
            basic_eval_task: The basic evaluation task to add.

        Returns:
            MetaEvaluator: The modified MetaEvaluator instance.
        """
        meta_evaluator.add_eval_task(basic_eval_task)
        return meta_evaluator

    # === Judge Management Tests ===

    def test_add_judge_success(self, meta_evaluator_with_task, sample_prompt):
        """Test successfully adding a judge."""
        judge_id = "test_judge"

        meta_evaluator_with_task.add_judge(
            judge_id=judge_id,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        # Verify judge was added
        assert judge_id in meta_evaluator_with_task.judge_registry
        judge = meta_evaluator_with_task.judge_registry[judge_id]
        assert judge.id == judge_id
        assert judge.llm_client_enum == LLMClientEnum.OPENAI
        assert judge.model == "gpt-4"
        assert judge.prompt == sample_prompt

    def test_add_judge_without_eval_task(self, meta_evaluator, sample_prompt):
        """Test adding judge fails when no eval_task is set."""
        with pytest.raises(
            ValueError, match="eval_task must be set before adding judges"
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
        meta_evaluator_with_task.add_judge(
            judge_id=judge_id,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        # Try to add again without override
        with pytest.raises(JudgeAlreadyExistsException):
            meta_evaluator_with_task.add_judge(
                judge_id=judge_id,
                llm_client_enum=LLMClientEnum.OPENAI,
                model="gpt-4",
                prompt=sample_prompt,
            )

    def test_add_judge_override_existing(self, meta_evaluator_with_task, sample_prompt):
        """Test overriding existing judge."""
        judge_id = "test_judge"

        # Add judge first time
        meta_evaluator_with_task.add_judge(
            judge_id=judge_id,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

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

        meta_evaluator_with_task.add_judge(
            judge_id=judge_id,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        retrieved_judge = meta_evaluator_with_task.get_judge(judge_id)
        assert retrieved_judge.id == judge_id
        assert retrieved_judge.model == "gpt-4"

    def test_get_judge_not_found(self, meta_evaluator_with_task):
        """Test retrieving non-existent judge raises exception."""
        with pytest.raises(JudgeNotFoundException):
            meta_evaluator_with_task.get_judge("non_existent_judge")

    def test_get_judge_list_empty(self, meta_evaluator_with_task):
        """Test getting judge list when empty."""
        judge_list = meta_evaluator_with_task.get_judge_list()
        assert judge_list == []

    def test_get_judge_list_with_judges(self, meta_evaluator_with_task, sample_prompt):
        """Test getting judge list with multiple judges."""
        judge_ids = ["judge1", "judge2", "judge3"]

        for judge_id in judge_ids:
            meta_evaluator_with_task.add_judge(
                judge_id=judge_id,
                llm_client_enum=LLMClientEnum.OPENAI,
                model="gpt-4",
                prompt=sample_prompt,
            )

        judge_list = meta_evaluator_with_task.get_judge_list()
        assert len(judge_list) == 3
        retrieved_ids = [judge_id for judge_id, _ in judge_list]
        assert set(retrieved_ids) == set(judge_ids)

    def test_load_judges_from_yaml_success(self, meta_evaluator_with_task, tmp_path):
        """Test successfully loading judges from YAML file."""
        # Create prompt file
        prompt_file = tmp_path / "test_prompt.md"
        prompt_file.write_text("You are a helpful evaluator.")

        # Create YAML file
        yaml_content = f"""judges:
  - id: test_judge
    llm_client: openai
    model: gpt-4
    prompt_file: {prompt_file}
"""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)

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
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        prompt_file = prompt_dir / "test_prompt.md"
        prompt_file.write_text("You are a helpful evaluator.")

        # Create YAML file with relative path
        yaml_content = """judges:
  - id: test_judge
    llm_client: openai
    model: gpt-4
    prompt_file: prompts/test_prompt.md
"""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)

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
            ValueError, match="eval_task must be set before loading judges from YAML"
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

        with pytest.raises(InvalidYAMLStructureException):
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

        with pytest.raises(InvalidYAMLStructureException):
            meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

    def test_load_judges_from_yaml_invalid_llm_client(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading judges with invalid LLM client type."""
        prompt_file = tmp_path / "test_prompt.md"
        prompt_file.write_text("Test prompt")

        yaml_content = f"""judges:
  - id: test_judge
    llm_client: invalid_client
    model: gpt-4
    prompt_file: {prompt_file}
"""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="Invalid llm_client"):
            meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

    def test_load_judges_from_yaml_prompt_file_not_found(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading judges with non-existent prompt file."""
        yaml_content = """judges:
  - id: test_judge
    llm_client: openai
    model: gpt-4
    prompt_file: non_existent_prompt.md
"""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(PromptFileNotFoundException):
            meta_evaluator_with_task.load_judges_from_yaml(str(yaml_file))

    def test_load_judges_from_yaml_multiple_judges(
        self, meta_evaluator_with_task, tmp_path
    ):
        """Test loading multiple judges from YAML."""
        # Create prompt files
        prompt1 = tmp_path / "prompt1.md"
        prompt1.write_text("Prompt 1")
        prompt2 = tmp_path / "prompt2.md"
        prompt2.write_text("Prompt 2")

        yaml_content = f"""judges:
  - id: judge1
    llm_client: openai
    model: gpt-4
    prompt_file: {prompt1}
  - id: judge2
    llm_client: openai
    model: gpt-3.5-turbo
    prompt_file: {prompt2}
"""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)

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
        meta_evaluator_with_task.add_judge(
            judge_id="test_judge",
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        # Create YAML with same judge ID
        prompt_file = tmp_path / "new_prompt.md"
        prompt_file.write_text("New prompt content")

        yaml_content = f"""judges:
  - id: test_judge
    llm_client: openai
    model: gpt-3.5-turbo
    prompt_file: {prompt_file}
"""
        yaml_file = tmp_path / "judges.yaml"
        yaml_file.write_text(yaml_content)

        # Load with override
        meta_evaluator_with_task.load_judges_from_yaml(
            str(yaml_file), override_existing=True
        )

        # Verify judge was overridden
        judge = meta_evaluator_with_task.judge_registry["test_judge"]
        assert judge.model == "gpt-3.5-turbo"
        assert judge.prompt.prompt == "New prompt content"
