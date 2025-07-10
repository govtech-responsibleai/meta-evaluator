"""Judge management functionality for MetaEvaluator."""

import yaml
from pathlib import Path
from typing import Optional
from pydantic import ValidationError, BaseModel

from ..eval_task import EvalTask
from ..llm_client.models import LLMClientEnum
from ..judge import Judge
from ..common.models import Prompt
from .exceptions import (
    JudgeAlreadyExistsException,
    JudgeNotFoundException,
    InvalidYAMLStructureException,
    PromptFileNotFoundException,
)


class JudgeConfig(BaseModel):
    """Pydantic model for validating a single judge configuration from YAML."""

    id: str
    llm_client: str
    model: str
    prompt_file: str


class JudgeConfigList(BaseModel):
    """Pydantic model for validating the entire judges YAML configuration."""

    judges: list[JudgeConfig]


class JudgesMixin:
    """Mixin class for MetaEvaluator judge management functionality."""

    # Type hints for attributes that will be provided by MetaEvaluator
    eval_task: Optional[EvalTask]

    def __init__(self, *args, **kwargs):
        """Initialize judge registry."""
        super().__init__(*args, **kwargs)
        self.judge_registry = {}

    def add_judge(
        self,
        judge_id: str,
        llm_client_enum: LLMClientEnum,
        model: str,
        prompt: Prompt,
        override_existing: bool = False,
    ) -> None:
        """Add a judge to the evaluator programmatically.

        Args:
            judge_id: Unique identifier for the judge.
            llm_client_enum: The LLM client enum to use.
            model: The model name to use.
            prompt: The Prompt object containing the evaluation instructions.
            override_existing: Whether to override existing judge. Defaults to False.

        Raises:
            JudgeAlreadyExistsException: If judge already exists and override_existing is False.
            ValueError: If eval_task is not set.
        """
        if self.eval_task is None:
            raise ValueError("eval_task must be set before adding judges")

        if judge_id in self.judge_registry and not override_existing:
            raise JudgeAlreadyExistsException(judge_id)

        judge = Judge(
            id=judge_id,
            eval_task=self.eval_task,
            llm_client_enum=llm_client_enum,
            model=model,
            prompt=prompt,
        )

        self.judge_registry[judge_id] = judge

    def get_judge(self, judge_id: str) -> Judge:
        """Get a judge from the registry by ID.

        Args:
            judge_id: The judge ID to retrieve.

        Returns:
            Judge: The requested Judge instance.

        Raises:
            JudgeNotFoundException: If the judge ID is not found in the registry.
        """
        if judge_id not in self.judge_registry:
            raise JudgeNotFoundException(judge_id)

        return self.judge_registry[judge_id]

    def get_judge_list(self) -> list[tuple[str, Judge]]:
        """Get a list of judge tuples (id, judge).

        Returns:
            List of judge tuples.
        """
        return list(self.judge_registry.items())

    def load_judges_from_yaml(
        self,
        yaml_file: str,
        override_existing: bool = False,
    ) -> None:
        """Load judges from a YAML configuration file.

        The YAML file should have the following structure:
        ```yaml
        judges:
          - id: judge_id_1
            llm_client: openai
            model: gpt-4
            prompt_file: /absolute/path/to/toxicity_prompt.md
          - id: judge_id_2
            llm_client: azure_openai
            model: gpt-4
            prompt_file: ./relative/path/to/relevance_prompt.txt
        ```

        Args:
            yaml_file: Absolute or relative path to the YAML configuration file.
            override_existing: Whether to override existing judges. Defaults to False.

        Raises:
            FileNotFoundError: If the YAML file is not found.
            InvalidYAMLStructureException: If the YAML structure is invalid.
            ValueError: If eval_task is not set or if llm_client value is invalid.
        """
        if self.eval_task is None:
            raise ValueError("eval_task must be set before loading judges from YAML")

        # Resolve YAML file path (can be absolute or relative)
        yaml_path = Path(yaml_file)

        # Load and validate YAML
        try:
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        except yaml.YAMLError as e:
            raise InvalidYAMLStructureException(f"Invalid YAML syntax: {e}")

        # Validate YAML structure
        try:
            judges_config = JudgeConfigList.model_validate(yaml_data)
        except ValidationError as e:
            raise InvalidYAMLStructureException(f"YAML validation failed: {e}")

        # Load each judge
        for judge_config in judges_config.judges:
            self._load_single_judge_from_config(
                judge_config, override_existing, yaml_path
            )

    def _load_single_judge_from_config(
        self,
        judge_config: JudgeConfig,
        override_existing: bool,
        yaml_path: Path,
    ) -> None:
        """Load a single judge from YAML configuration.

        Args:
            judge_config: The judge configuration from YAML.
            override_existing: Whether to override existing judges.
            yaml_path: Path to the YAML file (for resolving relative prompt paths).

        Raises:
            ValueError: If llm_client value is invalid.
        """
        # Parse LLM client enum
        try:
            llm_client_enum = LLMClientEnum(judge_config.llm_client)
        except ValueError:
            valid_clients = [e.value for e in LLMClientEnum]
            raise ValueError(
                f"Invalid llm_client '{judge_config.llm_client}'. "
                f"Valid options: {valid_clients}"
            )

        # Load prompt file from absolute or relative path
        prompt = self._load_prompt_from_file(judge_config.prompt_file, yaml_path)

        # Use add_judge method to avoid duplication
        self.add_judge(
            judge_id=judge_config.id,
            llm_client_enum=llm_client_enum,
            model=judge_config.model,
            prompt=prompt,
            override_existing=override_existing,
        )

    def _load_prompt_from_file(
        self, prompt_file: str, yaml_path: Optional[Path] = None
    ) -> Prompt:
        """Load prompt content from file using absolute or relative path.

        Args:
            prompt_file: Absolute or relative path to the prompt file.
            yaml_path: Optional path to YAML file for resolving relative prompt paths.

        Returns:
            Prompt: The loaded Prompt object.

        Raises:
            PromptFileNotFoundException: If prompt file cannot be found.
        """
        # Resolve prompt file path
        prompt_path = Path(prompt_file)

        if prompt_path.is_absolute():
            # Use absolute path as-is
            resolved_prompt_path = prompt_path
        else:
            # Resolve relative path
            if yaml_path is not None:
                # Relative to YAML file directory
                resolved_prompt_path = yaml_path.parent / prompt_path
            else:
                # Relative to current working directory
                resolved_prompt_path = prompt_path

        # Load prompt content
        try:
            with open(resolved_prompt_path, "r", encoding="utf-8") as f:
                prompt_content = f.read().strip()
        except FileNotFoundError:
            raise PromptFileNotFoundException(str(resolved_prompt_path))

        # Create prompt with file stem as ID
        prompt_id = resolved_prompt_path.stem
        return Prompt(id=prompt_id, prompt=prompt_content)
