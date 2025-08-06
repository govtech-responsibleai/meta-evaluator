"""Judge management functionality for MetaEvaluator."""

import logging
import yaml
from pathlib import Path
from typing import Optional, Union, Literal, TYPE_CHECKING
from pydantic import ValidationError, BaseModel
from datetime import datetime
import uuid

if TYPE_CHECKING:
    from .base import Paths

from ..eval_task import EvalTask
from ..llm_client.models import LLMClientEnum
from ..judge import Judge
from ..common.models import Prompt
from ..data import EvalData
from ..results import JudgeResults
from .exceptions import (
    JudgeAlreadyExistsError,
    JudgeNotFoundError,
    InvalidYAMLStructureError,
    PromptFileNotFoundError,
    EvalTaskNotFoundError,
    EvalDataNotFoundError,
    LLMClientNotConfiguredError,
    JudgeExecutionError,
    ResultsSaveError,
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
    data: Optional[EvalData]
    paths: "Paths"
    logger: logging.Logger

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
            JudgeAlreadyExistsError: If judge already exists and override_existing is False.
            EvalTaskNotFoundError: If eval_task is not set.
        """
        if self.eval_task is None:
            raise EvalTaskNotFoundError("eval_task must be set before adding judges")

        if judge_id in self.judge_registry and not override_existing:
            raise JudgeAlreadyExistsError(judge_id)

        judge = Judge(
            id=judge_id,
            eval_task=self.eval_task,
            llm_client_enum=llm_client_enum,
            model=model,
            prompt=prompt,
        )

        self.judge_registry[judge_id] = judge

        self.logger.info(
            f"Added judge '{judge_id}' using {llm_client_enum.value} client with model '{model}'"
        )

    def get_judge(self, judge_id: str) -> Judge:
        """Get a judge from the registry by ID.

        Args:
            judge_id: The judge ID to retrieve.

        Returns:
            Judge: The requested Judge instance.

        Raises:
            JudgeNotFoundError: If the judge ID is not found in the registry.
        """
        if judge_id not in self.judge_registry:
            raise JudgeNotFoundError(judge_id)

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
            InvalidYAMLStructureError: If the YAML structure is invalid.
            EvalTaskNotFoundError: If eval_task is not set.
        """
        self.logger.info(f"Loading judges from YAML file: {yaml_file}")

        if self.eval_task is None:
            raise EvalTaskNotFoundError(
                "eval_task must be set before loading judges from YAML"
            )

        # Resolve YAML file path (can be absolute or relative)
        yaml_path = Path(yaml_file)

        # Load and validate YAML
        try:
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        except yaml.YAMLError as e:
            raise InvalidYAMLStructureError(f"Invalid YAML syntax: {e}")

        # Validate YAML structure
        try:
            judges_config = JudgeConfigList.model_validate(yaml_data)
            self.logger.info(
                f"Found {len(judges_config.judges)} judge configurations in YAML"
            )
        except ValidationError as e:
            raise InvalidYAMLStructureError(f"YAML validation failed: {e}")

        # Load each judge
        for judge_config in judges_config.judges:
            self._load_single_judge_from_config(
                judge_config, override_existing, yaml_path
            )

        self.logger.info(
            f"Successfully loaded {len(judges_config.judges)} judges from YAML"
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
                f"Invalid llm_client '{judge_config.llm_client}'. Valid options: {valid_clients}"
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
            PromptFileNotFoundError: If prompt file cannot be found.
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
            raise PromptFileNotFoundError(str(resolved_prompt_path))

        # Create prompt with file stem as ID
        prompt_id = resolved_prompt_path.stem
        return Prompt(id=prompt_id, prompt=prompt_content)

    def run_judges(
        self,
        judge_ids: Optional[Union[str, list[str]]] = None,
        run_id: Optional[str] = None,
        save_results: bool = True,
        results_format: Literal["json", "csv", "parquet"] = "json",
    ) -> dict[str, JudgeResults]:
        """Execute evaluations using specified judges and save results.

        This function runs evaluations on the loaded data using the specified judges.
        Each judge will evaluate the data using its configured LLM client and prompt.
        Results are automatically saved to the project's results directory.

        Args:
            judge_ids: Judge ID(s) to run. Can be:
                - None: Run all judges in the registry
                - str: Run a single judge by ID
                - list[str]: Run specified judges by ID
            run_id: Unique identifier for this evaluation run. If None,
                auto-generates a timestamp-based ID.
            save_results: Whether to save results to disk. Defaults to True.
            results_format: Format for saving results files. Defaults to "json".

        Returns:
            dict[str, JudgeResults]: Dictionary mapping judge IDs to their results.

        Raises:
            EvalTaskNotFoundError: If eval_task is not set.
            EvalDataNotFoundError: If data is not set.
            JudgeNotFoundError: If specified judge_ids don't exist or no judges are available to run.
            LLMClientNotConfiguredError: If required LLM client is not configured.
            JudgeExecutionError: If judge execution fails.
            ResultsSaveError: If saving results fails.

        Example:
            >>> # Run all judges
            >>> results = evaluator.run_judges()

            >>> # Run specific judges
            >>> results = evaluator.run_judges(judge_ids=["judge_1", "judge_2"])

            >>> # Run with custom run ID
            >>> results = evaluator.run_judges(run_id="experiment_1")
        """
        # Validate prerequisites
        if self.eval_task is None:
            raise EvalTaskNotFoundError("eval_task must be set before running judges")

        if self.data is None:
            raise EvalDataNotFoundError("data must be set before running judges")

        # Determine which judges to run
        if judge_ids is None:
            # Run all judges
            judges_to_run = list(self.judge_registry.keys())
        elif isinstance(judge_ids, str):
            # Run single judge
            judges_to_run = [judge_ids]
        else:
            # Run specified judges
            judges_to_run = judge_ids

        # Validate that all specified judges exist
        missing_judges = [
            j_id for j_id in judges_to_run if j_id not in self.judge_registry
        ]
        if missing_judges:
            raise JudgeNotFoundError(
                f"Judge IDs not found in registry: {missing_judges}"
            )

        # Check if any judges are available
        if not judges_to_run:
            raise JudgeNotFoundError("No judges available to run")

        # Generate run ID if not provided
        if run_id is None:
            run_id = (
                f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            )

        # Ensure results directory exists
        self.paths.ensure_directories()

        self.logger.info(
            f"Starting judge evaluation run '{run_id}' with {len(judges_to_run)} judge(s): {', '.join(judges_to_run)}"
        )

        # Run each judge and collect results
        results = {}

        for judge_id in judges_to_run:
            judge = self.judge_registry[judge_id]

            # Get the appropriate LLM client for this judge
            llm_client = getattr(self, "client_registry", {}).get(judge.llm_client_enum)
            if llm_client is None:
                raise LLMClientNotConfiguredError(judge_id, judge.llm_client_enum.value)

            # Run the evaluation
            try:
                self.logger.info(f"Running evaluation for judge '{judge_id}'...")
                judge_results = judge.evaluate_eval_data(
                    eval_data=self.data,
                    llm_client=llm_client,
                    run_id=f"{run_id}_{judge_id}",
                )
                results[judge_id] = judge_results
                self.logger.info(
                    f"Judge '{judge_id}' completed: {judge_results.succeeded_count}/{judge_results.total_count} successful evaluations"
                )
            except Exception as e:
                raise JudgeExecutionError(judge_id, str(e))

            # Save results if requested
            if save_results:
                try:
                    self._save_judge_results(
                        judge_results=judge_results,
                        judge_id=judge_id,
                        run_id=run_id,
                        results_format=results_format,
                    )
                    self.logger.info(
                        f"Saved results for judge '{judge_id}' to results directory"
                    )
                except Exception as e:
                    raise ResultsSaveError(judge_id, run_id, str(e))

        self.logger.info(
            f"Judge evaluation run '{run_id}' completed successfully with {len(results)} judge(s)"
        )
        return results

    def _save_judge_results(
        self,
        judge_results: JudgeResults,
        judge_id: str,
        run_id: str,
        results_format: Literal["json", "csv", "parquet"],
    ) -> None:
        """Save judge results to disk.

        Args:
            judge_results: The JudgeResults object to save.
            judge_id: ID of the judge.
            run_id: ID of the run.
            results_format: Format for saving results.
        """
        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_filename = f"{run_id}_{judge_id}_{timestamp}_results.{results_format}"
        state_filename = f"{run_id}_{judge_id}_{timestamp}_state.json"

        # Save results using save_state method
        state_filepath = self.paths.results / state_filename
        judge_results.save_state(str(state_filepath), results_format, data_filename)
