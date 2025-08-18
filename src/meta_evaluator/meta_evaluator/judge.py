"""Judge management functionality for MetaEvaluator."""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from .base import Paths

from ..common.models import Prompt
from ..data import EvalData
from ..eval_task import EvalTask
from ..judge import Judge
from ..results import JudgeResults
from .exceptions import (
    EvalDataNotFoundError,
    EvalTaskNotFoundError,
    InvalidYAMLStructureError,
    JudgeAlreadyExistsError,
    JudgeExecutionError,
    JudgeNotFoundError,
    PromptFileNotFoundError,
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
    """Mixin providing judge handling functionality for MetaEvaluator.

    This mixin class manages the creation, configuration, and execution of judge
    evaluations. It handles loading judge configurations from YAML files, executing
    judge evaluations against evaluation data, and saving the results. Judges are
    LLM-based evaluators that assess text content or LLM outputs according to
    specified evaluation tasks.

    The mixin supports:
    - Loading judge configurations from YAML files with validation
    - Creating Judge instances with prompts and LLM clients
    - Executing evaluations against datasets (full or sampled)
    - Saving judge results with proper metadata and state tracking

    Judge Configuration Requirements:
        - id: Unique identifier for the judge
        - llm_client: LLM client type (openai, azure, etc.)
        - model: Model name to use
        - prompt_file: Path to system prompt file

    Attributes:
        eval_task (Optional[EvalTask]): Inherited evaluation task configuration.
        data (Optional[EvalData]): Inherited evaluation dataset.
        judge_registry (dict): Inherited judge registry.
        paths (Paths): Inherited project directory structure.
        logger (logging.Logger): Inherited logger instance.

    Examples:
        >>> evaluator = MetaEvaluator()
        >>> evaluator.add_evaluation_data("data.csv", name="test_data")
        >>> evaluator.add_evaluation_task(task_schemas={"toxicity": ["toxic", "non_toxic"]})
        >>>
        >>> # Load judges from YAML configuration
        >>> evaluator.load_judges_from_yaml("judges_config.yaml")
        >>>
        >>> # Execute specific judge
        >>> results = evaluator.run_judge("judge_1", run_on_sample=True)
        >>>
        >>> # Execute all configured judges
        >>> evaluator.run_all_judges()
    """

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
        llm_client: str,
        model: str,
        prompt: Prompt,
        override: bool = False,
    ) -> None:
        """Add a judge to the evaluator programmatically.

        Args:
            judge_id: Unique identifier for the judge.
            llm_client: The LLM client to use.
            model: The model name to use.
            prompt: The Prompt object containing the evaluation instructions.
            override: Whether to override existing judge. Defaults to False.

        Raises:
            JudgeAlreadyExistsError: If judge already exists and override is False.
            EvalTaskNotFoundError: If eval_task is not set.
        """
        if self.eval_task is None:
            raise EvalTaskNotFoundError("eval_task must be set before adding judges")

        if judge_id in self.judge_registry and not override:
            raise JudgeAlreadyExistsError(judge_id)

        judge = Judge(
            id=judge_id,
            eval_task=self.eval_task,
            llm_client=llm_client,
            model=model,
            prompt=prompt,
        )

        self.judge_registry[judge_id] = judge

        self.logger.info(
            f"Added judge '{judge_id}' using {llm_client} client with model '{model}'"
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
        override: bool = False,
        async_mode: bool = False,
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
            llm_client: azure
            model: gpt-4
            prompt_file: ./relative/path/to/relevance_prompt.txt
        ```

        Args:
            yaml_file: Absolute or relative path to the YAML configuration file.
            override: Whether to override existing judges. Defaults to False.
            async_mode: Whether to load judges for async operation. Defaults to False.
                       When True, judges will be configured with async methods.

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
                judge_config, override, yaml_path, async_mode
            )

        self.logger.info(
            f"Successfully loaded {len(judges_config.judges)} judges from YAML"
        )

    def _load_single_judge_from_config(
        self,
        judge_config: JudgeConfig,
        override: bool,
        yaml_path: Path,
        async_mode: bool = False,
    ) -> None:
        """Load a single judge from YAML configuration.

        Args:
            judge_config: The judge configuration from YAML.
            override: Whether to override existing judges.
            yaml_path: Path to the YAML file (for resolving relative prompt paths).
            async_mode: Whether to load judges for async operation.
        """
        # Load prompt file from absolute or relative path
        prompt = self._load_prompt_from_file(judge_config.prompt_file, yaml_path)

        # Use add_judge method to avoid duplication
        self.add_judge(
            judge_id=judge_config.id,
            llm_client=judge_config.llm_client,
            model=judge_config.model,
            prompt=prompt,
            override=override,
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

    def _validate_and_prepare_judges_run(
        self, judge_ids: Optional[Union[str, list[str]]], run_id: Optional[str]
    ) -> tuple[list[str], str]:
        """Validate prerequisites and prepare judges list for execution.

        Args:
            judge_ids: Judge ID(s) to run.
            run_id: Run identifier.

        Returns:
            tuple: (judges_to_run, final_run_id)

        Raises:
            EvalTaskNotFoundError: If eval_task is not set.
            EvalDataNotFoundError: If data is not set.
            JudgeNotFoundError: If specified judges don't exist.
        """
        # Validate prerequisites
        if self.eval_task is None:
            raise EvalTaskNotFoundError("eval_task must be set before running judges")
        if self.data is None:
            raise EvalDataNotFoundError("data must be set before running judges")

        # Determine which judges to run
        if judge_ids is None:
            judges_to_run = list(self.judge_registry.keys())
        elif isinstance(judge_ids, str):
            judges_to_run = [judge_ids]
        else:
            judges_to_run = judge_ids

        # Validate judges exist and are available
        missing_judges = [
            j_id for j_id in judges_to_run if j_id not in self.judge_registry
        ]
        if missing_judges:
            raise JudgeNotFoundError(
                f"Judge IDs not found in registry: {missing_judges}"
            )
        if not judges_to_run:
            raise JudgeNotFoundError("No judges available to run")

        # Generate run ID if not provided
        if run_id is None:
            run_id = (
                f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            )

        self.paths.ensure_directories()

        return judges_to_run, run_id

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
        judges_to_run, run_id = self._validate_and_prepare_judges_run(judge_ids, run_id)

        self.logger.info(
            f"Starting judge evaluation run '{run_id}' with {len(judges_to_run)} judge(s): {', '.join(judges_to_run)}"
        )

        # Run each judge and collect results
        results = {}
        for judge_id in judges_to_run:
            judge = self.judge_registry[judge_id]

            # Run the evaluation
            try:
                self.logger.info(f"Running evaluation for judge '{judge_id}'...")
                judge_results = judge.evaluate_eval_data(
                    eval_data=self.data,
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

    async def run_judges_async(
        self,
        judge_ids: Optional[Union[str, list[str]]] = None,
        run_id: Optional[str] = None,
        save_results: bool = True,
        results_format: Literal["json", "csv", "parquet"] = "json",
    ) -> dict[str, JudgeResults]:
        """Execute evaluations using specified judges with parallel processing and save results.

        This is an async version of run_judges that executes multiple judges concurrently
        for improved performance. All judges run in parallel, and results are saved
        concurrently as well.

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
        """
        judges_to_run, run_id = self._validate_and_prepare_judges_run(judge_ids, run_id)

        self.logger.info(
            f"Starting async judge evaluation run '{run_id}' with {len(judges_to_run)} judge(s): {', '.join(judges_to_run)}"
        )

        # Define function to run single judge evaluation in parallel
        async def run_single_judge(judge_id: str) -> tuple[str, JudgeResults]:
            """Run evaluation for a single judge and return results.

            Args:
                judge_id: ID of the judge.

            Returns:
                tuple[str, JudgeResults]: Tuple containing the judge ID and its results.

            Raises:
                JudgeExecutionError: If the judge execution fails.
            """
            judge = self.judge_registry[judge_id]

            # Run the evaluation
            try:
                self.logger.info(f"Running async evaluation for judge '{judge_id}'...")
                judge_results = await judge.evaluate_eval_data_async(
                    eval_data=self.data,
                    run_id=f"{run_id}_{judge_id}",
                )
                self.logger.info(
                    f"Judge '{judge_id}' completed: {judge_results.succeeded_count}/{judge_results.total_count} successful evaluations"
                )
                return judge_id, judge_results
            except Exception as e:
                raise JudgeExecutionError(judge_id, str(e))

        # Execute all judges in parallel
        judge_tasks = [run_single_judge(judge_id) for judge_id in judges_to_run]
        judge_results_list = await asyncio.gather(*judge_tasks)

        # Collect results into dictionary
        results = {}
        for judge_id, judge_results in judge_results_list:
            results[judge_id] = judge_results

        # Save results if requested
        if save_results:
            # Define function to save single judge results in parallel
            async def save_single_judge_results(
                judge_id: str, judge_results: JudgeResults
            ):
                """Save results for a single judge.

                Args:
                    judge_id: ID of the judge.
                    judge_results: The JudgeResults object to save.

                Raises:
                    ResultsSaveError: If saving results fails.
                """
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

            # Save all results in parallel
            save_tasks = [
                save_single_judge_results(judge_id, judge_results)
                for judge_id, judge_results in results.items()
            ]
            await asyncio.gather(*save_tasks)

        self.logger.info(
            f"Async judge evaluation run '{run_id}' completed successfully with {len(results)} judge(s)"
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

    def validate_judge_registry(self) -> None:
        """Validate that judge_registry contains only Judge instances.

        Raises:
            JudgeNotFoundError: If judge_registry contains non-Judge values.
        """
        for judge_id, judge_instance in self.judge_registry.items():
            if not isinstance(judge_instance, Judge):
                raise JudgeNotFoundError(
                    f"Judge registry contains non-Judge instance for key '{judge_id}': "
                    f"got {type(judge_instance).__name__}, expected Judge"
                )
