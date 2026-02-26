"""Judge management functionality for MetaEvaluator."""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ValidationError
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

if TYPE_CHECKING:
    from .base import Paths

from ..common.async_utils import sync_wrapper
from ..common.models import Prompt
from ..data import EvalData, SampleEvalData
from ..eval_task import EvalTask
from ..judge import Judge
from ..results import JudgeResults, JudgeResultsBuilder
from ..results.enums import EvaluationStatusEnum
from .exceptions import (
    EvalDataNotFoundError,
    EvalTaskNotFoundError,
    InvalidYAMLStructureError,
    JudgeAlreadyExistsError,
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
    temperature: float | None = None
    extra_headers: dict[str, str] | None = None


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
    eval_task: EvalTask | None
    data: EvalData | None
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
        on_duplicate: Literal["skip", "overwrite"] | None = None,
        temperature: float | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        """Add a judge to the evaluator programmatically.

        Args:
            judge_id: Unique identifier for the judge.
            llm_client: The LLM client to use.
            model: The model name to use.
            prompt: The Prompt object containing the evaluation instructions.
            on_duplicate: How to handle duplicate judge IDs. Options:
                - None: Throw JudgeAlreadyExistsError if judge exists
                - "skip": Skip adding if judge already exists
                - "overwrite": Replace existing judge
            temperature: Sampling temperature for the LLM. If None, uses the
                model's default temperature.
            extra_headers: Additional HTTP headers to pass to the LLM API call.

        Raises:
            JudgeAlreadyExistsError: If judge already exists and on_duplicate is None.
            EvalTaskNotFoundError: If eval_task is not set.
        """
        if self.eval_task is None:
            raise EvalTaskNotFoundError("eval_task must be set before adding judges")

        # Handle duplicate judge IDs
        if judge_id in self.judge_registry:
            if on_duplicate is None:
                raise JudgeAlreadyExistsError(judge_id)

            elif on_duplicate == "skip":
                self.logger.info(f"Skipping judge '{judge_id}' - already exists")
                return

            elif on_duplicate == "overwrite":
                self.logger.info(f"Overwriting existing judge '{judge_id}'")
            # Continue to create judge for overwrite case

        judge = Judge(
            id=judge_id,
            eval_task=self.eval_task,
            llm_client=llm_client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            extra_headers=extra_headers,
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
        on_duplicate: Literal["skip", "overwrite"] | None = None,
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
            on_duplicate: How to handle duplicate judge IDs. Options:
                - None: Throw JudgeAlreadyExistsError if judge exists
                - "skip": Skip adding if judge already exists
                - "overwrite": Replace existing judge
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
                judge_config=judge_config,
                yaml_path=yaml_path,
                on_duplicate=on_duplicate,
                async_mode=async_mode,
            )

        self.logger.info(
            f"Successfully loaded {len(judges_config.judges)} judges from YAML"
        )

    def _load_single_judge_from_config(
        self,
        judge_config: JudgeConfig,
        yaml_path: Path,
        on_duplicate: Literal["skip", "overwrite"] | None,
        async_mode: bool = False,
    ) -> None:
        """Load a single judge from YAML configuration.

        Args:
            judge_config: The judge configuration from YAML.
            yaml_path: Path to the YAML file (for resolving relative prompt paths).
            on_duplicate: How to handle duplicate judge IDs.
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
            on_duplicate=on_duplicate,
            temperature=judge_config.temperature,
            extra_headers=judge_config.extra_headers,
        )

    def _load_prompt_from_file(
        self, prompt_file: str, yaml_path: Path | None = None
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

    def _is_classification_task(self) -> bool:
        """Check whether all task schemas are classification (no free-form schemas).

        Returns:
            bool: True if every task schema has predefined outcomes (not None).
        """
        if self.eval_task is None:
            return False
        return all(
            outcomes is not None for outcomes in self.eval_task.task_schemas.values()
        )

    def _majority_vote_outcomes(
        self, outcomes_list: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Compute majority-voted label for each task across multiple outcome dicts.

        For each task, counts label occurrences and picks the most frequent one.
        In case of a tie, the first-occurring label in the list wins.

        Args:
            outcomes_list: List of outcome dicts mapping task name to label string.

        Returns:
            dict[str, str]: Majority-voted label for each task.
        """
        assert self.eval_task is not None
        result: dict[str, str] = {}
        for task_name in self.eval_task.task_schemas:
            labels = [
                o[task_name]
                for o in outcomes_list
                if task_name in o and o[task_name] is not None
            ]
            if not labels:
                continue
            counts: dict[str, int] = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            max_count = max(counts.values())
            # First-occurrence tie-breaking: iterate in original order
            for label in labels:
                if counts[label] == max_count:
                    result[task_name] = label
                    break
        return result

    def _aggregate_outcomes(
        self, outcomes_list: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Aggregate outcomes from multiple consistency runs per task.

        For classification tasks (non-None schemas), uses majority voting with
        first-occurrence tie-breaking. For free-form tasks (None schemas),
        concatenates all outputs with indexed run markers.

        Args:
            outcomes_list: List of outcome dicts mapping task name to label/text.

        Returns:
            dict[str, str]: Aggregated outcome for each task.
        """
        assert self.eval_task is not None
        result: dict[str, str] = {}
        for task_name, schema in self.eval_task.task_schemas.items():
            labels = [
                o[task_name]
                for o in outcomes_list
                if task_name in o and o[task_name] is not None
            ]
            if not labels:
                continue
            if schema is None:
                # Free-form task: concatenate outputs with run markers
                parts = [
                    f"<RUN {i}>\n{label}" for i, label in enumerate(labels, start=1)
                ]
                result[task_name] = "\n===\n".join(parts)
            else:
                # Classification task: majority vote with first-occurrence tie-breaking
                counts: dict[str, int] = {}
                for label in labels:
                    counts[label] = counts.get(label, 0) + 1
                max_count = max(counts.values())
                for label in labels:
                    if counts[label] == max_count:
                        result[task_name] = label
                        break
        return result

    def _aggregate_consistency_results(
        self,
        all_results: list[JudgeResults],
        judge: Judge,
        run_id: str,
    ) -> JudgeResults:
        """Aggregate N JudgeResults from consistency runs using majority voting.

        For each row, collects task outcomes from all successful runs and selects
        the majority label per task. Ties are broken by first occurrence. Token
        counts and call durations are summed across all successful runs.

        Rows that were skipped in all runs remain skipped. Rows that never
        succeeded in any run retain the error type from the first failing run.

        Args:
            all_results: List of JudgeResults from N consistency runs.
            judge: The Judge instance used for the evaluations.
            run_id: Run identifier for the returned aggregated JudgeResults.

        Returns:
            JudgeResults: A single aggregated result with majority-voted labels.
        """
        assert self.data is not None
        assert self.data.id_column is not None

        expected_ids = self.data.data[self.data.id_column].to_list()

        builder = JudgeResultsBuilder(
            run_id=run_id,
            judge_id=judge.id,
            llm_client=judge.llm_client,
            model_used=judge.model,
            task_schemas=judge.eval_task.task_schemas,
            expected_ids=expected_ids,
            required_tasks=judge.eval_task.get_required_tasks(),
            is_sampled_run=isinstance(self.data, SampleEvalData),
        )

        # Collect per-original_id info from all runs
        id_info: dict[Any, dict[str, Any]] = {}
        for run_result in all_results:
            for row in run_result.results_data.iter_rows(named=True):
                orig_id = row["original_id"]
                status = row["status"]

                if orig_id not in id_info:
                    id_info[orig_id] = {
                        "successes": [],
                        "skip": False,
                        "first_error": None,
                        "agg_prompt_tokens": 0,
                        "agg_completion_tokens": 0,
                        "agg_total_tokens": 0,
                        "agg_duration": 0.0,
                        "last_llm_response": None,
                    }

                if status == EvaluationStatusEnum.SKIPPED.value:
                    id_info[orig_id]["skip"] = True
                elif status == EvaluationStatusEnum.SUCCESS.value:
                    outcomes = {
                        task: row[task]
                        for task in judge.eval_task.task_schemas
                        if task in row
                    }
                    id_info[orig_id]["successes"].append(outcomes)
                    id_info[orig_id]["agg_prompt_tokens"] += (
                        row.get("llm_prompt_tokens") or 0
                    )
                    id_info[orig_id]["agg_completion_tokens"] += (
                        row.get("llm_completion_tokens") or 0
                    )
                    id_info[orig_id]["agg_total_tokens"] += (
                        row.get("llm_total_tokens") or 0
                    )
                    id_info[orig_id]["agg_duration"] += (
                        row.get("llm_call_duration_seconds") or 0.0
                    )
                    id_info[orig_id]["last_llm_response"] = row.get(
                        "llm_raw_response_content"
                    )
                elif id_info[orig_id]["first_error"] is None:
                    id_info[orig_id]["first_error"] = row

        # Build aggregated rows for each expected ID
        for i, orig_id in enumerate(expected_ids):
            sample_example_id = f"{run_id}_agg_{i + 1}"
            info = id_info.get(orig_id)

            if info is None:
                builder.create_other_error_row(
                    sample_example_id=sample_example_id,
                    original_id=orig_id,
                    error=Exception(
                        f"No results found for ID '{orig_id}' across consistency runs"
                    ),
                )
                continue

            if info["successes"]:
                majority_outcomes = self._aggregate_outcomes(info["successes"])
                builder.create_success_row(
                    sample_example_id=sample_example_id,
                    original_id=orig_id,
                    outcomes=majority_outcomes,
                    llm_raw_response_content=info["last_llm_response"] or "",
                    llm_prompt_tokens=info["agg_prompt_tokens"],
                    llm_completion_tokens=info["agg_completion_tokens"],
                    llm_total_tokens=info["agg_total_tokens"],
                    llm_call_duration_seconds=info["agg_duration"],
                )
            elif info["skip"]:
                builder.create_skipped_row(
                    sample_example_id=sample_example_id,
                    original_id=orig_id,
                )
            elif info["first_error"] is not None:
                error_row = info["first_error"]
                error = Exception(error_row.get("error_message") or "Unknown error")
                status = error_row.get("status", "")
                if status == EvaluationStatusEnum.LLM_ERROR.value:
                    builder.create_llm_error_row(
                        sample_example_id=sample_example_id,
                        original_id=orig_id,
                        error=error,
                    )
                elif status == EvaluationStatusEnum.PARSING_ERROR.value:
                    builder.create_parsing_error_row(
                        sample_example_id=sample_example_id,
                        original_id=orig_id,
                        error=error,
                        llm_raw_response_content=error_row.get(
                            "llm_raw_response_content"
                        )
                        or "",
                        llm_prompt_tokens=error_row.get("llm_prompt_tokens") or 0,
                        llm_completion_tokens=error_row.get("llm_completion_tokens")
                        or 0,
                        llm_total_tokens=error_row.get("llm_total_tokens") or 0,
                        llm_call_duration_seconds=error_row.get(
                            "llm_call_duration_seconds"
                        )
                        or 0.0,
                    )
                else:
                    builder.create_other_error_row(
                        sample_example_id=sample_example_id,
                        original_id=orig_id,
                        error=error,
                    )
            else:
                builder.create_other_error_row(
                    sample_example_id=sample_example_id,
                    original_id=orig_id,
                    error=Exception(
                        f"No usable results for ID '{orig_id}' across consistency runs"
                    ),
                )

        return builder.complete()

    def _validate_and_prepare_judges_run(
        self, judge_ids: str | list[str] | None, run_id: str | None
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
        judge_ids: str | list[str] | None = None,
        run_id: str | None = None,
        save_results: bool = True,
        results_format: Literal["json", "csv", "parquet"] = "json",
        skip_duplicates: bool = False,
        consistency: int = 1,
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
            skip_duplicates: Whether to skip judges that already have results in the
                project directory. Works like overwrite=False, except it doesn't overwrite
                and instead collects all results. The load_all_judge_results method
                handles how to deal with duplicate results (currently takes most recent).
                - False: Run judges regardless and log duplicate warnings
                - True: Skip judges with existing results and log warnings
            consistency: Number of times to run each judge per row and take the
                majority label. Must be >= 1. When consistency=1 (default), each
                row is evaluated once. When consistency > 1, the judge runs that
                many times per row and the majority label is selected (first
                occurrence wins ties). Only applicable when all task schemas are
                classification tasks (no free-form schemas).

        Returns:
            dict[str, JudgeResults]: Dictionary mapping judge IDs to their results.

        Raises:
            ResultsSaveError: If saving results fails.
            ValueError: If consistency < 1.

        Example:
            >>> # Run all judges, skipping those with existing results
            >>> results = evaluator.run_judges()

            >>> # Run specific judges
            >>> results = evaluator.run_judges(judge_ids=["judge_1", "judge_2"])

            >>> # Run with custom run ID
            >>> results = evaluator.run_judges(run_id="experiment_1")

            >>> # Force run all judges even if results exist
            >>> results = evaluator.run_judges(skip_duplicates=False)

            >>> # Run each judge 3 times and take the majority label
            >>> results = evaluator.run_judges(consistency=3)
        """
        judges_to_run, run_id = self._validate_and_prepare_judges_run(judge_ids, run_id)

        # Validate consistency parameter
        if consistency < 1:
            raise ValueError(f"consistency must be >= 1, got {consistency}")

        # Get set of existing judge IDs from results directory
        existing_judge_ids = self._get_existing_judge_ids()

        # Filter judges based on skip_duplicates setting
        final_judges_to_run = []
        for judge_id in judges_to_run:
            if judge_id in existing_judge_ids:
                if skip_duplicates:
                    self.logger.info(
                        f"Skipping judge '{judge_id}' - existing results found in project directory"
                    )
                    continue
                else:
                    self.logger.warning(
                        f"Judge '{judge_id}' has existing results but will be run again (skip_duplicates=False)"
                    )
            final_judges_to_run.append(judge_id)

        if not final_judges_to_run:
            self.logger.info("No judges to run after checking for duplicates")
            return {}

        self.logger.info(
            f"Starting judge evaluation run '{run_id}' with {len(final_judges_to_run)} judge(s): {', '.join(final_judges_to_run)}"
        )

        # Run each judge and collect results
        results = {}
        for judge_id in tqdm(
            final_judges_to_run, desc="Evaluating judges", unit="judge"
        ):
            judge = self.judge_registry[judge_id]

            if consistency > 1:
                # Run N consistency rounds sequentially and aggregate
                self.logger.info(
                    f"Running {consistency} consistency rounds for judge '{judge_id}'..."
                )
                all_run_results = []
                for c_idx in range(consistency):
                    c_run_id = f"{run_id}_{judge_id}_c{c_idx + 1}"
                    self.logger.info(
                        f"  Consistency round {c_idx + 1}/{consistency} for judge '{judge_id}'..."
                    )
                    c_results = judge.evaluate_eval_data(
                        eval_data=self.data,
                        run_id=c_run_id,
                    )
                    all_run_results.append(c_results)
                judge_results = self._aggregate_consistency_results(
                    all_results=all_run_results,
                    judge=judge,
                    run_id=f"{run_id}_{judge_id}",
                )
            else:
                # Run the evaluation once (default)
                self.logger.info(f"Running evaluation for judge '{judge_id}'...")
                judge_results = judge.evaluate_eval_data(
                    eval_data=self.data,
                    run_id=f"{run_id}_{judge_id}",
                )
            results[judge_id] = judge_results

            # Calculate success and non-success counts
            non_success_count = (
                judge_results.total_count - judge_results.succeeded_count
            )
            success_pct = (
                (judge_results.succeeded_count / judge_results.total_count * 100)
                if judge_results.total_count > 0
                else 0
            )
            non_success_pct = (
                (non_success_count / judge_results.total_count * 100)
                if judge_results.total_count > 0
                else 0
            )

            self.logger.info(
                f"Judge '{judge_id}' completed: {judge_results.succeeded_count}/{judge_results.total_count} successful ({success_pct:.1f}%), {non_success_count} non-successful ({non_success_pct:.1f}%)"
            )

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

    @sync_wrapper
    def run_judges_async(self, *args, **kwargs):
        """Wrapper for _run_judges_async that handles asyncio internally.

        Returns:
            dict[str, JudgeResults]: Dictionary mapping judge IDs to their results.
        """
        return self._run_judges_async(*args, **kwargs)

    async def _run_judges_async(
        self,
        judge_ids: str | list[str] | None = None,
        run_id: str | None = None,
        save_results: bool = True,
        results_format: Literal["json", "csv", "parquet"] = "json",
        skip_duplicates: bool = False,
        consistency: int = 1,
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
            skip_duplicates: Whether to skip judges that already have results in the
                project directory. Works like overwrite=False, except it doesn't overwrite
                and instead collects all results. The load_all_judge_results method
                handles how to deal with duplicate results (currently takes most recent).
                - False: Run judges regardless and log duplicate warnings
                - True: Skip judges with existing results and log warnings
            consistency: Number of times to run each judge per row and take the
                majority label. Must be >= 1. When consistency=1 (default), each
                row is evaluated once. When consistency > 1, the judge runs that
                many times per row concurrently and the majority label is selected
                (first occurrence wins ties). Only applicable when all task schemas
                are classification tasks (no free-form schemas).

        Returns:
            dict[str, JudgeResults]: Dictionary mapping judge IDs to their results.

        Raises:
            ValueError: If consistency < 1.
        """
        judges_to_run, run_id = self._validate_and_prepare_judges_run(judge_ids, run_id)

        # Validate consistency parameter
        if consistency < 1:
            raise ValueError(f"consistency must be >= 1, got {consistency}")

        # Get set of existing judge IDs from results directory
        existing_judge_ids = self._get_existing_judge_ids()

        # Filter judges based on skip_duplicates setting
        final_judges_to_run = []
        for judge_id in judges_to_run:
            if judge_id in existing_judge_ids:
                if skip_duplicates:
                    self.logger.info(
                        f"Skipping judge '{judge_id}' - existing results found in project directory"
                    )
                    continue
                else:
                    self.logger.warning(
                        f"Judge '{judge_id}' has existing results but will be run again (skip_duplicates=False)"
                    )
            final_judges_to_run.append(judge_id)

        if not final_judges_to_run:
            self.logger.info("No judges to run after checking for duplicates")
            return {}

        self.logger.info(
            f"Starting async judge evaluation run '{run_id}' with {len(final_judges_to_run)} judge(s): {', '.join(final_judges_to_run)}"
        )

        # Define function to run single judge evaluation in parallel
        async def run_single_judge(judge_id: str) -> tuple[str, JudgeResults]:
            """Run evaluation for a single judge and return results.

            Args:
                judge_id: ID of the judge.

            Returns:
                tuple[str, JudgeResults]: Tuple containing the judge ID and its results.
            """
            judge = self.judge_registry[judge_id]

            if consistency > 1:
                # Run N consistency rounds concurrently and aggregate
                self.logger.info(
                    f"Running {consistency} consistency rounds (async) for judge '{judge_id}'..."
                )
                consistency_tasks = [
                    judge.evaluate_eval_data_async(
                        eval_data=self.data,
                        run_id=f"{run_id}_{judge_id}_c{c_idx + 1}",
                    )
                    for c_idx in range(consistency)
                ]
                all_run_results = list(await asyncio.gather(*consistency_tasks))
                judge_results = self._aggregate_consistency_results(
                    all_results=all_run_results,
                    judge=judge,
                    run_id=f"{run_id}_{judge_id}",
                )
            else:
                # Run the evaluation once (default)
                self.logger.info(f"Running async evaluation for judge '{judge_id}'...")
                judge_results = await judge.evaluate_eval_data_async(
                    eval_data=self.data,
                    run_id=f"{run_id}_{judge_id}",
                )

            # Calculate success and non-success counts
            non_success_count = (
                judge_results.total_count - judge_results.succeeded_count
            )
            success_pct = (
                (judge_results.succeeded_count / judge_results.total_count * 100)
                if judge_results.total_count > 0
                else 0
            )
            non_success_pct = (
                (non_success_count / judge_results.total_count * 100)
                if judge_results.total_count > 0
                else 0
            )

            self.logger.info(
                f"Judge '{judge_id}' completed: {judge_results.succeeded_count}/{judge_results.total_count} successful ({success_pct:.1f}%), {non_success_count} non-successful ({non_success_pct:.1f}%)"
            )
            return judge_id, judge_results

        # Execute all judges in parallel with progress bar
        judge_tasks = [run_single_judge(judge_id) for judge_id in final_judges_to_run]
        judge_results_list = await async_tqdm.gather(
            *judge_tasks, desc="Evaluating judges", unit="judge"
        )

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

    def _get_existing_judge_ids(self) -> set[str]:
        """Get set of judge IDs that already have results in the project directory.

        Returns:
            set[str]: Set of judge IDs found in existing result files.
        """
        existing_judge_ids = set()

        if not self.paths.results.exists():
            return existing_judge_ids

        # Look for all *_state.json files in results directory
        state_files = list(self.paths.results.glob("*_state.json"))

        for state_file in state_files:
            try:
                # Load the state file to get judge_id
                judge_results = JudgeResults.load_state(str(state_file))
                existing_judge_ids.add(judge_results.judge_id)
            except Exception as e:
                # Skip files that can't be loaded as judge results
                self.logger.debug(
                    f"Could not load judge results from {state_file.name}: {e}"
                )
                continue

        return existing_judge_ids

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
