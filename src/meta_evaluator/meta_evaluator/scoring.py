"""Mixin for scoring and comparison functionality including results loading."""

import logging
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Union

if TYPE_CHECKING:
    from .base import Paths

import polars as pl

from ..data import EvalData
from ..eval_task import EvalTask
from ..results import HumanAnnotationResults, JudgeResults
from ..scores import BaseScoringResult, MetricsConfig
from .exceptions import (
    EvalTaskNotFoundError,
    IncompatibleTaskError,
    InsufficientDataError,
    ScoringConfigError,
)


class ScoringMixin:
    """Mixin providing orchestration of scorers for human vs judge alignment evaluation.

    This mixin class orchestrates the comparison between judge evaluations and human
    annotations using various scoring metrics. It handles loading both judge results
    and human annotation results, then computes alignment scores using different
    scoring implementations (accuracy, Cohen's kappa, text similarity, etc.).

    The mixin manages the complete scoring workflow:
    - Loading judge results from evaluation runs
    - Loading human annotation results from annotation sessions
    - Validating task compatibility between results and scorers
    - Computing alignment scores between judge and human evaluations
    - Aggregating and saving scoring results

    Scoring Process:
        1. Load judge and human results with task schema validation
        2. Configure scoring metrics based on task types
        3. Compute alignment scores for each judge-human pairing
        4. Generate aggregate visualizations and reports
        5. Save individual and aggregate scoring results

    Attributes:
        eval_task (Optional[EvalTask]): Inherited evaluation task configuration.
        data (Optional[EvalData]): Inherited evaluation dataset.
        paths (Paths): Inherited project directory structure.
        logger (logging.Logger): Inherited logger instance.

    Examples:
        >>> evaluator = MetaEvaluator()
        >>> evaluator.add_evaluation_task(task_schemas={"toxicity": ["toxic", "non_toxic"]})
        >>>
        >>> # Load results from evaluation runs
        >>> judge_results = evaluator.load_all_judge_results()
        >>> human_results = evaluator.load_all_human_results()
        >>>
        >>> # Configure and run scoring
        >>> evaluator.configure_scoring_metrics([
        ...     MetricConfig(metric_type="accuracy", task_names=["toxicity"])
        ... ])
        >>> evaluator.compute_scores(save_results=True)
    """

    # Type hints for attributes that will be provided by MetaEvaluator
    eval_task: Optional[EvalTask]
    data: Optional[EvalData]
    paths: "Paths"
    logger: logging.Logger

    def __init__(self, *args, **kwargs):
        """Initialize scoring mixin."""
        super().__init__(*args, **kwargs)

    # ===== RESULTS LOADING METHODS =====

    def load_all_judge_results(self) -> Dict[str, JudgeResults]:
        """Load all judge results from the project's results directory.

        Searches for all *_state.json files in the results directory and attempts
        to load them as judge results. Files that fail to load are skipped with
        a warning logged.

        Returns:
            Dict[str, JudgeResults]: Dictionary mapping run_ids to their loaded JudgeResults objects.
        """
        results = {}

        # Find all state files in results directory
        if not self.paths.results.exists():  # type: ignore
            self.logger.warning(
                f"Results directory does not exist: {self.paths.results}"
            )  # type: ignore
            return results

        state_files = list(self.paths.results.glob("*_state.json"))  # type: ignore

        for state_file in state_files:
            try:
                # Load directly using absolute path from glob
                judge_results = JudgeResults.load_state(str(state_file))
                # Use run_id as key
                key = judge_results.run_id
                results[key] = judge_results
                self.logger.info(f"Loaded judge results from {state_file.name}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to load judge results from {state_file.name}: {e}"
                )
                continue

        return results

    def load_all_human_results(self) -> Dict[str, HumanAnnotationResults]:
        """Load all human annotation results from the project's annotations directory.

        Searches for all *_metadata.json files in the annotations directory and attempts
        to load them as human annotation results. Files that fail to load are skipped
        with a warning logged.

        Returns:
            Dict[str, HumanAnnotationResults]: Dictionary mapping run_ids to their loaded HumanAnnotationResults objects.
        """
        results = {}

        # Find all metadata files in annotations directory
        if not self.paths.annotations.exists():  # type: ignore
            self.logger.warning(
                f"Annotations directory does not exist: {self.paths.annotations}"  # type: ignore
            )
            return results

        metadata_files = list(self.paths.annotations.glob("*_metadata.json"))  # type: ignore

        for metadata_file in metadata_files:
            try:
                human_results = HumanAnnotationResults.load_state(str(metadata_file))
                # Use run_id as key
                key = human_results.run_id
                results[key] = human_results
                self.logger.info(
                    f"Loaded human annotation results from {metadata_file.name}"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to load human annotation results from {metadata_file.name}: {e}"
                )
                continue

        return results

    # ===== DATA PREPROCESSING METHODS FOR SCORER =====

    def _extract_outcomes_for_task(
        self, results: JudgeResults | HumanAnnotationResults, task_name: str
    ) -> pl.DataFrame:
        """Extract successful outcomes for a specific task from results, filtering out null values.

        Returns:
            pl.DataFrame: A DataFrame containing the successful outcomes for the task.
        """
        successful_data = results.get_successful_results()

        if successful_data.is_empty():
            return pl.DataFrame({"original_id": [], task_name: []})

        # Select the columns and filter out null values for the task
        task_data = successful_data.select(["original_id", task_name])

        # Filter out rows where the task value is null
        task_data = task_data.filter(pl.col(task_name).is_not_null())

        return task_data

    def _collect_all_outcomes(
        self,
        results_dict: Mapping[str, Union[JudgeResults, HumanAnnotationResults]],
        task_names: List[str],
        id_column: str,  # "judge_id" or "annotator_id"
    ) -> List[pl.DataFrame]:
        """Collect outcomes from all results for all tasks.

        Returns:
            List[pl.DataFrame]: A list of DataFrames containing the outcomes for all tasks.
        """
        all_outcomes = []

        for result in results_dict.values():
            # Get the appropriate ID value based on result type
            id_value = (
                result.judge_id if id_column == "judge_id" else result.annotator_id  # type: ignore
            )

            # Extract outcomes for all tasks for this result
            for task_name in task_names:
                outcomes = self._extract_outcomes_for_task(result, task_name)
                if not outcomes.is_empty():
                    # Add metadata columns
                    outcomes_with_metadata = outcomes.with_columns(
                        [
                            pl.lit(id_value).alias(id_column),
                            pl.lit(task_name).alias("task_name"),
                        ]
                    ).rename({task_name: "task_value"})
                    all_outcomes.append(outcomes_with_metadata)

        return all_outcomes

    def _find_common_ids(
        self, consolidated_judge_df: pl.DataFrame, consolidated_human_df: pl.DataFrame
    ) -> dict[str, set]:
        """Find original_ids that exist in both judge and human results.

        Returns:
            set: A set of original_ids that exist in both judge and human results.
        """
        judge_ids = set(consolidated_judge_df["original_id"].unique())
        human_ids = set(consolidated_human_df["original_id"].unique())
        common_ids = judge_ids & human_ids  # Use set intersection operator
        judge_only_ids = judge_ids - human_ids
        human_only_ids = human_ids - judge_ids
        return {
            "common_ids": common_ids,
            "judge_only_ids": judge_only_ids,
            "human_only_ids": human_only_ids,
        }

    def _align_results_by_id(
        self,
        judge_results: Dict[str, JudgeResults],
        human_results: Dict[str, HumanAnnotationResults],
        task_names: List[str],
    ) -> tuple:
        """Align judge and human outcomes by original_id across all tasks.

        Args:
            judge_results: Dictionary mapping run_ids to judge evaluation results
            human_results: Dictionary mapping run_ids to human annotation results
            task_names: List of task names to align

        Returns:
            tuple: Tuple containing consolidated judge and human DataFrames

        Raises:
            InsufficientDataError: If no aligned judge-human data is found.
        """
        # Collect all outcomes from judge and human results
        judge_outcomes = self._collect_all_outcomes(
            judge_results, task_names, "judge_id"
        )
        human_outcomes = self._collect_all_outcomes(
            human_results, task_names, "annotator_id"
        )

        if not judge_outcomes:
            raise InsufficientDataError("No judge outcomes found")

        if not human_outcomes:
            raise InsufficientDataError("No human outcomes found")

        # Combine all outcomes into single DataFrames
        consolidated_judge_df = pl.concat(judge_outcomes)
        consolidated_human_df = pl.concat(human_outcomes)

        # Find common IDs and differences for sanity check
        id_sets = self._find_common_ids(consolidated_judge_df, consolidated_human_df)
        common_ids = id_sets["common_ids"]
        judge_only_ids = id_sets["judge_only_ids"]
        human_only_ids = id_sets["human_only_ids"]

        # Log sanity check information
        self.logger.info(f"Judge IDs not found in human: {len(judge_only_ids)}")
        self.logger.info(f"Human IDs not found in judge: {len(human_only_ids)}")
        self.logger.info(f"Common IDs for alignment: {len(common_ids)}")

        if not common_ids:
            raise InsufficientDataError("No aligned judge-human data found")

        consolidated_judge_df = consolidated_judge_df.filter(
            pl.col("original_id").is_in(list(common_ids))
        )
        consolidated_human_df = consolidated_human_df.filter(
            pl.col("original_id").is_in(list(common_ids))
        )

        return consolidated_judge_df, consolidated_human_df

    def _extract_task_schemas(
        self, judge_results: Dict[str, JudgeResults], task_names: List[str]
    ) -> dict:
        """Extract task schemas from judge results.

        Returns:
            dict: A dictionary mapping task names to their schemas.

        Raises:
            EvalTaskNotFoundError: If a task is not found in the judge results schemas.
        """
        first_judge_result = next(iter(judge_results.values()))
        task_schemas = {}
        for task_name in task_names:
            if task_name not in first_judge_result.task_schemas:
                raise EvalTaskNotFoundError(
                    f"Task '{task_name}' not found in judge results schemas"
                )
            task_schemas[task_name] = first_judge_result.task_schemas.get(task_name)
        return task_schemas

    def _validate_scorer_compatibility(self, scorer, task_schemas: dict) -> None:
        """Validate that a scorer can handle all tasks.

        Raises:
            IncompatibleTaskError: If a scorer cannot handle a task.
        """
        for task_name, task_schema in task_schemas.items():
            if not scorer.can_score_task(task_schema):
                raise IncompatibleTaskError(
                    f"Scorer '{scorer.scorer_name}' cannot score task '{task_name}'"
                )

    # ===== SCORING AND COMPARISON METHODS =====

    def _run_scorer_aggregations(
        self,
        results: Dict[str, List[BaseScoringResult]],
        comparison_config: MetricsConfig,
    ) -> None:
        """Run aggregation for scorers that support it.

        Args:
            results: Dictionary mapping scorer names to lists of results
            comparison_config: Configuration containing metrics and their scorers
        """
        scores_dir = str(self.paths.scores)

        # Check each scorer type for aggregation capability
        processed_scorers = set()
        for metric_config in comparison_config.metrics:
            scorer = metric_config.scorer
            scorer_name = scorer.scorer_name

            # Skip if we already processed this scorer type
            if scorer_name in processed_scorers:
                continue

            # Check if scorer has aggregation capability
            scorer_results = results.get(scorer_name, [])
            if scorer_results:
                self.logger.info(
                    f"Running aggregation for {scorer_name} scorer with {len(scorer_results)} results"
                )
                try:
                    scorer.aggregate_results(scorer_results, scores_dir)
                except Exception as e:
                    self.logger.warning(
                        f"Warning: Failed to run aggregation for {scorer_name}: {e}"
                    )
            else:
                self.logger.warning(
                    f"Scorer {scorer_name} does not support aggregation"
                )

            processed_scorers.add(scorer_name)

    def compare(
        self,
        comparison_config: MetricsConfig,
        judge_results: Optional[Dict[str, JudgeResults]] = None,
        human_results: Optional[Dict[str, HumanAnnotationResults]] = None,
    ) -> Dict[str, List[BaseScoringResult]]:
        """Compare judge and human results using configured metrics.

        Args:
            comparison_config: Configuration specifying which metrics to run and on which tasks
            judge_results: Dictionary mapping run_ids to judge evaluation results.
                If None, loads all judge results from the project's results directory.
            human_results: Dictionary mapping run_ids to human annotation results.
                If None, loads all human results from the project's annotations directory.

        Returns:
            Dict[str, List[BaseScoringResult]]: Dictionary mapping scorer names to lists of results

        Raises:
            ScoringConfigError: If no metrics configured, no results found, or scoring fails
            InsufficientDataError: If insufficient data is available for scoring
        """
        self.logger.info(
            f"Starting comparison with {len(comparison_config.metrics)} metrics"
        )

        # Validate comparison configuration
        if not comparison_config.metrics:
            raise ScoringConfigError("No metrics configured for comparison")

        for i, metric_config in enumerate(comparison_config.metrics):
            if not metric_config.task_names:
                raise ScoringConfigError(f"No task names specified for metric {i}")

        # Load results if not provided
        if judge_results is None:
            self.logger.info("Loading all judge results from results directory")
            judge_results = self.load_all_judge_results()
        if human_results is None:
            self.logger.info("Loading all human results from annotations directory")
            human_results = self.load_all_human_results()

        # Validate we have results
        if not judge_results:
            raise InsufficientDataError("No judge results provided or found")
        if not human_results:
            raise InsufficientDataError("No human results provided or found")

        self.logger.info(
            f"Comparing {len(judge_results)} judge result sets with {len(human_results)} human result sets"
        )

        results = {}  # Dict[str, List[BaseScoringResult]]

        # Process each metric configuration
        for i, metric_config in enumerate(comparison_config.metrics, 1):
            self.logger.info(
                f"Processing metric {i}/{len(comparison_config.metrics)}: {metric_config.scorer.scorer_name} on tasks {metric_config.task_names}"
            )
            # Extract task schemas for this metric
            task_schemas = self._extract_task_schemas(
                judge_results, metric_config.task_names
            )

            # Validate scorer compatibility
            self._validate_scorer_compatibility(metric_config.scorer, task_schemas)

            # Align data for this metric's tasks
            consolidated_judge_df, consolidated_human_df = self._align_results_by_id(
                judge_results, human_results, metric_config.task_names
            )

            # Compute score for each judge
            all_judges = consolidated_judge_df["judge_id"].unique().to_list()
            self.logger.info(
                f"Computing scores for {len(all_judges)} judges: {all_judges}"
            )

            for j, judge_id in enumerate(all_judges, 1):
                self.logger.info(
                    f"  Computing score for judge {j}/{len(all_judges)}: {judge_id}"
                )
                # Filter judge data for this specific judge
                consolidated_judge_subset = consolidated_judge_df.filter(
                    pl.col("judge_id") == judge_id
                )

                # Compute the score for this judge
                score_result = metric_config.scorer.compute_score(
                    judge_id=judge_id,
                    consolidated_judge_df=consolidated_judge_subset,
                    consolidated_human_df=consolidated_human_df,
                    task_names=metric_config.task_names,
                    task_schemas=task_schemas,
                )

                # Group results by scorer name
                scorer_name = metric_config.scorer.scorer_name
                if scorer_name not in results:
                    results[scorer_name] = []
                results[scorer_name].append(score_result)

        # Run scorer aggregations after all individual scoring is complete
        self.logger.info("Running scorer aggregations for comparison results")
        self._run_scorer_aggregations(results, comparison_config)

        total_results = sum(len(scorer_results) for scorer_results in results.values())
        self.logger.info(
            f"Comparison completed successfully. Generated {total_results} total scoring results across {len(results)} scorer types"
        )

        return results
