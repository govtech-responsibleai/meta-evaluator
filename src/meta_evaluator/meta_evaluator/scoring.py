"""Mixin for scoring and comparison functionality including results loading."""

import logging
from typing import Dict, List, Optional, Union, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Paths

import polars as pl

from ..eval_task import EvalTask
from ..data import EvalData
from ..results import JudgeResults, HumanAnnotationResults
from ..scores import MetricsConfig, BaseScoringResult


class ScoringMixin:
    """Mixin providing methods for loading results and performing scoring comparisons."""

    # Type hints for attributes that will be provided by MetaEvaluator
    eval_task: Optional[EvalTask]
    data: Optional[EvalData]
    paths: "Paths"

    # ===== RESULTS LOADING METHODS =====

    def load_all_judge_results(self) -> Dict[str, JudgeResults]:
        """Load all judge results from the project's results directory.

        Searches for all *_state.json files in the results directory and attempts
        to load them as judge results. Files that fail to load are skipped with
        a warning logged.

        Returns:
            Dict[str, JudgeResults]: Dictionary mapping run_ids to their loaded JudgeResults objects.
        """
        logger = logging.getLogger(__name__)
        results = {}

        # Find all state files in results directory
        if not self.paths.results.exists():  # type: ignore
            logger.warning(f"Results directory does not exist: {self.paths.results}")  # type: ignore
            return results

        state_files = list(self.paths.results.glob("*_state.json"))  # type: ignore

        for state_file in state_files:
            try:
                # Load directly using absolute path from glob
                judge_results = JudgeResults.load_state(str(state_file))
                # Use run_id as key
                key = judge_results.run_id
                results[key] = judge_results
                logger.info(f"Loaded judge results from {state_file.name}")
            except Exception as e:
                logger.warning(
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
        logger = logging.getLogger(__name__)
        results = {}

        # Find all metadata files in annotations directory
        if not self.paths.annotations.exists():  # type: ignore
            logger.warning(
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
                logger.info(
                    f"Loaded human annotation results from {metadata_file.name}"
                )
            except Exception as e:
                logger.warning(
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
    ) -> set:
        """Find original_ids that exist in both judge and human results.

        Returns:
            set: A set of original_ids that exist in both judge and human results.
        """
        judge_ids = set(consolidated_judge_df["original_id"].unique())
        human_ids = set(consolidated_human_df["original_id"].unique())
        return judge_ids & human_ids  # Use set intersection operator

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
            ValueError: If no aligned judge-human data is found.
        """
        # Collect all outcomes from judge and human results
        judge_outcomes = self._collect_all_outcomes(
            judge_results, task_names, "judge_id"
        )
        human_outcomes = self._collect_all_outcomes(
            human_results, task_names, "annotator_id"
        )

        if not judge_outcomes or not human_outcomes:
            raise ValueError("No aligned judge-human data found")

        # Combine all outcomes into single DataFrames
        consolidated_judge_df = pl.concat(judge_outcomes)
        consolidated_human_df = pl.concat(human_outcomes)

        # Find and filter to common IDs
        common_ids = self._find_common_ids(consolidated_judge_df, consolidated_human_df)
        if not common_ids:
            raise ValueError("No aligned judge-human data found")

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
            ValueError: If a task is not found in the judge results schemas.
        """
        first_judge_result = next(iter(judge_results.values()))
        task_schemas = {}
        for task_name in task_names:
            if task_name not in first_judge_result.task_schemas:
                raise ValueError(
                    f"Task '{task_name}' not found in judge results schemas"
                )
            task_schemas[task_name] = first_judge_result.task_schemas.get(task_name)
        return task_schemas

    def _validate_scorer_compatibility(self, scorer, task_schemas: dict) -> None:
        """Validate that a scorer can handle all tasks.

        Raises:
            ValueError: If a scorer cannot handle a task.
        """
        for task_name, task_schema in task_schemas.items():
            if not scorer.can_score_task(task_schema):
                raise ValueError(
                    f"Scorer {scorer.scorer_name} cannot handle task {task_name}"
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
                print(
                    f"Running aggregation for {scorer_name} scorer with {len(scorer_results)} results"
                )
                try:
                    scorer.__class__.aggregate_results(scorer_results, scores_dir)
                    processed_scorers.add(scorer_name)
                except Exception as e:
                    print(f"Warning: Failed to run aggregation for {scorer_name}: {e}")
            else:
                print(f"Scorer {scorer_name} does not support aggregation")

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
            ValueError: If no metrics configured, no results found, or scoring fails
        """
        # Validate comparison configuration
        if not comparison_config.metrics:
            raise ValueError("No metrics configured for comparison")

        for i, metric_config in enumerate(comparison_config.metrics):
            if not metric_config.task_names:
                raise ValueError(f"No task names specified for metric {i}")

        # Load results if not provided
        if judge_results is None:
            judge_results = self.load_all_judge_results()
        if human_results is None:
            human_results = self.load_all_human_results()

        # Validate we have results
        if not judge_results:
            raise ValueError("No judge results provided or found")
        if not human_results:
            raise ValueError("No human results provided or found")

        results = {}  # Dict[str, List[BaseScoringResult]]

        # Process each metric configuration
        for metric_config in comparison_config.metrics:
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
            for judge_id in all_judges:
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
        self._run_scorer_aggregations(results, comparison_config)

        return results
