"""Mixin for scoring and comparison functionality including results loading."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

if TYPE_CHECKING:
    from .base import Paths

import numpy as np
import polars as pl

from ..common.async_utils import sync_wrapper
from ..data import EvalData
from ..eval_task import EvalTask
from ..results import HumanAnnotationResults, JudgeResults
from ..results.enums import EvaluationStatusEnum, HumanAnnotationStatusEnum
from ..results.serialization import (
    HumanAnnotationResultsSerializedState,
    JudgeResultsSerializedState,
)
from ..scores import BaseScoringResult, MetricConfig, MetricsConfig
from ..scores.enums import TaskAggregationMode
from .exceptions import (
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
        >>> alt_test_scorer = AltTestScorer(multiplicative_epsilon=True)
        >>> cohens_kappa_scorer = CohensKappaScorer()
        >>> config = MetricsConfig(
        >>>    metrics=[
        >>>        MetricConfig(scorer=alt_test_scorer,task_names=["task_1", "task_2"], task_strategy="multilabel"),
        >>>        MetricConfig(scorer=cohens_kappa_scorer, task_names=["task_1"], task_strategy="single"),
        >>>    ]
        >>> )
        >>> await evaluator.compare_async(config, judge_results=judge_results, human_results=human_results)
    """

    # Type hints for attributes that will be provided by MetaEvaluator
    eval_task: Optional[EvalTask]
    data: Optional[EvalData]
    paths: "Paths"
    logger: logging.Logger

    def __init__(self, *args, **kwargs):
        """Initialize scoring mixin."""
        super().__init__(*args, **kwargs)

    ##########################################################
    # RESULTS LOADING METHODS
    ##########################################################

    def load_all_judge_results(self) -> Dict[str, JudgeResults]:
        """Load all judge results from the project's results directory.

        Searches for all *_state.json files in the results directory and attempts
        to load them as judge results. Files that fail to load are skipped with
        a warning logged.

        When duplicate judge_ids are found, keeps the most recent run (highest run_id)
        and logs a warning about the duplicates.

        Returns:
            Dict[str, JudgeResults]: Dictionary mapping run_ids to their loaded JudgeResults objects.
        """
        results = {}
        judge_id_to_results = {}  # Track judge_id -> list of (run_id, JudgeResults) for duplicate detection

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
                run_id = judge_results.run_id
                judge_id = judge_results.judge_id

                # Track for duplicate detection
                if judge_id not in judge_id_to_results:
                    judge_id_to_results[judge_id] = []
                judge_id_to_results[judge_id].append((run_id, judge_results))

                self.logger.info(f"Loaded judge results from {state_file.name}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to load judge results from {state_file.name}: {e}"
                )
                continue

        # TODO: Handle duplicates according to requirements (e.g. take both, drop both, take earliest run, etc.)
        # Current default: keep most recent run_id for each judge_id
        # Without this, the last loaded judge results will overwrite the previous ones
        for judge_id, judge_results_list in judge_id_to_results.items():
            if len(judge_results_list) > 1:
                # Sort by run_id (assuming higher run_id means more recent)
                judge_results_list.sort(key=lambda x: x[0], reverse=True)
                most_recent_run_id, most_recent_result = judge_results_list[0]

                # Log warning about duplicates
                duplicate_run_ids = [run_id for run_id, _ in judge_results_list[1:]]
                self.logger.warning(
                    f"Found duplicate results for judge_id '{judge_id}'. "
                    f"Keeping most recent run_id '{most_recent_run_id}', "
                    f"skipping: {duplicate_run_ids}"
                )

                results[most_recent_run_id] = most_recent_result
            else:
                # No duplicates, add the single result
                run_id, judge_result = judge_results_list[0]
                results[run_id] = judge_result

        return results

    def load_all_human_results(self) -> Dict[str, HumanAnnotationResults]:
        """Load all human annotation results from the project's annotations directory.

        Searches for all *_metadata.json files in the annotations directory and attempts
        to load them as human annotation results. Files that fail to load are skipped
        with a warning logged.

        When duplicate annotator_ids are found, keeps the most recent run (highest run_id)
        and logs a warning about the duplicates.

        Returns:
            Dict[str, HumanAnnotationResults]: Dictionary mapping run_ids to their loaded HumanAnnotationResults objects.
        """
        results = {}
        annotator_id_to_results = {}  # Track annotator_id -> list of (run_id, HumanAnnotationResults) for duplicate detection

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
                run_id = human_results.run_id
                annotator_id = human_results.annotator_id

                # Track for duplicate detection
                if annotator_id not in annotator_id_to_results:
                    annotator_id_to_results[annotator_id] = []
                annotator_id_to_results[annotator_id].append((run_id, human_results))

                self.logger.info(
                    f"Loaded human annotation results from {metadata_file.name}"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to load human annotation results from {metadata_file.name}: {e}"
                )
                continue

        # TODO: Handle duplicates according to requirements (e.g. take both, drop both, take earliest run, etc.)
        # Current default: keep most recent run_id for each annotator_id
        # Without this, the last loaded human results will overwrite the previous ones
        for annotator_id, human_results_list in annotator_id_to_results.items():
            if len(human_results_list) > 1:
                # Sort by run_id (assuming higher run_id means more recent)
                human_results_list.sort(key=lambda x: x[0], reverse=True)
                most_recent_run_id, most_recent_result = human_results_list[0]

                # Log warning about duplicates
                duplicate_run_ids = [run_id for run_id, _ in human_results_list[1:]]
                self.logger.warning(
                    f"Found duplicate results for annotator_id '{annotator_id}'. "
                    f"Keeping most recent run_id '{most_recent_run_id}', "
                    f"skipping: {duplicate_run_ids}"
                )

                results[most_recent_run_id] = most_recent_result
            else:
                # No duplicates, add the single result
                run_id, human_result = human_results_list[0]
                results[run_id] = human_result

        return results

    ##############################
    # DATA PREPROCESSING METHODS #
    ##############################

    def _filter_and_align_data(
        self,
        judge_result: JudgeResults,
        human_results: Dict[str, HumanAnnotationResults],
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Filter judge and human results for successful results and common IDs.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: (judge_df, human_df) with success results and common IDs
        """
        # Get successful results only
        judge_success_df = judge_result.get_successful_results()

        # Combine all human successful results
        human_success_dfs = []
        for human_result in human_results.values():
            human_df = human_result.get_successful_results()
            if not human_df.is_empty():
                human_df = human_df.with_columns(
                    pl.lit(human_result.annotator_id).alias("human_id")
                )
                human_success_dfs.append(human_df)

        if not human_success_dfs:
            return pl.DataFrame(), pl.DataFrame()

        combined_human_df = pl.concat(human_success_dfs)

        # Find common IDs
        judge_ids = set(judge_success_df["original_id"].unique())
        human_ids = set(combined_human_df["original_id"].unique())
        common_ids = judge_ids & human_ids

        if not common_ids:
            return pl.DataFrame(), pl.DataFrame()

        # Filter to common IDs
        filtered_judge_df = judge_success_df.filter(
            pl.col("original_id").is_in(list(common_ids))
        )
        filtered_human_df = combined_human_df.filter(
            pl.col("original_id").is_in(list(common_ids))
        )

        return filtered_judge_df, filtered_human_df

    def _preprocess_task_data(
        self,
        judge_df: pl.DataFrame,
        human_df: pl.DataFrame,
        task_names: List[str],
        aggregation_mode: TaskAggregationMode,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Preprocess data based on aggregation mode.

        This is used to determine the format of the task labels for the scorer.
        There are 3 possible formats:
        - TaskAggregationMode.SINGLE: Single task (Evaluate a single column only. e.g. "toxicity")
        - TaskAggregationMode.MULTILABEL: Multilabel column (Evaluate multiple columns, but treat as a single multilabel. e.g. ["hateful", "insults", "self_harm])
        - TaskAggregationMode.MULTITASK: Multiple task columns (Evaluate multiple columns, and calculate scores for each column, then take the average. e.g. ["toxicity_1", "toxicity_2"])

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: (judge_data, human_data) with columns (original_id, label) and (original_id, human_id, label)
        """
        if aggregation_mode == TaskAggregationMode.SINGLE:
            # Select single task column
            task_name = task_names[0]
            judge_data = judge_df.select(["original_id", task_name]).rename(
                {task_name: "label"}
            )
            human_data = human_df.select(["original_id", "human_id", task_name]).rename(
                {task_name: "label"}
            )

        elif aggregation_mode == TaskAggregationMode.MULTILABEL:
            # Create multilabel column
            judge_data = self._create_multilabel_column(
                judge_df, task_names, include_human_id=False
            )
            human_data = self._create_multilabel_column(
                human_df, task_names, include_human_id=True
            )

        else:  # MULTITASK - keep task columns as-is, will be processed separately
            judge_data = judge_df
            human_data = human_df

        return judge_data, human_data

    def _create_multilabel_column(
        self, df: pl.DataFrame, task_names: List[str], include_human_id: bool = False
    ) -> pl.DataFrame:
        """Combine task columns into a single multilabel colum for MULTILABEL aggregation mode.

        Returns:
            pl.DataFrame: DataFrame with aggregated label column.
        """
        group_cols = ["original_id"]
        if include_human_id:
            group_cols.append("human_id")

        # Select task columns and melt to long format
        task_df = df.select(
            ["original_id"] + (["human_id"] if include_human_id else []) + task_names
        )

        # Melt task columns into rows
        melted = task_df.melt(
            id_vars=group_cols,
            value_vars=task_names,
            variable_name="task_name",
            value_name="task_value",
        )

        # Group and combine task values into list
        def convert_values_to_strings(values_list):
            converted = []
            for value in values_list:
                if value is None:
                    converted.append(None)
                elif isinstance(value, list):
                    converted.append(str(value))
                else:
                    converted.append(str(value))
            return converted

        multilabel_df = (
            melted.group_by(group_cols)
            .agg([pl.col("task_value").alias("label_list")])
            .with_columns(
                [
                    pl.col("label_list")
                    .map_elements(
                        convert_values_to_strings, return_dtype=pl.List(pl.Utf8)
                    )
                    .alias("label")
                ]
            )
            .drop("label_list")
        )

        return multilabel_df

    def _average_multitask_results(
        self, task_scoring_results: List[BaseScoringResult]
    ) -> BaseScoringResult:
        """Average multiple task results for MULTITASK aggregation mode.

        Returns:
            BaseScoringResult: Averaged result across all tasks.

        Raises:
            ValueError: If the results list is empty.
        """
        if not task_scoring_results:
            raise ValueError("Cannot average empty results list")

        first_result = task_scoring_results[0]

        # Average scores across tasks
        averaged_scores = {}
        for score_key in first_result.scores.keys():
            if score_key == "winning_rate" and isinstance(
                first_result.scores[score_key], dict
            ):
                # Special handling for alt-test WR dict
                wr_values = {}
                for eps in first_result.scores[score_key].keys():
                    values = [
                        result.scores[score_key][eps]
                        for result in task_scoring_results
                        if score_key in result.scores
                    ]
                    wr_values[eps] = float(np.nanmean(values))
                averaged_scores[score_key] = wr_values
            else:
                # Simple averaging for other metrics
                values = [
                    result.scores[score_key]
                    for result in task_scoring_results
                    if score_key in result.scores
                ]
                averaged_scores[score_key] = float(np.nanmean(values))

        # Sum up comparisons
        total_comparisons = sum(
            result.num_comparisons for result in task_scoring_results
        )
        total_failed = sum(result.failed_comparisons for result in task_scoring_results)

        return BaseScoringResult(
            scorer_name=first_result.scorer_name,
            task_name=f"{len(task_scoring_results)}_tasks_avg",
            judge_id=first_result.judge_id,
            scores=averaged_scores,
            metadata={
                "task_names": [result.task_name for result in task_scoring_results],
                "scoring_method": "average_across_tasks",
                "ground_truth_method": first_result.metadata.get(
                    "ground_truth_method", ""
                ),
            },
            aggregation_mode=TaskAggregationMode.MULTITASK,
            num_comparisons=total_comparisons,
            failed_comparisons=total_failed,
        )

    ##################################
    # SCORING AND COMPARISON METHODS #
    ##################################

    def _get_first_non_null_value(self, label_column):
        """Get first non-null sample value from label column for scorer compatibility validation.

        Args:
            label_column: Polars Series containing label values

        Returns:
            First non-null value from label column, or None if no non-null values
        """
        if label_column is None or len(label_column) == 0:
            return None

        sample_values = label_column.drop_nulls()
        if len(sample_values) > 0:
            # Get the first element and extract the actual Python value
            first_value = sample_values.item(0)
            self.logger.info(f"SAMPLE: {first_value} {type(first_value)}")

            # For multilabel cases, first_value might be a list/series of labels
            # In that case, we need to return the structure as-is for compatibility checking
            if isinstance(first_value, (list, tuple)):
                return first_value
            elif hasattr(first_value, "to_list"):  # Polars Series
                return first_value.to_list()
            else:
                return first_value
        else:
            return None

    # Parallelize different unique scorer configurations
    async def _run_scoring_async(
        self,
        metric_config: MetricConfig,
        judge_results: Dict[str, JudgeResults],
        human_results: Dict[str, HumanAnnotationResults],
    ) -> Tuple[MetricConfig, List[BaseScoringResult]]:
        """Run a single metric configuration with different judges asynchronously.

        Args:
            metric_config: Metric configuration to run
            judge_results: Dictionary mapping judge IDs to JudgeResults objects
            human_results: Dictionary mapping human IDs to HumanAnnotationResults objects

        Returns:
            Tuple[MetricConfig, List[BaseScoringResult]]: (metric_config, list of BaseScoringResult objects)
        """
        self.logger.info(
            f"Processing metric: {metric_config.scorer.scorer_name} on tasks {metric_config.task_names}"
        )

        # Parallelize different judges within this scorer
        judge_tasks = []
        for judge_result in judge_results.values():
            task = self._process_single_judge_async(
                metric_config, judge_result, human_results
            )
            judge_tasks.append(task)

        judge_scoring_results = await asyncio.gather(
            *judge_tasks
        )  # No return_exceptions=True

        # All results are BaseScoringResult objects (some might be NaN)
        return metric_config, judge_scoring_results

    # Parallelize different judges within a scorer
    async def _process_single_judge_async(
        self,
        metric_config: MetricConfig,
        judge_result: JudgeResults,
        human_results: Dict[str, HumanAnnotationResults],
    ) -> BaseScoringResult:
        """Process a single judge async with a single scorer configuration. Always returns BaseScoringResult.

        Args:
            metric_config: Metric configuration to run
            judge_result: JudgeResults object
            human_results: Dictionary mapping human IDs to HumanAnnotationResults objects

        Returns:
            BaseScoringResult: Scoring result for the judge

        Raises:
            IncompatibleTaskError: If the scorer cannot handle the task type.
            InsufficientDataError: If there is insufficient data for scoring.
        """
        judge_id = judge_result.judge_id

        try:
            # Step 1: Filter and align data
            judge_df, human_df = self._filter_and_align_data(
                judge_result, human_results
            )

            if judge_df.is_empty() or human_df.is_empty():
                raise InsufficientDataError(f"No aligned data for judge {judge_id}")

            # Step 2: Validate minimum human annotators
            num_humans = len(human_df["human_id"].unique())
            min_required = metric_config.scorer.min_human_annotators

            if num_humans < min_required:
                raise InsufficientDataError(
                    f"Scorer '{metric_config.scorer.scorer_name}' requires at least {min_required} "
                    f"human annotators, but only {num_humans} found for judge {judge_id}"
                )

            # Step 3: Process based on aggregation mode
            if metric_config.aggregation_mode == TaskAggregationMode.MULTITASK:
                # MULTITASK - process each task separately, then average.
                task_scoring_results = []
                for task_name in metric_config.task_names:
                    task_judge_data, task_human_data = self._preprocess_task_data(
                        judge_df, human_df, [task_name], TaskAggregationMode.SINGLE
                    )

                    if task_judge_data.is_empty():
                        self.logger.warning(f"No data for task {task_name}, skipping")
                        continue

                    # Validate scorer compatibility
                    sample_label = self._get_first_non_null_value(
                        task_judge_data["label"]
                    )
                    if not metric_config.scorer.can_score_task(sample_label):
                        raise IncompatibleTaskError(
                            f"Scorer '{metric_config.scorer.scorer_name}' cannot score task '{task_name}' data format"
                        )

                    task_scoring_result = (
                        await metric_config.scorer.compute_score_async(
                            task_judge_data,
                            task_human_data,
                            task_name,
                            judge_id,
                            TaskAggregationMode.SINGLE,
                            metric_config.annotator_aggregation,
                        )
                    )
                    task_scoring_results.append(task_scoring_result)

                if not task_scoring_results:
                    raise InsufficientDataError(
                        f"No valid task results for judge {judge_id}"
                    )

                # Average the task results
                return self._average_multitask_results(task_scoring_results)

            else:
                # SINGLE or MULTILABEL - process once
                processed_judge_data, processed_human_data = self._preprocess_task_data(
                    judge_df,
                    human_df,
                    metric_config.task_names,
                    metric_config.aggregation_mode,
                )

                # Validate scorer compatibility
                sample_label = self._get_first_non_null_value(
                    processed_judge_data["label"]
                )
                if not metric_config.scorer.can_score_task(sample_label):
                    raise IncompatibleTaskError(
                        f"Scorer '{metric_config.scorer.scorer_name}' cannot score the processed data format"
                    )

                # Set task name based on aggregation mode
                if metric_config.aggregation_mode == TaskAggregationMode.SINGLE:
                    display_task_name = metric_config.task_names[0]
                else:  # MULTILABEL
                    display_task_name = (
                        f"multilabel_{len(metric_config.task_names)}_tasks"
                    )

                result = await metric_config.scorer.compute_score_async(
                    processed_judge_data,
                    processed_human_data,
                    display_task_name,
                    judge_id,
                    metric_config.aggregation_mode,
                    metric_config.annotator_aggregation,
                )
                return result

        except Exception as e:
            self.logger.error(f"Failed to process judge {judge_id}: {e}")
            # Return NaN result instead of failing
            return BaseScoringResult(
                scorer_name=metric_config.scorer.scorer_name,
                task_name="error",
                judge_id=judge_id,
                scores={},
                metadata={"error": str(e)},
                aggregation_mode=metric_config.aggregation_mode,
                num_comparisons=0,
                failed_comparisons=1,
            )

    def _save_all_scorer_results(
        self,
        results: Dict[str, Tuple[MetricConfig, List[BaseScoringResult]]],
    ) -> None:
        """Save individual results for each metric configuration.

        Args:
            results: Dictionary mapping unique metric names to (metric_config, results) tuples
        """
        scores_dir = str(self.paths.scores)

        # Save results for each unique metric configuration
        for unique_name, (metric_config, scorer_results) in results.items():
            scorer = metric_config.scorer

            if scorer_results:
                self.logger.info(
                    f"Saving {len(scorer_results)} results for {unique_name}"
                )
                try:
                    scorer.save_results(scorer_results, scores_dir, unique_name)
                except Exception as e:
                    self.logger.error(f"Failed to save results for {unique_name}: {e}")
            else:
                self.logger.warning(f"No results to save for {unique_name}")

    def _aggregate_all_scorer_results(
        self,
        results: Dict[str, Tuple[MetricConfig, List[BaseScoringResult]]],
    ) -> None:
        """Run aggregation for each metric configuration.

        Args:
            results: Dictionary mapping unique metric names to (metric_config, results) tuples
        """
        scores_dir = str(self.paths.scores)

        # Run aggregation for each metric configuration
        for unique_name, (metric_config, scorer_results) in results.items():
            scorer = metric_config.scorer

            if scorer_results:
                self.logger.info(
                    f"Running aggregation for {unique_name} with {len(scorer_results)} results"
                )
                try:
                    scorer.aggregate_results(scorer_results, scores_dir, unique_name)
                except Exception as e:
                    self.logger.warning(
                        f"Warning: Failed to run aggregation for {unique_name}: {e}"
                    )

    @sync_wrapper
    def compare_async(self, *args, **kwargs):
        """Synchronous wrapper for compare_async that handles asyncio internally.

        Returns:
            Dict[str, Tuple[MetricConfig, List[BaseScoringResult]]]: Dictionary mapping unique names to (metric_config, results) tuples.
        """
        return self._compare_async(*args, **kwargs)

    async def _compare_async(
        self,
        comparison_config: MetricsConfig,
        judge_results: Optional[Dict[str, JudgeResults]] = None,
        human_results: Optional[Dict[str, HumanAnnotationResults]] = None,
    ) -> Dict[str, Tuple[MetricConfig, List[BaseScoringResult]]]:
        """Main method to compare judge and human results using configured metrics.

        Handles:
        - Validating comparison configuration
        - Loading judge and human results
        - Running scoring for each metric configuration
        - Saving individual judge results
        - Running aggregations for each metric configuration for all judges

        Args:
            comparison_config: Configuration specifying which metrics to run and on which tasks
            judge_results: Dictionary mapping run_ids to judge evaluation results.
                If None, loads all judge results from the project's results directory.
            human_results: Dictionary mapping run_ids to human annotation results.
                If None, loads all human results from the project's annotations directory.

        Returns:
            Dict[str, Tuple[MetricConfig, List[BaseScoringResult]]]: Dictionary mapping unique names to (metric_config, results) tuples

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

        # Parallelize different scorers and judges
        scorer_tasks = []
        for metric_config in comparison_config.metrics:
            task = self._run_scoring_async(metric_config, judge_results, human_results)
            scorer_tasks.append(task)

        scorer_results = await asyncio.gather(*scorer_tasks, return_exceptions=True)

        # Combine results from all scorers with unique names
        results = {}  # Dict[str, Tuple[MetricConfig, List[BaseScoringResult]]]
        for scorer_result in scorer_results:
            if isinstance(scorer_result, Exception):
                self.logger.error(f"Scorer failed: {scorer_result}")
                continue

            metric_config, scorer_results_list = scorer_result  # type: ignore

            # Create unique name to differentiate between scorer + task_names + aggregation_mode
            unique_name = metric_config.get_unique_name()

            results[unique_name] = (metric_config, scorer_results_list)

        # Save individual results for each scorer
        self.logger.info("Saving individual scoring results")
        self._save_all_scorer_results(results)

        # Run scorer aggregations after all individual scoring is complete
        self.logger.info("Running scorer aggregations for comparison results")
        self._aggregate_all_scorer_results(results)

        # Log total results and metric configurations
        total_results = sum(
            len(scorer_results_list) for _, scorer_results_list in results.values()
        )
        total_metric_configs = len(comparison_config.metrics)
        self.logger.info(
            f"Async comparison completed successfully. Generated {total_results} total scoring results from {total_metric_configs} metric configurations"
        )

        return results

    ##########################################################
    # EXTERNAL RESULTS LOADING METHODS
    ##########################################################

    def add_external_judge_results(
        self,
        file_path: str,
        judge_id: str,
        llm_client: str = "external",
        model_used: str = "unknown",
        run_id: Optional[str] = None,
        data_format: Optional[Literal["json", "csv", "parquet"]] = None,
        **metadata_kwargs,
    ) -> None:
        """Add judge results from external data source to the project.

        By default assumes all results are successful. Provide specific counts
        via metadata_kwargs if your data has failures (e.g. succeeded_count=80,
        llm_error_count=5, etc.).

        Args:
            file_path: Path to data file containing results
            judge_id: Unique identifier for the judge
            llm_client: LLM provider used (defaults to "external")
            model_used: Model name used (defaults to "unknown")
            run_id: Unique run identifier (auto-generated if None)
            data_format: Format of file (auto-detected from extension if None)
            **metadata_kwargs: Additional metadata including status counts

        Raises:
            ValueError: If eval_task is not set or if file format cannot be detected/validated
        """
        # 1. Validate prerequisites
        if self.eval_task is None:
            raise ValueError("eval_task must be set before importing external results")

        # 2. Load DataFrame from file
        detected_format = self._extract_data_format(file_path, data_format)
        df = EvalData.load_data(file_path, detected_format)

        # 3. Validate and prepare results data (for JudgeResultRow schema)
        actual_run_id = run_id or self._generate_run_id()
        df = self._validate_judge_results_data(
            df, judge_id, actual_run_id, **metadata_kwargs
        )

        # 4. Extract status counts from metadata_kwargs (default: all imported)
        total_count = len(df)
        status_counts = {
            "succeeded_count": metadata_kwargs.get("succeeded_count", total_count),
            "skipped_count": metadata_kwargs.get("skipped_count", 0),
            "partial_count": metadata_kwargs.get("partial_count", 0),
            "llm_error_count": metadata_kwargs.get("llm_error_count", 0),
            "parsing_error_count": metadata_kwargs.get("parsing_error_count", 0),
            "other_error_count": metadata_kwargs.get("other_error_count", 0),
        }

        # 5. Create serialized state
        state = JudgeResultsSerializedState(
            run_id=actual_run_id,
            judge_id=judge_id,
            task_schemas=self.eval_task.task_schemas,
            llm_client=llm_client,
            model_used=model_used,
            timestamp_local=datetime.now(),
            total_count=total_count,
            **status_counts,
            is_sampled_run=False,
            data_file="",  # Not applicable for external
            data_format=detected_format,
            **{k: v for k, v in metadata_kwargs.items() if k not in status_counts},
        )

        # 6. Use existing deserialize method
        judge_results = JudgeResults.deserialize(df, state)

        # 7. Always save to project
        save_path = (
            self.paths.results
            / f"{judge_results.run_id}_{judge_results.judge_id}_external_state.json"
        )
        self.paths.results.mkdir(exist_ok=True)
        judge_results.save_state(str(save_path))

        self.logger.info(f"Added external judge results for '{judge_id}' to project")

    def add_external_annotation_results(
        self,
        file_path: str,
        annotator_id: str,
        run_id: Optional[str] = None,
        data_format: Optional[Literal["json", "csv", "parquet"]] = None,
        **metadata_kwargs,
    ) -> None:
        """Add human annotation results from external data source to the project.

        By default assumes all results are successful. Provide specific counts
        via metadata_kwargs if your data has failures (e.g. succeeded_count=80,
        error_count=5).

        Args:
            file_path: Path to data file containing results
            annotator_id: Unique identifier for the annotator
            run_id: Unique run identifier (auto-generated if None)
            data_format: Format of file (auto-detected from extension if None)
            **metadata_kwargs: Additional metadata including status counts

        Raises:
            ValueError: If eval_task is not set or if file format cannot be detected/validated
        """
        # 1. Validate prerequisites
        if self.eval_task is None:
            raise ValueError("eval_task must be set before importing external results")

        # 2. Load DataFrame from file
        detected_format = self._extract_data_format(file_path, data_format)
        df = EvalData.load_data(file_path, detected_format)

        # 3. Validate and prepare results data (for HumanAnnotationResultRow schema)
        actual_run_id = run_id or self._generate_run_id()
        df = self._validate_annotation_results_data(
            df, annotator_id, actual_run_id, **metadata_kwargs
        )

        # 4. Extract status counts from metadata_kwargs (default: all success)
        total_count = len(df)
        status_counts = {
            "succeeded_count": metadata_kwargs.get("succeeded_count", total_count),
            "error_count": metadata_kwargs.get("error_count", 0),
        }

        # 5. Create serialized state
        state = HumanAnnotationResultsSerializedState(
            run_id=actual_run_id,
            annotator_id=annotator_id,
            task_schemas=self.eval_task.task_schemas,
            timestamp_local=datetime.now(),
            total_count=total_count,
            is_sampled_run=False,
            data_file="",  # Not applicable for external
            data_format=detected_format,
            **status_counts,
            **{k: v for k, v in metadata_kwargs.items() if k not in status_counts},
        )

        # 6. Use existing deserialize method
        human_results = HumanAnnotationResults.deserialize(df, state)

        # 7. Always save to project
        save_path = (
            self.paths.annotations
            / f"{human_results.run_id}_{human_results.annotator_id}_external_metadata.json"
        )
        self.paths.annotations.mkdir(exist_ok=True)
        human_results.save_state(str(save_path))

        self.logger.info(
            f"Added external human results for '{annotator_id}' to project"
        )

    # ===== HELPER METHODS FOR EXTERNAL RESULTS =====

    def _extract_data_format(
        self, file_path: str, data_format: Optional[str]
    ) -> Literal["json", "csv", "parquet"]:
        """Extract or validate data format for external files.

        Args:
            file_path: Path to the external file
            data_format: Optional data format override

        Returns:
            The validated data format (json, csv, or parquet)

        Raises:
            ValueError: If format cannot be auto-detected or is unsupported
        """
        if data_format is None:
            # Auto-detect from file extension
            if file_path.endswith(".csv"):
                return "csv"
            elif file_path.endswith(".json"):
                return "json"
            elif file_path.endswith(".parquet"):
                return "parquet"
            else:
                raise ValueError(
                    f"Cannot auto-detect format for {file_path}. Specify data_format."
                )
        else:
            # Validate provided format
            if data_format not in ["json", "csv", "parquet"]:
                raise ValueError(f"Unsupported format: {data_format}")
            return data_format  # type: ignore

    def _validate_judge_results_data(
        self, df: pl.DataFrame, judge_id: str, run_id: str, **metadata_kwargs
    ) -> pl.DataFrame:
        """Validate and prepare results data for JudgeResultRow schema.

        Args:
            df: DataFrame to validate and prepare
            judge_id: Judge identifier
            run_id: Run identifier
            **metadata_kwargs: Additional metadata

        Returns:
            pl.DataFrame: DataFrame with all required JudgeResultRow columns

        Raises:
            ValueError: If required columns are missing
        """
        # For external data import, users only need to provide original_id and task columns
        # We auto-generate the system-required columns from function parameters
        assert self.eval_task is not None  # Already validated in calling method
        required_cols = [
            "original_id",  # Only require original_id from user
        ] + list(self.eval_task.task_schemas.keys())

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Auto-generate system columns from function parameters
        # sample_example_id: Originally was a per-run unique identifier (e.g. "sample_1", "sample_2")
        # For imported external data, we simplify by setting it equal to original_id
        if "sample_example_id" not in df.columns:
            df = df.with_columns(pl.col("original_id").alias("sample_example_id"))

        # run_id: Use the run_id parameter passed to add_external_judge_results()
        if "run_id" not in df.columns:
            df = df.with_columns(pl.lit(run_id).alias("run_id"))

        # judge_id: Use the judge_id parameter passed to add_external_judge_results()
        if "judge_id" not in df.columns:
            df = df.with_columns(pl.lit(judge_id).alias("judge_id"))

        # Add missing fields (status/error + LLM diagnostic fields) - all None for external data
        # Set status to SUCCESS for all imported external data
        if "status" not in df.columns:
            df = df.with_columns(
                pl.lit(EvaluationStatusEnum.SUCCESS.value).alias("status")
            )

        missing_fields = [
            ("error_message", pl.String),
            ("error_details_json", pl.String),
            ("llm_raw_response_content", pl.String),
            ("llm_prompt_tokens", pl.Int64),
            ("llm_completion_tokens", pl.Int64),
            ("llm_total_tokens", pl.Int64),
            ("llm_call_duration_seconds", pl.Float64),
        ]

        for field_name, field_type in missing_fields:
            if field_name not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=field_type).alias(field_name))

        return df

    def _validate_annotation_results_data(
        self, df: pl.DataFrame, annotator_id: str, run_id: str, **metadata_kwargs
    ) -> pl.DataFrame:
        """Validate and prepare results data for HumanAnnotationResultRow schema.

        Args:
            df: DataFrame to validate and prepare
            annotator_id: Annotator identifier
            run_id: Run identifier
            **metadata_kwargs: Additional metadata

        Returns:
            pl.DataFrame: DataFrame with all required HumanAnnotationResultRow columns

        Raises:
            ValueError: If required columns are missing
        """
        # For external data import, users only need to provide original_id and task columns
        # We auto-generate the system-required columns from function parameters
        assert self.eval_task is not None  # Already validated in calling method
        required_cols = [
            "original_id",  # Only require original_id from user
        ] + list(self.eval_task.task_schemas.keys())

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Auto-generate system columns from function parameters
        # sample_example_id: Originally was a per-run unique identifier (e.g. "sample_1", "sample_2")
        # For imported external data, we simplify by setting it equal to original_id
        if "sample_example_id" not in df.columns:
            df = df.with_columns(pl.col("original_id").alias("sample_example_id"))

        # run_id: Use the run_id parameter passed to add_external_annotation_results()
        if "run_id" not in df.columns:
            df = df.with_columns(pl.lit(run_id).alias("run_id"))

        # annotator_id: Use the annotator_id parameter passed to add_external_annotation_results()
        if "annotator_id" not in df.columns:
            df = df.with_columns(pl.lit(annotator_id).alias("annotator_id"))

        # Add missing fields (status/error + annotation diagnostic) - all None for external data
        # Set status to SUCCESS for all imported external data
        if "status" not in df.columns:
            df = df.with_columns(
                pl.lit(HumanAnnotationStatusEnum.SUCCESS.value).alias("status")
            )

        missing_fields = [
            ("error_message", pl.String),
            ("error_details_json", pl.String),
            ("annotation_timestamp", pl.Datetime),
        ]

        for field_name, field_type in missing_fields:
            if field_name not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=field_type).alias(field_name))

        return df

    def _generate_run_id(self) -> str:
        """Generate unique run ID for external results.

        Returns:
            A unique run ID string with timestamp and UUID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"external_{timestamp}_{short_uuid}"
