"""Tests for ScoringMixin data processing functionality."""

import os
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest

from meta_evaluator.llm_client.enums import LLMClientEnum
from meta_evaluator.meta_evaluator import MetaEvaluator
from meta_evaluator.meta_evaluator.exceptions import (
    EvalTaskNotFoundError,
    IncompatibleTaskError,
    InsufficientDataError,
    ScoringConfigError,
)
from meta_evaluator.results import (
    HumanAnnotationResults,
    HumanAnnotationResultsBuilder,
    JudgeResults,
    JudgeResultsBuilder,
)
from meta_evaluator.scores import (
    AccuracyScorer,
    MetricConfig,
    MetricsConfig,
)
from meta_evaluator.scores.metrics.agreement.alt_test import AltTestScorer

# Note: All fixtures are now available from conftest.py


class TestDataProcessing:
    """Test data processing methods in ScoringMixin."""

    def test_extract_outcomes_for_task(self, mock_evaluator, judge_results_1):
        """Test extracting outcomes for a specific task."""
        result = mock_evaluator._extract_outcomes_for_task(judge_results_1, "task1")

        expected = pl.DataFrame(
            {"original_id": ["1", "2", "3", "4"], "task1": ["A", "B", "A", "C"]}
        )

        assert result.equals(expected)

    def test_extract_outcomes_for_task_empty(self, mock_evaluator):
        """Test extracting outcomes when no successful results."""
        empty_results = Mock(spec=JudgeResults)
        empty_results.get_successful_results.return_value = pl.DataFrame()

        result = mock_evaluator._extract_outcomes_for_task(empty_results, "task1")

        expected = pl.DataFrame({"original_id": [], "task1": []})
        assert result.equals(expected)

    def test_collect_all_outcomes_judge(self, mock_evaluator, judge_results_dict):
        """Test collecting outcomes from judge results."""
        task_names = ["task1"]

        outcomes = mock_evaluator._collect_all_outcomes(
            judge_results_dict, task_names, "judge_id"
        )

        assert len(outcomes) == 2  # Two judges
        result_df = outcomes[0]

        expected_columns = ["original_id", "task_value", "judge_id", "task_name"]
        assert set(result_df.columns) == set(expected_columns)
        assert result_df["judge_id"].to_list() == [
            "judge_1",
            "judge_1",
            "judge_1",
            "judge_1",
        ]
        assert result_df["task_name"].to_list() == ["task1", "task1", "task1", "task1"]

    def test_collect_all_outcomes_human(self, mock_evaluator, human_results_dict):
        """Test collecting outcomes from human results."""
        task_names = ["task1"]

        outcomes = mock_evaluator._collect_all_outcomes(
            human_results_dict, task_names, "annotator_id"
        )

        assert len(outcomes) == 2  # Two separate runs for human_1 and human_2

        # Check human_1 results
        human_1_df = outcomes[0]
        expected_columns = ["original_id", "task_value", "annotator_id", "task_name"]
        assert set(human_1_df.columns) == set(expected_columns)
        assert human_1_df["annotator_id"].to_list() == [
            "human_1",
            "human_1",
            "human_1",
            "human_1",
        ]

        # Check human_2 results
        human_2_df = outcomes[1]
        assert set(human_2_df.columns) == set(expected_columns)
        assert human_2_df["annotator_id"].to_list() == [
            "human_2",
            "human_2",
            "human_2",
            "human_2",
        ]

    def test_find_common_ids(self, mock_evaluator):
        """Test finding common IDs between judge and human results."""
        consolidated_judge_df = pl.DataFrame({"original_id": ["1", "2", "3", "4"]})
        consolidated_human_df = pl.DataFrame({"original_id": ["2", "3", "4", "5"]})

        id_sets = mock_evaluator._find_common_ids(
            consolidated_judge_df, consolidated_human_df
        )

        assert id_sets["common_ids"] == {"2", "3", "4"}
        assert id_sets["judge_only_ids"] == {"1"}
        assert id_sets["human_only_ids"] == {"5"}

    def test_align_results_by_id(
        self, mock_evaluator, judge_results_dict, human_results_dict
    ):
        """Test aligning judge and human results by ID."""
        task_names = ["task1"]

        consolidated_judge_df, consolidated_human_df = (
            mock_evaluator._align_results_by_id(
                judge_results_dict, human_results_dict, task_names
            )
        )

        # All results should have the same original_ids
        judge_ids = set(consolidated_judge_df["original_id"].to_list())
        human_ids = set(consolidated_human_df["original_id"].to_list())
        assert judge_ids == human_ids
        assert judge_ids == {"1", "2", "3", "4"}

    def test_align_results_by_id_no_common_ids(self, mock_evaluator):
        """Test alignment when no common IDs exist."""
        judge_results = Mock(spec=JudgeResults)
        judge_results.judge_id = "judge_1"
        judge_results.get_successful_results.return_value = pl.DataFrame(
            {"original_id": ["1", "2"], "task1": ["A", "B"]}
        )

        human_results = Mock(spec=HumanAnnotationResults)
        human_results.annotator_id = "human_1"
        human_results.get_successful_results.return_value = pl.DataFrame(
            {"original_id": ["3", "4"], "task1": ["C", "D"]}
        )

        judge_dict = {"run_1": judge_results}
        human_dict = {"run_1": human_results}

        with pytest.raises(
            InsufficientDataError, match="No aligned judge-human data found"
        ):
            mock_evaluator._align_results_by_id(judge_dict, human_dict, ["task1"])

    def test_empty_dataframes(self, mock_evaluator):
        """Test handling of empty DataFrames."""
        judge_results = Mock(spec=JudgeResults)
        judge_results.judge_id = "judge_1"
        judge_results.get_successful_results.return_value = pl.DataFrame(
            {"original_id": [], "task1": []}
        )

        human_results = Mock(spec=HumanAnnotationResults)
        human_results.annotator_id = "human_1"
        human_results.get_successful_results.return_value = pl.DataFrame(
            {"original_id": [], "task1": []}
        )

        judge_dict = {"run_1": judge_results}
        human_dict = {"run_1": human_results}

        with pytest.raises(InsufficientDataError, match="No judge outcomes found"):
            mock_evaluator._align_results_by_id(judge_dict, human_dict, ["task1"])

    def test_extract_task_schemas(self, mock_evaluator, judge_results_dict):
        """Test extracting task schemas from judge results."""
        task_names = ["task1", "task2", "safety"]

        schemas = mock_evaluator._extract_task_schemas(judge_results_dict, task_names)

        expected = {
            "task1": ["A", "B", "C"],
            "task2": None,
            "safety": ["SAFE", "UNSAFE"],
        }
        assert schemas == expected

    def test_validate_scorer_compatibility_valid(self, mock_evaluator):
        """Test scorer compatibility validation - valid case."""
        scorer = AccuracyScorer()
        task_schemas = {"task1": ["A", "B", "C"]}  # Classification task

        # Should not raise any exception
        mock_evaluator._validate_scorer_compatibility(scorer, task_schemas)

    def test_validate_scorer_compatibility_invalid(self, mock_evaluator):
        """Test scorer compatibility validation - invalid case."""
        scorer = AccuracyScorer()
        task_schemas = {"task1": None}  # Free-form text task

        with pytest.raises(IncompatibleTaskError, match="Incompatible task"):
            mock_evaluator._validate_scorer_compatibility(scorer, task_schemas)

    def test_extract_task_schemas_missing_task(
        self, mock_evaluator, judge_results_dict
    ):
        """Test extracting task schemas when a task is not found."""
        task_names = ["task1", "missing_task"]

        with pytest.raises(
            EvalTaskNotFoundError,
            match="Task 'missing_task' not found in judge results schemas",
        ):
            mock_evaluator._extract_task_schemas(judge_results_dict, task_names)

    def test_extract_outcomes_for_task_null_filtering(self, mock_evaluator):
        """Test that _extract_outcomes_for_task filters out null/None values."""
        # Create mock results with null values
        judge_results = Mock(spec=JudgeResults)
        judge_results.get_successful_results.return_value = pl.DataFrame(
            {
                "original_id": ["1", "2", "3", "4"],
                "task1": ["A", None, "C", "D"],
                "task2": ["text1", "text2", None, "text4"],
            }
        )

        # Test task1 - should filter out row with None value
        result = mock_evaluator._extract_outcomes_for_task(judge_results, "task1")
        expected = pl.DataFrame(
            {"original_id": ["1", "3", "4"], "task1": ["A", "C", "D"]}
        )
        assert result.equals(expected)

        # Test task2 - should filter out row with None value
        result = mock_evaluator._extract_outcomes_for_task(judge_results, "task2")
        expected = pl.DataFrame(
            {"original_id": ["1", "2", "4"], "task2": ["text1", "text2", "text4"]}
        )
        assert result.equals(expected)

        # Test all null values - should return empty DataFrame
        judge_results.get_successful_results.return_value = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "task1": [None, None, None],
            }
        )
        result = mock_evaluator._extract_outcomes_for_task(judge_results, "task1")
        expected = pl.DataFrame({"original_id": [], "task1": []})
        assert result.equals(expected)


class TestResultsLoading:
    """Test suite for MetaEvaluator results loading methods."""

    # Note: Fixtures are now available from conftest.py

    # === Judge Results Loading Tests ===

    def test_load_all_judge_results_empty_directory(self, meta_evaluator):
        """Test loading all judge results when results directory is empty."""
        meta_evaluator.paths.results.mkdir(parents=True, exist_ok=True)
        results = meta_evaluator.load_all_judge_results()
        assert results == {}

    def test_load_all_judge_results_no_directory(self, meta_evaluator):
        """Test loading all judge results when results directory doesn't exist."""
        if meta_evaluator.paths.results.exists():
            shutil.rmtree(meta_evaluator.paths.results)
        results = meta_evaluator.load_all_judge_results()
        assert results == {}

    def test_load_all_judge_results_absolute_directory(
        self, tmp_path, completed_judge_results
    ):
        """Test loading judge results with absolute project directory."""
        project_dir = str(tmp_path / "abs_project")
        meta_evaluator = MetaEvaluator(project_dir=project_dir)

        state_file = meta_evaluator.paths.results / "test_judge_state.json"
        completed_judge_results.save_state(str(state_file), data_format="json")

        results = meta_evaluator.load_all_judge_results()
        assert len(results) == 1
        assert "test_run" in results
        assert results["test_run"].judge_id == "test_judge"

    def test_load_all_judge_results_relative_directory(
        self, tmp_path, completed_judge_results
    ):
        """Test loading judge results with relative project directory."""
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            project_dir = "./rel_project"
            meta_evaluator = MetaEvaluator(project_dir=project_dir)

            state_file = meta_evaluator.paths.results / "test_judge_state.json"
            completed_judge_results.save_state(str(state_file), data_format="json")

            results = meta_evaluator.load_all_judge_results()
            assert len(results) == 1
            assert "test_run" in results
            assert results["test_run"].judge_id == "test_judge"
        finally:
            os.chdir(original_cwd)

    def test_load_all_judge_results_multiple_files(self, meta_evaluator):
        """Test loading all judge results with multiple valid files."""
        for i, judge_id in enumerate(["judge_1", "judge_2"]):
            builder = JudgeResultsBuilder(
                run_id=f"test_run_{i}",
                judge_id=judge_id,
                llm_client_enum=LLMClientEnum.OPENAI,
                model_used="gpt-4",
                task_schemas={"sentiment": ["positive", "negative"]},
                expected_ids=[f"id{i}"],
            )
            builder.create_success_row(
                sample_example_id=f"test_{i}",
                original_id=f"id{i}",
                outcomes={"sentiment": "positive"},
                llm_raw_response_content="positive",
                llm_prompt_tokens=10,
                llm_completion_tokens=5,
                llm_total_tokens=15,
                llm_call_duration_seconds=1.0,
            )
            judge_results = builder.complete()
            state_file = meta_evaluator.paths.results / f"test_{judge_id}_state.json"
            judge_results.save_state(str(state_file))

        results = meta_evaluator.load_all_judge_results()
        assert len(results) == 2
        assert "test_run_0" in results
        assert "test_run_1" in results
        assert results["test_run_0"].judge_id == "judge_1"
        assert results["test_run_1"].judge_id == "judge_2"

    def test_load_all_judge_results_invalid_files(
        self, meta_evaluator, completed_judge_results
    ):
        """Test loading all judge results with some invalid files that should be skipped."""
        valid_state_file = meta_evaluator.paths.results / "valid_judge_state.json"
        completed_judge_results.save_state(str(valid_state_file))

        # Create an invalid state file in results directory
        invalid_state_file = meta_evaluator.paths.results / "invalid_state.json"
        invalid_state_file.write_text("{ invalid json }")

        results = meta_evaluator.load_all_judge_results()
        assert len(results) == 1
        assert "test_run" in results
        assert results["test_run"].judge_id == "test_judge"

    # === Human Results Loading Tests ===

    def test_load_all_human_results_empty_directory(self, meta_evaluator):
        """Test loading all human results when annotations directory is empty."""
        meta_evaluator.paths.annotations.mkdir(parents=True, exist_ok=True)
        results = meta_evaluator.load_all_human_results()
        assert results == {}

    def test_load_all_human_results_no_directory(self, meta_evaluator):
        """Test loading all human results when annotations directory doesn't exist."""
        if meta_evaluator.paths.annotations.exists():
            shutil.rmtree(meta_evaluator.paths.annotations)
        results = meta_evaluator.load_all_human_results()
        assert results == {}

    def test_load_all_human_results_absolute_directory(
        self, tmp_path, completed_human_results
    ):
        """Test loading human results with absolute project directory."""
        project_dir = str(tmp_path / "abs_project")
        meta_evaluator = MetaEvaluator(project_dir=project_dir)

        state_file = meta_evaluator.paths.annotations / "test_annotator_metadata.json"
        completed_human_results.save_state(str(state_file))

        results = meta_evaluator.load_all_human_results()
        assert len(results) == 1
        assert "test_annotation_run" in results
        assert results["test_annotation_run"].annotator_id == "test_annotator"

    def test_load_all_human_results_relative_directory(
        self, tmp_path, completed_human_results
    ):
        """Test loading human results with relative project directory."""
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            project_dir = "./rel_project"
            meta_evaluator = MetaEvaluator(project_dir=project_dir)

            state_file = (
                meta_evaluator.paths.annotations / "test_annotator_metadata.json"
            )
            completed_human_results.save_state(str(state_file))

            results = meta_evaluator.load_all_human_results()
            assert len(results) == 1
            assert "test_annotation_run" in results
            assert results["test_annotation_run"].annotator_id == "test_annotator"
        finally:
            os.chdir(original_cwd)

    def test_load_all_human_results_multiple_files(self, meta_evaluator):
        """Test loading all human results with multiple valid files."""
        for i, annotator_id in enumerate(["annotator_1", "annotator_2"]):
            builder = HumanAnnotationResultsBuilder(
                run_id=f"test_annotation_run_{i}",
                annotator_id=annotator_id,
                task_schemas={"accuracy": ["accurate", "inaccurate"]},
                expected_ids=[f"id{i}"],
            )
            builder.create_success_row(
                sample_example_id=f"test_{i}",
                original_id=f"id{i}",
                outcomes={"accuracy": "accurate"},
                annotation_timestamp=datetime.now(),
            )
            human_results = builder.complete()
            state_file = (
                meta_evaluator.paths.annotations / f"test_{annotator_id}_metadata.json"
            )
            human_results.save_state(str(state_file))

        results = meta_evaluator.load_all_human_results()
        assert len(results) == 2
        assert "test_annotation_run_0" in results
        assert "test_annotation_run_1" in results
        assert results["test_annotation_run_0"].annotator_id == "annotator_1"
        assert results["test_annotation_run_1"].annotator_id == "annotator_2"

    def test_load_all_human_results_invalid_files(
        self, meta_evaluator, completed_human_results
    ):
        """Test loading all human results with some invalid files that should be skipped."""
        valid_state_file = (
            meta_evaluator.paths.annotations / "valid_annotator_metadata.json"
        )
        completed_human_results.save_state(str(valid_state_file))

        # Create an invalid metadata file
        invalid_state_file = meta_evaluator.paths.annotations / "invalid_metadata.json"
        invalid_state_file.write_text("{ invalid json }")

        results = meta_evaluator.load_all_human_results()
        assert len(results) == 1
        assert "test_annotation_run" in results
        assert results["test_annotation_run"].annotator_id == "test_annotator"


class TestScoring:
    """Test the compare method in ScoringMixin."""

    def test_compare_no_metrics_configured(self, mock_evaluator):
        """Test comparison when no metrics are configured."""
        config = MetricsConfig(metrics=[])

        with pytest.raises(
            ScoringConfigError, match="No metrics configured for comparison"
        ):
            mock_evaluator.compare(config)

    def test_compare_no_task_names_specified(self, mock_evaluator):
        """Test comparison when no task names are specified for a metric."""
        mock_scorer = AccuracyScorer()  # Use a real scorer
        config = MetricsConfig(
            metrics=[MetricConfig(scorer=mock_scorer, task_names=[])]
        )

        with pytest.raises(
            ScoringConfigError, match="No task names specified for metric 0"
        ):
            mock_evaluator.compare(config)

    def test_compare_success(
        self, mock_evaluator, judge_results_dict, human_results_dict
    ):
        """Test successful comparison."""
        # Mock the load methods
        mock_evaluator.load_all_judge_results = Mock(return_value=judge_results_dict)
        mock_evaluator.load_all_human_results = Mock(return_value=human_results_dict)

        # Use a real scorer
        scorer = AccuracyScorer()

        # Create config
        config = MetricsConfig(
            metrics=[MetricConfig(scorer=scorer, task_names=["task1"])]
        )

        results = mock_evaluator.compare(config)

        assert len(results) == 1
        assert "accuracy" in results
        # Should have 2 results: one for each judge
        assert len(results["accuracy"]) == 2
        # Check first result
        result1 = results["accuracy"][0]
        assert result1.scorer_name == "accuracy"
        assert result1.task_name == "task1"
        assert result1.judge_id in ["judge_1", "judge_2"]
        assert isinstance(result1.score, float)
        assert 0.0 <= result1.score <= 1.0
        # Check second result
        result2 = results["accuracy"][1]
        assert result2.scorer_name == "accuracy"
        assert result2.task_name == "task1"
        assert result2.judge_id in ["judge_1", "judge_2"]
        assert isinstance(result2.score, float)
        assert 0.0 <= result2.score <= 1.0

    def test_compare_no_judge_results(self, mock_evaluator):
        """Test comparison when no judge results found."""
        mock_evaluator.load_all_judge_results = Mock(return_value={})
        mock_evaluator.load_all_human_results = Mock(return_value={"run_1": Mock()})

        config = MetricsConfig(
            metrics=[MetricConfig(scorer=AccuracyScorer(), task_names=["task1"])]
        )

        with pytest.raises(
            InsufficientDataError, match="No judge results provided or found"
        ):
            mock_evaluator.compare(config)

    def test_compare_no_human_results(self, mock_evaluator):
        """Test comparison when no human results found."""
        mock_evaluator.load_all_judge_results = Mock(return_value={"run_1": Mock()})
        mock_evaluator.load_all_human_results = Mock(return_value={})

        config = MetricsConfig(
            metrics=[MetricConfig(scorer=AccuracyScorer(), task_names=["task1"])]
        )

        with pytest.raises(
            InsufficientDataError, match="No human results provided or found"
        ):
            mock_evaluator.compare(config)

    def test_aggregation_with_mixed_scorers(
        self, judge_results_dict, human_results_dict, mock_evaluator
    ):
        """Test aggregation with scorers that both support aggregation."""
        # Mock the load methods
        mock_evaluator.load_all_judge_results = Mock(return_value=judge_results_dict)
        mock_evaluator.load_all_human_results = Mock(return_value=human_results_dict)

        # Use both AltTestScorer and AccuracyScorer (both support aggregation)
        alt_test_scorer = AltTestScorer()
        alt_test_scorer.min_instances_per_human = 1
        alt_test_scorer.min_humans_per_instance = 1

        accuracy_scorer = AccuracyScorer()

        # Create config with multiple metrics
        config = MetricsConfig(
            metrics=[
                MetricConfig(scorer=alt_test_scorer, task_names=["safety"]),
                MetricConfig(scorer=accuracy_scorer, task_names=["task1"]),
            ]
        )

        results = mock_evaluator.compare(config)

        # Verify results were generated for both scorers
        assert "alt_test" in results
        assert "accuracy" in results
        # Should have 2 results each: one for each judge
        assert len(results["alt_test"]) == 2
        assert len(results["accuracy"]) == 2

        # Verify aggregation directory was created for AltTestScorer
        alt_test_dir = Path(mock_evaluator.paths.scores) / "alt_test"
        assert alt_test_dir.exists()

        # Verify aggregation files were created (AltTestScorer creates 3 plots)
        expected_plots = [
            "aggregate_winning_rates.png",
            "aggregate_advantage_probabilities.png",
            "aggregate_human_vs_llm_advantage.png",
        ]
        for plot_name in expected_plots:
            plot_path = alt_test_dir / plot_name
            assert plot_path.exists(), f"Missing aggregate plot: {plot_name}"

        # Verify aggregation directory was created for AccuracyScorer
        accuracy_dir = Path(mock_evaluator.paths.scores) / "accuracy"
        assert accuracy_dir.exists()

        # Verify individual result files were saved for AccuracyScorer
        result_files = list(accuracy_dir.glob("*_result.json"))
        assert len(result_files) == 2  # One for each judge
