"""Tests for ScoringMixin data processing functionality."""

import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from meta_evaluator.meta_evaluator import MetaEvaluator
from meta_evaluator.meta_evaluator.scoring import ScoringMixin
from meta_evaluator.results import (
    HumanAnnotationResults,
    HumanAnnotationResultsBuilder,
    JudgeResults,
    JudgeResultsBuilder,
)
from meta_evaluator.scores import (
    AccuracyScorer,
    BaseScoringResult,
    MetricConfig,
    MetricsConfig,
)
from meta_evaluator.scores.enums import TaskAggregationMode
from meta_evaluator.scores.metrics.agreement.alt_test import AltTestScorer
from meta_evaluator.scores.metrics.agreement.iaa import CohensKappaScorer
from meta_evaluator.scores.metrics.text_comparison.text_similarity import (
    TextSimilarityScorer,
)

from .conftest import create_mock_human_metadata_file, create_mock_judge_state_file

# Note: All fixtures are now available from conftest.py


class TestDataProcessing:
    """Test data processing methods in ScoringMixin."""

    def test_filter_and_align_data(
        self, mock_evaluator, judge_results_1, human_results_dict
    ):
        """Test filtering and aligning judge and human data."""
        judge_df, human_df = mock_evaluator._filter_and_align_data(
            judge_results_1, human_results_dict
        )

        # Should have common IDs from both judge and human results
        assert set(judge_df["original_id"].to_list()) == {"1", "2", "3", "4"}
        assert set(human_df["original_id"].to_list()) == {"1", "2", "3", "4"}
        # Human df should have human_id column
        assert "human_id" in human_df.columns
        assert "task1" in judge_df.columns
        assert "task1" in human_df.columns

    def test_filter_and_align_data_empty(self, mock_evaluator):
        """Test filtering when no successful results."""
        # Create mock judge results with empty successful results
        judge_result = Mock(spec=JudgeResults)
        judge_result.get_successful_results.return_value = pl.DataFrame()

        # Create mock human results
        human_result = Mock(spec=HumanAnnotationResults)
        human_result.annotator_id = "human_1"
        human_result.get_successful_results.return_value = pl.DataFrame()

        human_results = {"run_1": human_result}

        judge_df, human_df = mock_evaluator._filter_and_align_data(
            judge_result, human_results
        )

        # Should return empty DataFrames
        assert judge_df.is_empty()
        assert human_df.is_empty()

    def test_preprocess_task_data_single(self, mock_evaluator, simple_task_data):
        """Test preprocessing data for single task."""
        judge_df = simple_task_data["judge"]
        human_df = simple_task_data["human"]

        processed_judge, processed_human = mock_evaluator._preprocess_task_data(
            judge_df, human_df, ["task1"], TaskAggregationMode.SINGLE
        )

        # Should rename task column to "label"
        assert "label" in processed_judge.columns
        assert "label" in processed_human.columns
        assert processed_judge["label"].to_list() == ["A", "B", "A"]
        assert processed_human["label"].to_list() == ["A", "B", "A"]


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
                llm_client="openai",
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

    def test_load_all_judge_results_with_duplicates_keeps_most_recent(
        self, meta_evaluator, mock_judge_state_template, caplog
    ):
        """Test load_all_judge_results with duplicate judge_ids keeps the most recent run."""
        results_dir = meta_evaluator.paths.results
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create state files for same judge with different run_ids
        judge_configs = [
            {"judge_id": "judge1", "run_id": "run_001"},  # Older run
            {"judge_id": "judge1", "run_id": "run_002"},  # Newer run (higher run_id)
            {"judge_id": "judge2", "run_id": "run_003"},  # Different judge
        ]

        for config in judge_configs:
            create_mock_judge_state_file(
                results_dir,
                config["judge_id"],
                config["run_id"],
                mock_judge_state_template,
            )

        with patch("meta_evaluator.results.JudgeResults.load_state") as mock_load:

            def mock_load_side_effect(file_path):
                mock_results = Mock(spec=JudgeResults)
                if "run_001" in str(file_path):
                    mock_results.judge_id = "judge1"
                    mock_results.run_id = "run_001"
                elif "run_002" in str(file_path):
                    mock_results.judge_id = "judge1"
                    mock_results.run_id = "run_002"
                elif "run_003" in str(file_path):
                    mock_results.judge_id = "judge2"
                    mock_results.run_id = "run_003"
                return mock_results

            mock_load.side_effect = mock_load_side_effect

            # Load results
            results = meta_evaluator.load_all_judge_results()

            # Should have 2 results (most recent for judge1, and judge2)
            assert len(results) == 2

            # Should keep run_002 (higher run_id) for judge1 and skip run_001
            judge1_result = None
            for run_id, result in results.items():
                if result.judge_id == "judge1":
                    judge1_result = result
                    break

            assert judge1_result is not None
            assert judge1_result.run_id == "run_002"  # Most recent

            # Should have judge2 results
            assert "run_003" in results
            assert results["run_003"].judge_id == "judge2"

            # Verify warning log
            assert "Found duplicate results for judge_id 'judge1'" in caplog.text
            assert "Keeping most recent run_id 'run_002'" in caplog.text
            assert "skipping: ['run_001']" in caplog.text

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

    def test_load_all_human_results_with_duplicates_keeps_most_recent(
        self, meta_evaluator, mock_human_state_template, caplog
    ):
        """Test load_all_human_results with duplicate annotator_ids keeps the most recent run."""
        annotations_dir = meta_evaluator.paths.annotations
        annotations_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata files for same annotator with different run_ids
        human_configs = [
            {"annotator_id": "human1", "run_id": "run_001"},  # Older run
            {
                "annotator_id": "human1",
                "run_id": "run_002",
            },  # Newer run (higher run_id)
            {"annotator_id": "human2", "run_id": "run_003"},  # Different annotator
        ]

        for config in human_configs:
            create_mock_human_metadata_file(
                annotations_dir,
                config["annotator_id"],
                config["run_id"],
                mock_human_state_template,
            )

        with patch(
            "meta_evaluator.results.HumanAnnotationResults.load_state"
        ) as mock_load:

            def mock_load_side_effect(file_path):
                mock_results = Mock(spec=HumanAnnotationResults)
                if "run_001" in str(file_path):
                    mock_results.annotator_id = "human1"
                    mock_results.run_id = "run_001"
                elif "run_002" in str(file_path):
                    mock_results.annotator_id = "human1"
                    mock_results.run_id = "run_002"
                elif "run_003" in str(file_path):
                    mock_results.annotator_id = "human2"
                    mock_results.run_id = "run_003"
                return mock_results

            mock_load.side_effect = mock_load_side_effect

            # Load results
            results = meta_evaluator.load_all_human_results()

            # Should have 2 results (most recent for human1, and human2)
            assert len(results) == 2

            # Should keep run_002 (higher run_id) for human1 and skip run_001
            human1_result = None
            for run_id, result in results.items():
                if result.annotator_id == "human1":
                    human1_result = result
                    break

            assert human1_result is not None
            assert human1_result.run_id == "run_002"  # Most recent

            # Should have human2 results
            assert "run_003" in results
            assert results["run_003"].annotator_id == "human2"

            # Verify warning log
            assert "Found duplicate results for annotator_id 'human1'" in caplog.text
            assert "Keeping most recent run_id 'run_002'" in caplog.text
            assert "skipping: ['run_001']" in caplog.text


class TestExternalResultsLoading:
    """Test external judge and human results loading functionality."""

    # ===== Judge Results =====

    @pytest.mark.parametrize("data_format", ["json", "csv", "parquet"])
    def test_judge_results_file_format_success(
        self, evaluator_with_task, valid_judge_data, tmp_path, data_format
    ):
        """Test successfully loading judge results from different file formats."""
        file_path = tmp_path / f"judge_results.{data_format}"

        # Write data in specified format
        if data_format == "json":
            valid_judge_data.write_json(file_path)
        elif data_format == "csv":
            valid_judge_data.write_csv(file_path)
        elif data_format == "parquet":
            valid_judge_data.write_parquet(file_path)

        judge_id = f"external_judge_{data_format}"

        # Load external results
        evaluator_with_task.add_external_judge_results(
            file_path=str(file_path),
            judge_id=judge_id,
            llm_client="test_client",
            model_used="test_model",
        )

        # Verify results were saved to project directory
        results_dir = Path(evaluator_with_task.project_dir) / "results"
        assert results_dir.exists()

        # Check that state file was created
        state_files = list(results_dir.glob(f"*_{judge_id}_external_state.json"))
        assert len(state_files) == 1

        # Verify we can load the results back
        all_results = evaluator_with_task.load_all_judge_results()
        assert len(all_results) == 1

        # Verify data integrity
        result = list(all_results.values())[0]
        assert result.judge_id == judge_id
        successful_results = result.get_successful_results()
        assert len(successful_results) == 3
        assert "sentiment" in successful_results.columns
        assert "quality" in successful_results.columns

    def test_add_external_judge_results_integration(
        self, evaluator_with_task, valid_judge_data, tmp_path
    ):
        """Test that add_external_judge_results leads to successful load_all_judge_results() call."""
        csv_path = tmp_path / "judge_results.csv"
        valid_judge_data.write_csv(csv_path)

        judge_id = "integration_test_judge"

        # Add external results
        evaluator_with_task.add_external_judge_results(
            file_path=str(csv_path),
            judge_id=judge_id,
            llm_client="test_client",
            model_used="test_model",
        )

        # Verify the file is found in project_dir
        results_dir = Path(evaluator_with_task.project_dir) / "results"
        state_files = list(results_dir.glob(f"*_{judge_id}_external_state.json"))
        assert len(state_files) == 1

        # Verify state file has correct structure
        with open(state_files[0]) as f:
            state_data = json.load(f)
        assert state_data["judge_id"] == judge_id
        assert state_data["llm_client"] == "test_client"
        assert state_data["model_used"] == "test_model"

        # Test successful load_all_judge_results() call
        all_results = evaluator_with_task.load_all_judge_results()
        assert len(all_results) == 1

        result = list(all_results.values())[0]
        assert result.judge_id == judge_id
        assert result.llm_client == "test_client"
        assert result.model_used == "test_model"
        assert result.total_count == 3

    @pytest.mark.parametrize(
        "columns_to_remove,test_description",
        [
            (["original_id"], "missing original_id column"),
            (["sentiment"], "missing task column"),
            (["sentiment", "quality"], "missing multiple task columns"),
            (
                ["original_id", "sentiment"],
                "missing multiple required columns",
            ),
        ],
    )
    def test_validate_judge_results_data_required_columns(
        self,
        evaluator_with_task,
        valid_judge_data,
        tmp_path,
        columns_to_remove,
        test_description,
    ):
        """Test that missing required columns throw appropriate errors."""
        # Remove specified columns from valid data
        invalid_data = valid_judge_data.drop(columns_to_remove)

        csv_path = tmp_path / "invalid_judge_results.csv"
        invalid_data.write_csv(csv_path)

        with pytest.raises(ValueError, match="Missing required columns"):
            evaluator_with_task.add_external_judge_results(
                file_path=str(csv_path), judge_id="invalid_judge"
            )

    def test_validate_judge_results_data_missing_fields(
        self, evaluator_with_task, valid_judge_data, tmp_path
    ):
        """Test that optional fields are auto-generated when missing from user CSV."""
        # The simplified valid_judge_data already only has required fields
        # Test that system fields are auto-generated properly
        csv_path = tmp_path / "minimal_judge_results.csv"
        valid_judge_data.write_csv(csv_path)

        # Should not raise an error
        evaluator_with_task.add_external_judge_results(
            file_path=str(csv_path),
            judge_id="minimal_judge",
            llm_client="test_client",
            model_used="test_model",
        )

        # Verify results were processed correctly
        all_results = evaluator_with_task.load_all_judge_results()
        result = list(all_results.values())[0]

        # Check that DataFrame has all expected columns
        results_df = result.results_data

        # Status should be set to "success" for all external data
        assert "status" in results_df.columns
        assert (results_df["status"] == "success").all()

        # Other optional fields should be filled with None
        expected_none_columns = [
            "error_message",
            "error_details_json",
            "llm_raw_response_content",
            "llm_prompt_tokens",
            "llm_completion_tokens",
            "llm_total_tokens",
            "llm_call_duration_seconds",
        ]

        for col in expected_none_columns:
            assert col in results_df.columns
            # Verify that the column contains only None values
            assert results_df[col].is_null().all()

    # ===== Human Results =====

    @pytest.mark.parametrize("data_format", ["json", "csv", "parquet"])
    def test_human_results_file_format_success(
        self, evaluator_with_task, valid_human_data, tmp_path, data_format
    ):
        """Test successfully loading human results from different file formats."""
        file_path = tmp_path / f"human_results.{data_format}"

        # Write data in specified format
        if data_format == "json":
            valid_human_data.write_json(file_path)
        elif data_format == "csv":
            valid_human_data.write_csv(file_path)
        elif data_format == "parquet":
            valid_human_data.write_parquet(file_path)

        annotator_id = f"external_annotator_{data_format}"

        # Load external results
        evaluator_with_task.add_external_annotation_results(
            file_path=str(file_path),
            annotator_id=annotator_id,
        )

        # Verify results were saved to project directory
        annotations_dir = Path(evaluator_with_task.project_dir) / "annotations"
        assert annotations_dir.exists()

        # Check that metadata file was created
        metadata_files = list(
            annotations_dir.glob(f"*_{annotator_id}_external_metadata.json")
        )
        assert len(metadata_files) == 1

        # Verify we can load the results back
        all_results = evaluator_with_task.load_all_human_results()
        assert len(all_results) == 1

        # Verify data integrity
        result = list(all_results.values())[0]
        assert result.annotator_id == annotator_id
        successful_results = result.get_successful_results()
        assert len(successful_results) == 3
        assert "sentiment" in successful_results.columns
        assert "quality" in successful_results.columns

    def test_add_external_annotation_results_integration(
        self, evaluator_with_task, valid_human_data, tmp_path
    ):
        """Test that add_external_annotation_results leads to successful load_all_human_results() call."""
        csv_path = tmp_path / "human_results.csv"
        valid_human_data.write_csv(csv_path)

        annotator_id = "integration_test_annotator"

        # Add external results
        evaluator_with_task.add_external_annotation_results(
            file_path=str(csv_path),
            annotator_id=annotator_id,
        )

        # Verify the file is found in project_dir
        annotations_dir = Path(evaluator_with_task.project_dir) / "annotations"
        metadata_files = list(
            annotations_dir.glob(f"*_{annotator_id}_external_metadata.json")
        )
        assert len(metadata_files) == 1

        # Verify metadata file has correct structure
        with open(metadata_files[0]) as f:
            metadata = json.load(f)
        assert metadata["annotator_id"] == annotator_id

        # Test successful load_all_human_results() call
        all_results = evaluator_with_task.load_all_human_results()
        assert len(all_results) == 1

        result = list(all_results.values())[0]
        assert result.annotator_id == annotator_id
        assert result.total_count == 3

    @pytest.mark.parametrize(
        "columns_to_remove,test_description",
        [
            (["original_id"], "missing original_id column"),
            (["sentiment"], "missing task column"),
            (["sentiment", "quality"], "missing multiple task columns"),
            (
                ["original_id", "sentiment"],
                "missing multiple required columns",
            ),
        ],
    )
    def test_validate_annotation_results_data_required_columns(
        self,
        evaluator_with_task,
        valid_human_data,
        tmp_path,
        columns_to_remove,
        test_description,
    ):
        """Test that missing required columns throw appropriate errors."""
        # Remove specified columns from valid data
        invalid_data = valid_human_data.drop(columns_to_remove)

        csv_path = tmp_path / "invalid_human_results.csv"
        invalid_data.write_csv(csv_path)

        with pytest.raises(ValueError, match="Missing required columns"):
            evaluator_with_task.add_external_annotation_results(
                file_path=str(csv_path), annotator_id="invalid_annotator"
            )

    def test_validate_annotation_results_data_missing_fields(
        self, evaluator_with_task, valid_human_data, tmp_path
    ):
        """Test that optional fields are auto-generated when missing from user CSV."""
        # The simplified valid_human_data already only has required fields
        # Test that system fields are auto-generated properly
        csv_path = tmp_path / "minimal_human_results.csv"
        valid_human_data.write_csv(csv_path)

        # Should not raise an error
        evaluator_with_task.add_external_annotation_results(
            file_path=str(csv_path),
            annotator_id="minimal_annotator",
        )

        # Verify results were processed correctly
        all_results = evaluator_with_task.load_all_human_results()
        result = list(all_results.values())[0]

        # Check that DataFrame has all expected columns
        results_df = result.results_data

        # Status should be set to "success" for all external data
        assert "status" in results_df.columns
        assert (results_df["status"] == "success").all()

        # Other optional fields should be filled with None
        expected_none_columns = [
            "error_message",
            "error_details_json",
            "annotation_timestamp",
        ]

        for col in expected_none_columns:
            assert col in results_df.columns
            # Verify that the column contains only None values
            assert results_df[col].is_null().all()


class TestScoring:
    """Test the ScoringMixin methods."""

    @patch.object(ScoringMixin, "_process_single_judge_async")
    def test_run_scoring_async_calls(
        self, mock_process_judge, mock_evaluator, judge_results_dict, human_results_dict
    ):
        """Test _run_scoring_async calls _process_single_judge_async for number of judges times asynchronously."""
        scorer = AccuracyScorer()
        config = MetricConfig(
            scorer=scorer,
            task_names=["task1"],
            aggregation_name="single",
        )

        # Mock the process_single_judge_async method
        mock_result1 = BaseScoringResult(
            scorer_name="accuracy",
            task_name="task1",
            judge_id="judge_1",
            scores={"accuracy": 0.8},
            metadata={},
            aggregation_mode=TaskAggregationMode.SINGLE,
            num_comparisons=10,
            failed_comparisons=0,
        )
        mock_result2 = BaseScoringResult(
            scorer_name="accuracy",
            task_name="task1",
            judge_id="judge_2",
            scores={"accuracy": 0.7},
            metadata={},
            aggregation_mode=TaskAggregationMode.SINGLE,
            num_comparisons=8,
            failed_comparisons=0,
        )

        # Create async mock that returns different results for each judge
        mock_process_judge.side_effect = [mock_result1, mock_result2]

        # Run the method
        result = asyncio.run(
            mock_evaluator._run_scoring_async(
                config, judge_results_dict, human_results_dict
            )
        )

        # Verify _process_single_judge_async was called for each judge
        assert mock_process_judge.call_count == 2

        # Verify return structure
        metric_config, scoring_results = result
        assert metric_config == config
        assert len(scoring_results) == 2
        assert scoring_results[0].judge_id == "judge_1"
        assert scoring_results[1].judge_id == "judge_2"

    def test_filter_and_align_data(
        self, mock_evaluator, judge_results_1, human_results_dict
    ):
        """Test _filter_and_align_data method."""
        judge_df, human_df = mock_evaluator._filter_and_align_data(
            judge_results_1, human_results_dict
        )

        # Should have common IDs from both judge and human results
        assert set(judge_df["original_id"].to_list()) == {"1", "2", "3", "4"}
        assert set(human_df["original_id"].to_list()) == {"1", "2", "3", "4"}
        # Human df should have human_id column
        assert "human_id" in human_df.columns
        assert "task1" in judge_df.columns
        assert "task1" in human_df.columns

    def test_preprocess_task_data_multilabel(
        self, mock_evaluator, multilabel_task_data
    ):
        """Test _preprocess_task_data for TaskAggregationMode.MULTILABEL aggregates the columns."""
        judge_df = multilabel_task_data["judge"]
        human_df = multilabel_task_data["human"]

        processed_judge, processed_human = mock_evaluator._preprocess_task_data(
            judge_df, human_df, ["task1", "task2"], TaskAggregationMode.MULTILABEL
        )

        # Should have label column with aggregated values
        assert "label" in processed_judge.columns
        assert "label" in processed_human.columns
        # Values should be lists of strings (check the actual list, not series)
        judge_labels = processed_judge["label"].to_list()
        human_labels = processed_human["label"].to_list()

        assert all(isinstance(label, list) for label in judge_labels)
        assert all(isinstance(label, list) for label in human_labels)

        # Each list should contain the aggregated task values (order may vary)
        assert len(judge_labels) == 2
        assert len(human_labels) == 2
        assert ["A", "X"] in judge_labels
        assert ["B", "Y"] in judge_labels
        assert ["A", "X"] in human_labels
        assert ["B", "Y"] in human_labels

    @pytest.mark.asyncio
    async def test_process_single_judge_async_calls_multitask(
        self, mock_evaluator, judge_results_1, human_results_dict
    ):
        """Test _process_single_judge_async calls compute_score_async for number of tasks times when multitask."""
        # Use a real AccuracyScorer to avoid pydantic validation issues
        scorer = AccuracyScorer()

        config = MetricConfig(
            scorer=scorer,
            task_names=["task1", "task2"],
            aggregation_name="multitask",
        )

        # Call the actual method
        result = await mock_evaluator._process_single_judge_async(
            config, judge_results_1, human_results_dict
        )

        # Result should be aggregated for multi-task
        assert "2_tasks_avg" in result.task_name
        assert result.scorer_name == "accuracy"
        assert result.aggregation_mode == TaskAggregationMode.MULTITASK

    @pytest.mark.asyncio
    async def test_process_single_judge_async_calls_single(
        self, mock_evaluator, judge_results_1, human_results_dict
    ):
        """Test _process_single_judge_async calls compute_score_async once when single/multilabel."""
        # Use a real AccuracyScorer to avoid pydantic validation issues
        scorer = AccuracyScorer()

        config = MetricConfig(
            scorer=scorer,
            task_names=["task1"],
            aggregation_name="single",
        )

        # Call the actual method
        result = await mock_evaluator._process_single_judge_async(
            config, judge_results_1, human_results_dict
        )

        # Result should be the direct result for single task
        assert result.task_name == "task1"
        assert result.scorer_name == "accuracy"

    def test_save_all_scorer_results_different_scorers(
        self, mock_evaluator, different_scorer_configs
    ):
        """Test _save_all_scorer_results with different scorers."""
        # Set up mock evaluator paths
        mock_evaluator.paths.scores = Path(mock_evaluator.paths.scores)

        # Call the method
        mock_evaluator._save_all_scorer_results(different_scorer_configs)

        # Verify scorer directories were created and files saved
        accuracy_dir = mock_evaluator.paths.scores / "accuracy"
        kappa_dir = mock_evaluator.paths.scores / "cohens_kappa"
        alt_test_dir = mock_evaluator.paths.scores / "alt_test"

        assert accuracy_dir.exists()
        assert kappa_dir.exists()
        assert alt_test_dir.exists()

        # Verify result files were saved
        accuracy_files = list(accuracy_dir.glob("**/*_result.json"))
        kappa_files = list(kappa_dir.glob("**/*_result.json"))
        alt_test_files = list(alt_test_dir.glob("**/*_result.json"))

        assert len(accuracy_files) == 2  # 2 judges
        assert len(kappa_files) == 1  # 1 judge
        assert len(alt_test_files) == 1  # 1 judge

    def test_save_all_scorer_results_different_aggregation_modes(
        self, mock_evaluator, multi_aggregation_configs
    ):
        """Test _save_all_scorer_results with same scorer but different aggregation modes."""
        # Set up mock evaluator paths
        mock_evaluator.paths.scores = Path(mock_evaluator.paths.scores)

        # Call the method
        mock_evaluator._save_all_scorer_results(multi_aggregation_configs)

        # Verify accuracy directory was created
        accuracy_dir = mock_evaluator.paths.scores / "accuracy"
        assert accuracy_dir.exists()

        # Verify 3 result files were saved (one for each aggregation mode)
        result_files = list(accuracy_dir.glob("**/*_result.json"))
        assert len(result_files) == 3

    def test_aggregate_all_scorer_results_different_scorers(
        self, mock_evaluator, different_scorer_configs
    ):
        """Test _aggregate_all_scorer_results with different scorers."""
        # Set up mock evaluator paths
        mock_evaluator.paths.scores = Path(mock_evaluator.paths.scores)

        # Call the method
        mock_evaluator._aggregate_all_scorer_results(different_scorer_configs)

        # Verify scorer directories were created
        accuracy_dir = mock_evaluator.paths.scores / "accuracy"
        kappa_dir = mock_evaluator.paths.scores / "cohens_kappa"
        alt_test_dir = mock_evaluator.paths.scores / "alt_test"

        assert accuracy_dir.exists()
        assert kappa_dir.exists()
        assert alt_test_dir.exists()

        # Verify plots were generated
        accuracy_plots = list(accuracy_dir.glob("**/*.png"))
        kappa_plots = list(kappa_dir.glob("**/*.png"))
        alt_test_plots = list(alt_test_dir.glob("**/*.png"))

        assert len(accuracy_plots) == 1  # 1 bar plot for accuracy
        assert len(kappa_plots) == 1  # 1 bar plot for kappa
        assert len(alt_test_plots) == 3  # 3 plots for alt_test

    def test_aggregate_all_scorer_results_different_aggregation_modes(
        self, mock_evaluator, multi_aggregation_configs
    ):
        """Test _aggregate_all_scorer_results with same scorer but different aggregation modes."""
        # Set up mock evaluator paths
        mock_evaluator.paths.scores = Path(mock_evaluator.paths.scores)

        # Call the method
        mock_evaluator._aggregate_all_scorer_results(multi_aggregation_configs)

        # Verify accuracy directory was created
        accuracy_dir = mock_evaluator.paths.scores / "accuracy"
        assert accuracy_dir.exists()

        # Verify 3 plots were generated (one for each aggregation mode)
        plot_files = list(accuracy_dir.glob("**/*.png"))
        assert len(plot_files) == 3

    def test_get_first_non_null_value_returns_first_value(self, mock_evaluator):
        """Test _get_first_non_null_value returns first value."""
        # Test with simple values
        series = pl.Series(["A", "B", "C"])
        result = mock_evaluator._get_first_non_null_value(series)
        assert result == "A"

        # Test with null values
        series_with_nulls = pl.Series([None, "B", "C"])
        result = mock_evaluator._get_first_non_null_value(series_with_nulls)
        assert result == "B"

        # Test with all nulls
        all_nulls = pl.Series([None, None, None])
        result = mock_evaluator._get_first_non_null_value(all_nulls)
        assert result is None

    @pytest.mark.parametrize(
        "scorer_class,scorer_name,task_names,aggregation_name",
        [
            # AccuracyScorer tests
            (AccuracyScorer, "accuracy", ["task1"], "single"),
            (
                AccuracyScorer,
                "accuracy",
                ["task1", "safety"],
                "multitask",
            ),
            (
                AccuracyScorer,
                "accuracy",
                ["task1", "safety"],
                "multilabel",
            ),
            # CohensKappaScorer tests
            (CohensKappaScorer, "cohens_kappa", ["task1"], "single"),
            (
                CohensKappaScorer,
                "cohens_kappa",
                ["task1", "safety"],
                "multitask",
            ),
            (
                CohensKappaScorer,
                "cohens_kappa",
                ["task1", "safety"],
                "multilabel",
            ),
            # AltTestScorer tests
            (AltTestScorer, "alt_test", ["task1"], "single"),
            (
                AltTestScorer,
                "alt_test",
                ["task1", "safety"],
                "multitask",
            ),
            (
                AltTestScorer,
                "alt_test",
                ["task1", "safety"],
                "multilabel",
            ),
            # TextSimilarityScorer tests
            (TextSimilarityScorer, "text_similarity", ["task2"], "single"),
            (
                TextSimilarityScorer,
                "text_similarity",
                ["task1", "task2"],
                "multitask",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_process_single_judge_async_parametrized(
        self,
        mock_evaluator,
        judge_results_dict,
        human_results_dict,
        scorer_class,
        scorer_name,
        task_names,
        aggregation_name,
    ):
        """Test _process_single_judge_async with different combinations of metrics, scorers and aggregation modes."""
        # Create real scorer instances
        if scorer_class == AltTestScorer:
            scorer = AltTestScorer()
            scorer.min_instances_per_human = 1  # Set low threshold for testing
        else:
            scorer = scorer_class()

        # Create metric config
        config = MetricConfig(
            scorer=scorer, task_names=task_names, aggregation_name=aggregation_name
        )

        # Get first judge result for testing
        judge_result = list(judge_results_dict.values())[0]

        # Call the method
        result = await mock_evaluator._process_single_judge_async(
            config, judge_result, human_results_dict
        )

        # Verify result structure
        assert result.scorer_name == scorer_name
        assert result.judge_id == judge_result.judge_id

        # Verify task name based on aggregation mode
        if config.aggregation_mode == TaskAggregationMode.SINGLE:
            assert result.task_name == task_names[0]
        elif (
            config.aggregation_mode == TaskAggregationMode.MULTITASK
            and len(task_names) > 1
        ):
            assert "tasks_avg" in result.task_name
        elif (
            config.aggregation_mode == TaskAggregationMode.MULTILABEL
            and len(task_names) > 1
        ):
            assert "multilabel" in result.task_name

    @pytest.mark.asyncio
    async def test_compare_async_success(
        self, mock_evaluator, judge_results_dict, human_results_dict
    ):
        """Test successful compare_async."""
        # Mock the load methods
        mock_evaluator.load_all_judge_results = Mock(return_value=judge_results_dict)
        mock_evaluator.load_all_human_results = Mock(return_value=human_results_dict)

        # Use a real scorer
        scorer = AccuracyScorer()

        # Create config
        config = MetricsConfig(
            metrics=[
                MetricConfig(
                    scorer=scorer, task_names=["task1"], aggregation_name="single"
                )
            ]
        )

        # Run comparison
        results = await mock_evaluator._compare_async(config)

        # Verify results structure
        assert len(results) == 1
        unique_name = list(results.keys())[0]
        metric_config, scoring_results = results[unique_name]

        # Should have results for each judge
        assert len(scoring_results) == 2
        for result in scoring_results:
            assert result.scorer_name == "accuracy"
            assert result.judge_id in ["judge_1", "judge_2"]
            assert isinstance(result.scores.get("accuracy"), float)


@pytest.mark.parametrize(
    "scorer_class,num_humans",
    [
        (CohensKappaScorer, 1),  # Requires 2, test with 1
        (AltTestScorer, 2),  # Requires 3, test with 2
        # Note: We skip testing scorers that only need 1 human annotator (AccuracyScorer,
        # TextSimilarityScorer, SemanticSimilarityScorer) because when testing with 0 humans,
        # they hit the "No aligned data" error before the min_human_annotators validation
    ],
)
def test_min_human_annotators_validation_failure(
    mock_evaluator, scorer_class, num_humans
):
    """Test that scorers fail when insufficient human annotators are provided."""
    # Create scorer instance
    scorer = scorer_class()

    # Create metric config
    metric_config = MetricConfig(
        scorer=scorer, task_names=["task1"], aggregation_name="single"
    )

    # Create mock data with specified number of humans
    judge_result = Mock(spec=JudgeResults)
    judge_result.judge_id = "test_judge"

    # Setup mock data - always create judge data
    judge_df = pl.DataFrame({"original_id": ["1", "2"], "task1": ["A", "B"]})
    judge_result.get_successful_results.return_value = judge_df

    human_results = {}
    for i in range(num_humans):
        human_result = Mock(spec=HumanAnnotationResults)
        human_result.annotator_id = f"human_{i + 1}"
        human_result.get_successful_results.return_value = pl.DataFrame(
            {
                "original_id": ["1", "2"],
                "human_id": [f"human_{i + 1}", f"human_{i + 1}"],
                "task1": ["A", "B"],
            }
        )
        human_results[f"run_{i + 1}"] = human_result

    # Should return BaseScoringResult with error indicators when insufficient humans
    result = asyncio.run(
        mock_evaluator._process_single_judge_async(
            metric_config, judge_result, human_results
        )
    )

    # Verify the result indicates failure due to insufficient human annotators
    assert result.task_name == "error"
    assert result.failed_comparisons == 1
    assert result.num_comparisons == 0

    def test_compare_sync_wrapper(
        self, tmp_path, judge_results_dict, human_results_dict
    ):
        """Test that compare_async wrapper works without async/await."""
        # Use a real MetaEvaluator instance
        from meta_evaluator.meta_evaluator import MetaEvaluator

        evaluator = MetaEvaluator(project_dir=str(tmp_path / "test_project"))

        # Mock the load methods
        evaluator.load_all_judge_results = Mock(return_value=judge_results_dict)
        evaluator.load_all_human_results = Mock(return_value=human_results_dict)

        # Use a real scorer
        scorer = AccuracyScorer()

        # Create config
        config = MetricsConfig(
            metrics=[
                MetricConfig(
                    scorer=scorer, task_names=["task1"], aggregation_name="single"
                )
            ]
        )

        # Run comparison using sync wrapper (no await needed)
        results = evaluator.compare_async(config)

        # Verify results structure
        assert len(results) == 1
        unique_name = list(results.keys())[0]
        metric_config, scoring_results = results[unique_name]

        # Should have results for each judge
        assert len(scoring_results) == 2
        for result in scoring_results:
            assert result.scorer_name == "accuracy"
            assert result.judge_id in ["judge_1", "judge_2"]
            assert isinstance(result.scores.get("accuracy"), float)
