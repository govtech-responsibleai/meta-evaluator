"""Tests for agreement metrics."""

import pytest
import numpy as np
import polars as pl
from meta_evaluator.scores.exceptions import AltTestInsufficientAnnotationsError


# Basic fixtures are now provided by conftest.py


class TestCohensKappaScorer:
    """Test CohensKappaScorer functionality."""

    def test_can_score_classification(self, cohens_kappa_scorer):
        """Test that CohensKappaScorer can handle classification tasks."""
        # Should accept classification tasks
        assert cohens_kappa_scorer.can_score_task(["A", "B", "C"]) is True
        assert cohens_kappa_scorer.can_score_task(["positive", "negative"]) is True
        assert cohens_kappa_scorer.can_score_task(["class1"]) is True  # Single class

    def test_cannot_score_free_form(self, cohens_kappa_scorer):
        """Test that CohensKappaScorer cannot handle free-form text tasks."""
        # Should reject text tasks
        assert cohens_kappa_scorer.can_score_task(None) is False

    def test_compute_classification_kappa_perfect_agreement(
        self, cohens_kappa_scorer, basic_judge_df, basic_human_df
    ):
        """Test Cohen's kappa with perfect agreement on classification task."""
        kappa = cohens_kappa_scorer._compute_classification_kappa(
            basic_judge_df, basic_human_df, "task1"
        )
        assert kappa == 1.0

    def test_compute_classification_kappa_no_agreement(self, cohens_kappa_scorer):
        """Test Cohen's kappa with no agreement on classification task."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "judge_id": ["judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": ["A", "B", "C"],
            }
        )

        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "annotator_id": ["human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": ["C", "A", "B"],
            }
        )

        kappa = cohens_kappa_scorer._compute_classification_kappa(
            judge_df, human_df, "task1"
        )
        # Should be around 0 or negative for no agreement
        assert kappa <= 0.1

    def test_compute_classification_kappa_partial_agreement(self, cohens_kappa_scorer):
        """Test Cohen's kappa with partial agreement on classification task."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3", "4"],
                "judge_id": ["judge_1", "judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task1", "task1"],
                "task_value": ["A", "B", "C", "A"],
            }
        )

        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3", "4"],
                "annotator_id": ["human_1", "human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task1", "task1"],
                "task_value": ["A", "B", "A", "A"],  # 3 out of 4 match
            }
        )

        kappa = cohens_kappa_scorer._compute_classification_kappa(
            judge_df, human_df, "task1"
        )
        # Should be positive but less than 1
        assert 0.0 < kappa < 1.0

    def test_compute_score_single_task(
        self, cohens_kappa_scorer, sample_judge_df, sample_human_df
    ):
        """Test compute_score method for single classification task."""
        task_schemas = {"task1": ["A", "B", "C"]}
        result = cohens_kappa_scorer.compute_score(
            "judge_1", sample_judge_df, sample_human_df, ["task1"], task_schemas
        )

        assert result.scorer_name == "cohens_kappa"
        assert result.task_name == "task1"
        assert result.judge_id == "judge_1"
        assert result.score == 1.0
        assert result.metadata["scoring_method"] == "single_task"

    def test_compute_score_multi_task(self, cohens_kappa_scorer):
        """Test compute_score method for multiple tasks."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3", "4", "1", "2", "3", "4"],
                "judge_id": ["judge_1"] * 8,
                "task_name": [
                    "task1",
                    "task1",
                    "task1",
                    "task1",
                    "task2",
                    "task2",
                    "task2",
                    "task2",
                ],
                "task_value": ["A", "B", "C", "D", "X", "Y", "Z", "W"],
            }
        )

        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3", "4", "1", "2", "3", "4"],
                "annotator_id": ["human_1"] * 8,
                "task_name": [
                    "task1",
                    "task1",
                    "task1",
                    "task1",
                    "task2",
                    "task2",
                    "task2",
                    "task2",
                ],
                "task_value": [
                    "A",
                    "A",
                    "C",
                    "D",  # task1: Partial agreement with judge above
                    "X",
                    "Y",
                    "Z",
                    "W",  # task2: Perfect agreement with judge above
                ],
            }
        )

        task_schemas = {"task1": ["A", "B", "C"], "task2": ["X", "Y", "Z"]}
        result = cohens_kappa_scorer.compute_score(
            "judge_1", judge_df, human_df, ["task1", "task2"], task_schemas
        )

        assert result.scorer_name == "cohens_kappa"
        assert result.task_name == "2_tasks_avg"
        assert result.judge_id == "judge_1"
        assert 0.0 < result.score < 1.0  # Average of partial and perfect agreement
        assert result.metadata["scoring_method"] == "average_across_tasks"

    def test_all_null_task_values(
        self, cohens_kappa_scorer, null_values_judge_df, null_values_human_df
    ):
        """Test when all judge/human values are None for a task."""
        judge_df = null_values_judge_df
        human_df = null_values_human_df

        # Should return np.nan when all values are null
        kappa = cohens_kappa_scorer._compute_classification_kappa(
            judge_df, human_df, "task1"
        )
        assert np.isnan(kappa)

    def test_mixed_null_and_valid_data(self, cohens_kappa_scorer):
        """Test some valid, some null values in same task."""
        judge_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "judge_id": ["judge_1", "judge_1", "judge_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": [None, "B", "C"],
            }
        )

        human_df = pl.DataFrame(
            {
                "original_id": ["1", "2", "3"],
                "annotator_id": ["human_1", "human_1", "human_1"],
                "task_name": ["task1", "task1", "task1"],
                "task_value": ["A", None, "C"],
            }
        )

        # Should compute kappa only on valid pairs (should be only 1 valid pair: "C"=="C")
        kappa = cohens_kappa_scorer._compute_classification_kappa(
            judge_df, human_df, "task1"
        )
        # With only 1 valid pair, should return np.nan (Cohen's kappa requires >=2 samples)
        assert np.isnan(kappa)

    def test_compute_classification_kappa_multiple_humans(
        self, cohens_kappa_scorer, basic_judge_df, multi_human_df
    ):
        """Test Cohen's kappa with multiple human annotators."""
        kappa = cohens_kappa_scorer._compute_classification_kappa(
            basic_judge_df, multi_human_df, "task1"
        )
        # Should handle multiple humans gracefully
        assert isinstance(kappa, float)
        assert 0.0 <= kappa <= 1.0

    def test_aggregate_results_saves_individual_results(
        self, cohens_kappa_scorer, tmp_path
    ):
        """Test that aggregate_results saves individual results as JSON files."""
        from meta_evaluator.scores.base_scoring_result import BaseScoringResult

        # Create mock results for 2 judges
        results = [
            BaseScoringResult(
                scorer_name="cohens_kappa",
                task_name="task1",
                judge_id="judge_1",
                score=0.75,
                metadata={
                    "task_names": ["task1"],
                    "task_schemas": {"task1": ["A", "B", "C"]},
                },
            ),
            BaseScoringResult(
                scorer_name="cohens_kappa",
                task_name="task1",
                judge_id="judge_2",
                score=0.68,
                metadata={
                    "task_names": ["task1"],
                    "task_schemas": {"task1": ["A", "B", "C"]},
                },
            ),
        ]

        scores_dir = str(tmp_path)

        # Call aggregate_results
        cohens_kappa_scorer.aggregate_results(results, scores_dir)

        # Verify cohens_kappa directory was created
        cohens_kappa_dir = tmp_path / "cohens_kappa"
        assert cohens_kappa_dir.exists()

        # Verify individual result files were saved
        result_files = list(cohens_kappa_dir.glob("*_result.json"))
        assert len(result_files) == 2

        # Verify file naming convention
        expected_files = {"judge_1_task1_result.json", "judge_2_task1_result.json"}
        actual_files = {f.name for f in result_files}
        assert actual_files == expected_files

        # Verify we can load back the results
        for result_file in result_files:
            loaded_result = BaseScoringResult.load_state(str(result_file))
            assert loaded_result.scorer_name == "cohens_kappa"
            assert loaded_result.task_name == "task1"
            assert loaded_result.judge_id in ["judge_1", "judge_2"]
            assert isinstance(loaded_result.score, float)

    def test_aggregate_results_handles_empty_list(self, cohens_kappa_scorer, tmp_path):
        """Test that aggregate_results handles empty results list gracefully."""
        scores_dir = str(tmp_path)

        # Should not crash with empty list
        cohens_kappa_scorer.aggregate_results([], scores_dir)

        # Should not create directory
        cohens_kappa_dir = tmp_path / "cohens_kappa"
        assert not cohens_kappa_dir.exists()


class TestAltTestScorer:
    """Test AltTestScorer functionality."""

    # Single task fixtures are now provided by conftest.py

    # Multi-task fixtures are now provided by conftest.py

    # Text task fixtures are now provided by conftest.py

    def test_can_score_any_task(self, alt_test_scorer):
        """Test that AltTestScorer can handle any task type."""
        assert alt_test_scorer.can_score_task(["A", "B", "C"]) is True  # Classification
        assert alt_test_scorer.can_score_task(None) is True  # Text

    def test_convert_single_classification_annotations(
        self, alt_test_scorer, single_task_judge_df, single_task_human_df
    ):
        """Test annotation conversion for single classification task."""
        task_names = ["safety"]
        task_schemas = {"safety": ["SAFE", "UNSAFE"]}

        # Test judge conversion
        judge_annotations = alt_test_scorer._convert_judge_to_alttest_format(
            single_task_judge_df, task_names, task_schemas
        )
        expected_judge = {"1": "SAFE", "2": "UNSAFE", "3": "SAFE", "4": "UNSAFE"}
        assert judge_annotations == expected_judge

        # Test human conversion
        human_annotations = alt_test_scorer._convert_humans_to_alttest_format(
            single_task_human_df, task_names, task_schemas
        )
        expected_human = {
            "human_1": {"1": "SAFE", "2": "UNSAFE", "3": "SAFE", "4": "SAFE"},
            "human_2": {"1": "SAFE", "2": "SAFE", "3": "UNSAFE", "4": "UNSAFE"},
        }
        assert human_annotations == expected_human

    def test_convert_multilabel_annotations(
        self, alt_test_scorer, multi_task_judge_df, multi_task_human_df
    ):
        """Test annotation conversion for multi-label classification task."""
        task_names = ["safety", "toxicity"]
        task_schemas = {
            "safety": ["SAFE", "UNSAFE"],
            "toxicity": ["TOXIC", "NON_TOXIC"],
        }

        # Test judge conversion
        judge_annotations = alt_test_scorer._convert_judge_to_alttest_format(
            multi_task_judge_df, task_names, task_schemas
        )
        expected_judge = {
            "1": ["SAFE", "NON_TOXIC"],
            "2": ["UNSAFE", "TOXIC"],
            "3": ["UNSAFE", "TOXIC"],
            "4": ["SAFE", "NON_TOXIC"],
            "5": ["SAFE", "NON_TOXIC"],
            "6": ["UNSAFE", "TOXIC"],
        }
        assert judge_annotations == expected_judge

        # Test human conversion
        human_annotations = alt_test_scorer._convert_humans_to_alttest_format(
            multi_task_human_df, task_names, task_schemas
        )
        expected_human = {
            "human_1": {
                "1": ["SAFE", "NON_TOXIC"],
                "2": ["SAFE", "NON_TOXIC"],
                "3": ["UNSAFE", "TOXIC"],
                "4": ["UNSAFE", "TOXIC"],
                "5": ["SAFE", "NON_TOXIC"],
                "6": ["SAFE", "NON_TOXIC"],
            },
            "human_2": {
                "1": ["SAFE", "NON_TOXIC"],
                "2": ["UNSAFE", "TOXIC"],
                "3": ["UNSAFE", "TOXIC"],
                "4": ["SAFE", "NON_TOXIC"],
                "5": ["SAFE", "NON_TOXIC"],
                "6": ["UNSAFE", "TOXIC"],
            },
        }
        assert human_annotations == expected_human

    def test_convert_text_annotations(
        self, alt_test_scorer, text_task_judge_df, text_task_human_df
    ):
        """Test annotation conversion for text task."""
        task_names = ["summary"]
        task_schemas = {"summary": None}  # Text task

        # Test judge conversion
        judge_annotations = alt_test_scorer._convert_judge_to_alttest_format(
            text_task_judge_df, task_names, task_schemas
        )
        expected_judge = {"1": "Good", "2": "Bad", "3": "OK"}
        assert judge_annotations == expected_judge

        # Test human conversion
        human_annotations = alt_test_scorer._convert_humans_to_alttest_format(
            text_task_human_df, task_names, task_schemas
        )
        expected_human = {
            "human_1": {
                "1": "Good",
                "2": "Bad",
                "3": "OK",
                "4": "Great",
                "5": "Poor",
                "6": "OK",
            },
            "human_2": {
                "1": "Good",
                "2": "Bad",
                "3": "OK",
                "4": "Great",
                "5": "Poor",
                "6": "OK",
            },
        }
        assert human_annotations == expected_human

    def test_determine_scoring_function(self, alt_test_scorer):
        """Test automatic scoring function determination."""
        # Single classification task -> accuracy
        assert (
            alt_test_scorer._determine_scoring_function(
                ["safety"], {"safety": ["SAFE", "UNSAFE"]}
            )
            == "accuracy"
        )

        # Multi-label classification -> jaccard_similarity
        assert (
            alt_test_scorer._determine_scoring_function(
                ["safety", "toxicity"],
                {"safety": ["SAFE", "UNSAFE"], "toxicity": ["TOXIC", "NON_TOXIC"]},
            )
            == "jaccard_similarity"
        )

        # Text task -> accuracy
        assert (
            alt_test_scorer._determine_scoring_function(["summary"], {"summary": None})
            == "accuracy"
        )

        # Mixed task (text + classification) -> accuracy
        assert (
            alt_test_scorer._determine_scoring_function(
                ["summary", "safety"], {"summary": None, "safety": ["SAFE", "UNSAFE"]}
            )
            == "accuracy"
        )

    def test_alt_test_single_task_metrics(
        self, alt_test_scorer, single_task_judge_df, single_task_human_df
    ):
        """Test that alt-test produces correct metrics for single task."""
        task_names = ["safety"]
        task_schemas = {"safety": ["SAFE", "UNSAFE"]}

        result = alt_test_scorer.compute_score(
            "judge_1",
            single_task_judge_df,
            single_task_human_df,
            task_names,
            task_schemas,
        )

        # Check basic properties
        assert result.scorer_name == "alt_test"
        assert result.task_name == "safety"
        assert result.judge_id == "judge_1"
        assert isinstance(result.score, float)  # winning_rate
        assert 0.0 <= result.score <= 1.0

        # Check metadata
        assert "advantage_probability" in result.metadata
        assert "human_advantage_probabilities" in result.metadata
        assert "scoring_function" in result.metadata
        assert result.metadata["scoring_function"] == "accuracy"

    def test_alt_test_multi_task_metrics(
        self, alt_test_scorer, multi_task_judge_df, multi_task_human_df
    ):
        """Test that alt-test produces correct metrics for multi-task."""
        task_names = ["safety", "toxicity"]
        task_schemas = {
            "safety": ["SAFE", "UNSAFE"],
            "toxicity": ["TOXIC", "NON_TOXIC"],
        }

        result = alt_test_scorer.compute_score(
            "judge_1",
            multi_task_judge_df,
            multi_task_human_df,
            task_names,
            task_schemas,
        )

        # Check basic properties
        assert result.scorer_name == "alt_test"
        assert result.task_name == "2_tasks_combined"
        assert result.judge_id == "judge_1"
        assert isinstance(result.score, float)  # winning_rate
        assert 0.0 <= result.score <= 1.0

        # Check metadata
        assert result.metadata["scoring_function"] == "jaccard_similarity"

    def test_is_multilabel_task(self, alt_test_scorer):
        """Test multilabel task detection."""
        # Single classification task
        assert (
            alt_test_scorer._is_multilabel_task(
                ["safety"], {"safety": ["SAFE", "UNSAFE"]}
            )
            is False
        )

        # Multiple classification tasks (multilabel)
        assert (
            alt_test_scorer._is_multilabel_task(
                ["safety", "toxicity"],
                {"safety": ["SAFE", "UNSAFE"], "toxicity": ["TOXIC", "NON_TOXIC"]},
            )
            is True
        )

        # Text task
        assert (
            alt_test_scorer._is_multilabel_task(["summary"], {"summary": None}) is False
        )

        # Mixed tasks
        assert (
            alt_test_scorer._is_multilabel_task(
                ["summary", "safety"], {"summary": None, "safety": ["SAFE", "UNSAFE"]}
            )
            is False
        )

    def test_alt_test_core_algorithm_success(self, alt_test_scorer):
        """Test that the core alt-test algorithm produces reasonable results when thresholds are met."""
        # Create simple test data where judge perfectly matches one human
        judge_annotations = {"1": "A", "2": "B", "3": "A"}
        human_annotations = {
            "human_1": {"1": "A", "2": "B", "3": "A"},  # Perfect match
            "human_2": {"1": "B", "2": "A", "3": "B"},  # Complete mismatch
        }

        winning_rate, advantage_prob, human_advantage_probs = alt_test_scorer._alt_test(
            judge_annotations, human_annotations, "accuracy"
        )

        # Should return reasonable values
        assert 0.0 <= winning_rate <= 1.0
        assert 0.0 <= advantage_prob <= 1.0
        assert len(human_advantage_probs) == 2

        # Each human should have advantage probability data
        for human_id in ["human_1", "human_2"]:
            assert human_id in human_advantage_probs
            llm_adv, human_adv = human_advantage_probs[human_id]
            assert 0.0 <= llm_adv <= 1.0
            assert 0.0 <= human_adv <= 1.0

    def test_alt_test_core_algorithm_insufficient_data(self, alt_test_scorer):
        """Test that the core alt-test algorithm raises error when data is insufficient."""
        # Set min_instances_per_human to 10
        alt_test_scorer.min_instances_per_human = 10

        # Create simple test data where judge perfectly matches one human
        judge_annotations = {"1": "A", "2": "B", "3": "A"}
        human_annotations = {
            "human_1": {"1": "A", "2": "B", "3": "A"},  # Perfect match
            "human_2": {"1": "B", "2": "A", "3": "B"},  # Complete mismatch
        }

        # With only 3 instances and min_instances_per_human = 10, should raise error
        with pytest.raises(
            AltTestInsufficientAnnotationsError,
            match="No annotators meet the minimum threshold",
        ):
            alt_test_scorer._alt_test(judge_annotations, human_annotations, "accuracy")

    def test_aggregate_results_generates_plots(
        self, alt_test_scorer, single_task_judge_df, single_task_human_df, tmp_path
    ):
        """Test that aggregate_results generates 3 aggregate plots."""
        task_names = ["safety"]
        task_schemas = {"safety": ["SAFE", "UNSAFE"]}

        # Create multiple scoring results (simulating multiple judges)
        result1 = alt_test_scorer.compute_score(
            "judge_1",
            single_task_judge_df,
            single_task_human_df,
            task_names,
            task_schemas,
        )
        result2 = alt_test_scorer.compute_score(
            "judge_2",
            single_task_judge_df,
            single_task_human_df,
            task_names,
            task_schemas,
        )

        results = [result1, result2]

        scores_dir = str(tmp_path)

        # Call aggregate_results
        alt_test_scorer.aggregate_results(results, scores_dir)

        # Check that alt_test directory was created
        alt_test_dir = tmp_path / "alt_test"
        assert alt_test_dir.exists()

        # Check that 3 aggregate plots were generated
        expected_aggregate_plots = [
            "aggregate_winning_rates.png",
            "aggregate_advantage_probabilities.png",
            "aggregate_human_vs_llm_advantage.png",
        ]

        for plot_name in expected_aggregate_plots:
            plot_path = alt_test_dir / plot_name
            assert plot_path.exists(), f"Missing aggregate plot: {plot_name}"

    def test_aggregate_results_single_judge(
        self, alt_test_scorer, single_task_judge_df, single_task_human_df, tmp_path
    ):
        """Test that aggregate_results works with single judge."""
        task_names = ["safety"]
        task_schemas = {"safety": ["SAFE", "UNSAFE"]}

        result = alt_test_scorer.compute_score(
            "judge_1",
            single_task_judge_df,
            single_task_human_df,
            task_names,
            task_schemas,
        )

        scores_dir = str(tmp_path)

        # Should work with single judge too
        alt_test_scorer.aggregate_results([result], scores_dir)

        # Check that plots were generated
        alt_test_dir = tmp_path / "alt_test"
        assert alt_test_dir.exists()

        # Should still generate all 3 plots even with 1 judge
        expected_plots = [
            "aggregate_winning_rates.png",
            "aggregate_advantage_probabilities.png",
            "aggregate_human_vs_llm_advantage.png",
        ]

        for plot_name in expected_plots:
            plot_path = alt_test_dir / plot_name
            assert plot_path.exists(), f"Missing plot: {plot_name}"

    def test_aggregate_results_saves_individual_results(
        self, alt_test_scorer, single_task_judge_df, single_task_human_df, tmp_path
    ):
        """Test that aggregate_results saves individual results as JSON files."""
        task_names = ["safety"]
        task_schemas = {"safety": ["SAFE", "UNSAFE"]}

        # Create multiple scoring results (simulating multiple judges)
        result1 = alt_test_scorer.compute_score(
            "judge_1",
            single_task_judge_df,
            single_task_human_df,
            task_names,
            task_schemas,
        )
        result2 = alt_test_scorer.compute_score(
            "judge_2",
            single_task_judge_df.with_columns(pl.lit("judge_2").alias("judge_id")),
            single_task_human_df,
            task_names,
            task_schemas,
        )

        results = [result1, result2]
        scores_dir = str(tmp_path)

        # Call aggregate_results
        alt_test_scorer.aggregate_results(results, scores_dir)

        # Verify alt_test directory was created
        alt_test_dir = tmp_path / "alt_test"
        assert alt_test_dir.exists()

        # Verify individual result files were saved
        result_files = list(alt_test_dir.glob("*_result.json"))
        assert len(result_files) == 2

        # Verify file naming convention
        expected_files = {"judge_1_safety_result.json", "judge_2_safety_result.json"}
        actual_files = {f.name for f in result_files}
        assert actual_files == expected_files

        # Verify we can load back the results
        from meta_evaluator.scores.base_scoring_result import BaseScoringResult

        for result_file in result_files:
            loaded_result = BaseScoringResult.load_state(str(result_file))
            assert loaded_result.scorer_name == "alt_test"
            assert loaded_result.task_name == "safety"
            assert loaded_result.judge_id in ["judge_1", "judge_2"]
            assert isinstance(loaded_result.score, float)
            assert "advantage_probability" in loaded_result.metadata
            assert "scoring_function" in loaded_result.metadata

    def test_aggregate_results_empty_list(self, alt_test_scorer, tmp_path):
        """Test that aggregate_results handles empty results list gracefully."""
        scores_dir = str(tmp_path)

        # Should not crash with empty list
        alt_test_scorer.aggregate_results([], scores_dir)

        # Should not create directory
        alt_test_dir = tmp_path / "alt_test"
        assert not alt_test_dir.exists()
