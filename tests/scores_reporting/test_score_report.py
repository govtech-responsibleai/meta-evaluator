"""Tests for ScoreReport functionality."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from meta_evaluator.scores import MetricConfig, MetricsConfig
from meta_evaluator.scores.base_scoring_result import BaseScoringResult
from meta_evaluator.scores.enums import TaskAggregationMode
from meta_evaluator.scores.metrics.agreement.alt_test import AltTestScorer
from meta_evaluator.scores.metrics.classification.classification_scorer import (
    ClassificationScorer,
)
from meta_evaluator.scores.metrics.text_comparison.text_similarity import (
    TextSimilarityScorer,
)
from meta_evaluator.scores_reporting.score_report import ScoreReport


class TestScoreReport:
    """Test ScoreReport functionality with mock data."""

    @pytest.fixture
    def temp_scores_dir(self):
        """Create temporary scores directory with mock results.

        Yields:
            tuple[Path, MetricsConfig]: Tuple containing the scores directory and metrics configuration
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            scores_dir = Path(temp_dir)

            # Create metrics config
            accuracy_scorer = ClassificationScorer(metric="accuracy")
            alt_test_scorer = AltTestScorer()
            text_sim_scorer = TextSimilarityScorer()

            metrics_config = MetricsConfig(
                metrics=[
                    MetricConfig(
                        scorer=accuracy_scorer,
                        task_names=["rejection"],
                        task_strategy="single",
                    ),
                    MetricConfig(
                        scorer=alt_test_scorer,
                        task_names=["rejection"],
                        task_strategy="single",
                    ),
                    MetricConfig(
                        scorer=text_sim_scorer,
                        task_names=["explanation"],
                        task_strategy="single",
                    ),
                ]
            )

            # Create mock results for each metric configuration
            judges = ["judge1", "judge2"]

            for metric_config in metrics_config.metrics:
                unique_name = metric_config.get_unique_name()
                scorer_name = metric_config.scorer.scorer_name

                # Create directory structure: scores/{scorer_name}/{unique_name}/
                result_dir = scores_dir / scorer_name / unique_name
                result_dir.mkdir(parents=True, exist_ok=True)

                for judge_id in judges:
                    # Create different scores based on scorer type
                    if scorer_name == "classification_accuracy":
                        scores = {"accuracy": 0.8 if judge_id == "judge1" else 0.7}
                    elif scorer_name == "alt_test":
                        scores = {
                            "winning_rate": {
                                "0.20": 0.6 if judge_id == "judge1" else 0.4
                            },
                            "advantage_probability": 0.75
                            if judge_id == "judge1"
                            else 0.65,
                        }
                    elif scorer_name == "text_similarity":
                        scores = {"similarity": 0.85 if judge_id == "judge1" else 0.75}
                    else:
                        scores = {}

                    # Create BaseScoringResult
                    result = BaseScoringResult(
                        scorer_name=scorer_name,
                        task_name=metric_config.task_names[0],
                        judge_id=judge_id,
                        scores=scores,
                        aggregation_mode=TaskAggregationMode.SINGLE,
                        num_comparisons=10,
                        failed_comparisons=0,
                    )

                    # Save result to JSON file
                    result_file = result_dir / f"{judge_id}_result.json"
                    result.save_state(str(result_file))

            yield scores_dir, metrics_config

    def test_summary_report(self, temp_scores_dir):
        """Test that summary report contains all expected labels and they are displayed correctly."""
        scores_dir, metrics_config = temp_scores_dir

        # Create ScoreReport
        report = ScoreReport(scores_dir, metrics_config)

        # Generate report
        df = report.generate()

        # Verify DataFrame structure
        assert len(df) == 2  # 2 judges
        assert "judge_id" in df.columns

        # Expected columns: judge_id + 4 metric columns
        # - accuracy_xxx_single
        # - alt_test_xxx_single_winning_rate
        # - alt_test_xxx_single_advantage_prob
        # - text_similarity_xxx_single
        expected_column_count = 5  # judge_id + 4 metric columns
        assert len(df.columns) == expected_column_count

        # Verify all judges are present
        judge_ids = df["judge_id"].to_list()
        assert set(judge_ids) == {"judge1", "judge2"}

        # Verify judge1 has highest scores (as set in mock data)
        judge1_row = df.filter(pl.col("judge_id") == "judge1").to_dicts()[0]

        # Find accuracy column and verify score
        accuracy_cols = [
            col for col in df.columns if col.startswith("classification_accuracy_")
        ]
        assert len(accuracy_cols) == 1
        assert judge1_row[accuracy_cols[0]] == 0.8

        # Find alt_test columns and verify scores
        alt_test_winning_cols = [col for col in df.columns if "winning_rate" in col]
        alt_test_advantage_cols = [col for col in df.columns if "advantage_prob" in col]
        assert len(alt_test_winning_cols) == 1
        assert len(alt_test_advantage_cols) == 1
        assert judge1_row[alt_test_winning_cols[0]] == 0.6
        assert judge1_row[alt_test_advantage_cols[0]] == 0.75

        # Find text_similarity column and verify score
        text_sim_cols = [
            col for col in df.columns if col.startswith("text_similarity_")
        ]
        assert len(text_sim_cols) == 1
        assert judge1_row[text_sim_cols[0]] == 0.85

        # Verify HTML report generation works
        html_report = report.save("test_report.html", format="html")
        assert "<table" in html_report
        assert "judge1" in html_report
        assert "judge2" in html_report
        assert "best-score" in html_report

        # Verify CSV report generation works
        csv_report = report.save("test_report.csv", format="csv")
        assert "judge_id" in csv_report
        assert "judge1" in csv_report
        assert "judge2" in csv_report
