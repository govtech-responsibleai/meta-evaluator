"""Tests for metrics configuration validation."""

import pytest

from meta_evaluator.scores.exceptions import InvalidAggregationModeError
from meta_evaluator.scores.metrics_config import MetricConfig


class TestMetricConfigValidation:
    """Test class for MetricConfig validation rules."""

    def test_single_aggregation_with_one_task_succeeds(self, accuracy_scorer):
        """Test that 'single' aggregation mode works with exactly 1 task name."""
        config = MetricConfig(
            scorer=accuracy_scorer, task_names=["sentiment"], aggregation_name="single"
        )
        assert config.aggregation_name == "single"
        assert len(config.task_names) == 1

    def test_single_aggregation_with_multiple_tasks_fails(self, accuracy_scorer):
        """Test that 'single' aggregation mode fails with multiple task names."""
        with pytest.raises(InvalidAggregationModeError) as exc_info:
            MetricConfig(
                scorer=accuracy_scorer,
                task_names=["sentiment", "toxicity"],
                aggregation_name="single",
            )

        error_message = str(exc_info.value)
        assert (
            "Aggregation mode 'single' can only be used with exactly 1 task name"
            in error_message
        )
        assert "got 2 task names" in error_message
        assert "sentiment" in error_message
        assert "toxicity" in error_message

    def test_single_aggregation_with_zero_tasks_fails(self, accuracy_scorer):
        """Test that 'single' aggregation mode fails with zero task names."""
        with pytest.raises(InvalidAggregationModeError) as exc_info:
            MetricConfig(
                scorer=accuracy_scorer, task_names=[], aggregation_name="single"
            )

        error_message = str(exc_info.value)
        assert (
            "Aggregation mode 'single' can only be used with exactly 1 task name"
            in error_message
        )
        assert "got 0 task names" in error_message

    def test_multilabel_aggregation_with_multiple_tasks_succeeds(self, accuracy_scorer):
        """Test that 'multilabel' aggregation mode works with multiple task names."""
        config = MetricConfig(
            scorer=accuracy_scorer,
            task_names=["sentiment", "toxicity"],
            aggregation_name="multilabel",
        )
        assert config.aggregation_name == "multilabel"
        assert len(config.task_names) == 2

    def test_multilabel_aggregation_with_one_task_fails(self, accuracy_scorer):
        """Test that 'multilabel' aggregation mode fails with single task name."""
        with pytest.raises(InvalidAggregationModeError) as exc_info:
            MetricConfig(
                scorer=accuracy_scorer,
                task_names=["sentiment"],
                aggregation_name="multilabel",
            )

        error_message = str(exc_info.value)
        assert (
            "Aggregation mode 'multilabel' and 'multitask' can only be used with more than 1 task name"
            in error_message
        )
        assert "got 1 task names" in error_message
        assert "sentiment" in error_message

    def test_multilabel_aggregation_with_zero_tasks_fails(self, accuracy_scorer):
        """Test that 'multilabel' aggregation mode fails with zero task names."""
        with pytest.raises(InvalidAggregationModeError) as exc_info:
            MetricConfig(
                scorer=accuracy_scorer, task_names=[], aggregation_name="multilabel"
            )

        error_message = str(exc_info.value)
        assert (
            "Aggregation mode 'multilabel' and 'multitask' can only be used with more than 1 task name"
            in error_message
        )
        assert "got 0 task names" in error_message

    def test_multitask_aggregation_with_multiple_tasks_succeeds(self, accuracy_scorer):
        """Test that 'multitask' aggregation mode works with multiple task names."""
        config = MetricConfig(
            scorer=accuracy_scorer,
            task_names=["sentiment", "toxicity", "safety"],
            aggregation_name="multitask",
        )
        assert config.aggregation_name == "multitask"
        assert len(config.task_names) == 3

    def test_multitask_aggregation_with_one_task_fails(self, accuracy_scorer):
        """Test that 'multitask' aggregation mode fails with single task name."""
        with pytest.raises(InvalidAggregationModeError) as exc_info:
            MetricConfig(
                scorer=accuracy_scorer,
                task_names=["sentiment"],
                aggregation_name="multitask",
            )

        error_message = str(exc_info.value)
        assert (
            "Aggregation mode 'multilabel' and 'multitask' can only be used with more than 1 task name"
            in error_message
        )
        assert "got 1 task names" in error_message
        assert "sentiment" in error_message

    def test_multitask_aggregation_with_zero_tasks_fails(self, accuracy_scorer):
        """Test that 'multitask' aggregation mode fails with zero task names."""
        with pytest.raises(InvalidAggregationModeError) as exc_info:
            MetricConfig(
                scorer=accuracy_scorer, task_names=[], aggregation_name="multitask"
            )

        error_message = str(exc_info.value)
        assert (
            "Aggregation mode 'multilabel' and 'multitask' can only be used with more than 1 task name"
            in error_message
        )
        assert "got 0 task names" in error_message

    def test_aggregation_mode_computed_field_still_works(self, accuracy_scorer):
        """Test that the aggregation_mode computed field still works after validation."""
        config = MetricConfig(
            scorer=accuracy_scorer, task_names=["sentiment"], aggregation_name="single"
        )

        # Test that the computed field works
        from meta_evaluator.scores.enums import TaskAggregationMode

        assert config.aggregation_mode == TaskAggregationMode.SINGLE

    def test_get_unique_name_still_works(self, accuracy_scorer):
        """Test that get_unique_name method still works after validation."""
        config = MetricConfig(
            scorer=accuracy_scorer, task_names=["sentiment"], aggregation_name="single"
        )

        unique_name = config.get_unique_name()
        assert isinstance(unique_name, str)
        assert "accuracy" in unique_name
        assert "1tasks" in unique_name
        assert "single" in unique_name or "SINGLE" in unique_name
