"""Test suite for JudgeResultsBuilder with comprehensive path coverage."""

import pytest
from datetime import datetime

from meta_evaluator.judge.models import (
    JudgeResultsConfig,
    JudgeResultsBuilder,
)
from meta_evaluator.llm_client import LLMClientEnum


class TestJudgeResultsBuilder:
    """Comprehensive test suite for JudgeResultsBuilder achieving 100% path coverage."""

    @pytest.fixture
    def base_config(self) -> JudgeResultsConfig:
        """Provides a valid JudgeResultsConfig for testing.

        Returns:
            JudgeResultsConfig: A valid configuration instance.
        """
        return JudgeResultsConfig(
            run_id="test_run_123",
            judge_id="test_judge_1",
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            llm_client_enum=LLMClientEnum.OPENAI,
            model_used="gpt-4",
            timestamp_local=datetime.now(),
            is_sampled_run=False,
            expected_ids=["id1", "id2", "id3"],
        )

    @pytest.fixture
    def multi_task_config(self) -> JudgeResultsConfig:
        """Provides a config with multiple tasks.

        Returns:
            JudgeResultsConfig: A configuration with multiple tasks.
        """
        return JudgeResultsConfig(
            run_id="multi_task_run",
            judge_id="multi_judge",
            task_schemas={
                "sentiment": ["positive", "negative", "neutral"],
                "toxicity": ["toxic", "non_toxic"],
            },
            llm_client_enum=LLMClientEnum.OPENAI,
            model_used="gpt-4",
            timestamp_local=datetime.now(),
            is_sampled_run=True,
            expected_ids=["id1", "id2"],
        )

    @pytest.fixture
    def single_task_config(self) -> JudgeResultsConfig:
        """Provides a config with single task.

        Returns:
            JudgeResultsConfig: A configuration with single task.
        """
        return JudgeResultsConfig(
            run_id="single_task_run",
            judge_id="single_judge",
            task_schemas={"sentiment": ["positive", "negative"]},
            llm_client_enum=LLMClientEnum.ANTHROPIC,
            model_used="claude-3",
            timestamp_local=datetime.now(),
            is_sampled_run=False,
            expected_ids=["id1"],
        )

    # === Initialization Tests ===

    def test_initialization_happy_path(self, base_config):
        """Test successful builder initialization."""
        builder = JudgeResultsBuilder(base_config)

        assert builder.config == base_config
        assert builder.total_count == 3
        assert builder.completed_count == 0
        assert not builder.is_complete

    def test_dynamic_class_creation_single_task(self, single_task_config):
        """Test _create_result_row_class with single task."""
        builder = JudgeResultsBuilder(single_task_config)
        result_class = builder._result_row_class

        # Verify the class has the task field
        assert hasattr(result_class, "model_fields")
        assert "sentiment" in result_class.model_fields

    def test_dynamic_class_creation_multiple_tasks(self, multi_task_config):
        """Test _create_result_row_class with multiple tasks."""
        builder = JudgeResultsBuilder(multi_task_config)
        result_class = builder._result_row_class

        # Verify the class has all task fields
        assert "sentiment" in result_class.model_fields
        assert "toxicity" in result_class.model_fields

    # === Property Access Tests ===

    def test_completed_count_property(self, base_config):
        """Test completed_count property."""
        builder = JudgeResultsBuilder(base_config)
        assert builder.completed_count == 0

        # Add a success row
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response="positive sentiment",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            call_duration=1.0,
        )
        assert builder.completed_count == 1

    def test_total_count_property(self, base_config):
        """Test total_count property."""
        builder = JudgeResultsBuilder(base_config)
        assert builder.total_count == 3

    def test_is_complete_property_false(self, base_config):
        """Test is_complete property returns False when incomplete."""
        builder = JudgeResultsBuilder(base_config)
        assert not builder.is_complete

    def test_is_complete_property_true(self, single_task_config):
        """Test is_complete property returns True when complete."""
        builder = JudgeResultsBuilder(single_task_config)

        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response="positive",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            call_duration=1.0,
        )
        assert builder.is_complete

    # === Success Row Creation Tests ===

    def test_create_success_row_happy_path(self, base_config):
        """Test successful creation of success row."""
        builder = JudgeResultsBuilder(base_config)

        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response="positive sentiment detected",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            call_duration=1.5,
        )

        assert builder.completed_count == 1

    def test_create_success_row_missing_tasks_error(self, base_config):
        """Test error when missing tasks in success row."""
        builder = JudgeResultsBuilder(base_config)

        with pytest.raises(
            ValueError, match="Success row must contain outcomes for ALL tasks"
        ):
            builder.create_success_row(
                sample_example_id="test_1",
                original_id="id1",
                outcomes={},  # Missing sentiment task
                llm_raw_response="response",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                call_duration=1.0,
            )

    def test_create_success_row_extra_tasks_error(self, base_config):
        """Test error when extra tasks in success row."""
        builder = JudgeResultsBuilder(base_config)

        with pytest.raises(
            ValueError, match="Success row must contain outcomes for ALL tasks"
        ):
            builder.create_success_row(
                sample_example_id="test_1",
                original_id="id1",
                outcomes={"sentiment": "positive", "extra_task": "value"},
                llm_raw_response="response",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                call_duration=1.0,
            )

    # === Partial Row Creation Tests ===

    def test_create_partial_row_happy_path(self, multi_task_config):
        """Test successful creation of partial row."""
        builder = JudgeResultsBuilder(multi_task_config)

        builder.create_partial_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},  # Missing toxicity
            error_message="Failed to parse toxicity",
            llm_raw_response="partial response",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            call_duration=1.0,
        )

        assert builder.completed_count == 1

    def test_create_partial_row_invalid_task_names_error(self, base_config):
        """Test error when invalid task names in partial row."""
        builder = JudgeResultsBuilder(base_config)

        with pytest.raises(ValueError, match="Invalid task names"):
            builder.create_partial_row(
                sample_example_id="test_1",
                original_id="id1",
                outcomes={"invalid_task": "value"},
                error_message="error",
                llm_raw_response="response",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                call_duration=1.0,
            )

    # === Skipped Row Creation Tests ===

    def test_create_skipped_row_happy_path(self, base_config):
        """Test successful creation of skipped row."""
        builder = JudgeResultsBuilder(base_config)

        builder.create_skipped_row(
            sample_example_id="test_1",
            original_id="id1",
        )

        assert builder.completed_count == 1

    # === LLM Error Row Creation Tests ===

    def test_create_llm_error_row_happy_path(self, base_config):
        """Test successful creation of LLM error row."""
        builder = JudgeResultsBuilder(base_config)

        error = Exception("API timeout")
        builder.create_llm_error_row(
            sample_example_id="test_1",
            original_id="id1",
            error=error,
        )

        assert builder.completed_count == 1

    # === Parsing Error Row Creation Tests ===

    def test_create_parsing_error_row_happy_path(self, base_config):
        """Test successful creation of parsing error row."""
        builder = JudgeResultsBuilder(base_config)

        error = ValueError("Invalid JSON")
        builder.create_parsing_error_row(
            sample_example_id="test_1",
            original_id="id1",
            error=error,
            llm_raw_response="malformed response",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            call_duration=1.0,
        )

        assert builder.completed_count == 1

    # === Other Error Row Creation Tests ===

    def test_create_other_error_row_happy_path(self, base_config):
        """Test successful creation of other error row."""
        builder = JudgeResultsBuilder(base_config)

        error = RuntimeError("Unexpected error")
        builder.create_other_error_row(
            sample_example_id="test_1",
            original_id="id1",
            error=error,
        )

        assert builder.completed_count == 1

    # === Validation Tests (_validate_and_store) ===

    def test_validate_and_store_invalid_original_id_error(self, base_config):
        """Test error when original_id not in expected_ids."""
        builder = JudgeResultsBuilder(base_config)

        with pytest.raises(
            ValueError, match="Unexpected original_id 'invalid_id' not in expected IDs"
        ):
            builder.create_success_row(
                sample_example_id="test_1",
                original_id="invalid_id",  # Not in expected_ids
                outcomes={"sentiment": "positive"},
                llm_raw_response="response",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                call_duration=1.0,
            )

    def test_validate_and_store_duplicate_original_id_error(self, base_config):
        """Test error when original_id already exists."""
        builder = JudgeResultsBuilder(base_config)

        # Add first row
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response="response",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            call_duration=1.0,
        )

        # Try to add duplicate
        with pytest.raises(
            ValueError, match="Result for original_id 'id1' already exists"
        ):
            builder.create_success_row(
                sample_example_id="test_2",
                original_id="id1",  # Duplicate
                outcomes={"sentiment": "negative"},
                llm_raw_response="response",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                call_duration=1.0,
            )

    def test_validate_and_store_happy_path(self, base_config):
        """Test successful validation and storage."""
        builder = JudgeResultsBuilder(base_config)

        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response="response",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            call_duration=1.0,
        )

        assert "id1" in builder._results
        assert builder.completed_count == 1

    # === Completion Tests ===

    def test_complete_happy_path(self, single_task_config):
        """Test successful completion and JudgeResults creation."""
        builder = JudgeResultsBuilder(single_task_config)

        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response="positive sentiment",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            call_duration=1.0,
        )

        results = builder.complete()

        assert results.run_id == "single_task_run"
        assert results.judge_id == "single_judge"
        assert results.total_count == 1
        assert results.succeeded_count == 1
        assert len(results.results_data) == 1

    def test_complete_missing_results_error(self, base_config):
        """Test error when not all expected results received."""
        builder = JudgeResultsBuilder(base_config)

        # Only add one result, but expecting 3
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response="response",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            call_duration=1.0,
        )

        with pytest.raises(ValueError, match="Missing results for IDs"):
            builder.complete()

    def test_complete_status_count_calculation(self, base_config):
        """Test correct status count calculation in completion."""
        builder = JudgeResultsBuilder(base_config)

        # Add different types of results
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response="response",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            call_duration=1.0,
        )

        builder.create_llm_error_row(
            sample_example_id="test_2",
            original_id="id2",
            error=Exception("API error"),
        )

        builder.create_skipped_row(
            sample_example_id="test_3",
            original_id="id3",
        )

        results = builder.complete()

        assert results.succeeded_count == 1
        assert results.llm_error_count == 1
        assert results.skipped_count == 1
        assert results.total_count == 3

    # === Edge Case Tests ===

    def test_edge_case_single_task_schema(self, single_task_config):
        """Test builder with single task schema."""
        builder = JudgeResultsBuilder(single_task_config)

        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response="response",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            call_duration=1.0,
        )

        results = builder.complete()
        assert len(results.task_schema) == 1
        assert "sentiment" in results.task_schema

    def test_edge_case_multiple_task_schemas(self, multi_task_config):
        """Test builder with multiple task schemas."""
        builder = JudgeResultsBuilder(multi_task_config)

        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive", "toxicity": "non_toxic"},
            llm_raw_response="response",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            call_duration=1.0,
        )

        builder.create_partial_row(
            sample_example_id="test_2",
            original_id="id2",
            outcomes={"sentiment": "negative"},  # Missing toxicity
            error_message="Failed to parse toxicity",
            llm_raw_response="partial response",
            prompt_tokens=8,
            completion_tokens=3,
            total_tokens=11,
            call_duration=0.8,
        )

        results = builder.complete()
        assert len(results.task_schema) == 2
        assert results.succeeded_count == 1
        assert results.partial_count == 1

    def test_edge_case_mix_of_row_types(self, base_config):
        """Test builder with mix of different row types."""
        builder = JudgeResultsBuilder(base_config)

        # Success
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response="response",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            call_duration=1.0,
        )

        # Error
        builder.create_llm_error_row(
            sample_example_id="test_2",
            original_id="id2",
            error=Exception("error"),
        )

        # Skipped
        builder.create_skipped_row(
            sample_example_id="test_3",
            original_id="id3",
        )

        results = builder.complete()
        assert results.succeeded_count == 1
        assert results.llm_error_count == 1
        assert results.skipped_count == 1

    def test_edge_case_all_same_row_type_success(self, base_config):
        """Test builder with all success rows."""
        builder = JudgeResultsBuilder(base_config)

        for i, original_id in enumerate(["id1", "id2", "id3"], 1):
            builder.create_success_row(
                sample_example_id=f"test_{i}",
                original_id=original_id,
                outcomes={"sentiment": "positive"},
                llm_raw_response="response",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                call_duration=1.0,
            )

        results = builder.complete()
        assert results.succeeded_count == 3
        assert results.llm_error_count == 0
        assert results.skipped_count == 0

    def test_edge_case_all_same_row_type_error(self, base_config):
        """Test builder with all error rows."""
        builder = JudgeResultsBuilder(base_config)

        for i, original_id in enumerate(["id1", "id2", "id3"], 1):
            builder.create_llm_error_row(
                sample_example_id=f"test_{i}",
                original_id=original_id,
                error=Exception(f"error {i}"),
            )

        results = builder.complete()
        assert results.succeeded_count == 0
        assert results.llm_error_count == 3
        assert results.skipped_count == 0
