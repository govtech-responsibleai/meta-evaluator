"""Test suite for JudgeResultsBuilder with comprehensive path coverage."""

import pytest

from meta_evaluator.results import JudgeResultsBuilder
from meta_evaluator.llm_client import LLMClientEnum


class TestJudgeResultsBuilder:
    """Comprehensive test suite for JudgeResultsBuilder achieving 100% path coverage."""

    @pytest.fixture
    def base_builder(self) -> JudgeResultsBuilder:
        """Provides a valid JudgeResultsBuilder for testing.

        Returns:
            JudgeResultsBuilder: A valid builder instance.
        """
        return JudgeResultsBuilder(
            run_id="test_run_123",
            judge_id="test_judge_1",
            llm_client_enum=LLMClientEnum.OPENAI,
            model_used="gpt-4",
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            is_sampled_run=False,
        )

    @pytest.fixture
    def multi_task_builder(self) -> JudgeResultsBuilder:
        """Provides a builder with multiple tasks.

        Returns:
            JudgeResultsBuilder: A builder with multiple tasks.
        """
        return JudgeResultsBuilder(
            run_id="multi_task_run",
            judge_id="multi_judge",
            llm_client_enum=LLMClientEnum.OPENAI,
            model_used="gpt-4",
            task_schemas={
                "sentiment": ["positive", "negative", "neutral"],
                "toxicity": ["toxic", "non_toxic"],
            },
            is_sampled_run=True,
        )

    @pytest.fixture
    def single_task_builder(self) -> JudgeResultsBuilder:
        """Provides a builder with single task.

        Returns:
            JudgeResultsBuilder: A builder with single task.
        """
        return JudgeResultsBuilder(
            run_id="single_task_run",
            judge_id="single_judge",
            llm_client_enum=LLMClientEnum.ANTHROPIC,
            model_used="claude-3",
            task_schemas={"sentiment": ["positive", "negative"]},
            is_sampled_run=False,
        )

    # === Initialization Tests ===

    def test_initialization_happy_path(self, base_builder):
        """Test successful builder initialization."""
        assert base_builder.run_id == "test_run_123"
        assert base_builder.evaluator_id == "test_judge_1"
        assert base_builder.llm_client_enum == LLMClientEnum.OPENAI
        assert base_builder.model_used == "gpt-4"
        assert base_builder.task_schemas == {
            "sentiment": ["positive", "negative", "neutral"]
        }
        assert base_builder.is_sampled_run is False
        assert base_builder.total_count == 0  # No expected_ids provided
        assert base_builder.completed_count == 0
        assert base_builder.is_complete

    def test_initialization_with_expected_ids(self):
        """Test builder initialization with expected IDs."""
        builder = JudgeResultsBuilder(
            run_id="test_run_123",
            judge_id="test_judge_1",
            llm_client_enum=LLMClientEnum.OPENAI,
            model_used="gpt-4",
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            is_sampled_run=False,
            expected_ids=["id1", "id2", "id3"],
        )

        assert builder.total_count == 3
        assert builder.completed_count == 0
        assert not builder.is_complete

    # === Property Access Tests ===

    def test_completed_count_property(self, base_builder):
        """Test completed_count property."""
        assert base_builder.completed_count == 0

        # Add a success row
        base_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response_content="positive sentiment",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.0,
        )
        assert base_builder.completed_count == 1

    def test_total_count_property(self, base_builder):
        """Test total_count property."""
        assert base_builder.total_count == 0  # No expected_ids

    def test_total_count_property_with_expected_ids(self):
        """Test total_count property with expected IDs."""
        builder = JudgeResultsBuilder(
            run_id="test_run_123",
            judge_id="test_judge_1",
            llm_client_enum=LLMClientEnum.OPENAI,
            model_used="gpt-4",
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            is_sampled_run=False,
            expected_ids=["id1", "id2", "id3"],
        )
        assert builder.total_count == 3

    def test_is_complete_property_false(self):
        """Test is_complete property returns False when incomplete."""
        builder = JudgeResultsBuilder(
            run_id="test_run_123",
            judge_id="test_judge_1",
            llm_client_enum=LLMClientEnum.OPENAI,
            model_used="gpt-4",
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            is_sampled_run=False,
            expected_ids=["id1", "id2", "id3"],
        )
        assert not builder.is_complete

    def test_is_complete_property_true(self, single_task_builder):
        """Test is_complete property returns True when complete."""
        # No expected_ids, so always complete
        assert single_task_builder.is_complete

    def test_is_complete_property_true_with_expected_ids(self):
        """Test is_complete property returns True when all expected results received."""
        builder = JudgeResultsBuilder(
            run_id="test_run_123",
            judge_id="test_judge_1",
            llm_client_enum=LLMClientEnum.OPENAI,
            model_used="gpt-4",
            task_schemas={"sentiment": ["positive", "negative"]},
            is_sampled_run=False,
            expected_ids=["id1"],
        )

        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response_content="positive",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.0,
        )
        assert builder.is_complete

    # === Success Row Creation Tests ===

    def test_create_success_row_happy_path(self, base_builder):
        """Test successful creation of success row."""
        base_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response_content="positive sentiment detected",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.5,
        )

        assert base_builder.completed_count == 1

    def test_create_success_row_missing_tasks_error(self, base_builder):
        """Test error when missing tasks in success row."""
        with pytest.raises(
            ValueError, match="Success row must contain outcomes for ALL tasks"
        ):
            base_builder.create_success_row(
                sample_example_id="test_1",
                original_id="id1",
                outcomes={},  # Missing sentiment task
                llm_raw_response_content="response",
                llm_prompt_tokens=10,
                llm_completion_tokens=5,
                llm_total_tokens=15,
                llm_call_duration_seconds=1.0,
            )

    def test_create_success_row_extra_tasks_error(self, base_builder):
        """Test error when extra tasks in success row."""
        with pytest.raises(
            ValueError, match="Success row must contain outcomes for ALL tasks"
        ):
            base_builder.create_success_row(
                sample_example_id="test_1",
                original_id="id1",
                outcomes={"sentiment": "positive", "extra_task": "value"},
                llm_raw_response_content="response",
                llm_prompt_tokens=10,
                llm_completion_tokens=5,
                llm_total_tokens=15,
                llm_call_duration_seconds=1.0,
            )

    # === Partial Row Creation Tests ===

    def test_create_partial_row_happy_path(self, multi_task_builder):
        """Test successful creation of partial row."""
        multi_task_builder.create_partial_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},  # Missing toxicity
            error_message="Failed to parse toxicity",
            llm_raw_response_content="partial response",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.0,
        )

        assert multi_task_builder.completed_count == 1

    def test_create_partial_row_invalid_task_names_error(self, base_builder):
        """Test error when invalid task names in partial row."""
        with pytest.raises(ValueError, match="Invalid task names"):
            base_builder.create_partial_row(
                sample_example_id="test_1",
                original_id="id1",
                outcomes={"invalid_task": "value"},
                error_message="error",
                llm_raw_response_content="response",
                llm_prompt_tokens=10,
                llm_completion_tokens=5,
                llm_total_tokens=15,
                llm_call_duration_seconds=1.0,
            )

    # === Skipped Row Creation Tests ===

    def test_create_skipped_row_happy_path(self, base_builder):
        """Test successful creation of skipped row."""
        base_builder.create_skipped_row(
            sample_example_id="test_1",
            original_id="id1",
        )

        assert base_builder.completed_count == 1

    # === LLM Error Row Creation Tests ===

    def test_create_llm_error_row_happy_path(self, base_builder):
        """Test successful creation of LLM error row."""
        error = Exception("API timeout")
        base_builder.create_llm_error_row(
            sample_example_id="test_1",
            original_id="id1",
            error=error,
        )

        assert base_builder.completed_count == 1

    # === Parsing Error Row Creation Tests ===

    def test_create_parsing_error_row_happy_path(self, base_builder):
        """Test successful creation of parsing error row."""
        error = ValueError("Invalid JSON")
        base_builder.create_parsing_error_row(
            sample_example_id="test_1",
            original_id="id1",
            error=error,
            llm_raw_response_content="malformed response",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.0,
        )

        assert base_builder.completed_count == 1

    # === Other Error Row Creation Tests ===

    def test_create_other_error_row_happy_path(self, base_builder):
        """Test successful creation of other error row."""
        error = RuntimeError("Unexpected error")
        base_builder.create_other_error_row(
            sample_example_id="test_1",
            original_id="id1",
            error=error,
        )

        assert base_builder.completed_count == 1

    # === Validation Tests (_validate_and_store) ===

    def test_validate_and_store_invalid_original_id_error(self):
        """Test error when original_id not in expected_ids."""
        builder = JudgeResultsBuilder(
            run_id="test_run_123",
            judge_id="test_judge_1",
            llm_client_enum=LLMClientEnum.OPENAI,
            model_used="gpt-4",
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            is_sampled_run=False,
            expected_ids=["id1", "id2", "id3"],
        )

        with pytest.raises(
            ValueError, match="Unexpected original_id 'invalid_id' not in expected IDs"
        ):
            builder.create_success_row(
                sample_example_id="test_1",
                original_id="invalid_id",  # Not in expected_ids
                outcomes={"sentiment": "positive"},
                llm_raw_response_content="response",
                llm_prompt_tokens=10,
                llm_completion_tokens=5,
                llm_total_tokens=15,
                llm_call_duration_seconds=1.0,
            )

    def test_validate_and_store_duplicate_original_id_error(self, base_builder):
        """Test error when original_id already exists."""
        # Add first row
        base_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response_content="response",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.0,
        )

        # Try to add duplicate
        with pytest.raises(
            ValueError, match="Result for original_id 'id1' already exists"
        ):
            base_builder.create_success_row(
                sample_example_id="test_2",
                original_id="id1",  # Duplicate
                outcomes={"sentiment": "negative"},
                llm_raw_response_content="response",
                llm_prompt_tokens=10,
                llm_completion_tokens=5,
                llm_total_tokens=15,
                llm_call_duration_seconds=1.0,
            )

    def test_validate_and_store_happy_path(self, base_builder):
        """Test successful validation and storage."""
        base_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response_content="response",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.0,
        )

        assert "id1" in base_builder._results
        assert base_builder.completed_count == 1

    # === Completion Tests ===

    def test_complete_happy_path(self, single_task_builder):
        """Test successful completion and JudgeResults creation."""
        single_task_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response_content="positive sentiment",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.0,
        )

        results = single_task_builder.complete()

        assert results.run_id == "single_task_run"
        assert results.judge_id == "single_judge"
        assert results.total_count == 1
        assert results.succeeded_count == 1
        assert len(results.results_data) == 1

    def test_complete_missing_results_error(self):
        """Test error when not all expected results received."""
        builder = JudgeResultsBuilder(
            run_id="test_run_123",
            judge_id="test_judge_1",
            llm_client_enum=LLMClientEnum.OPENAI,
            model_used="gpt-4",
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            is_sampled_run=False,
            expected_ids=["id1", "id2", "id3"],
        )

        # Only add one result, but expecting 3
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response_content="response",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.0,
        )

        with pytest.raises(ValueError, match="Missing results for IDs"):
            builder.complete()

    def test_complete_status_count_calculation(self):
        """Test correct status count calculation in completion."""
        builder = JudgeResultsBuilder(
            run_id="test_run_123",
            judge_id="test_judge_1",
            llm_client_enum=LLMClientEnum.OPENAI,
            model_used="gpt-4",
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            is_sampled_run=False,
            expected_ids=["id1", "id2", "id3"],
        )

        # Add different types of results
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response_content="response",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.0,
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

    def test_edge_case_single_task_schema(self, single_task_builder):
        """Test builder with single task schema."""
        single_task_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response_content="response",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.0,
        )

        results = single_task_builder.complete()
        assert len(results.task_schemas) == 1
        assert "sentiment" in results.task_schemas

    def test_edge_case_multiple_task_schemas(self, multi_task_builder):
        """Test builder with multiple task schemas."""
        multi_task_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive", "toxicity": "non_toxic"},
            llm_raw_response_content="response",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.0,
        )

        multi_task_builder.create_partial_row(
            sample_example_id="test_2",
            original_id="id2",
            outcomes={"sentiment": "negative"},  # Missing toxicity
            error_message="Failed to parse toxicity",
            llm_raw_response_content="partial response",
            llm_prompt_tokens=8,
            llm_completion_tokens=3,
            llm_total_tokens=11,
            llm_call_duration_seconds=0.8,
        )

        results = multi_task_builder.complete()
        assert len(results.task_schemas) == 2
        assert results.succeeded_count == 1
        assert results.partial_count == 1

    def test_edge_case_mix_of_row_types(self, base_builder):
        """Test builder with mix of different row types."""
        # Success
        base_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response_content="response",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.0,
        )

        # Error
        base_builder.create_llm_error_row(
            sample_example_id="test_2",
            original_id="id2",
            error=Exception("error"),
        )

        # Skipped
        base_builder.create_skipped_row(
            sample_example_id="test_3",
            original_id="id3",
        )

        results = base_builder.complete()
        assert results.succeeded_count == 1
        assert results.llm_error_count == 1
        assert results.skipped_count == 1

    def test_edge_case_all_same_row_type_success(self, base_builder):
        """Test builder with all success rows."""
        for i, original_id in enumerate(["id1", "id2", "id3"], 1):
            base_builder.create_success_row(
                sample_example_id=f"test_{i}",
                original_id=original_id,
                outcomes={"sentiment": "positive"},
                llm_raw_response_content="response",
                llm_prompt_tokens=10,
                llm_completion_tokens=5,
                llm_total_tokens=15,
                llm_call_duration_seconds=1.0,
            )

        results = base_builder.complete()
        assert results.succeeded_count == 3
        assert results.llm_error_count == 0
        assert results.skipped_count == 0

    def test_edge_case_all_same_row_type_error(self, base_builder):
        """Test builder with all error rows."""
        for i, original_id in enumerate(["id1", "id2", "id3"], 1):
            base_builder.create_llm_error_row(
                sample_example_id=f"test_{i}",
                original_id=original_id,
                error=Exception(f"error {i}"),
            )

        results = base_builder.complete()
        assert results.succeeded_count == 0
        assert results.llm_error_count == 3
        assert results.skipped_count == 0
