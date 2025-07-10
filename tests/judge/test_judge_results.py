"""Test suite for JudgeResults and JudgeResultsBuilder with comprehensive path coverage."""

import pytest

from meta_evaluator.results import JudgeResults, JudgeResultsBuilder
from meta_evaluator.llm_client import LLMClientEnum
from meta_evaluator.results.judge_results import (
    INVALID_JSON_STRUCTURE_MSG,
    STATE_FILE_NOT_FOUND_MSG,
)


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
            expected_ids=["id1", "id2", "id3"],
            is_sampled_run=False,
        )

    @pytest.fixture
    def multi_task_builder(self) -> JudgeResultsBuilder:
        """Provides a builder with multiple tasks and ids.

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
            expected_ids=["id1", "id2", "id3", "id4", "id5"],
            is_sampled_run=True,
        )

    @pytest.fixture
    def single_task_builder(self) -> JudgeResultsBuilder:
        """Provides a builder with single task and id.

        Returns:
            JudgeResultsBuilder: A builder with single task.
        """
        return JudgeResultsBuilder(
            run_id="single_task_run",
            judge_id="single_judge",
            llm_client_enum=LLMClientEnum.ANTHROPIC,
            model_used="claude-3",
            task_schemas={"sentiment": ["positive", "negative"]},
            expected_ids=["id1"],
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
        assert base_builder.total_count == 3
        assert base_builder.completed_count == 0
        assert not base_builder.is_complete

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
        """Test total_count property with expected IDs."""
        assert base_builder.total_count == 3

    def test_is_complete_property(self, single_task_builder):
        """Test is_complete property in various scenarios."""
        assert not single_task_builder.is_complete

        # After completion
        single_task_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive"},
            llm_raw_response_content="positive",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.0,
        )
        assert single_task_builder.is_complete

    # === Row Creation Tests ===

    def test_create_various_row_types(self, multi_task_builder):
        """Test creation of different row types."""
        # Success row
        multi_task_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"sentiment": "positive", "toxicity": "non_toxic"},
            llm_raw_response_content="positive sentiment detected",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.5,
        )

        # Partial row
        multi_task_builder.create_partial_row(
            sample_example_id="test_2",
            original_id="id2",
            outcomes={"sentiment": "positive"},  # Missing toxicity
            error_message="Failed to parse toxicity",
            llm_raw_response_content="partial response",
            llm_prompt_tokens=10,
            llm_completion_tokens=5,
            llm_total_tokens=15,
            llm_call_duration_seconds=1.0,
        )

        # Skipped row
        multi_task_builder.create_skipped_row(
            sample_example_id="test_3",
            original_id="id3",
        )

        # LLM error row
        multi_task_builder.create_llm_error_row(
            sample_example_id="test_4",
            original_id="id4",
            error=Exception("API timeout"),
        )

        # Other error row
        multi_task_builder.create_other_error_row(
            sample_example_id="test_5",
            original_id="id5",
            error=RuntimeError("Unexpected error"),
        )

        assert multi_task_builder.completed_count == 5

    def test_create_row_validation_errors(self, base_builder):
        """Test validation errors for row creation."""
        # Missing tasks in success row
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

        # Extra tasks in success row
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

        # Invalid task names in partial row
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

    # === Validation Tests (_validate_and_store) ===

    def test_validate_and_store_invalid_original_id_error(self, base_builder):
        """Test that creating a row with an original_id not in expected_ids raises an error."""
        with pytest.raises(
            ValueError, match="Unexpected original_id 'invalid_id' not in expected IDs"
        ):
            base_builder.create_success_row(
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
        """Test that adding a row with a duplicate original_id raises an error."""
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

    def test_complete_missing_results_error(self, base_builder):
        """Test error when not all expected results received."""
        # Only add one result, but expecting 3
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

        with pytest.raises(ValueError, match="Missing results for IDs"):
            base_builder.complete()

    def test_complete_status_count_calculation(self, base_builder):
        """Test correct status count calculation in completion."""
        # Add different types of results
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

        base_builder.create_llm_error_row(
            sample_example_id="test_2",
            original_id="id2",
            error=Exception("API error"),
        )

        base_builder.create_skipped_row(
            sample_example_id="test_3",
            original_id="id3",
        )

        results = base_builder.complete()

        assert results.succeeded_count == 1
        assert results.llm_error_count == 1
        assert results.skipped_count == 1
        assert results.total_count == 3

    def test_builder_empty_expected_ids_error(self):
        """Test that empty expected_ids raises an error."""
        with pytest.raises(ValueError, match="expected_ids cannot be empty"):
            JudgeResultsBuilder(
                run_id="test_run_123",
                judge_id="test_judge_1",
                llm_client_enum=LLMClientEnum.OPENAI,
                model_used="gpt-4",
                task_schemas={"sentiment": ["positive", "negative", "neutral"]},
                expected_ids=[],  # Empty expected_ids should raise error
                is_sampled_run=False,
            )


class TestJudgeResultsSerialization:
    """Comprehensive test suite for JudgeResults serialization functionality."""

    @pytest.fixture
    def example_task_schemas(self):
        """Provide example task schemas for testing.

        Returns:
            dict: Example task schemas.
        """
        return {"task1": ["yes", "no"], "task2": ["good", "bad"]}

    @pytest.fixture
    def judge_results_builder(self, example_task_schemas):
        """Provide a JudgeResultsBuilder for testing.

        Returns:
            JudgeResultsBuilder: A JudgeResultsBuilder for testing.
        """
        return JudgeResultsBuilder(
            run_id="run_001",
            judge_id="judge_001",
            llm_client_enum=LLMClientEnum.OPENAI,
            model_used="gpt-4",
            task_schemas=example_task_schemas,
            expected_ids=["id1", "id2", "id3", "id4"],
            is_sampled_run=False,
        )

    @pytest.fixture
    def sample_judge_results(self, judge_results_builder):
        """Create sample JudgeResults for testing.

        Returns:
            JudgeResults: A sample JudgeResults for testing.
        """
        builder = judge_results_builder

        # Add successful results
        builder.create_success_row(
            sample_example_id="sample_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            llm_raw_response_content="Response 1",
            llm_prompt_tokens=100,
            llm_completion_tokens=50,
            llm_total_tokens=150,
            llm_call_duration_seconds=2.5,
        )

        builder.create_success_row(
            sample_example_id="sample_2",
            original_id="id2",
            outcomes={"task1": "no", "task2": "bad"},
            llm_raw_response_content="Response 2",
            llm_prompt_tokens=120,
            llm_completion_tokens=60,
            llm_total_tokens=180,
            llm_call_duration_seconds=3.0,
        )

        # Add partial result
        builder.create_partial_row(
            sample_example_id="sample_3",
            original_id="id3",
            outcomes={"task1": "yes"},  # Only partial outcomes
            error_message="Could not parse task2",
            llm_raw_response_content="Response 3",
            llm_prompt_tokens=110,
            llm_completion_tokens=55,
            llm_total_tokens=165,
            llm_call_duration_seconds=2.8,
        )

        # Add error results
        builder.create_llm_error_row(
            sample_example_id="sample_4",
            original_id="id4",
            error=Exception("LLM API failed"),
        )

        return builder.complete()

    # -------------------------
    # Persistence Tests
    # -------------------------

    ### write_data and load_data Tests

    @pytest.mark.parametrize("data_format", ["json", "csv", "parquet"])
    def test_write_and_load_data_formats(
        self, tmp_path, sample_judge_results, data_format
    ):
        """Test that write_data and load_data work for all supported formats."""
        file_path = tmp_path / f"judge_results.{data_format}"

        # Write data
        sample_judge_results.write_data(str(file_path), data_format=data_format)

        # Verify file was created
        assert file_path.exists()

        # Load data
        loaded_df = JudgeResults.load_data(str(file_path), data_format=data_format)

        # Verify data integrity
        assert len(loaded_df) == len(sample_judge_results.results_data)
        assert set(loaded_df.columns) == set(sample_judge_results.results_data.columns)

        # Verify specific data points
        assert "task1" in loaded_df.columns
        assert "task2" in loaded_df.columns
        assert "llm_raw_response_content" in loaded_df.columns
        assert "llm_prompt_tokens" in loaded_df.columns

    def test_serialization_error_handling(self, tmp_path, sample_judge_results):
        """Test serialization error handling for unsupported formats and missing files."""
        # Test unsupported format for write_data
        file_path = tmp_path / "results.xml"
        write_data_method = getattr(sample_judge_results, "write_data")
        with pytest.raises((ValueError, Exception)) as exc_info:
            write_data_method(str(file_path), data_format="xml")
        assert "xml" in str(exc_info.value)

        # Test unsupported format for load_data
        file_path.write_text("<xml>dummy</xml>")
        load_data_method = getattr(JudgeResults, "load_data")
        with pytest.raises((ValueError, Exception)) as exc_info:
            load_data_method(str(file_path), data_format="xml")
        assert "xml" in str(exc_info.value)

        # Test missing file
        with pytest.raises(Exception):
            JudgeResults.load_data("nonexistent_file.json", data_format="json")

    ### save_state and load_state Tests

    @pytest.mark.parametrize("data_format", ["json", "csv", "parquet"])
    def test_save_and_load_state_formats(
        self, tmp_path, sample_judge_results, data_format
    ):
        """Test save_state and load_state work for all data formats."""
        state_file = tmp_path / "judge_state.json"

        # Save state
        sample_judge_results.save_state(str(state_file), data_format=data_format)

        # Verify state file was created
        assert state_file.exists()

        # Verify data file was created
        data_file = tmp_path / f"judge_state_data.{data_format}"
        assert data_file.exists()

        # Load state
        loaded_results = JudgeResults.load_state(str(state_file))

        # Verify loaded results match original
        assert loaded_results.run_id == sample_judge_results.run_id
        assert loaded_results.judge_id == sample_judge_results.judge_id
        assert loaded_results.llm_client_enum == sample_judge_results.llm_client_enum
        assert loaded_results.model_used == sample_judge_results.model_used
        assert loaded_results.task_schemas == sample_judge_results.task_schemas
        assert loaded_results.total_count == sample_judge_results.total_count
        assert loaded_results.succeeded_count == sample_judge_results.succeeded_count
        assert loaded_results.partial_count == sample_judge_results.partial_count
        assert loaded_results.llm_error_count == sample_judge_results.llm_error_count
        assert loaded_results.is_sampled_run == sample_judge_results.is_sampled_run

        # Verify DataFrame data
        assert len(loaded_results.results_data) == len(
            sample_judge_results.results_data
        )
        assert set(loaded_results.results_data.columns) == set(
            sample_judge_results.results_data.columns
        )

    def test_save_state_custom_data_filename(self, tmp_path, sample_judge_results):
        """Test save_state with custom data filename."""
        state_file = tmp_path / "custom_state.json"
        custom_data_filename = "my_custom_data.json"

        sample_judge_results.save_state(
            str(state_file), data_format="json", data_filename=custom_data_filename
        )

        # Verify custom data file was created
        data_file = tmp_path / custom_data_filename
        assert data_file.exists()

        # Verify loading works with custom filename
        loaded_results = JudgeResults.load_state(str(state_file))
        assert loaded_results.run_id == sample_judge_results.run_id

    def test_save_state_data_filename_extension_validation(
        self, tmp_path, sample_judge_results
    ):
        """Test that save_state validates data filename extension matches format."""
        state_file = tmp_path / "state.json"

        with pytest.raises(
            ValueError, match="data_filename extension.*must match data_format"
        ):
            sample_judge_results.save_state(
                str(state_file), data_format="json", data_filename="wrong_extension.csv"
            )

    def test_save_state_creates_directory(self, tmp_path, sample_judge_results):
        """Test that save_state creates parent directories if they don't exist."""
        nested_path = tmp_path / "nested" / "directory"
        state_file = nested_path / "state.json"

        sample_judge_results.save_state(str(state_file), data_format="json")

        # Verify directory was created
        assert nested_path.exists()
        assert state_file.exists()

        # Verify loading works
        loaded_results = JudgeResults.load_state(str(state_file))
        assert loaded_results.run_id == sample_judge_results.run_id

    def test_load_state_error_handling(self, tmp_path, sample_judge_results):
        """Test load_state error handling for various failure scenarios."""
        # Test missing file
        with pytest.raises(FileNotFoundError, match=STATE_FILE_NOT_FOUND_MSG):
            JudgeResults.load_state("nonexistent_state.json")

        # Test invalid JSON
        state_file = tmp_path / "invalid.json"
        state_file.write_text("{invalid json")
        with pytest.raises(ValueError, match=INVALID_JSON_STRUCTURE_MSG):
            JudgeResults.load_state(str(state_file))

        # Test missing required keys
        state_file = tmp_path / "incomplete.json"
        state_file.write_text(
            '{"metadata": {}}'
        )  # Missing data_format and data_filename
        with pytest.raises(ValueError, match=INVALID_JSON_STRUCTURE_MSG):
            JudgeResults.load_state(str(state_file))

        # Test missing data file
        state_file = tmp_path / "state.json"
        sample_judge_results.save_state(str(state_file), data_format="json")

        # Delete the data file
        data_file = tmp_path / "state_data.json"
        data_file.unlink()

        with pytest.raises(Exception):  # Will raise FileNotFoundError or similar
            JudgeResults.load_state(str(state_file))

    # -------------------------
    # Integration Tests
    # -------------------------

    def test_serialization_preserves_judge_specific_fields(
        self, tmp_path, sample_judge_results
    ):
        """Test that serialization preserves judge-specific fields like LLM diagnostics."""
        # Test via save_state/load_state
        state_file = tmp_path / "test_state.json"
        sample_judge_results.save_state(str(state_file), data_format="json")
        reconstructed = JudgeResults.load_state(str(state_file))

        # Verify judge-specific fields are preserved
        assert reconstructed.judge_id == sample_judge_results.judge_id
        assert reconstructed.llm_client_enum == sample_judge_results.llm_client_enum
        assert reconstructed.model_used == sample_judge_results.model_used
        assert reconstructed.partial_count == sample_judge_results.partial_count
        assert reconstructed.llm_error_count == sample_judge_results.llm_error_count
        assert (
            reconstructed.parsing_error_count
            == sample_judge_results.parsing_error_count
        )
        assert reconstructed.other_error_count == sample_judge_results.other_error_count

        # Verify LLM diagnostic fields are preserved in DataFrame
        success_results = reconstructed.get_successful_results()
        if not success_results.is_empty():
            assert "llm_raw_response_content" in success_results.columns
            assert "llm_prompt_tokens" in success_results.columns
            assert "llm_completion_tokens" in success_results.columns
            assert "llm_total_tokens" in success_results.columns
            assert "llm_call_duration_seconds" in success_results.columns

    def test_complete_serialization_workflow(self, tmp_path, sample_judge_results):
        """Test complete serialization workflow: save_state -> load_state."""
        state_file = tmp_path / "workflow_test.json"

        # Step 1: Save state
        sample_judge_results.save_state(str(state_file), data_format="json")

        # Step 2: Load state
        loaded_results = JudgeResults.load_state(str(state_file))

        # Verify final result matches original
        assert loaded_results.run_id == sample_judge_results.run_id
        assert loaded_results.judge_id == sample_judge_results.judge_id
        assert loaded_results.total_count == sample_judge_results.total_count
        assert loaded_results.succeeded_count == sample_judge_results.succeeded_count
        assert len(loaded_results.results_data) == len(
            sample_judge_results.results_data
        )
