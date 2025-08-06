"""Unit and integration tests for HumanAnnotationResults.

Covers:
- Model validation (BaseResultRow field selection, config validation, ID validation)
- DataFrame operations (row creation, filtering successful/failed results)
- Persistence (write/load data in multiple formats, save/load state)
- Builder logic (row creation, error handling, completion, duplicate prevention)
"""

import pytest
from datetime import datetime

from meta_evaluator.results import (
    HumanAnnotationResults,
    HumanAnnotationResultsBuilder,
    HumanAnnotationResultRow,
)
from meta_evaluator.results.human_results import (
    INVALID_JSON_STRUCTURE_MSG,
    STATE_FILE_NOT_FOUND_MSG,
)
from meta_evaluator.results.exceptions import (
    BuilderInitializationError,
    IncompleteResultsError,
    InvalidFileError,
    MismatchedTasksError,
    ResultsDataFormatError,
    TaskNotFoundError,
)


class TestHumanAnnotationResultsBuilder:
    """Comprehensive test suite for HumanAnnotationResultsBuilder."""

    @pytest.fixture
    def builder(self) -> HumanAnnotationResultsBuilder:
        """Provide a HumanAnnotationResultsBuilder instance for testing.

        Returns:
            HumanAnnotationResultsBuilder: A builder instance ready to create
                                        annotation result rows for testing.
        """
        return HumanAnnotationResultsBuilder(
            run_id="run_001",
            annotator_id="annotator_1",
            task_schemas={"task1": ["yes", "no"], "task2": ["good", "bad"]},
            expected_ids=["id1", "id2"],
            is_sampled_run=False,
        )

    @pytest.fixture
    def single_task_builder(self) -> HumanAnnotationResultsBuilder:
        """Provides a builder with single task and id.

        Returns:
            HumanAnnotationResultsBuilder: A builder with single task.
        """
        return HumanAnnotationResultsBuilder(
            run_id="single_task_run",
            annotator_id="annotator_1",
            task_schemas={"task1": ["yes", "no"]},
            expected_ids=["id1"],
            is_sampled_run=False,
        )

    # === Initialization Tests ===

    def test_initialization_happy_path(self, builder):
        """Test successful builder initialization."""
        assert builder.run_id == "run_001"
        assert builder.evaluator_id == "annotator_1"
        assert builder.task_schemas == {
            "task1": ["yes", "no"],
            "task2": ["good", "bad"],
        }
        assert builder.is_sampled_run is False
        assert builder.total_count == 2  # expected_ids provided
        assert builder.completed_count == 0
        assert not builder.is_complete

    # === Property Access Tests ===

    def test_completed_count_property(self, builder):
        """Test completed_count property."""
        assert builder.completed_count == 0

        # Add a success row
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )
        assert builder.completed_count == 1

    def test_total_count_property(self, builder):
        """Test total_count property with expected IDs."""
        assert builder.total_count == 2

    def test_is_complete_property(self, builder):
        """Test is_complete property in various scenarios."""
        assert not builder.is_complete

        # After adding first result
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )
        assert not builder.is_complete  # Still incomplete (needs 2)

        # After adding second result
        builder.create_success_row(
            sample_example_id="test_2",
            original_id="id2",
            outcomes={"task1": "no", "task2": "bad"},
            annotation_timestamp=datetime.now(),
        )
        assert builder.is_complete

    # === Row Creation Tests ===

    def test_create_various_row_types(self, builder):
        """Test creation of different row types."""
        # Success row
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )
        assert builder.completed_count == 1
        # Check that the row contains the correct outcomes
        row = builder._results["id1"]
        assert row.task1 == "yes"
        assert row.task2 == "good"

        # Error row
        builder.create_error_row(
            sample_example_id="test_2",
            original_id="id2",
            error_message="Test error",
        )
        assert builder.completed_count == 2

    def test_create_row_validation_errors(self, builder):
        """Test validation errors for row creation."""
        # Missing tasks in success row
        with pytest.raises(MismatchedTasksError):
            builder.create_success_row(
                sample_example_id="test_1",
                original_id="id1",
                outcomes={"task1": "yes"},  # Missing task2
                annotation_timestamp=datetime.now(),
            )

        # Extra tasks in success row
        with pytest.raises(MismatchedTasksError):
            builder.create_success_row(
                sample_example_id="test_1",
                original_id="id1",
                outcomes={"task1": "yes", "task2": "good", "extra_task": "value"},
                annotation_timestamp=datetime.now(),
            )

    # === Validation Tests (_validate_and_store) ===

    def test_validate_and_store_invalid_original_id_error(self, builder):
        """Test that creating a row with an original_id not in expected_ids raises an error."""
        with pytest.raises(BuilderInitializationError):
            builder.create_success_row(
                sample_example_id="sample_1",
                original_id="not_in_expected",
                outcomes={"task1": "yes", "task2": "good"},
                annotation_timestamp=datetime.now(),
            )

    def test_validate_and_store_duplicate_original_id_error(self, builder):
        """Test that adding a row with a duplicate original_id raises an error."""
        builder.create_success_row(
            sample_example_id="sample_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )
        with pytest.raises(BuilderInitializationError):
            builder.create_success_row(
                sample_example_id="sample_2",
                original_id="id1",  # Duplicate
                outcomes={"task1": "no", "task2": "bad"},
                annotation_timestamp=datetime.now(),
            )

    def test_get_successful_and_failed_results(self, builder):
        """Test filtering of successful and failed results."""
        builder.create_success_row(
            sample_example_id="sample_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )
        # Simulate an error row
        builder.create_error_row(
            sample_example_id="sample_2",
            original_id="id2",
            error_message="Test error",
        )
        results = builder.complete()
        successful = results.get_successful_results()
        failed = results.get_failed_results()
        assert successful.shape[0] == 1
        assert failed.shape[0] == 1

    # === Completion Tests ===

    def test_complete_happy_path(self, builder):
        """Test successful completion and HumanAnnotationResults creation."""
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )
        builder.create_success_row(
            sample_example_id="test_2",
            original_id="id2",
            outcomes={"task1": "no", "task2": "bad"},
            annotation_timestamp=datetime.now(),
        )

        results = builder.complete()

        assert results.run_id == "run_001"
        assert results.annotator_id == "annotator_1"
        assert results.total_count == 2
        assert results.succeeded_count == 2
        assert len(results.results_data) == 2

    def test_complete_missing_results_error(self):
        """Test error when not all expected results received."""
        builder = HumanAnnotationResultsBuilder(
            run_id="run_001",
            annotator_id="annotator_1",
            task_schemas={"task1": ["yes", "no"], "task2": ["good", "bad"]},
            expected_ids=["id1", "id2", "id3"],
            is_sampled_run=False,
        )

        # Only add one result, but expecting 3
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )

        with pytest.raises(IncompleteResultsError, match="Missing results for IDs"):
            builder.complete()

    def test_complete_no_rows_error(self, builder):
        """Test error when no rows added to builder."""
        with pytest.raises(IncompleteResultsError, match="No rows added to builder"):
            builder.complete()

    def test_complete_status_count_calculation(self, builder):
        """Test correct status count calculation in completion."""
        # Add different types of results
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )

        builder.create_error_row(
            sample_example_id="test_2",
            original_id="id2",
            error_message="Test error",
        )

        results = builder.complete()

        assert results.succeeded_count == 1
        assert results.error_count == 1
        assert results.total_count == 2

    def test_builder_empty_expected_ids_error(self):
        """Test that empty expected_ids raises an error."""
        with pytest.raises(
            BuilderInitializationError, match="expected_ids cannot be empty"
        ):
            HumanAnnotationResultsBuilder(
                run_id="run_001",
                annotator_id="annotator_1",
                task_schemas={"task1": ["yes", "no"], "task2": ["good", "bad"]},
                expected_ids=[],  # Empty expected_ids should raise error
                is_sampled_run=False,
            )

    # === Results Analysis Tests ===

    def test_results_model_methods(self, builder):
        """Test various results model methods."""
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )
        builder.create_error_row(
            sample_example_id="test_2",
            original_id="id2",
            error_message="Test error",
        )
        results = builder.complete()

        # Test various model methods
        assert results.get_evaluator_id() == "annotator_1"
        assert results.get_error_count() == 1
        assert results.get_base_result_row_class() == HumanAnnotationResultRow
        assert len(results) == 2

    def test_results_get_task_success_rate(self, builder):
        """Test get_task_success_rate calculation."""
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )
        builder.create_error_row(
            sample_example_id="test_2",
            original_id="id2",
            error_message="Test error",
        )
        results = builder.complete()

        # task1 has 1 non-null value out of 2 total
        assert results.get_task_success_rate("task1") == 0.5

        # task2 has 1 non-null value out of 2 total
        assert results.get_task_success_rate("task2") == 0.5

    def test_results_get_task_success_rate_invalid_task(self, single_task_builder):
        """Test get_task_success_rate with invalid task name."""
        single_task_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes"},
            annotation_timestamp=datetime.now(),
        )
        results = single_task_builder.complete()

        with pytest.raises(
            TaskNotFoundError, match="Task 'invalid_task' not found in task schema"
        ):
            results.get_task_success_rate("invalid_task")

    # === Field Tag Tests ===

    def test_field_tag_selection(self):
        """Test that HumanAnnotationResultRow tag-based field selection works as expected."""
        metadata_fields = HumanAnnotationResultRow.get_metadata_fields()
        error_fields = HumanAnnotationResultRow.get_error_fields()
        diagnostic_fields = HumanAnnotationResultRow.get_annotation_diagnostic_fields()
        assert "sample_example_id" in metadata_fields
        assert "error_message" in error_fields
        assert "annotation_timestamp" in diagnostic_fields

    def test_human_annotation_result_row_field_tags(self):
        """Test that HumanAnnotationResultRow field tag selection works."""
        metadata_fields = HumanAnnotationResultRow.get_metadata_fields()
        error_fields = HumanAnnotationResultRow.get_error_fields()
        diagnostic_fields = HumanAnnotationResultRow.get_annotation_diagnostic_fields()

        assert "sample_example_id" in metadata_fields
        assert "annotator_id" in metadata_fields
        assert "error_message" in error_fields
        assert "annotation_timestamp" in diagnostic_fields


class TestHumanAnnotationResultsSerialization:
    """Comprehensive test suite for HumanAnnotationResults serialization."""

    @pytest.fixture
    def builder(self):
        """Provide a HumanAnnotationResultsBuilder instance for testing.

        Returns:
            HumanAnnotationResultsBuilder: A builder instance ready to create
                                        annotation result rows for testing.
        """
        return HumanAnnotationResultsBuilder(
            run_id="run_001",
            annotator_id="annotator_1",
            task_schemas={"task1": ["yes", "no"], "task2": ["good", "bad"]},
            expected_ids=["id1", "id2"],
            is_sampled_run=False,
        )

    @pytest.fixture
    def single_task_builder(self) -> HumanAnnotationResultsBuilder:
        """Provides a builder with single task and id.

        Returns:
            HumanAnnotationResultsBuilder: A builder with single task.
        """
        return HumanAnnotationResultsBuilder(
            run_id="single_task_run",
            annotator_id="annotator_1",
            task_schemas={"task1": ["yes", "no"]},
            expected_ids=["id1"],
            is_sampled_run=False,
        )

    @pytest.mark.parametrize("data_format", ["json", "csv", "parquet"])
    def test_write_and_load_data(self, tmp_path, builder, data_format):
        """Test that write_data and load_data work for all supported formats."""
        builder.create_success_row(
            sample_example_id="sample_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )
        builder.create_success_row(
            sample_example_id="sample_2",
            original_id="id2",
            outcomes={"task1": "no", "task2": "bad"},
            annotation_timestamp=datetime.now(),
        )
        results = builder.complete()
        file_path = tmp_path / f"results.{data_format}"
        results.write_data(str(file_path), data_format=data_format)
        loaded_df = HumanAnnotationResults.load_data(
            str(file_path), data_format=data_format
        )
        assert loaded_df.shape[0] == 2
        assert "task1" in loaded_df.columns

    def test_save_and_load_state(self, tmp_path, builder):
        """Test that save_state and load_state work and preserve all data."""
        builder.create_success_row(
            sample_example_id="sample_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )
        builder.create_success_row(
            sample_example_id="sample_2",
            original_id="id2",
            outcomes={"task1": "no", "task2": "bad"},
            annotation_timestamp=datetime.now(),
        )
        results = builder.complete()
        state_file = tmp_path / "state.json"
        results.save_state(str(state_file), data_format="json")
        loaded = HumanAnnotationResults.load_state(str(state_file))
        assert loaded.run_id == results.run_id
        assert loaded.annotator_id == results.annotator_id
        assert loaded.succeeded_count == results.succeeded_count

    def test_serialization_error_handling(self, tmp_path, single_task_builder):
        """Test serialization error handling for unsupported formats and missing files."""
        single_task_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes"},
            annotation_timestamp=datetime.now(),
        )
        results = single_task_builder.complete()

        # Test unsupported format for write_data
        file_path = tmp_path / "results.xml"
        write_data_method = getattr(results, "write_data")
        with pytest.raises((ValueError, Exception)) as exc_info:
            write_data_method(str(file_path), data_format="xml")
        assert "xml" in str(exc_info.value)

        # Test unsupported format for load_data
        file_path.write_text("<xml>dummy</xml>")
        load_data_method = getattr(HumanAnnotationResults, "load_data")
        with pytest.raises((ValueError, Exception)) as exc_info:
            load_data_method(str(file_path), data_format="xml")
        assert "xml" in str(exc_info.value)

        # Test missing file
        with pytest.raises(Exception):
            HumanAnnotationResults.load_data(
                "nonexistent_file.json", data_format="json"
            )

    @pytest.mark.parametrize("data_format", ["json", "csv", "parquet"])
    def test_save_and_load_state_formats(self, tmp_path, builder, data_format):
        """Test save_state and load_state work for all data formats."""
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )
        builder.create_error_row(
            sample_example_id="test_2",
            original_id="id2",
            error_message="Test error",
        )
        results = builder.complete()

        state_file = tmp_path / "human_state.json"

        # Save state
        results.save_state(str(state_file), data_format=data_format)

        # Verify state file was created
        assert state_file.exists()

        # Verify data file was created
        data_file = tmp_path / f"human_state_data.{data_format}"
        assert data_file.exists()

        # Load state
        loaded_results = HumanAnnotationResults.load_state(str(state_file))

        # Verify loaded results match original
        assert loaded_results.run_id == results.run_id
        assert loaded_results.annotator_id == results.annotator_id
        assert loaded_results.task_schemas == results.task_schemas
        assert loaded_results.total_count == results.total_count
        assert loaded_results.succeeded_count == results.succeeded_count
        assert loaded_results.error_count == results.error_count
        assert loaded_results.is_sampled_run == results.is_sampled_run

        # Verify DataFrame data
        assert len(loaded_results.results_data) == len(results.results_data)
        assert set(loaded_results.results_data.columns) == set(
            results.results_data.columns
        )

    def test_save_state_custom_data_filename(self, tmp_path, single_task_builder):
        """Test save_state with custom data filename."""
        single_task_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes"},
            annotation_timestamp=datetime.now(),
        )
        results = single_task_builder.complete()

        state_file = tmp_path / "custom_state.json"
        custom_data_filename = "my_custom_data.json"

        results.save_state(
            str(state_file), data_format="json", data_filename=custom_data_filename
        )

        # Verify custom data file was created
        data_file = tmp_path / custom_data_filename
        assert data_file.exists()

        # Verify loading works with custom filename
        loaded_results = HumanAnnotationResults.load_state(str(state_file))
        assert loaded_results.run_id == results.run_id

    def test_save_state_data_filename_extension_validation(
        self, tmp_path, single_task_builder
    ):
        """Test that save_state validates data filename extension matches format."""
        single_task_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes"},
            annotation_timestamp=datetime.now(),
        )
        results = single_task_builder.complete()

        state_file = tmp_path / "state.json"

        with pytest.raises(
            ResultsDataFormatError,
            match="Unsupported results data format.*json.*for.*wrong_extension.csv",
        ):
            results.save_state(
                str(state_file), data_format="json", data_filename="wrong_extension.csv"
            )

    def test_save_state_creates_directory(self, tmp_path, single_task_builder):
        """Test that save_state creates parent directories if they don't exist."""
        single_task_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes"},
            annotation_timestamp=datetime.now(),
        )
        results = single_task_builder.complete()

        nested_path = tmp_path / "nested" / "directory"
        state_file = nested_path / "state.json"

        results.save_state(str(state_file), data_format="json")

        # Verify directory was created
        assert nested_path.exists()
        assert state_file.exists()

        # Verify loading works
        loaded_results = HumanAnnotationResults.load_state(str(state_file))
        assert loaded_results.run_id == results.run_id

    def test_load_state_error_handling(self, tmp_path, single_task_builder):
        """Test load_state error handling for various failure scenarios."""
        # Test missing file
        with pytest.raises(InvalidFileError, match=STATE_FILE_NOT_FOUND_MSG):
            HumanAnnotationResults.load_state("nonexistent_state.json")

        # Test invalid JSON
        state_file = tmp_path / "invalid.json"
        state_file.write_text("{invalid json")
        with pytest.raises(InvalidFileError, match=INVALID_JSON_STRUCTURE_MSG):
            HumanAnnotationResults.load_state(str(state_file))

        # Test missing required keys
        state_file = tmp_path / "incomplete.json"
        state_file.write_text(
            '{"metadata": {}}'
        )  # Missing data_format and data_filename
        with pytest.raises(InvalidFileError, match=INVALID_JSON_STRUCTURE_MSG):
            HumanAnnotationResults.load_state(str(state_file))

        # Test missing data file
        single_task_builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes"},
            annotation_timestamp=datetime.now(),
        )
        results = single_task_builder.complete()

        state_file = tmp_path / "state.json"
        results.save_state(str(state_file), data_format="json")

        # Delete the data file
        data_file = tmp_path / "state_data.json"
        data_file.unlink()

        with pytest.raises(Exception):  # Will raise FileNotFoundError or similar
            HumanAnnotationResults.load_state(str(state_file))

    # === Integration Tests ===

    def test_annotation_specific_field_preservation(self, tmp_path, builder):
        """Test that serialization preserves annotation-specific fields."""
        builder.create_success_row(
            sample_example_id="test_1",
            original_id="id1",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )
        builder.create_error_row(
            sample_example_id="test_2",
            original_id="id2",
            error_message="Test error",
            error_details_json='{"error_type": "validation"}',
        )
        results = builder.complete()

        # Test via save_state/load_state
        state_file = tmp_path / "test_state.json"
        results.save_state(str(state_file), data_format="json")
        reconstructed = HumanAnnotationResults.load_state(str(state_file))

        # Verify annotation-specific fields are preserved
        assert reconstructed.annotator_id == results.annotator_id
        assert reconstructed.error_count == results.error_count

        # Verify annotation diagnostic fields are preserved in DataFrame
        success_results = reconstructed.get_successful_results()
        if not success_results.is_empty():
            assert "annotation_timestamp" in success_results.columns
