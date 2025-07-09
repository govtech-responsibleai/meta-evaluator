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

# -------------------------
# Fixtures and Test Helpers
# -------------------------


@pytest.fixture
def example_task_schemas():
    """Provide example task schemas for testing.

    Returns:
        dict: A dictionary mapping task names to their valid outcome values.
              Example: {"task1": ["yes", "no"], "task2": ["good", "bad"]}
    """
    return {"task1": ["yes", "no"], "task2": ["good", "bad"]}


@pytest.fixture
def example_expected_ids():
    """Provide example expected IDs for testing.

    Returns:
        list: A list of expected sample IDs that should be annotated.
              Example: ["id1", "id2"]
    """
    return ["id1", "id2"]


@pytest.fixture
def example_builder(example_task_schemas):
    """Provide a complete HumanAnnotationResultsBuilder for testing.

    Args:
        example_task_schemas: Fixture providing task schemas

    Returns:
        HumanAnnotationResultsBuilder: A fully configured builder object with
                                     run_id="run_001", annotator_id="annotator_1",
                                     and other test values.
    """
    return HumanAnnotationResultsBuilder(
        run_id="run_001",
        annotator_id="annotator_1",
        task_schemas=example_task_schemas,
        is_sampled_run=False,
    )


@pytest.fixture
def builder(example_builder):
    """Provide a HumanAnnotationResultsBuilder instance for testing.

    Args:
        example_builder: Fixture providing the builder

    Returns:
        HumanAnnotationResultsBuilder: A builder instance ready to create
                                      annotation result rows for testing.
    """
    return example_builder


# -------------------------
# Model Validation Tests
# -------------------------


def test_original_id_in_expected_ids(example_task_schemas, example_expected_ids):
    """Test that creating a row with an original_id not in expected_ids raises an error."""
    builder = HumanAnnotationResultsBuilder(
        run_id="run_001",
        annotator_id="annotator_1",
        task_schemas=example_task_schemas,
        is_sampled_run=False,
        expected_ids=example_expected_ids,
    )
    with pytest.raises(ValueError):
        builder.create_success_row(
            sample_example_id="sample_1",
            original_id="not_in_expected",
            outcomes={"task1": "yes", "task2": "good"},
            annotation_timestamp=datetime.now(),
        )


def test_builder_initialization(example_builder):
    """Test that the builder is initialized correctly."""
    assert example_builder.run_id == "run_001"
    assert example_builder.evaluator_id == "annotator_1"
    assert len(example_builder.task_schemas) > 0


def test_error_when_original_id_already_exists(builder):
    """Test that adding a row with a duplicate original_id raises an error."""
    builder.create_success_row(
        sample_example_id="sample_1",
        original_id="id1",
        outcomes={"task1": "yes", "task2": "good"},
        annotation_timestamp=datetime.now(),
    )
    with pytest.raises(ValueError):
        builder.create_success_row(
            sample_example_id="sample_2",
            original_id="id1",  # Duplicate
            outcomes={"task1": "no", "task2": "bad"},
            annotation_timestamp=datetime.now(),
        )


def test_field_tag_selection():
    """Test that HumanAnnotationResultRow tag-based field selection works as expected."""
    metadata_fields = HumanAnnotationResultRow.get_metadata_fields()
    error_fields = HumanAnnotationResultRow.get_error_fields()
    diagnostic_fields = HumanAnnotationResultRow.get_annotation_diagnostic_fields()
    assert "sample_example_id" in metadata_fields
    assert "error_message" in error_fields
    assert "annotation_timestamp" in diagnostic_fields


# -------------------------
# DataFrame Operations Tests
# -------------------------


def test_create_success_row_matches_task_schema(builder):
    """Test that create_success_row creates a row matching the eval_task task_schema."""
    builder.create_success_row(
        sample_example_id="sample_1",
        original_id="id1",
        outcomes={"task1": "yes", "task2": "good"},
        annotation_timestamp=datetime.now(),
    )
    assert builder.completed_count == 1
    # Check that the row contains the correct outcomes
    row = builder._results["id1"]
    assert row.task1 == "yes"
    assert row.task2 == "good"


def test_complete_function(builder):
    """Test that the complete() function returns a HumanAnnotationResults object with correct counts."""
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
    assert isinstance(results, HumanAnnotationResults)
    assert results.succeeded_count == 2
    assert results.error_count == 0
    assert results.total_count == 2  # from expected_ids


def test_get_successful_and_failed_results(builder):
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


# -------------------------
# Persistence Tests
# -------------------------


@pytest.mark.parametrize("data_format", ["json", "csv", "parquet"])
def test_write_and_load_data(tmp_path, builder, data_format):
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


def test_save_and_load_state(tmp_path, builder):
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
