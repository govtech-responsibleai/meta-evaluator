"""Tests for multi-label results storage round-trips (Section 4)."""

import pytest

from meta_evaluator.eval_task import MultiLabelSchema
from meta_evaluator.results import HumanAnnotationResults, HumanAnnotationResultsBuilder


@pytest.fixture
def multilabel_results() -> HumanAnnotationResults:
    """HumanAnnotationResults with one multi-label and one single-select task.

    Returns:
        HumanAnnotationResults: Completed results with two annotated samples.
    """
    task_schemas = {
        "harm": MultiLabelSchema(outcomes=["hateful", "insults", "sexual"]),
        "sentiment": ["positive", "negative"],
    }
    builder = HumanAnnotationResultsBuilder(
        run_id="run1",
        annotator_id="ann1",
        task_schemas=task_schemas,
        expected_ids=["s1", "s2"],
        required_tasks=["harm", "sentiment"],
    )
    builder.create_success_row(
        sample_example_id="run1_0",
        original_id="s1",
        outcomes={"harm": ["hateful", "FALSE", "sexual"], "sentiment": "positive"},
    )
    builder.create_success_row(
        sample_example_id="run1_1",
        original_id="s2",
        outcomes={"harm": ["FALSE", "FALSE", "FALSE"], "sentiment": "negative"},
    )
    return builder.complete()


class TestMultiLabelResultsRoundTrip:
    """Parquet and CSV round-trips preserve the ordered vector and slot order."""

    def test_parquet_roundtrip(self, multilabel_results, tmp_path):
        """Parquet preserves the multi-label vector and slot order."""
        state_file = str(tmp_path / "results_metadata.json")
        multilabel_results.save_state(state_file=state_file, data_format="parquet")

        loaded = HumanAnnotationResults.load_state(state_file)
        harm = loaded.results_data.sort("original_id")["harm"].to_list()
        assert harm[0] == ["hateful", "FALSE", "sexual"]
        assert harm[1] == ["FALSE", "FALSE", "FALSE"]
        # The schema is retained as a MultiLabelSchema.
        assert isinstance(loaded.task_schemas["harm"], MultiLabelSchema)
        assert loaded.task_schemas["harm"].outcomes == [
            "hateful",
            "insults",
            "sexual",
        ]

    def test_csv_roundtrip(self, multilabel_results, tmp_path):
        """CSV JSON-encodes the vector on write and decodes it on read."""
        state_file = str(tmp_path / "results_metadata.json")
        multilabel_results.save_state(state_file=state_file, data_format="csv")

        loaded = HumanAnnotationResults.load_state(state_file)
        harm = loaded.results_data.sort("original_id")["harm"].to_list()
        assert harm[0] == ["hateful", "FALSE", "sexual"]
        assert harm[1] == ["FALSE", "FALSE", "FALSE"]
        # Single-select column is unaffected by the encoding.
        sentiment = loaded.results_data.sort("original_id")["sentiment"].to_list()
        assert sentiment == ["positive", "negative"]

    def test_csv_cell_is_json_encoded_on_disk(self, multilabel_results, tmp_path):
        """The multi-label CSV cell is a JSON array string on disk."""
        state_file = str(tmp_path / "results_metadata.json")
        multilabel_results.save_state(state_file=state_file, data_format="csv")

        csv_path = tmp_path / "results_metadata_data.csv"
        text = csv_path.read_text()
        # JSON-encoded vector appears as a quoted array in the CSV.
        assert '[""hateful"", ""FALSE"", ""sexual""]' in text or (
            '["hateful", "FALSE", "sexual"]' in text
        )
