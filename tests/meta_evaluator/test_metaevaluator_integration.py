"""Integration tests for MetaEvaluator with real objects and full workflows.

These tests use real EvalData/SampleEvalData objects and test complete save/load cycles
to verify that the serialization system works correctly in real-world scenarios.
"""

import json
import tempfile
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from meta_evaluator.common.models import Prompt
from meta_evaluator.data import EvalData, SampleEvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.meta_evaluator import MetaEvaluator
from meta_evaluator.meta_evaluator.exceptions import InvalidFileError


class TestMetaEvaluatorIntegration:
    """Integration tests for MetaEvaluator with real data and full workflows."""

    def create_real_eval_data(self, name: str = "integration_test") -> EvalData:
        """Create a real EvalData object with actual data.

        Returns:
            EvalData: A real EvalData instance with sample question/answer data.
        """
        df = pl.DataFrame(
            {
                "question": [
                    "What is 2+2?",
                    "What is the capital of France?",
                    "Who wrote Hamlet?",
                    "What is the square root of 16?",
                    "What year did World War II end?",
                ],
                "answer": ["4", "Paris", "Shakespeare", "4", "1945"],
                "category": ["math", "geography", "literature", "math", "history"],
                "difficulty": ["easy", "easy", "medium", "easy", "medium"],
            }
        )
        return EvalData(name=name, data=df)

    def create_real_sample_eval_data(self) -> SampleEvalData:
        """Create a real SampleEvalData object with actual sampling.

        Returns:
            SampleEvalData: A real SampleEvalData instance with stratified sampling.
        """
        # Create larger base dataset for meaningful sampling
        df = pl.DataFrame(
            {
                "question": [f"Question {i}" for i in range(100)],
                "topic": (["math", "science", "history", "literature"] * 25),
                "difficulty": (["easy", "medium", "hard"] * 33 + ["easy"]),
                "points": list(range(100)),
            }
        )
        base_data = EvalData(name="base_dataset", data=df)

        # Create real sample with actual sampling logic
        return base_data.stratified_sample_by_columns(
            columns=["topic", "difficulty"],
            sample_percentage=0.2,
            sample_name="integration_sample",
            seed=42,
        )

    def create_real_eval_task(self) -> EvalTask:
        """Create a real EvalTask object with actual task schemas.

        Returns:
            EvalTask: A real EvalTask instance with sample task schemas.
        """
        return EvalTask(
            task_schemas={
                "accuracy": ["correct", "incorrect"],
                "difficulty_agreement": None,  # Free form text output
            },
            prompt_columns=["question", "answer"],
            response_columns=["category", "difficulty"],
            answering_method="structured",
        )

    def create_test_prompt(self) -> Prompt:
        """Create a test prompt for judge evaluation.

        Returns:
            Prompt: A test prompt for evaluation tasks.
        """
        return Prompt(
            id="test_integration_prompt",
            prompt="Evaluate the accuracy and difficulty agreement of the given question-answer pair. "
            "For accuracy, determine if the answer is 'correct' or 'incorrect'. "
            "For difficulty_agreement, provide your assessment of the difficulty level.",
        )

    def add_test_judge(self, evaluator: MetaEvaluator) -> None:
        """Add a test judge to the evaluator.

        Args:
            evaluator: The MetaEvaluator instance to add the judge to.
        """
        test_prompt = self.create_test_prompt()
        evaluator.add_judge(
            judge_id="test_judge",
            llm_client="openai",
            model="gpt-4",
            prompt=test_prompt,
        )

    def save_and_load_evaluator(
        self,
        evaluator: MetaEvaluator,
        data_format: Literal["json", "csv", "parquet"] = "json",
        load_data: bool = True,
        load_task: bool = True,
    ) -> MetaEvaluator:
        """Helper to save and load a MetaEvaluator for round-trip testing.

        Returns:
            MetaEvaluator: A loaded MetaEvaluator instance from saved state.
        """
        # Save evaluator in its own project directory
        evaluator.save_state(
            state_filename="test_state.json",
            include_data=True,
            include_task=True,
            data_format=data_format,
        )

        # Load evaluator back from the same project directory
        return MetaEvaluator.load_state(
            project_dir=str(evaluator.project_dir),
            state_filename="test_state.json",
            load_data=load_data,
            load_task=load_task,
        )

    @pytest.mark.integration
    def test_full_roundtrip_with_real_evaldata_json(self, tmp_path):
        """Test complete save/load cycle with real EvalData in JSON format."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        # Create MetaEvaluator with real data
        evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        evaluator.add_data(eval_data)
        evaluator.add_eval_task(eval_task)
        self.add_test_judge(evaluator)

        # Test round-trip
        loaded = self.save_and_load_evaluator(evaluator, "json")

        assert loaded.data is not None

        # Verify complete reconstruction
        assert loaded.data.name == eval_data.name
        assert loaded.data.id_column == eval_data.id_column
        assert loaded.data.data.equals(eval_data.data)
        assert loaded.eval_task is not None
        assert isinstance(loaded.data, EvalData)
        assert not isinstance(loaded.data, SampleEvalData)

    @pytest.mark.integration
    def test_full_roundtrip_with_real_evaldata_csv(self, tmp_path):
        """Test complete save/load cycle with real EvalData in CSV format."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        evaluator.add_data(eval_data)
        evaluator.add_eval_task(eval_task)
        self.add_test_judge(evaluator)

        loaded = self.save_and_load_evaluator(evaluator, "csv")

        assert loaded.data is not None

        # Verify data integrity
        assert loaded.data.name == eval_data.name
        assert loaded.data.id_column == eval_data.id_column
        assert loaded.data.data.equals(eval_data.data)
        assert loaded.eval_task is not None

    @pytest.mark.integration
    def test_full_roundtrip_with_real_evaldata_parquet(self, tmp_path):
        """Test complete save/load cycle with real EvalData in Parquet format."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        evaluator.add_data(eval_data)
        evaluator.add_eval_task(eval_task)
        self.add_test_judge(evaluator)

        loaded = self.save_and_load_evaluator(evaluator, "parquet")

        assert loaded.data is not None

        # Verify data integrity
        assert loaded.data.name == eval_data.name
        assert loaded.data.id_column == eval_data.id_column
        assert loaded.data.data.equals(eval_data.data)
        assert loaded.eval_task is not None

    @pytest.mark.integration
    def test_judge_execution_integration(self, tmp_path):
        """Test actual judge execution with mocked LLM calls."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        evaluator.add_data(eval_data)
        evaluator.add_eval_task(eval_task)
        self.add_test_judge(evaluator)

        # Mock the LLM completion call to avoid actual API calls
        with patch("meta_evaluator.judge.sync_evaluator.completion") as mock_completion:
            # Mock a successful structured response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[
                0
            ].message.content = (
                '{"accuracy": "correct", "difficulty_agreement": "easy"}'
            )
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150
            mock_completion.return_value = mock_response

            # Mock supports_response_schema to return True
            with patch(
                "meta_evaluator.judge.sync_evaluator.supports_response_schema",
                return_value=True,
            ):
                # Run the judge
                results = evaluator.run_judges(save_results=False)

                # Verify results
                assert len(results) == 1
                assert "test_judge" in results

                judge_result = results["test_judge"]
                assert judge_result.total_count == len(eval_data.data)
                assert judge_result.succeeded_count > 0

    @pytest.mark.integration
    def test_sample_eval_data_roundtrip(self, tmp_path):
        """Test complete workflow with real SampleEvalData."""
        sample_data = self.create_real_sample_eval_data()
        eval_task = self.create_real_eval_task()

        evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        evaluator.add_data(sample_data)
        evaluator.add_eval_task(eval_task)
        self.add_test_judge(evaluator)

        # Test round-trip preserves sampling metadata
        loaded = self.save_and_load_evaluator(evaluator, "csv")

        # Verify it's still a SampleEvalData with correct metadata
        assert isinstance(loaded.data, SampleEvalData)
        assert loaded.data.sample_name == "integration_sample"
        assert loaded.data.stratification_columns == ["topic", "difficulty"]
        assert loaded.data.sample_percentage == 0.2
        assert loaded.data.seed == 42

        # Verify data integrity
        assert loaded.data.name == sample_data.name
        assert loaded.data.id_column == sample_data.id_column
        assert loaded.data.data.equals(sample_data.data)

    @pytest.mark.integration
    def test_full_roundtrip_with_eval_task(self, tmp_path):
        """Test complete save/load cycle with real EvalTask."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        # Create MetaEvaluator with real data and evaluation task
        evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        evaluator.add_data(eval_data)
        evaluator.add_eval_task(eval_task)
        self.add_test_judge(evaluator)

        # Test round-trip
        loaded = self.save_and_load_evaluator(evaluator, "json")

        # Verify evaluation task reconstruction
        assert loaded.eval_task is not None
        assert loaded.eval_task.task_schemas == eval_task.task_schemas
        assert loaded.eval_task.prompt_columns == eval_task.prompt_columns
        assert loaded.eval_task.response_columns == eval_task.response_columns
        assert loaded.eval_task.answering_method == eval_task.answering_method
        assert loaded.data is not None

    @pytest.mark.integration
    def test_file_system_integration(self, tmp_path):
        """Test that data files are actually created and readable."""
        eval_data = self.create_real_eval_data("file_test")
        eval_task = self.create_real_eval_task()

        evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        evaluator.add_data(eval_data)
        evaluator.add_eval_task(eval_task)
        self.add_test_judge(evaluator)

        # Save with parquet data format
        evaluator.save_state(
            state_filename="test_state.json",
            include_data=True,
            data_format="parquet",
            include_task=True,
        )

        # Verify files exist in the evaluator's project directory
        project_dir = evaluator.project_dir
        assert (project_dir / "test_state.json").exists()
        data_file = project_dir / "data" / "test_state_data.parquet"
        assert data_file.exists()

        # Verify data file is readable by polars directly
        loaded_df = pl.read_parquet(data_file)
        assert loaded_df.equals(eval_data.data)

        # Verify MetaEvaluator can load it
        loaded = MetaEvaluator.load_state(
            project_dir=str(project_dir),
            state_filename="test_state.json",
            load_data=True,
            load_task=True,
        )
        assert loaded.data is not None

        # Verify data integrity
        assert loaded.data.data.equals(eval_data.data)

    @pytest.mark.integration
    def test_cross_format_data_consistency(self, tmp_path):
        """Test that the same data produces equivalent results across formats."""
        eval_data = self.create_real_eval_data("consistency_test")
        eval_task = self.create_real_eval_task()

        evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        evaluator.add_data(eval_data)
        evaluator.add_eval_task(eval_task)
        self.add_test_judge(evaluator)

        formats: list[Literal["json", "csv", "parquet"]] = [
            "json",
            "csv",
            "parquet",
        ]
        loaded_evaluators = {}

        # Save and load in all formats
        for fmt in formats:
            loaded_evaluators[fmt] = self.save_and_load_evaluator(evaluator, fmt)

        # Verify data is identical across formats
        base_data = loaded_evaluators["json"].data.data
        for fmt in ["csv", "parquet"]:
            assert loaded_evaluators[fmt].data.data.equals(base_data), (
                f"Data mismatch between json and {fmt} formats"
            )

            # Verify metadata is also consistent
            json_meta = loaded_evaluators["json"].data
            fmt_meta = loaded_evaluators[fmt].data
            assert json_meta.name == fmt_meta.name
            assert json_meta.id_column == fmt_meta.id_column

    @pytest.mark.integration
    def test_missing_data_file_integration(self, tmp_path):
        """Test behavior when data file is deleted after saving."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        evaluator.add_data(eval_data)
        evaluator.add_eval_task(eval_task)
        self.add_test_judge(evaluator)

        # Save state first
        evaluator.save_state(
            state_filename="test_state.json",
            include_data=True,
            data_format="csv",
            include_task=True,
        )

        # Delete the data file from the project directory
        project_dir = evaluator.project_dir
        data_file = project_dir / "data" / "test_state_data.csv"
        data_file.unlink()

        # Should raise FileNotFoundError with clear message
        with pytest.raises(FileNotFoundError) as exc:
            MetaEvaluator.load_state(
                project_dir=str(project_dir),
                state_filename="test_state.json",
                load_data=True,
                load_task=True,
            )

        assert "test_state_data.csv" in str(exc.value)

    @pytest.mark.integration
    def test_corrupted_state_file_integration(self):
        """Test handling of corrupted JSON state files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = Path(tmp_dir) / "corrupted.json"
            # Write truncated/invalid JSON
            state_file.write_text('{"version": "1.0", "data":')

            with pytest.raises(InvalidFileError) as exc:
                MetaEvaluator.load_state(
                    project_dir=tmp_dir,
                    state_filename="corrupted.json",
                )

            assert "Invalid JSON structure" in str(exc.value)

    @pytest.mark.integration
    def test_skip_data_loading_integration(self, tmp_path):
        """Test loading state without data (load_data=False)."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        evaluator.add_data(eval_data)
        evaluator.add_eval_task(eval_task)
        self.add_test_judge(evaluator)

        # Save with data
        loaded_with_data = self.save_and_load_evaluator(
            evaluator, "json", load_data=True
        )
        loaded_without_data = self.save_and_load_evaluator(
            evaluator, "json", load_data=False
        )

        # Verify data behavior
        assert loaded_with_data.data is not None
        assert loaded_without_data.data is None

    @pytest.mark.integration
    def test_skip_eval_task_loading(self, tmp_path):
        """Test loading state without evaluation task (load_task=False)."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        evaluator.add_data(eval_data)
        evaluator.add_eval_task(eval_task)
        self.add_test_judge(evaluator)

        # Save with task
        loaded_with_task = self.save_and_load_evaluator(
            evaluator, "json", load_task=True
        )
        loaded_without_task = self.save_and_load_evaluator(
            evaluator, "json", load_task=False
        )
        # Verify tasks
        assert loaded_with_task.eval_task is not None
        assert loaded_without_task.eval_task is None

    @pytest.mark.integration
    def test_large_dataset_performance(self, tmp_path):
        """Test serialization with larger datasets to catch performance issues."""
        # Create larger dataset (1000 rows)
        df = pl.DataFrame(
            {
                "row_id": list(
                    range(1000)
                ),  # Use different name to avoid conflict with auto-generated 'id' column
                "question": [f"Question {i}" for i in range(1000)],
                "answer": [f"Answer {i}" for i in range(1000)],
                "category": (["math", "science", "history", "literature"] * 250),
                "difficulty": (["easy", "medium", "hard"] * 334)[:1000],
                "score": [i * 0.1 for i in range(1000)],
            }
        )
        eval_data = EvalData(name="large_dataset", data=df)
        eval_task = self.create_real_eval_task()

        evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        evaluator.add_data(eval_data)
        evaluator.add_eval_task(eval_task)
        self.add_test_judge(evaluator)

        # Test all formats work with larger data
        formats: list[Literal["json", "csv", "parquet"]] = [
            "json",
            "csv",
            "parquet",
        ]
        for fmt in formats:
            loaded = self.save_and_load_evaluator(evaluator, fmt)

            assert loaded.data is not None

            # Verify data integrity
            assert loaded.data.data.equals(eval_data.data)
            assert len(loaded.data.data) == 1000

    @pytest.mark.integration
    def test_custom_data_filename_integration(self, tmp_path):
        """Test custom data filename functionality."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        evaluator.add_data(eval_data)
        evaluator.add_eval_task(eval_task)
        self.add_test_judge(evaluator)

        custom_data_filename = "my_custom_data.parquet"

        # Save with custom filename
        evaluator.save_state(
            state_filename="test_state.json",
            include_data=True,
            data_format="parquet",
            data_filename=custom_data_filename,
        )

        # Verify custom filename is used in the project directory
        project_dir = evaluator.project_dir
        # Custom data files are saved in the data subdirectory
        custom_data_file = project_dir / "data" / custom_data_filename
        assert custom_data_file.exists()

        # Verify state file references custom filename
        state_file = project_dir / "test_state.json"
        with open(state_file) as f:
            state_data = json.load(f)
        assert state_data["data"]["data_file"] == custom_data_filename

        # Verify loading works with custom filename
        loaded = MetaEvaluator.load_state(
            project_dir=str(project_dir),
            state_filename="test_state.json",
            load_data=True,
        )
        assert loaded.data is not None
        assert loaded.data.data.equals(eval_data.data)

    @pytest.mark.integration
    def test_multiple_judges_integration(self, tmp_path):
        """Test integration with multiple judges."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        evaluator = MetaEvaluator(str(tmp_path / "test_project"))
        evaluator.add_data(eval_data)
        evaluator.add_eval_task(eval_task)

        # Add multiple judges
        test_prompt = self.create_test_prompt()
        evaluator.add_judge(
            judge_id="judge_1",
            llm_client="openai",
            model="gpt-4",
            prompt=test_prompt,
        )

        evaluator.add_judge(
            judge_id="judge_2",
            llm_client="openai",
            model="gpt-3.5-turbo",
            prompt=test_prompt,
        )

        # Test round-trip with multiple judges
        loaded = self.save_and_load_evaluator(evaluator, "json")

        # Verify data integrity
        assert loaded.data is not None
        assert loaded.data.data.equals(eval_data.data)
        assert loaded.eval_task is not None

        # Verify judges are preserved (they should be reconstructable from saved state)
        assert len(evaluator.judge_registry) == 2
        assert "judge_1" in evaluator.judge_registry
        assert "judge_2" in evaluator.judge_registry
