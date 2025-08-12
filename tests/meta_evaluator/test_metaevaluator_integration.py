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

from meta_evaluator.data import EvalData, SampleEvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.llm_client.azureopenai_client import (
    AzureOpenAIClient,
    AzureOpenAIConfig,
)
from meta_evaluator.llm_client.enums import LLMClientEnum
from meta_evaluator.llm_client.openai_client import OpenAIClient, OpenAIConfig
from meta_evaluator.llm_client.serialization import (
    AzureOpenAISerializedState,
    OpenAISerializedState,
)
from meta_evaluator.meta_evaluator import MetaEvaluator


class TestMetaEvaluatorIntegration:
    """Integration tests for MetaEvaluator with real data and full workflows."""

    @pytest.mark.integration
    def create_mock_openai_client(self):
        """Create a properly mocked OpenAI client that passes type checking.

        Returns:
            MagicMock: A mocked OpenAI client with properly configured mock config.
        """
        # Create proper mock config with serialize method
        mock_config = MagicMock(spec=OpenAIConfig)
        mock_config.default_model = "gpt-4"
        mock_config.default_embedding_model = "text-embedding-ada-002"
        mock_config.supports_structured_output = True
        mock_config.supports_logprobs = True

        # Mock the serialize method to return proper state
        mock_state = OpenAISerializedState(
            default_model="gpt-4",
            default_embedding_model="text-embedding-ada-002",
            supports_structured_output=True,
            supports_logprobs=True,
            supports_instructor=True,
        )
        mock_config.serialize.return_value = mock_state

        # Create mock client with config
        mock_client = MagicMock(spec=OpenAIClient)
        mock_client.config = mock_config
        return mock_client

    def create_mock_azure_openai_client(self):
        """Create a properly mocked Azure OpenAI client that passes type checking.

        Returns:
            MagicMock: A mocked Azure OpenAI client with properly configured mock config.
        """
        mock_config = MagicMock(spec=AzureOpenAIConfig)
        mock_config.endpoint = "https://test.openai.azure.com"
        mock_config.api_version = "2024-02-15-preview"
        mock_config.default_model = "gpt-4"
        mock_config.default_embedding_model = "text-embedding-ada-002"
        mock_config.supports_structured_output = True
        mock_config.supports_logprobs = True

        # Mock the serialize method to return proper state
        mock_state = AzureOpenAISerializedState(
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            default_model="gpt-4",
            default_embedding_model="text-embedding-ada-002",
            supports_structured_output=True,
            supports_logprobs=True,
            supports_instructor=True,
        )
        mock_config.serialize.return_value = mock_state

        # Create mock client with config
        mock_client = MagicMock(spec=AzureOpenAIClient)
        mock_client.config = mock_config
        return mock_client

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
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save evaluator
            evaluator.save_state(
                state_filename="test_state.json",
                include_data=True,
                include_task=True,
                data_format=data_format,
            )

            # Load evaluator back
            return MetaEvaluator.load_state(
                project_dir=tmp_dir,
                state_filename="test_state.json",
                load_data=load_data,
                load_task=load_task,
                openai_api_key="test-key",  # Mock key for integration tests
                azure_openai_api_key="test-azure-key",  # Mock Azure key for integration tests
            )

    @pytest.mark.integration
    def test_full_roundtrip_with_real_evaldata_json(self, tmp_path):
        """Test complete save/load cycle with real EvalData in JSON format."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        with patch(
            "meta_evaluator.llm_client.openai_client.OpenAIClient"
        ) as mock_client_class:
            mock_client_class.return_value = self.create_mock_openai_client()

            # Create MetaEvaluator with real data
            evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            evaluator.add_openai()
            evaluator.add_data(eval_data)
            evaluator.add_eval_task(eval_task)

            # Test round-trip
            loaded = self.save_and_load_evaluator(evaluator, "json")

            assert loaded.data is not None

            # Verify complete reconstruction
            assert loaded.data.name == eval_data.name
            assert loaded.data.id_column == eval_data.id_column
            assert loaded.data.data.equals(eval_data.data)
            assert loaded.eval_task is not None
            assert LLMClientEnum.OPENAI in loaded.client_registry
            assert isinstance(loaded.data, EvalData)
            assert not isinstance(loaded.data, SampleEvalData)

    @pytest.mark.integration
    def test_full_roundtrip_with_real_evaldata_csv(self, tmp_path):
        """Test complete save/load cycle with real EvalData in CSV format."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        with patch(
            "meta_evaluator.llm_client.openai_client.OpenAIClient"
        ) as mock_client_class:
            mock_client_class.return_value = self.create_mock_openai_client()

            evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            evaluator.add_openai()
            evaluator.add_data(eval_data)
            evaluator.add_eval_task(eval_task)

            loaded = self.save_and_load_evaluator(evaluator, "csv")

            assert loaded.data is not None

            # Verify data integrity
            assert loaded.data.name == eval_data.name
            assert loaded.data.id_column == eval_data.id_column
            assert loaded.data.data.equals(eval_data.data)
            assert loaded.eval_task is not None
            assert LLMClientEnum.OPENAI in loaded.client_registry

    @pytest.mark.integration
    def test_full_roundtrip_with_real_evaldata_parquet(self, tmp_path):
        """Test complete save/load cycle with real EvalData in Parquet format."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        with patch(
            "meta_evaluator.llm_client.openai_client.OpenAIClient"
        ) as mock_client_class:
            mock_client_class.return_value = self.create_mock_openai_client()

            evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            evaluator.add_openai()
            evaluator.add_data(eval_data)
            evaluator.add_eval_task(eval_task)

            loaded = self.save_and_load_evaluator(evaluator, "parquet")

            assert loaded.data is not None

            # Verify data integrity
            assert loaded.data.name == eval_data.name
            assert loaded.data.id_column == eval_data.id_column
            assert loaded.data.data.equals(eval_data.data)
            assert loaded.eval_task is not None
            assert LLMClientEnum.OPENAI in loaded.client_registry

    @pytest.mark.integration
    def test_multi_client_serialization(self, tmp_path):
        """Test serialization with multiple real LLM clients."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        with (
            patch(
                "meta_evaluator.llm_client.openai_client.OpenAIClient"
            ) as mock_openai_class,
            patch(
                "meta_evaluator.llm_client.azureopenai_client.AzureOpenAIClient"
            ) as mock_azure_class,
        ):
            mock_openai_class.return_value = self.create_mock_openai_client()
            mock_azure_class.return_value = self.create_mock_azure_openai_client()

            evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            evaluator.add_openai()
            evaluator.add_azure_openai()
            evaluator.add_data(eval_data)
            evaluator.add_eval_task(eval_task)

            loaded = self.save_and_load_evaluator(evaluator)

            # Verify both clients are reconstructed
            assert len(loaded.client_registry) == 2
            assert LLMClientEnum.OPENAI in loaded.client_registry
            assert LLMClientEnum.AZURE_OPENAI in loaded.client_registry

            assert loaded.data is not None
            assert loaded.eval_task is not None

            # Verify data is preserved
            assert loaded.data.name == eval_data.name
            assert loaded.data.data.equals(eval_data.data)

    @pytest.mark.integration
    def test_sample_eval_data_roundtrip(self, tmp_path):
        """Test complete workflow with real SampleEvalData."""
        sample_data = self.create_real_sample_eval_data()
        eval_task = self.create_real_eval_task()

        with patch(
            "meta_evaluator.llm_client.openai_client.OpenAIClient"
        ) as mock_client_class:
            mock_client_class.return_value = self.create_mock_openai_client()

            evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            evaluator.add_openai()
            evaluator.add_data(sample_data)
            evaluator.add_eval_task(eval_task)

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

        with patch(
            "meta_evaluator.llm_client.openai_client.OpenAIClient"
        ) as mock_client_class:
            mock_client_class.return_value = self.create_mock_openai_client()

            # Create MetaEvaluator with real data and evaluation task
            evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            evaluator.add_openai()
            evaluator.add_data(eval_data)
            evaluator.add_eval_task(eval_task)

            # Test round-trip
            loaded = self.save_and_load_evaluator(evaluator, "json")

            # Verify evaluation task reconstruction
            assert loaded.eval_task is not None
            assert loaded.eval_task.task_schemas == eval_task.task_schemas
            assert loaded.eval_task.prompt_columns == eval_task.prompt_columns
            assert loaded.eval_task.response_columns == eval_task.response_columns
            assert loaded.eval_task.answering_method == eval_task.answering_method
            assert loaded.data is not None
            assert LLMClientEnum.OPENAI in loaded.client_registry

    @pytest.mark.integration
    def test_file_system_integration(self, tmp_path):
        """Test that data files are actually created and readable."""
        eval_data = self.create_real_eval_data("file_test")
        eval_task = self.create_real_eval_task()

        with patch(
            "meta_evaluator.llm_client.openai_client.OpenAIClient"
        ) as mock_client_class:
            mock_client_class.return_value = self.create_mock_openai_client()
            evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            evaluator.add_openai()
            evaluator.add_data(eval_data)
            evaluator.add_eval_task(eval_task)

            with tempfile.TemporaryDirectory() as tmp_dir:
                # Save with parquet data format
                evaluator.save_state(
                    state_filename="test_state.json",
                    include_data=True,
                    data_format="parquet",
                    include_task=True,
                )

                # Verify files exist
                assert (Path(tmp_dir) / "test_state.json").exists()
                data_file = Path(tmp_dir) / "data" / "test_state_data.parquet"
                assert data_file.exists()

                # Verify data file is readable by polars directly
                loaded_df = pl.read_parquet(data_file)
                assert loaded_df.equals(eval_data.data)

                # Verify MetaEvaluator can load it
                loaded = MetaEvaluator.load_state(
                    project_dir=tmp_dir,
                    state_filename="test_state.json",
                    load_data=True,
                    load_task=True,
                    openai_api_key="test-key",
                    azure_openai_api_key="test-azure-key",
                )
                assert loaded.data is not None

                # Verify data integrity
                assert loaded.data.data.equals(eval_data.data)

    @pytest.mark.integration
    def test_cross_format_data_consistency(self, tmp_path):
        """Test that the same data produces equivalent results across formats."""
        eval_data = self.create_real_eval_data("consistency_test")
        eval_task = self.create_real_eval_task()

        with patch(
            "meta_evaluator.llm_client.openai_client.OpenAIClient"
        ) as mock_client_class:
            mock_client_class.return_value = self.create_mock_openai_client()
            evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            evaluator.add_openai()
            evaluator.add_data(eval_data)
            evaluator.add_eval_task(eval_task)

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

        with patch(
            "meta_evaluator.llm_client.openai_client.OpenAIClient"
        ) as mock_client_class:
            mock_client_class.return_value = self.create_mock_openai_client()
            evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            evaluator.add_openai()
            evaluator.add_data(eval_data)
            evaluator.add_eval_task(eval_task)

            with tempfile.TemporaryDirectory() as tmp_dir:
                state_file = Path(tmp_dir) / "state.json"
                evaluator.save_state(
                    state_filename="test_state.json",
                    include_data=True,
                    data_format="csv",
                    include_task=True,
                )

                # Delete the data file
                data_file = state_file.parent / "state_data.csv"
                data_file.unlink()

                # Should raise FileNotFoundError with clear message
                with pytest.raises(FileNotFoundError) as exc:
                    MetaEvaluator.load_state(
                        project_dir=tmp_dir,
                        state_filename="test_state.json",
                        load_data=True,
                        load_task=True,
                        openai_api_key="test-key",
                        azure_openai_api_key="test-azure-key",
                    )

                assert "state_data.csv" in str(exc.value)

    @pytest.mark.integration
    def test_corrupted_state_file_integration(self):
        """Test handling of corrupted JSON state files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = Path(tmp_dir) / "corrupted.json"
            # Write truncated/invalid JSON
            state_file.write_text('{"version": "1.0", "client_registry":')

            with pytest.raises(ValueError) as exc:
                MetaEvaluator.load_state(
                    project_dir=tmp_dir,
                    state_filename="corrupted.json",
                    openai_api_key="test-key",
                    azure_openai_api_key="test-azure-key",
                )

            assert "Invalid JSON structure" in str(exc.value)

    @pytest.mark.integration
    def test_skip_data_loading_integration(self, tmp_path):
        """Test loading state without data (load_data=False)."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        with patch(
            "meta_evaluator.llm_client.openai_client.OpenAIClient"
        ) as mock_client_class:
            mock_client_class.return_value = self.create_mock_openai_client()
            evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            evaluator.add_openai()
            evaluator.add_data(eval_data)
            evaluator.add_eval_task(eval_task)

            # Save with data
            loaded_with_data = self.save_and_load_evaluator(
                evaluator, "json", load_data=True
            )
            loaded_without_data = self.save_and_load_evaluator(
                evaluator, "json", load_data=False
            )

            # Verify client registry is loaded in both cases
            assert LLMClientEnum.OPENAI in loaded_with_data.client_registry
            assert LLMClientEnum.OPENAI in loaded_without_data.client_registry

            # Verify data behavior
            assert loaded_with_data.data is not None
            assert loaded_without_data.data is None

    @pytest.mark.integration
    def test_skip_eval_task_loading(self, tmp_path):
        """Test loading state without evaluation task (load_task=False)."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        with patch(
            "meta_evaluator.llm_client.openai_client.OpenAIClient"
        ) as mock_client_class:
            mock_client_class.return_value = self.create_mock_openai_client()

            evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            evaluator.add_openai()
            evaluator.add_data(eval_data)
            evaluator.add_eval_task(eval_task)

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

        with patch(
            "meta_evaluator.llm_client.openai_client.OpenAIClient"
        ) as mock_client_class:
            mock_client_class.return_value = self.create_mock_openai_client()
            evaluator = MetaEvaluator(str(tmp_path / "test_project"))
            evaluator.add_openai()
            evaluator.add_data(eval_data)
            evaluator.add_eval_task(eval_task)

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
    def test_custom_data_filename_integration(self):
        """Test custom data filename functionality."""
        eval_data = self.create_real_eval_data()
        eval_task = self.create_real_eval_task()

        with patch(
            "meta_evaluator.llm_client.openai_client.OpenAIClient"
        ) as mock_client_class:
            mock_client_class.return_value = self.create_mock_openai_client()
            evaluator = MetaEvaluator()
            evaluator.add_openai()
            evaluator.add_data(eval_data)
            evaluator.add_eval_task(eval_task)

            with tempfile.TemporaryDirectory() as tmp_dir:
                state_file = Path(tmp_dir) / "state.json"
                custom_data_filename = "my_custom_data.parquet"

                # Save with custom filename
                evaluator.save_state(
                    state_filename="test_state.json",
                    include_data=True,
                    data_format="parquet",
                    data_filename=custom_data_filename,
                )

                # Verify custom filename is used
                custom_data_file = state_file.parent / custom_data_filename
                assert custom_data_file.exists()

                # Verify state file references custom filename
                with open(state_file) as f:
                    state_data = json.load(f)
                assert state_data["data"]["data_file"] == custom_data_filename

                # Verify loading works with custom filename
                loaded = MetaEvaluator.load_state(
                    project_dir=tmp_dir,
                    state_filename="test_state.json",
                    load_data=True,
                    openai_api_key="test-key",
                    azure_openai_api_key="test-azure-key",
                )
                assert loaded.data is not None
                assert loaded.data.data.equals(eval_data.data)
