"""Test suite for the Judge class with comprehensive path coverage."""

import pytest
from unittest.mock import Mock
import polars as pl
from datetime import datetime

from meta_evaluator.judge import Judge
from meta_evaluator.judge.models import EvaluationStatusEnum
from meta_evaluator.judge.exceptions import IncorrectClientError
from meta_evaluator.data import EvalData
from meta_evaluator.evaluation_task import EvaluationTask
from meta_evaluator.llm_client import LLMClientEnum, LLMClient
from meta_evaluator.llm_client.models import (
    Message,
    RoleEnum,
    LLMResponse,
    LLMUsage,
    ParseResult,
    ParseError,
    ErrorType,
)
from meta_evaluator.llm_client.exceptions import LLMAPIError
from meta_evaluator.common.models import Prompt


class TestJudge:
    """Comprehensive test suite for Judge class achieving 100% path coverage."""

    @pytest.fixture
    def sample_eval_data(self) -> EvalData:
        """Provides a valid EvalData instance for testing.

        Returns:
            EvalData: A valid EvalData instance.
        """
        df = pl.DataFrame(
            {
                "question": ["What is 2+2?", "What is 3+3?", "What is 4+4?"],
                "model_response": ["Four", "Six", "Eight"],
                "difficulty": ["easy", "easy", "medium"],
            }
        )

        return EvalData(
            name="test_dataset",
            data=df,
            input_columns=["question"],
            output_columns=["model_response"],
            metadata_columns=["difficulty"],
        )

    @pytest.fixture
    def empty_eval_data(self) -> EvalData:
        """Provides an EvalData instance with empty DataFrame for testing.

        Returns:
            EvalData: An EvalData instance with empty DataFrame.
        """
        # Create a properly initialized EvalData then manually empty it
        df = pl.DataFrame({"question": ["temp"], "model_response": ["temp"]})

        eval_data = EvalData(
            name="empty_test",
            data=df,
            input_columns=["question"],
            output_columns=["model_response"],
        )

        # Replace with empty DataFrame to simulate the empty condition
        eval_data.data = df.clear()
        return eval_data

    @pytest.fixture
    def test_prompt(self) -> Prompt:
        """Provides a test prompt for judge configuration.

        Returns:
            Prompt: A test prompt instance.
        """
        return Prompt(
            id="test_prompt_1",
            prompt="Analyze the sentiment of the following text and respond with positive, negative, or neutral.",
        )

    @pytest.fixture
    def structured_evaluation_task(self) -> EvaluationTask:
        """Provides a structured evaluation task.

        Returns:
            EvaluationTask: A structured evaluation task instance.
        """
        return EvaluationTask(
            task_name="sentiment_analysis",
            outcomes=["positive", "negative", "neutral"],
            input_columns=["question"],
            output_columns=["model_response"],
            answering_method="structured",
        )

    @pytest.fixture
    def xml_evaluation_task(self) -> EvaluationTask:
        """Provides an XML evaluation task.

        Returns:
            EvaluationTask: An XML evaluation task instance.
        """
        return EvaluationTask(
            task_name="sentiment_analysis",
            outcomes=["positive", "negative", "neutral"],
            input_columns=["question"],
            output_columns=["model_response"],
            answering_method="xml",
        )

    @pytest.fixture
    def skip_evaluation_task(self) -> EvaluationTask:
        """Provides an evaluation task with skip function.

        Returns:
            EvaluationTask: An evaluation task with skip function.
        """

        def skip_medium_difficulty(row: dict) -> bool:
            return row.get("difficulty") == "medium"

        return EvaluationTask(
            task_name="sentiment_analysis",
            outcomes=["positive", "negative", "neutral"],
            input_columns=["question"],
            output_columns=["model_response"],
            answering_method="structured",
            skip_function=skip_medium_difficulty,
        )

    @pytest.fixture
    def error_skip_evaluation_task(self) -> EvaluationTask:
        """Provides an evaluation task with error-throwing skip function.

        Returns:
            EvaluationTask: An evaluation task with error-throwing skip function.
        """

        def error_skip_function(row: dict) -> bool:
            raise ValueError("Skip function failed")

        return EvaluationTask(
            task_name="sentiment_analysis",
            outcomes=["positive", "negative", "neutral"],
            input_columns=["question"],
            output_columns=["model_response"],
            answering_method="structured",
            skip_function=error_skip_function,
        )

    @pytest.fixture
    def mock_llm_client(self) -> Mock:
        """Provides a mock LLM client with correct enum value.

        Returns:
            Mock: A mock LLM client instance.
        """
        client = Mock(spec=LLMClient)
        client.enum_value = LLMClientEnum.OPENAI
        return client

    @pytest.fixture
    def wrong_llm_client(self) -> Mock:
        """Provides a mock LLM client with wrong enum value.

        Returns:
            Mock: A mock LLM client instance.
        """
        client = Mock(spec=LLMClient)
        client.enum_value = LLMClientEnum.ANTHROPIC
        return client

    @pytest.fixture
    def structured_judge(self, structured_evaluation_task, test_prompt) -> Judge:
        """Provides a Judge configured for structured output.

        Returns:
            Judge: A Judge instance configured for structured output.
        """
        return Judge(
            id="test_structured_judge",
            evaluation_task=structured_evaluation_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=test_prompt,
        )

    @pytest.fixture
    def xml_judge(self, xml_evaluation_task, test_prompt) -> Judge:
        """Provides a Judge configured for XML output.

        Returns:
            Judge: A Judge instance configured for XML output.
        """
        return Judge(
            id="test_xml_judge",
            evaluation_task=xml_evaluation_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=test_prompt,
        )

    @pytest.fixture
    def skip_judge(self, skip_evaluation_task, test_prompt) -> Judge:
        """Provides a Judge with skip function.

        Returns:
            Judge: A Judge instance with skip function.
        """
        return Judge(
            id="test_skip_judge",
            evaluation_task=skip_evaluation_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=test_prompt,
        )

    @pytest.fixture
    def error_skip_judge(self, error_skip_evaluation_task, test_prompt) -> Judge:
        """Provides a Judge with error-throwing skip function.

        Returns:
            Judge: A Judge instance with error-throwing skip function.
        """
        return Judge(
            id="test_error_skip_judge",
            evaluation_task=error_skip_evaluation_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=test_prompt,
        )

    # === Judge Validation Tests ===

    def test_judge_id_validation_valid(self, structured_evaluation_task, test_prompt):
        """Test that valid Judge IDs pass validation."""
        valid_ids = ["valid_id", "ValidID123", "test_123", "_underscore_start"]

        for valid_id in valid_ids:
            judge = Judge(
                id=valid_id,
                evaluation_task=structured_evaluation_task,
                llm_client_enum=LLMClientEnum.OPENAI,
                model="gpt-4",
                prompt=test_prompt,
            )
            assert judge.id == valid_id

    def test_judge_id_validation_invalid(self, structured_evaluation_task, test_prompt):
        """Test that invalid Judge IDs raise ValueError."""
        invalid_ids = ["invalid-id", "invalid id", "123invalid", "invalid.id", ""]

        for invalid_id in invalid_ids:
            with pytest.raises(
                ValueError,
                match="id must only contain alphanumeric characters and underscores",
            ):
                Judge(
                    id=invalid_id,
                    evaluation_task=structured_evaluation_task,
                    llm_client_enum=LLMClientEnum.OPENAI,
                    model="gpt-4",
                    prompt=test_prompt,
                )

    def test_judge_immutability(self, structured_judge):
        """Test that Judge instances are immutable after creation."""
        with pytest.raises(Exception):
            structured_judge.id = "new_id"

    # === Pre-Loop Exception Tests ===

    def test_incorrect_client_error(
        self, structured_judge, wrong_llm_client, sample_eval_data
    ):
        """Test that mismatched LLM client raises IncorrectClientError."""
        with pytest.raises(IncorrectClientError):
            structured_judge.evaluate_eval_data(
                sample_eval_data, wrong_llm_client, "test_run"
            )

    # === Skip Function Tests ===

    def test_skip_function_success(self, skip_judge, mock_llm_client, sample_eval_data):
        """Test successful skip function execution creates SKIPPED rows."""
        # Mock successful structured response for non-skipped rows
        mock_response = Mock()
        mock_response.result_outcome = "positive"
        mock_usage = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_llm_response = LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="gpt-4",
            messages=[Message(role=RoleEnum.ASSISTANT, content="positive")],
            usage=mock_usage,
        )
        mock_llm_client.prompt_with_structured_response.return_value = (
            mock_response,
            mock_llm_response,
        )

        results = skip_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "test_run"
        )

        # Should have 1 skipped row (medium difficulty) and 2 success rows
        assert results.skipped_examples_count == 1
        assert results.succeeded_examples_count == 2
        assert results.total_examples_count == 3

        skipped_rows = results.get_skipped_results()
        assert len(skipped_rows) == 1

    def test_skip_function_error(
        self, error_skip_judge, mock_llm_client, sample_eval_data
    ):
        """Test skip function errors are caught and create OTHER_ERROR rows."""
        results = error_skip_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "test_run"
        )

        # All rows should be OTHER_ERROR due to skip function failures
        assert results.other_error_examples_count == 3
        assert results.succeeded_examples_count == 0
        assert results.total_examples_count == 3

        error_rows = results.get_failed_results()
        assert len(error_rows) == 3

        for row in error_rows.iter_rows(named=True):
            assert row["status"] == EvaluationStatusEnum.OTHER_ERROR.value
            assert "Skip function failed" in row["error_message"]

    # === Structured Response Tests ===

    def test_structured_success_path(
        self, structured_judge, mock_llm_client, sample_eval_data
    ):
        """Test successful structured response evaluation creates SUCCESS rows."""
        # Mock successful structured response
        mock_response = Mock()
        mock_response.result_outcome = "positive"
        mock_usage = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_llm_response = LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="gpt-4",
            messages=[Message(role=RoleEnum.ASSISTANT, content="positive")],
            usage=mock_usage,
        )
        mock_llm_client.prompt_with_structured_response.return_value = (
            mock_response,
            mock_llm_response,
        )

        results = structured_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "test_run"
        )

        assert results.succeeded_examples_count == 3
        assert results.llm_error_examples_count == 0
        assert results.parsing_error_examples_count == 0
        assert results.other_error_examples_count == 0

        success_rows = results.get_successful_results()
        assert len(success_rows) == 3

        for row in success_rows.iter_rows(named=True):
            assert row["status"] == EvaluationStatusEnum.SUCCESS.value
            assert row["sentiment_analysis"] == "positive"

    def test_structured_llm_error(
        self, structured_judge, mock_llm_client, sample_eval_data
    ):
        """Test LLM API errors in structured mode create LLM_ERROR rows."""
        # Mock LLM API error
        mock_llm_client.prompt_with_structured_response.side_effect = LLMAPIError(
            "API Error", LLMClientEnum.OPENAI, Exception("Network error")
        )

        results = structured_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "test_run"
        )

        assert results.llm_error_examples_count == 3
        assert results.succeeded_examples_count == 0

        error_rows = results.get_failed_results()
        assert len(error_rows) == 3

        for row in error_rows.iter_rows(named=True):
            assert row["status"] == EvaluationStatusEnum.LLM_ERROR.value
            assert row["error_type"] == "LLM_API_ERROR"

    def test_structured_parsing_error(
        self, structured_judge, mock_llm_client, sample_eval_data
    ):
        """Test structured response parsing errors create PARSING_ERROR rows."""
        # Mock response without result_outcome attribute
        mock_response = Mock(spec=[])  # Empty spec means no attributes
        mock_usage = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_llm_response = LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="gpt-4",
            messages=[Message(role=RoleEnum.ASSISTANT, content="Raw LLM response")],
            usage=mock_usage,
        )
        mock_llm_client.prompt_with_structured_response.return_value = (
            mock_response,
            mock_llm_response,
        )

        results = structured_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "test_run"
        )

        assert results.parsing_error_examples_count == 3
        assert results.succeeded_examples_count == 0

        error_rows = results.get_failed_results()
        assert len(error_rows) == 3

        for row in error_rows.iter_rows(named=True):
            assert row["status"] == EvaluationStatusEnum.PARSING_ERROR.value
            assert row["error_type"] == "PARSING_ERROR"
            assert row["llm_raw_response_content"] == "Raw LLM response"

    def test_structured_other_error(
        self, structured_judge, mock_llm_client, sample_eval_data
    ):
        """Test unexpected errors in structured mode create OTHER_ERROR rows."""
        # Mock unexpected error during evaluation
        mock_llm_client.prompt_with_structured_response.side_effect = RuntimeError(
            "Unexpected error"
        )

        results = structured_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "test_run"
        )

        assert results.other_error_examples_count == 3
        assert results.succeeded_examples_count == 0

        error_rows = results.get_failed_results()
        for row in error_rows.iter_rows(named=True):
            assert row["status"] == EvaluationStatusEnum.OTHER_ERROR.value
            assert row["error_type"] == "OTHER_ERROR"
            assert "Unexpected error" in row["error_message"]

    # === XML Response Tests ===

    def test_xml_success_path(self, xml_judge, mock_llm_client, sample_eval_data):
        """Test successful XML response evaluation creates SUCCESS rows."""
        # Mock successful XML parsing
        mock_parse_result = ParseResult(data={"result_outcome": "positive"}, errors=[])
        mock_usage = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_llm_response = LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="gpt-4",
            messages=[
                Message(
                    role=RoleEnum.ASSISTANT,
                    content="<result_outcome>positive</result_outcome>",
                )
            ],
            usage=mock_usage,
        )
        mock_llm_client.prompt_with_xml_tags.return_value = (
            mock_parse_result,
            mock_llm_response,
        )

        results = xml_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "test_run"
        )

        assert results.succeeded_examples_count == 3
        assert results.parsing_error_examples_count == 0

        success_rows = results.get_successful_results()
        assert len(success_rows) == 3

        for row in success_rows.iter_rows(named=True):
            assert row["status"] == EvaluationStatusEnum.SUCCESS.value
            assert row["sentiment_analysis"] == "positive"

    def test_xml_llm_error(self, xml_judge, mock_llm_client, sample_eval_data):
        """Test LLM API errors in XML mode create LLM_ERROR rows."""
        # Mock LLM API error
        mock_llm_client.prompt_with_xml_tags.side_effect = LLMAPIError(
            "API Error", LLMClientEnum.OPENAI, Exception("Network error")
        )

        results = xml_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "test_run"
        )

        assert results.llm_error_examples_count == 3
        assert results.succeeded_examples_count == 0

    def test_xml_parsing_failure(self, xml_judge, mock_llm_client, sample_eval_data):
        """Test XML parsing failures create PARSING_ERROR rows."""
        # Mock failed XML parsing
        mock_parse_result = ParseResult(
            data={},
            errors=[
                ParseError(
                    error_type=ErrorType.TAG_NOT_FOUND,
                    tag_name="result_outcome",
                    message="Tag not found",
                )
            ],
        )
        mock_usage = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_llm_response = LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="gpt-4",
            messages=[
                Message(role=RoleEnum.ASSISTANT, content="Response without tags")
            ],
            usage=mock_usage,
        )
        mock_llm_client.prompt_with_xml_tags.return_value = (
            mock_parse_result,
            mock_llm_response,
        )

        results = xml_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "test_run"
        )

        assert results.parsing_error_examples_count == 3
        assert results.succeeded_examples_count == 0

        error_rows = results.get_failed_results()
        for row in error_rows.iter_rows(named=True):
            assert row["status"] == EvaluationStatusEnum.PARSING_ERROR.value
            assert "XML parsing failed" in row["error_message"]

    def test_xml_missing_tag_in_successful_parse(
        self, xml_judge, mock_llm_client, sample_eval_data
    ):
        """Test XML parsing success but missing result_outcome tag creates PARSING_ERROR rows."""
        # Mock successful parsing but missing the specific tag we need
        mock_parse_result = ParseResult(
            data={"other_tag": "value"},  # Missing result_outcome
            errors=[],
        )
        mock_usage = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_llm_response = LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="gpt-4",
            messages=[
                Message(role=RoleEnum.ASSISTANT, content="<other_tag>value</other_tag>")
            ],
            usage=mock_usage,
        )
        mock_llm_client.prompt_with_xml_tags.return_value = (
            mock_parse_result,
            mock_llm_response,
        )

        results = xml_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "test_run"
        )

        assert results.parsing_error_examples_count == 3
        assert results.succeeded_examples_count == 0

    def test_xml_other_error(self, xml_judge, mock_llm_client, sample_eval_data):
        """Test unexpected errors in XML mode create OTHER_ERROR rows."""
        # Mock unexpected error
        mock_llm_client.prompt_with_xml_tags.side_effect = RuntimeError(
            "Unexpected XML error"
        )

        results = xml_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "test_run"
        )

        assert results.other_error_examples_count == 3
        assert results.succeeded_examples_count == 0

    # === Mixed Outcome Tests ===

    def test_mixed_outcomes_structured(
        self, structured_judge, mock_llm_client, sample_eval_data
    ):
        """Test handling of mixed success/error outcomes in structured mode."""
        # Create different responses for each call
        responses = []

        # First call: Success
        mock_success = Mock()
        mock_success.result_outcome = "positive"
        mock_usage = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_llm_response = LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="gpt-4",
            messages=[Message(role=RoleEnum.ASSISTANT, content="positive")],
            usage=mock_usage,
        )
        responses.append((mock_success, mock_llm_response))

        # Second call: LLM Error
        responses.append(
            LLMAPIError("API Error", LLMClientEnum.OPENAI, Exception("Error"))
        )

        # Third call: Parsing Error (no result_outcome attribute)
        mock_parse_error = Mock(spec=[])
        mock_usage_2 = LLMUsage(prompt_tokens=8, completion_tokens=3, total_tokens=11)
        mock_llm_response_2 = LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="gpt-4",
            messages=[Message(role=RoleEnum.ASSISTANT, content="malformed")],
            usage=mock_usage_2,
        )
        responses.append((mock_parse_error, mock_llm_response_2))

        mock_llm_client.prompt_with_structured_response.side_effect = responses

        results = structured_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "test_run"
        )

        # Should have one of each type
        assert results.succeeded_examples_count == 1
        assert results.llm_error_examples_count == 1
        assert results.parsing_error_examples_count == 1
        assert results.total_examples_count == 3

    def test_mixed_outcomes_xml(self, xml_judge, mock_llm_client, sample_eval_data):
        """Test handling of mixed success/error outcomes in XML mode."""
        # Create different responses for each call
        responses = []

        # First call: Success
        mock_success_parse = ParseResult(data={"result_outcome": "positive"}, errors=[])
        mock_usage = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_llm_response = LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="gpt-4",
            messages=[
                Message(
                    role=RoleEnum.ASSISTANT,
                    content="<result_outcome>positive</result_outcome>",
                )
            ],
            usage=mock_usage,
        )
        responses.append((mock_success_parse, mock_llm_response))

        # Second call: LLM Error
        responses.append(
            LLMAPIError("API Error", LLMClientEnum.OPENAI, Exception("Error"))
        )

        # Third call: Parsing Error
        mock_parse_error = ParseResult(
            data={},
            errors=[
                ParseError(
                    error_type=ErrorType.TAG_NOT_FOUND,
                    tag_name="result_outcome",
                    message="Missing tag",
                )
            ],
        )
        mock_usage_2 = LLMUsage(prompt_tokens=8, completion_tokens=3, total_tokens=11)
        mock_llm_response_2 = LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="gpt-4",
            messages=[Message(role=RoleEnum.ASSISTANT, content="no tags")],
            usage=mock_usage_2,
        )
        responses.append((mock_parse_error, mock_llm_response_2))

        mock_llm_client.prompt_with_xml_tags.side_effect = responses

        results = xml_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "test_run"
        )

        # Should have mixed outcomes
        assert results.succeeded_examples_count == 1
        assert results.llm_error_examples_count == 1
        assert results.parsing_error_examples_count == 1
        assert results.total_examples_count == 3

    # === Result Validation Tests ===

    def test_judge_results_creation_and_validation(
        self, structured_judge, mock_llm_client, sample_eval_data
    ):
        """Test that JudgeResults is properly created and validated."""
        # Mock successful responses
        mock_response = Mock()
        mock_response.result_outcome = "positive"
        mock_usage = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_llm_response = LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="gpt-4",
            messages=[Message(role=RoleEnum.ASSISTANT, content="positive sentiment")],
            usage=mock_usage,
        )
        mock_llm_client.prompt_with_structured_response.return_value = (
            mock_response,
            mock_llm_response,
        )

        results = structured_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "test_run_123"
        )

        # Test JudgeResults metadata
        assert results.run_id == "test_run_123"
        assert results.judge_id == "test_structured_judge"
        assert results.task_name == "sentiment_analysis"
        assert results.task_outcomes == ["positive", "negative", "neutral"]
        assert results.llm_client_enum == LLMClientEnum.OPENAI
        assert results.model_used == "gpt-4"
        assert isinstance(results.timestamp_local, datetime)
        assert results.is_sampled_run is False

        # Test count consistency
        assert results.total_examples_count == 3
        total_processed = (
            results.skipped_examples_count
            + results.succeeded_examples_count
            + results.llm_error_examples_count
            + results.parsing_error_examples_count
            + results.other_error_examples_count
        )
        assert total_processed == results.total_examples_count

        # Test DataFrame structure
        assert isinstance(results.results_data, pl.DataFrame)
        assert len(results.results_data) == 3
        assert "sample_example_id" in results.results_data.columns
        assert "original_id" in results.results_data.columns
        assert "status" in results.results_data.columns
        assert "sentiment_analysis" in results.results_data.columns

    # === Integration Happy Path Tests ===

    def test_complete_structured_evaluation_workflow(
        self, structured_judge, mock_llm_client, sample_eval_data
    ):
        """Test complete successful evaluation workflow for structured responses."""
        # Mock all successful responses
        mock_response = Mock()
        mock_response.result_outcome = "neutral"
        mock_usage = LLMUsage(prompt_tokens=12, completion_tokens=6, total_tokens=18)
        mock_llm_response = LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="gpt-4",
            messages=[
                Message(role=RoleEnum.ASSISTANT, content="neutral sentiment detected")
            ],
            usage=mock_usage,
        )
        mock_llm_client.prompt_with_structured_response.return_value = (
            mock_response,
            mock_llm_response,
        )

        results = structured_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "happy_path_test"
        )

        # Verify successful completion
        assert results.succeeded_examples_count == 3
        assert results.llm_error_examples_count == 0
        assert results.parsing_error_examples_count == 0
        assert results.other_error_examples_count == 0
        assert results.skipped_examples_count == 0

        # Verify result data integrity
        successful_results = results.get_successful_results()
        assert len(successful_results) == 3

        for i, row in enumerate(successful_results.iter_rows(named=True), 1):
            assert row["sample_example_id"] == f"happy_path_test_{i}"
            assert row["original_id"] == f"id-{i}"
            assert row["run_id"] == "happy_path_test"
            assert row["judge_id"] == "test_structured_judge"
            assert row["task_name"] == "sentiment_analysis"
            assert row["status"] == EvaluationStatusEnum.SUCCESS.value
            assert row["sentiment_analysis"] == "neutral"
            assert row["llm_raw_response_content"] == "neutral sentiment detected"
            assert row["llm_prompt_tokens"] == 12
            assert row["llm_completion_tokens"] == 6
            assert row["llm_total_tokens"] == 18

    def test_complete_xml_evaluation_workflow(
        self, xml_judge, mock_llm_client, sample_eval_data
    ):
        """Test complete successful evaluation workflow for XML responses."""
        # Mock successful XML responses
        mock_parse_result = ParseResult(data={"result_outcome": "negative"}, errors=[])
        mock_usage = LLMUsage(prompt_tokens=15, completion_tokens=8, total_tokens=23)
        mock_llm_response = LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="gpt-4",
            messages=[
                Message(
                    role=RoleEnum.ASSISTANT,
                    content="<result_outcome>negative</result_outcome>",
                )
            ],
            usage=mock_usage,
        )
        mock_llm_client.prompt_with_xml_tags.return_value = (
            mock_parse_result,
            mock_llm_response,
        )

        results = xml_judge.evaluate_eval_data(
            sample_eval_data, mock_llm_client, "xml_happy_path"
        )

        # Verify successful completion
        assert results.succeeded_examples_count == 3
        assert results.llm_error_examples_count == 0
        assert results.parsing_error_examples_count == 0
        assert results.other_error_examples_count == 0
        assert results.skipped_examples_count == 0

        # Verify XML-specific result data
        successful_results = results.get_successful_results()
        for row in successful_results.iter_rows(named=True):
            assert row["sentiment_analysis"] == "negative"
            assert (
                row["llm_raw_response_content"]
                == "<result_outcome>negative</result_outcome>"
            )
            assert row["llm_prompt_tokens"] == 15
            assert row["llm_completion_tokens"] == 8
            assert row["llm_total_tokens"] == 23

    def test_comprehensive_count_consistency(self, skip_judge, mock_llm_client):
        """Test count consistency across all possible status combinations with larger dataset."""
        # Create larger dataset with more varied scenarios
        larger_df = pl.DataFrame(
            {
                "question": [f"Question {i}" for i in range(10)],
                "model_response": [f"Response {i}" for i in range(10)],
                "difficulty": ["easy", "medium", "hard"] * 3 + ["easy"],
            }
        )

        larger_eval_data = EvalData(
            name="comprehensive_test",
            data=larger_df,
            input_columns=["question"],
            output_columns=["model_response"],
            metadata_columns=["difficulty"],
        )

        # Create mixed responses to test all status types
        responses = []
        for i in range(10):
            if i % 4 == 0:  # Success
                mock_success = Mock()
                mock_success.result_outcome = "positive"
                mock_usage = LLMUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                )
                mock_llm_response = LLMResponse(
                    provider=LLMClientEnum.OPENAI,
                    model="gpt-4",
                    messages=[Message(role=RoleEnum.ASSISTANT, content="positive")],
                    usage=mock_usage,
                )
                responses.append((mock_success, mock_llm_response))
            elif i % 4 == 1:  # LLM Error
                responses.append(
                    LLMAPIError("API Error", LLMClientEnum.OPENAI, Exception("Error"))
                )
            elif i % 4 == 2:  # Parsing Error
                mock_parse_error = Mock(spec=[])
                mock_usage = LLMUsage(
                    prompt_tokens=8, completion_tokens=3, total_tokens=11
                )
                mock_llm_response = LLMResponse(
                    provider=LLMClientEnum.OPENAI,
                    model="gpt-4",
                    messages=[Message(role=RoleEnum.ASSISTANT, content="malformed")],
                    usage=mock_usage,
                )
                responses.append((mock_parse_error, mock_llm_response))
            else:  # Other Error
                responses.append(RuntimeError("Unexpected error"))

        mock_llm_client.prompt_with_structured_response.side_effect = responses

        results = skip_judge.evaluate_eval_data(
            larger_eval_data, mock_llm_client, "comprehensive_test"
        )

        # Verify count consistency
        total_processed = (
            results.skipped_examples_count
            + results.succeeded_examples_count
            + results.llm_error_examples_count
            + results.parsing_error_examples_count
            + results.other_error_examples_count
        )
        assert total_processed == results.total_examples_count
        assert results.total_examples_count == 10

        # Verify we have mixed outcomes (skip function will skip "medium" difficulty)
        assert results.skipped_examples_count > 0  # Should have some skipped
        assert results.succeeded_examples_count > 0  # Should have some successes
        assert results.llm_error_examples_count > 0  # Should have some LLM errors
        assert (
            results.parsing_error_examples_count > 0
        )  # Should have some parsing errors
        assert results.other_error_examples_count > 0  # Should have some other errors
