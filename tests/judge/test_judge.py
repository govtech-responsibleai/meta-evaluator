"""Test suite for Judge class with comprehensive path coverage."""

from typing import cast
from unittest.mock import Mock, create_autospec, patch

import polars as pl
import pytest
from pydantic import BaseModel

from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.judge.exceptions import IncorrectClientError
from meta_evaluator.judge.judge import Judge
from meta_evaluator.llm_client import (
    AsyncLLMClient,
    AsyncLLMClientEnum,
    LLMClient,
    LLMClientEnum,
)
from meta_evaluator.llm_client.exceptions import LLMAPIError
from meta_evaluator.llm_client.models import (
    ErrorType,
    ParseError,
    ParseResult,
    RoleEnum,
    TagConfig,
)
from meta_evaluator.results import JudgeResults, JudgeResultsBuilder


class TestJudge:
    """Comprehensive test suite for Judge class achieving 100% path coverage."""

    # === Validation Tests ===

    def test_validate_id_valid(self, basic_eval_task, sample_prompt):
        """Test successful ID validation with valid characters."""
        judge = Judge(
            id="valid_judge_123",
            eval_task=basic_eval_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )
        assert judge.id == "valid_judge_123"

    def test_validate_id_invalid_characters(self, basic_eval_task, sample_prompt):
        """Test ID validation failure with invalid characters."""
        with pytest.raises(
            ValueError,
            match="id must only contain alphanumeric characters and underscores",
        ):
            Judge(
                id="invalid-judge",
                eval_task=basic_eval_task,
                llm_client_enum=LLMClientEnum.OPENAI,
                model="gpt-4",
                prompt=sample_prompt,
            )

    def test_validate_id_starts_with_number(self, basic_eval_task, sample_prompt):
        """Test ID validation failure when starting with number."""
        with pytest.raises(
            ValueError,
            match="id must only contain alphanumeric characters and underscores",
        ):
            Judge(
                id="123_invalid",
                eval_task=basic_eval_task,
                llm_client_enum=LLMClientEnum.OPENAI,
                model="gpt-4",
                prompt=sample_prompt,
            )

    # === Helper Method Tests ===

    def test_get_xml_instructions_single_task(self, basic_judge):
        """Test XML instruction generation for single task."""
        instructions = basic_judge._get_xml_instructions()

        assert "<sentiment>" in instructions
        assert "YOUR_ANSWER_FOR_SENTIMENT" in instructions
        assert "positive, negative, neutral" in instructions

    def test_get_xml_instructions_multiple_tasks(
        self, multi_task_eval_task, sample_prompt
    ):
        """Test XML instruction generation for multiple tasks."""
        judge = Judge(
            id="multi_judge",
            eval_task=multi_task_eval_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )
        instructions = judge._get_xml_instructions()

        assert "<sentiment>" in instructions
        assert "<toxicity>" in instructions
        assert "positive, negative, neutral" in instructions
        assert "toxic, non_toxic" in instructions

    def test_create_system_message_without_xml(self, basic_judge):
        """Test system message creation without XML instructions."""
        message = basic_judge._create_system_message(include_xml_instructions=False)

        assert message.role == RoleEnum.SYSTEM
        assert message.content == "Evaluate the sentiment of the given text."
        assert "<sentiment>" not in message.content

    def test_create_system_message_with_xml(self, basic_judge):
        """Test system message creation with XML instructions."""
        message = basic_judge._create_system_message(include_xml_instructions=True)

        assert message.role == RoleEnum.SYSTEM
        assert "Evaluate the sentiment of the given text." in message.content
        assert "<sentiment>" in message.content

    def test_format_row_data_complete(self, sample_prompt):
        """Test row data formatting with both input and output columns."""
        task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="structured",
        )
        judge = Judge(
            id="test_judge",
            eval_task=task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        row = {"text": "Good movie", "response": "I liked it"}
        formatted = judge._format_row_data(row)

        assert "The prompts to be evaluated are text" in formatted
        assert "text: Good movie" in formatted
        assert "The responses to be evaluated are response" in formatted
        assert "response: I liked it" in formatted

    def test_format_row_data_both_input_output(self, basic_judge):
        """Test row data formatting with both input and output columns."""
        row = {"text": "Good movie", "response": "I liked it"}
        formatted = basic_judge._format_row_data(row)

        assert "The prompts to be evaluated are text" in formatted
        assert "text: Good movie" in formatted
        assert "The responses to be evaluated are response" in formatted
        assert "response: I liked it" in formatted

    def test_format_row_data_no_prompt_columns(self, sample_prompt):
        """Test row data formatting when prompt_columns is None."""
        task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative"]},
            prompt_columns=None,  # No prompt columns
            response_columns=["response"],
            answering_method="structured",
        )
        judge = Judge(
            id="test_judge_no_prompt",
            eval_task=task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        row = {"response": "bad movie"}
        formatted = judge._format_row_data(row)

        # Should not contain any input column information
        assert "The prompts to be evaluated are" not in formatted
        assert "text:" not in formatted
        # Should still contain output column information
        assert "The texts to be evaluated are response" in formatted
        assert "response: bad movie" in formatted

    def test_format_row_data_empty_prompt_columns(self, sample_prompt):
        """Test row data formatting when prompt_columns is empty list."""
        task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative"]},
            prompt_columns=[],  # Empty prompt columns
            response_columns=["response"],
            answering_method="structured",
        )
        judge = Judge(
            id="test_judge_empty_prompt",
            eval_task=task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        row = {"response": "bad movie"}
        formatted = judge._format_row_data(row)

        # Should not contain any input column information
        assert "The prompts to be evaluated are" not in formatted
        # Should still contain output column information
        assert "The texts to be evaluated are response" in formatted
        assert "response: bad movie" in formatted

    def test_get_dicts_as_generator(self, basic_judge, basic_eval_data):
        """Test generator functionality over eval data."""
        rows = list(basic_judge._get_dicts_as_generator(basic_eval_data))

        assert len(rows) == 3
        assert rows[0]["id"] == "1"
        assert rows[0]["text"] == "This movie is fantastic!"

    # === Structured Evaluation Tests ===

    def test_evaluate_row_structured_perfect_success(
        self, basic_judge, basic_eval_data, mock_llm_client, mock_llm_response
    ):
        """Test structured evaluation with perfect success."""

        # Create a real Pydantic model for the task
        class MockTaskClass(BaseModel):
            sentiment: str

        # Mock structured response
        structured_response = MockTaskClass(sentiment="positive")
        mock_llm_client.prompt_with_structured_response.return_value = (
            structured_response,
            mock_llm_response,
        )

        # Create mock builder with proper spec to satisfy beartype
        builder = create_autospec(JudgeResultsBuilder, instance=True)
        row = {"id": "1", "text": "Good movie", "response": "I liked it"}

        basic_judge._evaluate_row_structured(
            row=row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            task_class=MockTaskClass,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_success_row.assert_called_once()
        assert builder.create_success_row.call_args[1]["outcomes"] == {
            "sentiment": "positive"
        }

    def test_evaluate_row_structured_partial_success(
        self,
        multi_task_eval_task,
        sample_prompt,
        basic_eval_data,
        mock_llm_client,
        mock_llm_response,
    ):
        """Test structured evaluation with partial success."""
        judge = Judge(
            id="multi_judge",
            eval_task=multi_task_eval_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        # Create a real Pydantic model for testing
        class MockTaskClass(BaseModel):
            sentiment: str
            toxicity: str = ""

        # Mock structured response with missing toxicity
        structured_response = MockTaskClass(sentiment="positive")
        # Remove the toxicity attribute to simulate it being missing
        delattr(structured_response, "toxicity")

        mock_llm_client.prompt_with_structured_response.return_value = (
            structured_response,
            mock_llm_response,
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        row = {"id": "1", "text": "Good movie", "response": "I liked it"}

        judge._evaluate_row_structured(
            row=row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            task_class=MockTaskClass,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_partial_row.assert_called_once()
        assert builder.create_partial_row.call_args[1]["outcomes"] == {
            "sentiment": "positive"
        }

    def test_evaluate_row_structured_complete_failure(
        self,
        multi_task_eval_task,
        sample_prompt,
        basic_eval_data,
        mock_llm_client,
        mock_llm_response,
    ):
        """Test structured evaluation with complete failure."""
        judge = Judge(
            id="multi_judge",
            eval_task=multi_task_eval_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        # Create a real Pydantic model for testing
        class MockTaskClass(BaseModel):
            sentiment: str
            toxicity: str

        # Create structured response that will simulate complete failure
        structured_response = MockTaskClass(sentiment="positive", toxicity="non_toxic")
        # Remove both attributes to simulate complete failure
        delattr(structured_response, "sentiment")
        delattr(structured_response, "toxicity")

        mock_llm_client.prompt_with_structured_response.return_value = (
            structured_response,
            mock_llm_response,
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        row = {"id": "1", "text": "Good movie", "response": "I liked it"}

        judge._evaluate_row_structured(
            row=row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            task_class=MockTaskClass,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_parsing_error_row.assert_called_once()

    def test_evaluate_row_structured_attribute_error(
        self, basic_judge, basic_eval_data, mock_llm_client, mock_llm_response
    ):
        """Test structured evaluation with attribute error during parsing."""

        class MockTaskClass(BaseModel):
            sentiment: str

        # Create a structured response that will raise AttributeError during extraction
        class MockStructuredResponse:
            def __getattr__(self, name):
                if name == "sentiment":
                    raise AttributeError("No sentiment attribute")
                return Mock()

        structured_response = MockStructuredResponse()
        mock_llm_client.prompt_with_structured_response.return_value = (
            structured_response,
            mock_llm_response,
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        row = {"id": "1", "text": "Good movie", "response": "I liked it"}

        basic_judge._evaluate_row_structured(
            row=row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            task_class=MockTaskClass,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_parsing_error_row.assert_called_once()

    def test_evaluate_row_structured_llm_api_error(
        self, basic_judge, basic_eval_data, mock_llm_client
    ):
        """Test structured evaluation with LLM API error."""

        class MockTaskClass(BaseModel):
            sentiment: str

        mock_llm_client.prompt_with_structured_response.side_effect = LLMAPIError(
            "API timeout", LLMClientEnum.OPENAI, Exception("timeout")
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        row = {"id": "1", "text": "Good movie", "response": "I liked it"}

        basic_judge._evaluate_row_structured(
            row=row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            task_class=MockTaskClass,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_llm_error_row.assert_called_once()

    def test_evaluate_row_structured_unexpected_error(
        self, basic_judge, basic_eval_data, mock_llm_client
    ):
        """Test structured evaluation with unexpected error."""

        class MockTaskClass(BaseModel):
            sentiment: str

        mock_llm_client.prompt_with_structured_response.side_effect = RuntimeError(
            "Unexpected error"
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        row = {"id": "1", "text": "Good movie", "response": "I liked it"}

        basic_judge._evaluate_row_structured(
            row=row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            task_class=MockTaskClass,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_other_error_row.assert_called_once()

    # === XML Evaluation Tests ===

    def test_evaluate_row_xml_perfect_success(
        self, xml_judge, basic_eval_data, mock_llm_client, mock_llm_response
    ):
        """Test XML evaluation with perfect success."""
        parse_result = ParseResult(data={"sentiment": "positive"}, errors=[])
        mock_llm_client.prompt_with_xml_tags.return_value = (
            parse_result,
            mock_llm_response,
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        row = {"id": "1", "text": "Good movie", "response": "I liked it"}
        tag_configs = [
            TagConfig(
                name="sentiment",
                allowed_values=["positive", "negative", "neutral"],
                cardinality="one",
            )
        ]

        xml_judge._evaluate_row_xml(
            row=row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            tag_configs=tag_configs,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_success_row.assert_called_once()

    def test_evaluate_row_xml_partial_success(
        self,
        multi_task_eval_task,
        sample_prompt,
        basic_eval_data,
        mock_llm_client,
        mock_llm_response,
    ):
        """Test XML evaluation with partial success."""
        # Create XML judge with multiple tasks
        xml_task = EvalTask(
            task_schemas={
                "sentiment": ["positive", "negative", "neutral"],
                "toxicity": ["toxic", "non_toxic"],
            },
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="xml",
        )
        judge = Judge(
            id="xml_multi_judge",
            eval_task=xml_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        parse_result = ParseResult(
            data={"sentiment": "positive"},  # Missing toxicity
            errors=[
                ParseError(
                    error_type=ErrorType.TAG_NOT_FOUND,
                    tag_name="toxicity",
                    message="Failed to parse toxicity tag",
                )
            ],
        )
        mock_llm_client.prompt_with_xml_tags.return_value = (
            parse_result,
            mock_llm_response,
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        row = {"id": "1", "text": "Good movie", "response": "I liked it"}
        tag_configs = [
            TagConfig(
                name="sentiment",
                allowed_values=["positive", "negative", "neutral"],
                cardinality="one",
            ),
            TagConfig(
                name="toxicity",
                allowed_values=["toxic", "non_toxic"],
                cardinality="one",
            ),
        ]

        judge._evaluate_row_xml(
            row=row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            tag_configs=tag_configs,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_partial_row.assert_called_once()

    def test_evaluate_row_xml_complete_failure(
        self, xml_judge, basic_eval_data, mock_llm_client, mock_llm_response
    ):
        """Test XML evaluation with complete failure."""
        parse_result = ParseResult(
            data={},
            errors=[
                ParseError(
                    error_type=ErrorType.TAG_NOT_FOUND,
                    tag_name="sentiment",
                    message="Failed to parse XML",
                )
            ],
        )
        mock_llm_client.prompt_with_xml_tags.return_value = (
            parse_result,
            mock_llm_response,
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        row = {"id": "1", "text": "Good movie", "response": "I liked it"}
        tag_configs = [
            TagConfig(
                name="sentiment",
                allowed_values=["positive", "negative", "neutral"],
                cardinality="one",
            )
        ]

        xml_judge._evaluate_row_xml(
            row=row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            tag_configs=tag_configs,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_parsing_error_row.assert_called_once()

    def test_evaluate_row_xml_llm_api_error(
        self, xml_judge, basic_eval_data, mock_llm_client
    ):
        """Test XML evaluation with LLM API error."""
        mock_llm_client.prompt_with_xml_tags.side_effect = LLMAPIError(
            "API timeout", LLMClientEnum.OPENAI, Exception("timeout")
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        row = {"id": "1", "text": "Good movie", "response": "I liked it"}
        tag_configs = [
            TagConfig(
                name="sentiment",
                allowed_values=["positive", "negative", "neutral"],
                cardinality="one",
            )
        ]

        xml_judge._evaluate_row_xml(
            row=row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            tag_configs=tag_configs,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_llm_error_row.assert_called_once()

    def test_evaluate_row_xml_unexpected_error(
        self, xml_judge, basic_eval_data, mock_llm_client
    ):
        """Test XML evaluation with unexpected error."""
        mock_llm_client.prompt_with_xml_tags.side_effect = RuntimeError(
            "Unexpected error"
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        row = {"id": "1", "text": "Good movie", "response": "I liked it"}
        tag_configs = [
            TagConfig(
                name="sentiment",
                allowed_values=["positive", "negative", "neutral"],
                cardinality="one",
            )
        ]

        xml_judge._evaluate_row_xml(
            row=row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            tag_configs=tag_configs,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_other_error_row.assert_called_once()

    # === Main Evaluation Tests ===

    def test_evaluate_eval_data_client_mismatch(self, basic_judge, basic_eval_data):
        """Test evaluation with wrong LLM client type."""
        wrong_client = Mock(spec=LLMClient)
        wrong_client.enum_value = (
            LLMClientEnum.ANTHROPIC
        )  # Different from judge's OPENAI

        with pytest.raises(IncorrectClientError):
            basic_judge.evaluate_eval_data(basic_eval_data, wrong_client, "test_run")

    def test_evaluate_eval_data_structured_method(
        self, basic_judge, basic_eval_data, mock_llm_client, mock_llm_response
    ):
        """Test full evaluation run with structured method."""

        class MockTaskClass(BaseModel):
            sentiment: str

        # Mock structured response
        structured_response = MockTaskClass(sentiment="positive")
        mock_llm_client.prompt_with_structured_response.return_value = (
            structured_response,
            mock_llm_response,
        )

        with patch(
            "meta_evaluator.judge.judge.JudgeResultsBuilder"
        ) as mock_builder_class:
            mock_builder = create_autospec(JudgeResultsBuilder, instance=True)
            mock_results = create_autospec(JudgeResults, instance=True)
            mock_builder.complete.return_value = mock_results
            mock_builder_class.return_value = mock_builder

            result = basic_judge.evaluate_eval_data(
                basic_eval_data, mock_llm_client, "test_run"
            )

            # Verify builder was created and used
            mock_builder_class.assert_called_once()
            assert (
                mock_builder.create_success_row.call_count == 3
            )  # 3 rows in eval_data
            mock_builder.complete.assert_called_once()
            assert result == mock_results

    def test_evaluate_eval_data_xml_method(
        self, xml_judge, basic_eval_data, mock_llm_client, mock_llm_response
    ):
        """Test full evaluation run with XML method."""
        parse_result = ParseResult(data={"sentiment": "positive"}, errors=[])
        mock_llm_client.prompt_with_xml_tags.return_value = (
            parse_result,
            mock_llm_response,
        )

        with patch(
            "meta_evaluator.judge.judge.JudgeResultsBuilder"
        ) as mock_builder_class:
            mock_builder = create_autospec(JudgeResultsBuilder, instance=True)
            mock_results = create_autospec(JudgeResults, instance=True)
            mock_builder.complete.return_value = mock_results
            mock_builder_class.return_value = mock_builder

            result = xml_judge.evaluate_eval_data(
                basic_eval_data, mock_llm_client, "test_run"
            )

            # Verify builder was created and used
            mock_builder_class.assert_called_once()
            assert (
                mock_builder.create_success_row.call_count == 3
            )  # 3 rows in eval_data
            mock_builder.complete.assert_called_once()
            assert result == mock_results

    def test_evaluate_eval_data_with_skip_function_skip(
        self, basic_eval_task, sample_prompt, basic_eval_data, mock_llm_client
    ):
        """Test evaluation with skip function that triggers."""
        # Create task with skip function that skips all rows
        task_with_skip = EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="structured",
            skip_function=lambda row: True,  # Skip all rows
        )
        judge = Judge(
            id="skip_judge",
            eval_task=task_with_skip,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        with patch(
            "meta_evaluator.judge.judge.JudgeResultsBuilder"
        ) as mock_builder_class:
            mock_builder = create_autospec(JudgeResultsBuilder, instance=True)
            mock_results = create_autospec(JudgeResults, instance=True)
            mock_builder.complete.return_value = mock_results
            mock_builder_class.return_value = mock_builder

            result = judge.evaluate_eval_data(
                basic_eval_data, mock_llm_client, "test_run"
            )

            # All rows should be skipped
            assert mock_builder.create_skipped_row.call_count == 3
            assert mock_builder.create_success_row.call_count == 0
            assert result == mock_results

    def test_evaluate_eval_data_with_skip_function_no_skip(
        self,
        basic_eval_task,
        sample_prompt,
        basic_eval_data,
        mock_llm_client,
        mock_llm_response,
    ):
        """Test evaluation with skip function that doesn't trigger."""
        # Create task with skip function that skips no rows
        task_with_skip = EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="structured",
            skip_function=lambda row: False,  # Skip no rows
        )
        judge = Judge(
            id="no_skip_judge",
            eval_task=task_with_skip,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        class MockTaskClass(BaseModel):
            sentiment: str

        # Mock structured response
        structured_response = MockTaskClass(sentiment="positive")
        mock_llm_client.prompt_with_structured_response.return_value = (
            structured_response,
            mock_llm_response,
        )

        with patch(
            "meta_evaluator.judge.judge.JudgeResultsBuilder"
        ) as mock_builder_class:
            mock_builder = create_autospec(JudgeResultsBuilder, instance=True)
            mock_results = create_autospec(JudgeResults, instance=True)
            mock_builder.complete.return_value = mock_results
            mock_builder_class.return_value = mock_builder

            result = judge.evaluate_eval_data(
                basic_eval_data, mock_llm_client, "test_run"
            )

            # No rows should be skipped
            assert mock_builder.create_skipped_row.call_count == 0
            assert mock_builder.create_success_row.call_count == 3
            assert result == mock_results

    def test_evaluate_eval_data_no_skip_function(
        self, basic_judge, basic_eval_data, mock_llm_client, mock_llm_response
    ):
        """Test evaluation with no skip function defined."""

        class MockTaskClass(BaseModel):
            sentiment: str

        # Mock structured response
        structured_response = MockTaskClass(sentiment="positive")
        mock_llm_client.prompt_with_structured_response.return_value = (
            structured_response,
            mock_llm_response,
        )

        with patch(
            "meta_evaluator.judge.judge.JudgeResultsBuilder"
        ) as mock_builder_class:
            mock_builder = create_autospec(JudgeResultsBuilder, instance=True)
            mock_results = create_autospec(JudgeResults, instance=True)
            mock_builder.complete.return_value = mock_results
            mock_builder_class.return_value = mock_builder

            result = basic_judge.evaluate_eval_data(
                basic_eval_data, mock_llm_client, "test_run"
            )

            # No skip function, so no skips
            assert mock_builder.create_skipped_row.call_count == 0
            assert mock_builder.create_success_row.call_count == 3
            assert result == mock_results

    def test_evaluate_eval_data_outer_exception(
        self, basic_eval_task, sample_prompt, basic_eval_data, mock_llm_client
    ):
        """Test evaluation with error in row processing loop."""

        # Create task with skip function that raises exception
        def error_skip_function(row):
            raise RuntimeError("Skip function error")

        task_with_error = EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="structured",
            skip_function=error_skip_function,
        )
        judge = Judge(
            id="error_judge",
            eval_task=task_with_error,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        with patch(
            "meta_evaluator.judge.judge.JudgeResultsBuilder"
        ) as mock_builder_class:
            mock_builder = create_autospec(JudgeResultsBuilder, instance=True)
            mock_results = create_autospec(JudgeResults, instance=True)
            mock_builder.complete.return_value = mock_results
            mock_builder_class.return_value = mock_builder

            result = judge.evaluate_eval_data(
                basic_eval_data, mock_llm_client, "test_run"
            )

            # All rows should result in other errors
            assert mock_builder.create_other_error_row.call_count == 3
            assert result == mock_results

    def test_evaluate_eval_data_sampled_vs_regular(
        self, basic_judge, sample_eval_data, mock_llm_client, mock_llm_response
    ):
        """Test evaluation with SampleEvalData vs regular EvalData."""

        class MockTaskClass(BaseModel):
            sentiment: str

        # Mock structured response
        structured_response = MockTaskClass(sentiment="positive")
        mock_llm_client.prompt_with_structured_response.return_value = (
            structured_response,
            mock_llm_response,
        )

        with patch(
            "meta_evaluator.judge.judge.JudgeResultsBuilder"
        ) as mock_builder_class:
            mock_builder = create_autospec(JudgeResultsBuilder, instance=True)
            mock_results = create_autospec(JudgeResults, instance=True)
            mock_builder.complete.return_value = mock_results
            mock_builder_class.return_value = mock_builder

            result = basic_judge.evaluate_eval_data(
                sample_eval_data, mock_llm_client, "test_run"
            )

            # Verify config was created with is_sampled_run=True
            call_kwargs = mock_builder_class.call_args[1]
            assert call_kwargs["is_sampled_run"] is True
            assert result == mock_results

    def test_evaluate_eval_data_no_prompt_columns(
        self, sample_prompt, basic_eval_data, mock_llm_client, mock_llm_response
    ):
        """Test full evaluation run with no prompt columns."""
        # Create task with no prompt columns
        task_no_prompt = EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=None,  # No prompt columns
            response_columns=["response"],
            answering_method="structured",
        )
        judge = Judge(
            id="no_prompt_judge",
            eval_task=task_no_prompt,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        class MockTaskClass(BaseModel):
            sentiment: str

        # Mock structured response
        structured_response = MockTaskClass(sentiment="positive")
        mock_llm_client.prompt_with_structured_response.return_value = (
            structured_response,
            mock_llm_response,
        )

        with patch(
            "meta_evaluator.judge.judge.JudgeResultsBuilder"
        ) as mock_builder_class:
            mock_builder = create_autospec(JudgeResultsBuilder, instance=True)
            mock_results = create_autospec(JudgeResults, instance=True)
            mock_builder.complete.return_value = mock_results
            mock_builder_class.return_value = mock_builder

            result = judge.evaluate_eval_data(
                basic_eval_data, mock_llm_client, "test_run"
            )

            # Verify builder was created and used
            mock_builder_class.assert_called_once()
            assert (
                mock_builder.create_success_row.call_count == 3
            )  # 3 rows in eval_data
            mock_builder.complete.assert_called_once()
            assert result == mock_results

    # === Edge Cases ===

    def test_evaluate_eval_data_empty_dataset(self, mock_llm_client, sample_prompt):
        """Test evaluation with empty dataset."""
        # Test the empty dataset scenario by creating a Judge with a task
        # that skips all rows, effectively simulating an empty dataset

        task_with_skip = EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="structured",
            skip_function=lambda row: True,  # Skip all rows
        )

        skip_judge = Judge(
            id="test_judge_empty",
            eval_task=task_with_skip,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        # Create minimal eval data (since empty DataFrames aren't allowed)
        df = pl.DataFrame(
            {"id": ["1"], "text": ["Test"], "response": ["Test response"]}
        )
        local_eval_data = EvalData(
            name="test",
            data=df,
            id_column="id",
        )

        with patch(
            "meta_evaluator.judge.judge.JudgeResultsBuilder"
        ) as mock_builder_class:
            mock_builder = create_autospec(JudgeResultsBuilder, instance=True)
            mock_results = create_autospec(JudgeResults, instance=True)
            mock_builder.complete.return_value = mock_results
            mock_builder_class.return_value = mock_builder

            result = skip_judge.evaluate_eval_data(
                local_eval_data, mock_llm_client, "test_run"
            )

            # All rows should be skipped (simulating empty dataset behavior)
            assert mock_builder.create_skipped_row.call_count == 1
            assert mock_builder.create_success_row.call_count == 0
            mock_builder.complete.assert_called_once()
            assert result == mock_results

    def test_eval_task_free_form_outputs(self, sample_prompt):
        """Test EvalTask with free form outputs."""
        # Test mixed task schemas with both predefined and free form
        task = EvalTask(
            task_schemas={
                "sentiment": ["positive", "negative", "neutral"],  # Predefined
                "summary": None,  # Free form
                "explanation": None,  # Free form
            },
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="structured",
        )

        judge = Judge(
            id="free_form_judge",
            eval_task=task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        # Test task class creation
        TaskClass = task.create_task_class()

        # Test that we can create an instance with mixed field types
        instance = TaskClass(
            sentiment="positive",
            summary="This is a free form summary",
            explanation="This explains the sentiment",
        )

        # Use getattr for dynamically created attributes to avoid type checker issues
        assert getattr(instance, "sentiment") == "positive"
        assert getattr(instance, "summary") == "This is a free form summary"
        assert getattr(instance, "explanation") == "This explains the sentiment"

        # Test XML instructions include free form guidance
        xml_instructions = judge._get_xml_instructions()
        assert (
            "Valid values for sentiment are: positive, negative, neutral"
            in xml_instructions
        )
        assert "For summary, provide a free form text response" in xml_instructions
        assert "For explanation, provide a free form text response" in xml_instructions


# === ASYNC TESTS ===


class TestJudgeAsync:
    """Test suite for Judge async evaluation methods."""

    @pytest.mark.asyncio
    async def test_evaluate_eval_data_async_structured_method(
        self,
        basic_judge,
        sample_eval_data,
        mock_async_llm_client,
        mock_async_llm_response,
        mocker,
    ):
        """Test async evaluation with structured method and successful outcomes."""
        # Change to async client
        basic_judge = basic_judge.model_copy(
            update={"llm_client_enum": AsyncLLMClientEnum.OPENAI}
        )

        # Mock the async structured response method
        mock_response_data = type("TestResponse", (), {"sentiment": "positive"})()

        mock_async_llm_client.prompt_with_structured_response = mocker.AsyncMock(
            return_value=(mock_response_data, mock_async_llm_response)
        )

        # Mock batch processing
        mock_async_llm_client.prompt_with_structured_response_batch = mocker.AsyncMock(
            return_value=[(mock_response_data, mock_async_llm_response)]
            * len(sample_eval_data.data)
        )

        result = await basic_judge.evaluate_eval_data_async(
            eval_data=sample_eval_data,
            llm_client=mock_async_llm_client,
            run_id="test_async_run",
        )

        assert isinstance(result, JudgeResults)
        assert result.succeeded_count == len(sample_eval_data.data)
        assert result.total_count == len(sample_eval_data.data)

        # Verify batch method was called
        mock_async_llm_client.prompt_with_structured_response_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_eval_data_async_xml_method(
        self,
        xml_judge,
        sample_eval_data,
        mock_async_llm_client,
        mock_async_llm_response,
        mocker,
    ):
        """Test async evaluation with XML method and successful parsing."""
        # Change to async client
        xml_judge = xml_judge.model_copy(
            update={"llm_client_enum": AsyncLLMClientEnum.OPENAI}
        )

        # Mock the async XML tags method
        mock_parse_result = ParseResult(
            data={"sentiment": "positive"},
            errors=[],
        )

        mock_async_llm_client.prompt_with_xml_tags = mocker.AsyncMock(
            return_value=(mock_parse_result, mock_async_llm_response)
        )

        # Mock batch processing
        mock_async_llm_client.prompt_with_xml_tags_batch = mocker.AsyncMock(
            return_value=[(mock_parse_result, mock_async_llm_response)]
            * len(sample_eval_data.data)
        )

        result = await xml_judge.evaluate_eval_data_async(
            eval_data=sample_eval_data,
            llm_client=mock_async_llm_client,
            run_id="test_async_xml_run",
        )

        assert isinstance(result, JudgeResults)
        assert result.succeeded_count == len(sample_eval_data.data)
        assert result.total_count == len(sample_eval_data.data)

        # Verify batch method was called
        mock_async_llm_client.prompt_with_xml_tags_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_eval_data_async_with_skip_function(
        self,
        basic_judge,
        sample_eval_data,
        mock_async_llm_client,
        mock_async_llm_response,
        mocker,
    ):
        """Test async evaluation with skip function filtering some examples."""
        # Change to async client
        basic_judge = basic_judge.model_copy(
            update={"llm_client_enum": AsyncLLMClientEnum.OPENAI}
        )

        # Mock successful response
        mock_response_data = type("TestResponse", (), {"sentiment": "positive"})()

        # Skip first example only
        def skip_first_example(row_dict):
            return row_dict["sample_id"] == "1"

        # Set skip function on eval task
        basic_judge.eval_task.skip_function = skip_first_example

        # Mock batch processing - should only get called for non-skipped items
        expected_batch_size = len(sample_eval_data.data) - 1  # One item skipped
        mock_async_llm_client.prompt_with_structured_response_batch = mocker.AsyncMock(
            return_value=[(mock_response_data, mock_async_llm_response)]
            * expected_batch_size
        )

        result = await basic_judge.evaluate_eval_data_async(
            eval_data=sample_eval_data,
            llm_client=mock_async_llm_client,
            run_id="test_async_skip_run",
        )

        assert isinstance(result, JudgeResults)
        assert result.succeeded_count == expected_batch_size
        assert result.skipped_count == 1
        assert result.total_count == len(sample_eval_data.data)

    # === Batch Method Tests ===

    @pytest.mark.asyncio
    async def test_evaluate_rows_batch_structured_perfect_success(
        self,
        basic_judge,
        basic_eval_data,
        mock_async_llm_client,
        mock_async_llm_response,
        mocker,
    ):
        """Test batch structured evaluation with perfect success."""

        # Create a real Pydantic model for the task
        class MockTaskClass(BaseModel):
            sentiment: str

        # Mock structured response
        structured_response = MockTaskClass(sentiment="positive")

        # Mock batch method to return successful results
        mock_async_llm_client.prompt_with_structured_response_batch = mocker.AsyncMock(
            return_value=[(structured_response, mock_async_llm_response)]
        )

        # Create mock builder
        builder = create_autospec(JudgeResultsBuilder, instance=True)
        rows_to_evaluate = [
            ({"id": "1", "text": "Good movie", "response": "I liked it"}, "test_1")
        ]

        await basic_judge._evaluate_rows_batch_structured(
            rows_to_evaluate=rows_to_evaluate,
            eval_data=basic_eval_data,
            llm_client=mock_async_llm_client,
            builder=cast(JudgeResultsBuilder, builder),
            batch_size=10,
            max_concurrency=5,
        )

        # Verify batch method was called
        mock_async_llm_client.prompt_with_structured_response_batch.assert_called_once()
        builder.create_success_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_rows_batch_structured_partial_success(
        self,
        multi_task_eval_task,
        sample_prompt,
        basic_eval_data,
        mock_async_llm_client,
        mock_async_llm_response,
        mocker,
    ):
        """Test batch structured evaluation with partial success."""
        judge = Judge(
            id="multi_judge",
            eval_task=multi_task_eval_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        # Create a real Pydantic model for testing
        class MockTaskClass(BaseModel):
            sentiment: str
            toxicity: str = ""

        # Mock structured response with missing toxicity
        structured_response = MockTaskClass(sentiment="positive")
        # Remove the toxicity attribute to simulate it being missing
        delattr(structured_response, "toxicity")

        # Mock batch method
        mock_async_llm_client.prompt_with_structured_response_batch = mocker.AsyncMock(
            return_value=[(structured_response, mock_async_llm_response)]
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        rows_to_evaluate = [
            ({"id": "1", "text": "Good movie", "response": "I liked it"}, "test_1")
        ]

        await judge._evaluate_rows_batch_structured(
            rows_to_evaluate=rows_to_evaluate,
            eval_data=basic_eval_data,
            llm_client=mock_async_llm_client,
            builder=cast(JudgeResultsBuilder, builder),
            batch_size=10,
            max_concurrency=5,
        )

        builder.create_partial_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_rows_batch_structured_complete_failure(
        self,
        multi_task_eval_task,
        sample_prompt,
        basic_eval_data,
        mock_async_llm_client,
        mock_async_llm_response,
        mocker,
    ):
        """Test batch structured evaluation with complete failure."""
        judge = Judge(
            id="multi_judge",
            eval_task=multi_task_eval_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        # Create a real Pydantic model for testing
        class MockTaskClass(BaseModel):
            sentiment: str
            toxicity: str

        # Create structured response that will simulate complete failure
        structured_response = MockTaskClass(sentiment="positive", toxicity="non_toxic")
        # Remove both attributes to simulate complete failure
        delattr(structured_response, "sentiment")
        delattr(structured_response, "toxicity")

        # Mock batch method
        mock_async_llm_client.prompt_with_structured_response_batch = mocker.AsyncMock(
            return_value=[(structured_response, mock_async_llm_response)]
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        rows_to_evaluate = [
            ({"id": "1", "text": "Good movie", "response": "I liked it"}, "test_1")
        ]

        await judge._evaluate_rows_batch_structured(
            rows_to_evaluate=rows_to_evaluate,
            eval_data=basic_eval_data,
            llm_client=mock_async_llm_client,
            builder=cast(JudgeResultsBuilder, builder),
            batch_size=10,
            max_concurrency=5,
        )

        builder.create_parsing_error_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_rows_batch_structured_individual_llm_error(
        self, basic_judge, basic_eval_data, mock_async_llm_client, mocker
    ):
        """Test batch structured evaluation with individual LLM errors."""
        # Mock batch method to return an exception for one item
        llm_error = LLMAPIError(
            "API timeout", LLMClientEnum.OPENAI, Exception("timeout")
        )
        mock_async_llm_client.prompt_with_structured_response_batch = mocker.AsyncMock(
            return_value=[llm_error]  # Exception instead of (response, llm_response)
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        rows_to_evaluate = [
            ({"id": "1", "text": "Good movie", "response": "I liked it"}, "test_1")
        ]

        await basic_judge._evaluate_rows_batch_structured(
            rows_to_evaluate=rows_to_evaluate,
            eval_data=basic_eval_data,
            llm_client=mock_async_llm_client,
            builder=cast(JudgeResultsBuilder, builder),
            batch_size=10,
            max_concurrency=5,
        )

        builder.create_llm_error_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_rows_batch_structured_individual_unexpected_error(
        self, basic_judge, basic_eval_data, mock_async_llm_client, mocker
    ):
        """Test batch structured evaluation with individual unexpected errors."""
        # Mock batch method to return an unexpected exception
        unexpected_error = RuntimeError("Unexpected error")
        mock_async_llm_client.prompt_with_structured_response_batch = mocker.AsyncMock(
            return_value=[unexpected_error]
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        rows_to_evaluate = [
            ({"id": "1", "text": "Good movie", "response": "I liked it"}, "test_1")
        ]

        await basic_judge._evaluate_rows_batch_structured(
            rows_to_evaluate=rows_to_evaluate,
            eval_data=basic_eval_data,
            llm_client=mock_async_llm_client,
            builder=cast(JudgeResultsBuilder, builder),
            batch_size=10,
            max_concurrency=5,
        )

        builder.create_other_error_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_rows_batch_xml_perfect_success(
        self,
        xml_judge,
        basic_eval_data,
        mock_async_llm_client,
        mock_async_llm_response,
        mocker,
    ):
        """Test batch XML evaluation with perfect success."""
        parse_result = ParseResult(data={"sentiment": "positive"}, errors=[])

        # Mock batch method
        mock_async_llm_client.prompt_with_xml_tags_batch = mocker.AsyncMock(
            return_value=[(parse_result, mock_async_llm_response)]
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        rows_to_evaluate = [
            ({"id": "1", "text": "Good movie", "response": "I liked it"}, "test_1")
        ]

        await xml_judge._evaluate_rows_batch_xml(
            rows_to_evaluate=rows_to_evaluate,
            eval_data=basic_eval_data,
            llm_client=mock_async_llm_client,
            builder=cast(JudgeResultsBuilder, builder),
            batch_size=10,
            max_concurrency=5,
        )

        # Verify batch method was called
        mock_async_llm_client.prompt_with_xml_tags_batch.assert_called_once()
        builder.create_success_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_rows_batch_xml_partial_success(
        self,
        multi_task_eval_task,
        sample_prompt,
        basic_eval_data,
        mock_async_llm_client,
        mock_async_llm_response,
        mocker,
    ):
        """Test batch XML evaluation with partial success."""
        # Create XML judge with multiple tasks
        xml_task = EvalTask(
            task_schemas={
                "sentiment": ["positive", "negative", "neutral"],
                "toxicity": ["toxic", "non_toxic"],
            },
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="xml",
        )
        judge = Judge(
            id="xml_multi_judge",
            eval_task=xml_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        parse_result = ParseResult(
            data={"sentiment": "positive"},  # Missing toxicity
            errors=[
                ParseError(
                    error_type=ErrorType.TAG_NOT_FOUND,
                    tag_name="toxicity",
                    message="Failed to parse toxicity tag",
                )
            ],
        )

        # Mock batch method
        mock_async_llm_client.prompt_with_xml_tags_batch = mocker.AsyncMock(
            return_value=[(parse_result, mock_async_llm_response)]
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        rows_to_evaluate = [
            ({"id": "1", "text": "Good movie", "response": "I liked it"}, "test_1")
        ]

        await judge._evaluate_rows_batch_xml(
            rows_to_evaluate=rows_to_evaluate,
            eval_data=basic_eval_data,
            llm_client=mock_async_llm_client,
            builder=cast(JudgeResultsBuilder, builder),
            batch_size=10,
            max_concurrency=5,
        )

        builder.create_partial_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_rows_batch_xml_complete_failure(
        self,
        xml_judge,
        basic_eval_data,
        mock_async_llm_client,
        mock_async_llm_response,
        mocker,
    ):
        """Test batch XML evaluation with complete failure."""
        parse_result = ParseResult(
            data={},
            errors=[
                ParseError(
                    error_type=ErrorType.TAG_NOT_FOUND,
                    tag_name="sentiment",
                    message="Failed to parse XML",
                )
            ],
        )
        # Mock batch method
        mock_async_llm_client.prompt_with_xml_tags_batch = mocker.AsyncMock(
            return_value=[(parse_result, mock_async_llm_response)]
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        rows_to_evaluate = [
            ({"id": "1", "text": "Good movie", "response": "I liked it"}, "test_1")
        ]

        await xml_judge._evaluate_rows_batch_xml(
            rows_to_evaluate=rows_to_evaluate,
            eval_data=basic_eval_data,
            llm_client=mock_async_llm_client,
            builder=cast(JudgeResultsBuilder, builder),
            batch_size=10,
            max_concurrency=5,
        )

        builder.create_parsing_error_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_rows_batch_xml_individual_llm_error(
        self, xml_judge, basic_eval_data, mock_async_llm_client, mocker
    ):
        """Test batch XML evaluation with individual LLM errors."""
        # Mock batch method to return an exception
        llm_error = LLMAPIError(
            "API timeout", LLMClientEnum.OPENAI, Exception("timeout")
        )
        mock_async_llm_client.prompt_with_xml_tags_batch = mocker.AsyncMock(
            return_value=[llm_error]
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        rows_to_evaluate = [
            ({"id": "1", "text": "Good movie", "response": "I liked it"}, "test_1")
        ]

        await xml_judge._evaluate_rows_batch_xml(
            rows_to_evaluate=rows_to_evaluate,
            eval_data=basic_eval_data,
            llm_client=mock_async_llm_client,
            builder=cast(JudgeResultsBuilder, builder),
            batch_size=10,
            max_concurrency=5,
        )

        builder.create_llm_error_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_rows_batch_xml_individual_unexpected_error(
        self, xml_judge, basic_eval_data, mock_async_llm_client, mocker
    ):
        """Test batch XML evaluation with individual unexpected errors."""
        # Mock batch method to return an unexpected exception
        unexpected_error = RuntimeError("Unexpected error")
        mock_async_llm_client.prompt_with_xml_tags_batch = mocker.AsyncMock(
            return_value=[unexpected_error]
        )

        builder = create_autospec(JudgeResultsBuilder, instance=True)
        rows_to_evaluate = [
            ({"id": "1", "text": "Good movie", "response": "I liked it"}, "test_1")
        ]

        await xml_judge._evaluate_rows_batch_xml(
            rows_to_evaluate=rows_to_evaluate,
            eval_data=basic_eval_data,
            llm_client=mock_async_llm_client,
            builder=cast(JudgeResultsBuilder, builder),
            batch_size=10,
            max_concurrency=5,
        )

        builder.create_other_error_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_eval_data_async_skip_function_all_skipped(
        self, basic_judge, sample_eval_data, mock_async_llm_client, mocker
    ):
        """Test async evaluation when skip function skips all examples."""
        # Change to async client
        basic_judge = basic_judge.model_copy(
            update={"llm_client_enum": AsyncLLMClientEnum.OPENAI}
        )

        # Skip all examples
        def skip_all_examples(row_dict):
            return True

        # Set skip function on eval task
        basic_judge.eval_task.skip_function = skip_all_examples

        # Mock batch processing (should not be called when all items are skipped)
        mock_async_llm_client.prompt_with_structured_response_batch = mocker.AsyncMock(
            return_value=[]
        )

        result = await basic_judge.evaluate_eval_data_async(
            eval_data=sample_eval_data,
            llm_client=mock_async_llm_client,
            run_id="test_skip_all",
        )

        assert isinstance(result, JudgeResults)
        assert result.succeeded_count == 0
        assert result.skipped_count == len(sample_eval_data.data)
        assert result.total_count == len(sample_eval_data.data)

        # Verify batch method was NOT called since all items were skipped
        mock_async_llm_client.prompt_with_structured_response_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_eval_data_async_skip_function_partial_skip(
        self,
        basic_judge,
        sample_eval_data,
        mock_async_llm_client,
        mock_async_llm_response,
        mocker,
    ):
        """Test async evaluation with skip function that skips some examples."""
        # Change to async client
        basic_judge = basic_judge.model_copy(
            update={"llm_client_enum": AsyncLLMClientEnum.OPENAI}
        )

        # Mock successful response
        mock_response_data = type("TestResponse", (), {"sentiment": "positive"})()

        # Skip examples with even-numbered IDs
        def skip_even_ids(row_dict):
            return int(row_dict["sample_id"]) % 2 == 0

        # Set skip function on eval task
        basic_judge.eval_task.skip_function = skip_even_ids

        # Calculate expected non-skipped count
        total_items = len(sample_eval_data.data)
        expected_skipped = sum(1 for i in range(1, total_items + 1) if i % 2 == 0)
        expected_processed = total_items - expected_skipped

        # Mock batch processing
        mock_async_llm_client.prompt_with_structured_response_batch = mocker.AsyncMock(
            return_value=[(mock_response_data, mock_async_llm_response)]
            * expected_processed
        )

        result = await basic_judge.evaluate_eval_data_async(
            eval_data=sample_eval_data,
            llm_client=mock_async_llm_client,
            run_id="test_partial_skip",
        )

        assert isinstance(result, JudgeResults)
        assert result.succeeded_count == expected_processed
        assert result.skipped_count == expected_skipped
        assert result.total_count == total_items

        # Verify batch method was called with correct number of items
        call_args = (
            mock_async_llm_client.prompt_with_structured_response_batch.call_args
        )
        assert len(call_args[1]["items"]) == expected_processed

    @pytest.mark.asyncio
    async def test_evaluate_eval_data_async_incorrect_client_error(
        self, basic_judge, sample_eval_data, mocker
    ):
        """Test async evaluation raises error with incorrect LLM client enum."""
        # Create mock async client with different enum
        wrong_client = Mock(spec=AsyncLLMClient)
        wrong_client.enum_value = (
            AsyncLLMClientEnum.ANTHROPIC
        )  # Different from judge's OPENAI

        with pytest.raises(IncorrectClientError):
            await basic_judge.evaluate_eval_data_async(
                eval_data=sample_eval_data,
                llm_client=wrong_client,
                run_id="test_wrong_client",
            )

    @pytest.mark.asyncio
    async def test_evaluate_eval_data_async_mixed_batch_results(
        self,
        multi_task_eval_task,
        sample_prompt,
        sample_eval_data,
        mock_async_llm_client,
        mock_async_llm_response,
        mocker,
    ):
        """Test async evaluation with mixed success/failure results in batch."""
        judge = Judge(
            id="mixed_judge",
            eval_task=multi_task_eval_task,
            llm_client_enum=AsyncLLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=sample_prompt,
        )

        # Create mixed results: some successful, some failed
        mock_success_data = type(
            "TestResponse", (), {"sentiment": "positive", "toxicity": "non_toxic"}
        )

        # Mixed batch results: success, error (matching sample_eval_data which has 2 rows)
        mixed_results = [
            (mock_success_data, mock_async_llm_response),
            LLMAPIError("API Error", LLMClientEnum.OPENAI, Exception("timeout")),
        ]

        mock_async_llm_client.prompt_with_structured_response_batch = mocker.AsyncMock(
            return_value=mixed_results
        )

        result = await judge.evaluate_eval_data_async(
            eval_data=sample_eval_data,
            llm_client=mock_async_llm_client,
            run_id="test_mixed_results",
        )

        assert isinstance(result, JudgeResults)
        assert result.succeeded_count == 1  # One successful
        assert result.llm_error_count == 1  # One LLM error
        assert result.total_count == len(sample_eval_data.data)
