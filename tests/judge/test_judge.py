"""Test suite for Judge class with comprehensive path coverage."""

import pytest
from unittest.mock import Mock, patch, create_autospec
from typing import cast

from meta_evaluator.judge.judge import Judge
from meta_evaluator.judge.exceptions import IncorrectClientError
from meta_evaluator.judge.models import JudgeResultsBuilder, JudgeResults
from meta_evaluator.evaluation_task import EvaluationTask
from meta_evaluator.llm_client import LLMClientEnum, LLMClient
from meta_evaluator.llm_client.models import (
    Message,
    RoleEnum,
    LLMResponse,
    LLMUsage,
    TagConfig,
    ParseResult,
    ParseError,
    ErrorType,
)
from meta_evaluator.llm_client.exceptions import LLMAPIError
from meta_evaluator.common.models import Prompt
from meta_evaluator.data import EvalData, SampleEvalData
from pydantic import BaseModel
import polars as pl


class TestJudge:
    """Comprehensive test suite for Judge class achieving 100% path coverage."""

    @pytest.fixture
    def basic_evaluation_task(self) -> EvaluationTask:
        """Provides a basic evaluation task for testing.

        Returns:
            EvaluationTask: A basic evaluation task with sentiment analysis schema.
        """
        return EvaluationTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            input_columns=["text"],
            output_columns=["response"],
            answering_method="structured",
        )

    @pytest.fixture
    def xml_evaluation_task(self) -> EvaluationTask:
        """Provides an XML-based evaluation task for testing.

        Returns:
            EvaluationTask: An XML-based evaluation task with sentiment analysis schema.
        """
        return EvaluationTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            input_columns=["text"],
            output_columns=["response"],
            answering_method="xml",
        )

    @pytest.fixture
    def multi_task_evaluation_task(self) -> EvaluationTask:
        """Provides a multi-task evaluation task for testing.

        Returns:
            EvaluationTask: A multi-task evaluation task with sentiment and category schemas.
        """
        return EvaluationTask(
            task_schemas={
                "sentiment": ["positive", "negative", "neutral"],
                "toxicity": ["toxic", "non_toxic"],
            },
            input_columns=["text"],
            output_columns=["response"],
            answering_method="structured",
        )

    @pytest.fixture
    def basic_prompt(self) -> Prompt:
        """Provides a basic prompt for testing.

        Returns:
            Prompt: A basic prompt for sentiment evaluation.
        """
        return Prompt(
            id="basic_prompt", prompt="Evaluate the sentiment of the given text."
        )

    @pytest.fixture
    def basic_judge(self, basic_evaluation_task, basic_prompt) -> Judge:
        """Provides a basic judge configuration for testing.

        Returns:
            Judge: A basic judge configuration with structured output.
        """
        return Judge(
            id="test_judge_1",
            evaluation_task=basic_evaluation_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=basic_prompt,
        )

    @pytest.fixture
    def xml_judge(self, xml_evaluation_task, basic_prompt) -> Judge:
        """Provides an XML-based judge configuration for testing.

        Returns:
            Judge: An XML-based judge configuration.
        """
        return Judge(
            id="xml_judge_1",
            evaluation_task=xml_evaluation_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=basic_prompt,
        )

    @pytest.fixture
    def eval_data(self) -> EvalData:
        """Provides sample evaluation data for testing.

        Returns:
            EvalData: Sample evaluation data with text examples.
        """
        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "text": ["Good movie", "Bad movie", "Okay movie"],
                "response": ["I liked it", "I hated it", "It was fine"],
            }
        )
        return EvalData(
            name="test_data",
            data=df,
            id_column="id",
        )

    @pytest.fixture
    def sample_eval_data(self) -> SampleEvalData:
        """Provides sample evaluation data for testing.

        Returns:
            SampleEvalData: Sample evaluation data with text and target examples.
        """
        df = pl.DataFrame(
            {
                "id": ["1", "2"],
                "text": ["Good movie", "Bad movie"],
                "response": ["I liked it", "I hated it"],
            }
        )
        return SampleEvalData(
            name="test_sample",
            data=df,
            id_column="id",
            sample_name="test_sample",
            stratification_columns=[],
            sample_percentage=1.0,
            seed=42,
        )

    @pytest.fixture
    def mock_llm_client(self) -> Mock:
        """Provides a mock LLM client for testing.

        Returns:
            Mock: A mock LLM client configured for testing.
        """
        client = Mock(spec=LLMClient)
        client.enum_value = LLMClientEnum.OPENAI
        return client

    @pytest.fixture
    def mock_llm_response(self) -> LLMResponse:
        """Provides a mock LLM response for testing.

        Returns:
            LLMResponse: A mock LLM response with structured output.
        """
        return LLMResponse(
            provider=LLMClientEnum.OPENAI,
            model="gpt-4",
            messages=[Message(role=RoleEnum.ASSISTANT, content="positive")],
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    # === Validation Tests ===

    def test_validate_id_valid(self, basic_evaluation_task, basic_prompt):
        """Test successful ID validation with valid characters."""
        judge = Judge(
            id="valid_judge_123",
            evaluation_task=basic_evaluation_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=basic_prompt,
        )
        assert judge.id == "valid_judge_123"

    def test_validate_id_invalid_characters(self, basic_evaluation_task, basic_prompt):
        """Test ID validation failure with invalid characters."""
        with pytest.raises(
            ValueError,
            match="id must only contain alphanumeric characters and underscores",
        ):
            Judge(
                id="invalid-judge",
                evaluation_task=basic_evaluation_task,
                llm_client_enum=LLMClientEnum.OPENAI,
                model="gpt-4",
                prompt=basic_prompt,
            )

    def test_validate_id_starts_with_number(self, basic_evaluation_task, basic_prompt):
        """Test ID validation failure when starting with number."""
        with pytest.raises(
            ValueError,
            match="id must only contain alphanumeric characters and underscores",
        ):
            Judge(
                id="123_invalid",
                evaluation_task=basic_evaluation_task,
                llm_client_enum=LLMClientEnum.OPENAI,
                model="gpt-4",
                prompt=basic_prompt,
            )

    # === Helper Method Tests ===

    def test_get_xml_instructions_single_task(self, basic_judge):
        """Test XML instruction generation for single task."""
        instructions = basic_judge._get_xml_instructions()

        assert "<sentiment>" in instructions
        assert "YOUR_ANSWER_FOR_SENTIMENT" in instructions
        assert "positive, negative, neutral" in instructions

    def test_get_xml_instructions_multiple_tasks(
        self, multi_task_evaluation_task, basic_prompt
    ):
        """Test XML instruction generation for multiple tasks."""
        judge = Judge(
            id="multi_judge",
            evaluation_task=multi_task_evaluation_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=basic_prompt,
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

    def test_format_row_data_complete(self, basic_prompt):
        """Test row data formatting with both input and output columns."""
        task = EvaluationTask(
            task_schemas={"sentiment": ["positive", "negative"]},
            input_columns=["text"],
            output_columns=["response"],
            answering_method="structured",
        )
        judge = Judge(
            id="test_judge",
            evaluation_task=task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=basic_prompt,
        )

        row = {"text": "Good movie", "response": "I liked it"}
        formatted = judge._format_row_data(row)

        assert "The inputs given to the LLM were text" in formatted
        assert "text: Good movie" in formatted
        assert "The outputs given by the LLM were response" in formatted
        assert "response: I liked it" in formatted

    def test_format_row_data_both_input_output(self, basic_judge):
        """Test row data formatting with both input and output columns."""
        row = {"text": "Good movie", "response": "I liked it"}
        formatted = basic_judge._format_row_data(row)

        assert "The inputs given to the LLM were text" in formatted
        assert "text: Good movie" in formatted
        assert "The outputs given by the LLM were response" in formatted
        assert "response: I liked it" in formatted

    def test_get_dicts_as_generator(self, basic_judge, eval_data):
        """Test generator functionality over eval data."""
        rows = list(basic_judge._get_dicts_as_generator(eval_data))

        assert len(rows) == 3
        assert rows[0]["id"] == "1"
        assert rows[0]["text"] == "Good movie"

    # === Structured Evaluation Tests ===

    def test_evaluate_row_structured_perfect_success(
        self, basic_judge, eval_data, mock_llm_client, mock_llm_response
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
            eval_data=eval_data,
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
        multi_task_evaluation_task,
        basic_prompt,
        eval_data,
        mock_llm_client,
        mock_llm_response,
    ):
        """Test structured evaluation with partial success."""
        judge = Judge(
            id="multi_judge",
            evaluation_task=multi_task_evaluation_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=basic_prompt,
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
            eval_data=eval_data,
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
        multi_task_evaluation_task,
        basic_prompt,
        eval_data,
        mock_llm_client,
        mock_llm_response,
    ):
        """Test structured evaluation with complete failure."""
        judge = Judge(
            id="multi_judge",
            evaluation_task=multi_task_evaluation_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=basic_prompt,
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
            eval_data=eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            task_class=MockTaskClass,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_parsing_error_row.assert_called_once()

    def test_evaluate_row_structured_attribute_error(
        self, basic_judge, eval_data, mock_llm_client, mock_llm_response
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
            eval_data=eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            task_class=MockTaskClass,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_parsing_error_row.assert_called_once()

    def test_evaluate_row_structured_llm_api_error(
        self, basic_judge, eval_data, mock_llm_client
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
            eval_data=eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            task_class=MockTaskClass,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_llm_error_row.assert_called_once()

    def test_evaluate_row_structured_unexpected_error(
        self, basic_judge, eval_data, mock_llm_client
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
            eval_data=eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            task_class=MockTaskClass,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_other_error_row.assert_called_once()

    # === XML Evaluation Tests ===

    def test_evaluate_row_xml_perfect_success(
        self, xml_judge, eval_data, mock_llm_client, mock_llm_response
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
            eval_data=eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            tag_configs=tag_configs,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_success_row.assert_called_once()

    def test_evaluate_row_xml_partial_success(
        self,
        multi_task_evaluation_task,
        basic_prompt,
        eval_data,
        mock_llm_client,
        mock_llm_response,
    ):
        """Test XML evaluation with partial success."""
        # Create XML judge with multiple tasks
        xml_task = EvaluationTask(
            task_schemas={
                "sentiment": ["positive", "negative", "neutral"],
                "toxicity": ["toxic", "non_toxic"],
            },
            input_columns=["text"],
            output_columns=["response"],
            answering_method="xml",
        )
        judge = Judge(
            id="xml_multi_judge",
            evaluation_task=xml_task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=basic_prompt,
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
            eval_data=eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            tag_configs=tag_configs,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_partial_row.assert_called_once()

    def test_evaluate_row_xml_complete_failure(
        self, xml_judge, eval_data, mock_llm_client, mock_llm_response
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
            eval_data=eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            tag_configs=tag_configs,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_parsing_error_row.assert_called_once()

    def test_evaluate_row_xml_llm_api_error(
        self, xml_judge, eval_data, mock_llm_client
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
            eval_data=eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            tag_configs=tag_configs,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_llm_error_row.assert_called_once()

    def test_evaluate_row_xml_unexpected_error(
        self, xml_judge, eval_data, mock_llm_client
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
            eval_data=eval_data,
            sample_example_id="test_1",
            llm_client=mock_llm_client,
            tag_configs=tag_configs,
            builder=cast(JudgeResultsBuilder, builder),
        )

        builder.create_other_error_row.assert_called_once()

    # === Main Evaluation Tests ===

    def test_evaluate_eval_data_client_mismatch(self, basic_judge, eval_data):
        """Test evaluation with wrong LLM client type."""
        wrong_client = Mock(spec=LLMClient)
        wrong_client.enum_value = (
            LLMClientEnum.ANTHROPIC
        )  # Different from judge's OPENAI

        with pytest.raises(IncorrectClientError):
            basic_judge.evaluate_eval_data(eval_data, wrong_client, "test_run")

    def test_evaluate_eval_data_structured_method(
        self, basic_judge, eval_data, mock_llm_client, mock_llm_response
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
                eval_data, mock_llm_client, "test_run"
            )

            # Verify builder was created and used
            mock_builder_class.assert_called_once()
            assert (
                mock_builder.create_success_row.call_count == 3
            )  # 3 rows in eval_data
            mock_builder.complete.assert_called_once()
            assert result == mock_results

    def test_evaluate_eval_data_xml_method(
        self, xml_judge, eval_data, mock_llm_client, mock_llm_response
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
                eval_data, mock_llm_client, "test_run"
            )

            # Verify builder was created and used
            mock_builder_class.assert_called_once()
            assert (
                mock_builder.create_success_row.call_count == 3
            )  # 3 rows in eval_data
            mock_builder.complete.assert_called_once()
            assert result == mock_results

    def test_evaluate_eval_data_with_skip_function_skip(
        self, basic_evaluation_task, basic_prompt, eval_data, mock_llm_client
    ):
        """Test evaluation with skip function that triggers."""
        # Create task with skip function that skips all rows
        task_with_skip = EvaluationTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            input_columns=["text"],
            output_columns=["response"],
            answering_method="structured",
            skip_function=lambda row: True,  # Skip all rows
        )
        judge = Judge(
            id="skip_judge",
            evaluation_task=task_with_skip,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=basic_prompt,
        )

        with patch(
            "meta_evaluator.judge.judge.JudgeResultsBuilder"
        ) as mock_builder_class:
            mock_builder = create_autospec(JudgeResultsBuilder, instance=True)
            mock_results = create_autospec(JudgeResults, instance=True)
            mock_builder.complete.return_value = mock_results
            mock_builder_class.return_value = mock_builder

            result = judge.evaluate_eval_data(eval_data, mock_llm_client, "test_run")

            # All rows should be skipped
            assert mock_builder.create_skipped_row.call_count == 3
            assert mock_builder.create_success_row.call_count == 0
            assert result == mock_results

    def test_evaluate_eval_data_with_skip_function_no_skip(
        self,
        basic_evaluation_task,
        basic_prompt,
        eval_data,
        mock_llm_client,
        mock_llm_response,
    ):
        """Test evaluation with skip function that doesn't trigger."""
        # Create task with skip function that skips no rows
        task_with_skip = EvaluationTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            input_columns=["text"],
            output_columns=["response"],
            answering_method="structured",
            skip_function=lambda row: False,  # Skip no rows
        )
        judge = Judge(
            id="no_skip_judge",
            evaluation_task=task_with_skip,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=basic_prompt,
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

            result = judge.evaluate_eval_data(eval_data, mock_llm_client, "test_run")

            # No rows should be skipped
            assert mock_builder.create_skipped_row.call_count == 0
            assert mock_builder.create_success_row.call_count == 3
            assert result == mock_results

    def test_evaluate_eval_data_no_skip_function(
        self, basic_judge, eval_data, mock_llm_client, mock_llm_response
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
                eval_data, mock_llm_client, "test_run"
            )

            # No skip function, so no skips
            assert mock_builder.create_skipped_row.call_count == 0
            assert mock_builder.create_success_row.call_count == 3
            assert result == mock_results

    def test_evaluate_eval_data_outer_exception(
        self, basic_evaluation_task, basic_prompt, eval_data, mock_llm_client
    ):
        """Test evaluation with error in row processing loop."""

        # Create task with skip function that raises exception
        def error_skip_function(row):
            raise RuntimeError("Skip function error")

        task_with_error = EvaluationTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            input_columns=["text"],
            output_columns=["response"],
            answering_method="structured",
            skip_function=error_skip_function,
        )
        judge = Judge(
            id="error_judge",
            evaluation_task=task_with_error,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=basic_prompt,
        )

        with patch(
            "meta_evaluator.judge.judge.JudgeResultsBuilder"
        ) as mock_builder_class:
            mock_builder = create_autospec(JudgeResultsBuilder, instance=True)
            mock_results = create_autospec(JudgeResults, instance=True)
            mock_builder.complete.return_value = mock_results
            mock_builder_class.return_value = mock_builder

            result = judge.evaluate_eval_data(eval_data, mock_llm_client, "test_run")

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
            config_call = mock_builder_class.call_args[0][0]
            assert config_call.is_sampled_run is True
            assert result == mock_results

    # === Edge Cases ===

    def test_evaluate_eval_data_empty_dataset(self, mock_llm_client, basic_prompt):
        """Test evaluation with empty dataset."""
        # Test the empty dataset scenario by creating a Judge with a task
        # that skips all rows, effectively simulating an empty dataset

        task_with_skip = EvaluationTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            input_columns=["text"],
            output_columns=["response"],
            answering_method="structured",
            skip_function=lambda row: True,  # Skip all rows
        )

        skip_judge = Judge(
            id="test_judge_empty",
            evaluation_task=task_with_skip,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=basic_prompt,
        )

        # Create minimal eval data (since empty DataFrames aren't allowed)
        df = pl.DataFrame(
            {"id": ["1"], "text": ["Test"], "response": ["Test response"]}
        )
        eval_data = EvalData(
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
                eval_data, mock_llm_client, "test_run"
            )

            # All rows should be skipped (simulating empty dataset behavior)
            assert mock_builder.create_skipped_row.call_count == 1
            assert mock_builder.create_success_row.call_count == 0
            mock_builder.complete.assert_called_once()
            assert result == mock_results

    def test_evaluation_task_free_form_outputs(self, basic_prompt):
        """Test EvaluationTask with free form outputs."""
        # Test mixed task schemas with both predefined and free form
        task = EvaluationTask(
            task_schemas={
                "sentiment": ["positive", "negative", "neutral"],  # Predefined
                "summary": None,  # Free form
                "explanation": None,  # Free form
            },
            input_columns=["text"],
            output_columns=["response"],
            answering_method="structured",
        )

        judge = Judge(
            id="free_form_judge",
            evaluation_task=task,
            llm_client_enum=LLMClientEnum.OPENAI,
            model="gpt-4",
            prompt=basic_prompt,
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
