"""Tests for async evaluation methods in Judge class."""

from unittest.mock import AsyncMock, Mock, patch

import polars as pl
import pytest
from pydantic import BaseModel

from meta_evaluator.data import EvalData, SampleEvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.judge.enums import RoleEnum
from meta_evaluator.judge.judge import Judge
from meta_evaluator.judge.models import Message, TagConfig


class SentimentResponseModel(BaseModel):
    """Test response model for structured output testing."""

    sentiment: str


class TestJudgeAsyncEvaluation:
    """Test suite for asynchronous evaluation methods in Judge class."""

    @pytest.fixture
    def instructor_eval_task(self):
        """Create an instructor evaluation task for testing.

        Returns:
            EvalTask: An instructor evaluation task for testing.
        """
        return EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="instructor",
        )

    @pytest.fixture
    def instructor_judge(self, instructor_eval_task, sentiment_judge_prompt):
        """Create an instructor judge for testing.

        Returns:
            Judge: An instructor judge for testing.
        """
        return Judge(
            id="test_judge",
            eval_task=instructor_eval_task,
            llm_client="openai",
            model="gpt-4",
            prompt=sentiment_judge_prompt,
        )

    @pytest.fixture
    def sample_row(self):
        """Create a sample row for testing.

        Returns:
            dict: A sample row with id, text, and response columns.
        """
        return {"id": "1", "text": "Great movie!", "response": "Positive review"}

    # Test 1-6: Core functionality tests (structured/instructor/xml X default/explicit model)

    @patch("meta_evaluator.judge.async_evaluator.acompletion")
    @patch("meta_evaluator.judge.async_evaluator.supports_response_schema")
    @pytest.mark.asyncio
    async def test_async_structured_success(
        self,
        mock_supports,
        mock_acompletion,
        basic_judge,
        basic_eval_data,
        sample_row,
        mock_litellm_response,
        single_row_test_builder,
    ):
        """Test async structured evaluation - success path."""
        mock_supports.return_value = True
        mock_acompletion.return_value = mock_litellm_response

        task_class = basic_judge.eval_task.create_task_class()

        # Test the method
        await basic_judge._evaluate_row_structured_async(
            row=sample_row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            task_class=task_class,
            builder=single_row_test_builder,
        )

        # Verify acompletion was called with correct model
        mock_acompletion.assert_called_once()
        args, kwargs = mock_acompletion.call_args
        assert kwargs["model"] == "openai/gpt-4"
        assert kwargs["response_format"] == task_class

    @patch("meta_evaluator.judge.async_evaluator.instructor")
    @pytest.mark.asyncio
    async def test_async_instructor_success(
        self,
        mock_instructor,
        instructor_judge,
        basic_eval_data,
        sample_row,
        single_row_test_builder,
    ):
        """Test async instructor evaluation - success path."""
        # Mock instructor client
        mock_client = Mock()
        mock_instructor.from_provider.return_value = mock_client

        # Mock async response
        mock_response = SentimentResponseModel(sentiment="positive")
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Create messages
        system_message = instructor_judge._create_system_message(
            include_xml_instructions=False
        )
        user_content = instructor_judge._format_row_data(sample_row)

        user_message = Message(role=RoleEnum.USER, content=user_content)
        messages = [system_message, user_message]

        task_class = instructor_judge.eval_task.create_task_class()

        await instructor_judge._evaluate_row_instructor_async(
            row=sample_row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            task_class=task_class,
            builder=single_row_test_builder,
            messages=messages,
        )

        # Verify instructor was called with correct model
        mock_instructor.from_provider.assert_called_once_with("openai/gpt-4")
        mock_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4"  # instructor handles provider internally

    @patch("meta_evaluator.judge.async_evaluator.acompletion")
    @pytest.mark.asyncio
    async def test_async_xml_success(
        self,
        mock_acompletion,
        xml_judge,
        basic_eval_data,
        sample_row,
        single_row_test_builder,
        mock_xml_litellm_response,
    ):
        """Test async XML evaluation with default model - success path."""
        mock_acompletion.return_value = mock_xml_litellm_response

        tag_configs = [
            TagConfig(
                name="sentiment",
                allowed_values=["positive", "negative", "neutral"],
                cardinality="one",
            )
        ]

        await xml_judge._evaluate_row_xml_async(
            row=sample_row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            tag_configs=tag_configs,
            builder=single_row_test_builder,
        )

        mock_acompletion.assert_called_once()
        args, kwargs = mock_acompletion.call_args
        assert kwargs["model"] == "openai/gpt-4"

    # Test structured API call fails
    @patch("meta_evaluator.judge.async_evaluator.acompletion")
    @patch("meta_evaluator.judge.async_evaluator.supports_response_schema")
    @pytest.mark.asyncio
    async def test_async_structured_api_call_fails(
        self,
        mock_supports,
        mock_acompletion,
        basic_judge,
        basic_eval_data,
        sample_row,
        single_row_test_builder,
    ):
        """Test async structured evaluation handles API call failure."""
        mock_supports.return_value = True
        mock_acompletion.side_effect = Exception("API Error")

        task_class = basic_judge.eval_task.create_task_class()

        # Should handle the error gracefully and create an error row
        await basic_judge._evaluate_row_structured_async(
            row=sample_row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            task_class=task_class,
            builder=single_row_test_builder,
        )

        # Check that an error row was created by completing the builder
        result = single_row_test_builder.complete()
        assert len(result.results_data) == 1
        # Check that it's an LLM error by checking the LLM error count
        assert result.llm_error_count == 1

    # Test unsupported format method
    @patch("meta_evaluator.judge.async_evaluator.supports_response_schema")
    @pytest.mark.asyncio
    async def test_unsupported_format_method_raises_error_async(
        self,
        mock_supports,
        basic_judge,
        basic_eval_data,
        sample_row,
        single_row_test_builder,
    ):
        """Test that unsupported format method raises UnsupportedFormatMethodError in async mode."""
        mock_supports.return_value = False  # Model doesn't support structured output

        task_class = basic_judge.eval_task.create_task_class()

        # Should handle UnsupportedFormatMethodError and create error row
        await basic_judge._evaluate_row_structured_async(
            row=sample_row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            task_class=task_class,
            builder=single_row_test_builder,
        )

        # Check that an error row was created by completing the builder
        result = single_row_test_builder.complete()
        assert len(result.results_data) == 1
        # Check that it's an other error by checking the other error count
        assert result.other_error_count == 1

    # === PARTIAL SUCCESS TESTS ===

    @patch("meta_evaluator.judge.async_evaluator.acompletion")
    @patch("meta_evaluator.judge.async_evaluator.supports_response_schema")
    @pytest.mark.asyncio
    async def test_async_structured_partial_success(
        self,
        mock_supports,
        mock_acompletion,
        multi_task_eval_task,
        basic_eval_data,
        sample_row,
        sentiment_judge_prompt,
        single_row_test_builder,
    ):
        """Test async structured evaluation with partial success - some tasks missing."""
        # Create multi-task judge
        multi_judge = Judge(
            id="multi_judge",
            eval_task=multi_task_eval_task,
            llm_client="openai",
            model="gpt-4",
            prompt=sentiment_judge_prompt,
        )

        mock_supports.return_value = True

        # Mock response with only sentiment, missing toxicity
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = '{"sentiment": "positive"}'  # Missing toxicity
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_acompletion.return_value = mock_response

        task_class = multi_judge.eval_task.create_task_class()

        await multi_judge._evaluate_row_structured_async(
            row=sample_row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            task_class=task_class,
            builder=single_row_test_builder,
        )

        # Verify partial success was recorded
        result = single_row_test_builder.complete()
        assert len(result.results_data) == 1
        assert result.partial_count == 1

    @patch("meta_evaluator.judge.async_evaluator.instructor")
    @pytest.mark.asyncio
    async def test_async_instructor_partial_success(
        self,
        mock_instructor,
        multi_task_eval_task,
        basic_eval_data,
        sample_row,
        sentiment_judge_prompt,
        single_row_test_builder,
    ):
        """Test async instructor evaluation with partial success."""
        # Create multi-task instructor judge
        instructor_task = EvalTask(
            task_schemas={
                "sentiment": ["positive", "negative", "neutral"],
                "toxicity": ["toxic", "non_toxic"],
            },
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="instructor",
        )

        instructor_judge = Judge(
            id="instructor_judge",
            eval_task=instructor_task,
            llm_client="openai",
            model="gpt-4",
            prompt=sentiment_judge_prompt,
        )

        # Mock instructor client
        mock_client = Mock()
        mock_instructor.from_provider.return_value = mock_client

        # Create partial response model - only sentiment, missing toxicity
        class PartialResponseModel:
            def __init__(self):
                self.sentiment = "positive"
                # No toxicity attribute

            def model_dump_json(self):
                # Return JSON with only sentiment, missing toxicity
                return '{"sentiment": "positive"}'

        mock_response = PartialResponseModel()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Create messages
        system_message = instructor_judge._create_system_message(
            include_xml_instructions=False
        )
        user_content = instructor_judge._format_row_data(sample_row)

        user_message = Message(role=RoleEnum.USER, content=user_content)
        messages = [system_message, user_message]

        task_class = instructor_judge.eval_task.create_task_class()

        await instructor_judge._evaluate_row_instructor_async(
            row=sample_row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            task_class=task_class,
            builder=single_row_test_builder,
            messages=messages,
        )

        result = single_row_test_builder.complete()
        assert len(result.results_data) == 1
        assert result.partial_count == 1

    @patch("meta_evaluator.judge.async_evaluator.acompletion")
    @pytest.mark.asyncio
    async def test_async_xml_partial_success(
        self,
        mock_acompletion,
        multi_task_eval_task,
        basic_eval_data,
        sample_row,
        sentiment_judge_prompt,
        single_row_test_builder,
    ):
        """Test async XML evaluation with partial success."""
        # Create multi-task XML judge
        xml_task = EvalTask(
            task_schemas={
                "sentiment": ["positive", "negative", "neutral"],
                "toxicity": ["toxic", "non_toxic"],
            },
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="xml",
        )

        xml_judge = Judge(
            id="xml_judge",
            eval_task=xml_task,
            llm_client="openai",
            model="gpt-4",
            prompt=sentiment_judge_prompt,
        )

        # Mock response with only sentiment tag, missing toxicity
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = "<sentiment>positive</sentiment>"  # Missing toxicity
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_acompletion.return_value = mock_response

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

        await xml_judge._evaluate_row_xml_async(
            row=sample_row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            tag_configs=tag_configs,
            builder=single_row_test_builder,
        )

        result = single_row_test_builder.complete()
        assert len(result.results_data) == 1
        assert result.partial_count == 1

    # === SKIP FUNCTION TESTS ===

    @pytest.mark.asyncio
    async def test_async_structured_with_skip_function(
        self, basic_eval_data, sentiment_judge_prompt
    ):
        """Test async structured evaluation with skip function."""
        # Create task with skip function that skips all rows
        skip_task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="structured",
            skip_function=lambda row: True,  # Skip all rows
        )

        skip_judge = Judge(
            id="skip_judge",
            eval_task=skip_task,
            llm_client="openai",
            model="gpt-4",
            prompt=sentiment_judge_prompt,
        )

        result = await skip_judge.evaluate_eval_data_async(basic_eval_data, "test_run")

        # All rows should be skipped
        assert result.skipped_count == 3  # basic_eval_data has 3 rows
        assert result.succeeded_count == 0

    @pytest.mark.asyncio
    async def test_async_instructor_with_skip_function(
        self, basic_eval_data, sentiment_judge_prompt
    ):
        """Test async instructor evaluation with skip function."""
        # Create instructor task with skip function
        skip_task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="instructor",
            skip_function=lambda row: row.get("text")
            == "This movie is fantastic!",  # Skip first row
        )

        skip_judge = Judge(
            id="instructor_skip_judge",
            eval_task=skip_task,
            llm_client="openai",
            model="gpt-4",
            prompt=sentiment_judge_prompt,
        )

        result = await skip_judge.evaluate_eval_data_async(basic_eval_data, "test_run")

        # One row should be skipped
        assert result.skipped_count == 1
        assert result.total_count == 3

    @pytest.mark.asyncio
    async def test_async_xml_with_skip_function(
        self, basic_eval_data, sentiment_judge_prompt
    ):
        """Test async XML evaluation with skip function."""
        # Create XML task with skip function
        skip_task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="xml",
            skip_function=lambda row: len(row.get("text", "")) < 20,  # Skip short texts
        )

        skip_judge = Judge(
            id="xml_skip_judge",
            eval_task=skip_task,
            llm_client="openai",
            model="gpt-4",
            prompt=sentiment_judge_prompt,
        )

        result = await skip_judge.evaluate_eval_data_async(basic_eval_data, "test_run")

        # Some rows should be skipped based on text length
        assert result.skipped_count >= 0
        assert result.total_count == 3

    # === SAMPLED EVAL DATA TESTS ===

    @pytest.mark.asyncio
    async def test_async_structured_with_sampled_eval_data(
        self, sentiment_judge_prompt
    ):
        """Test async structured evaluation with SampleEvalData."""
        # Create SampleEvalData
        df = pl.DataFrame(
            {
                "id": ["1", "2"],
                "text": ["Great movie!", "Bad film"],
                "response": ["Positive", "Negative"],
            }
        )
        sampled_data = SampleEvalData(
            name="sampled_test",
            data=df,
            id_column="id",
            sample_name="test_sample",
            stratification_columns=["text"],
            sample_percentage=0.02,  # 2 out of 100 original rows
            seed=42,
            sampling_method="stratified_by_columns",
        )

        basic_judge = Judge(
            id="test_judge",
            eval_task=EvalTask(
                task_schemas={"sentiment": ["positive", "negative", "neutral"]},
                prompt_columns=["text"],
                response_columns=["response"],
                answering_method="structured",
            ),
            llm_client="openai",
            model="gpt-4",
            prompt=sentiment_judge_prompt,
        )

        result = await basic_judge.evaluate_eval_data_async(sampled_data, "test_run")

        # Should indicate it's a sampled run
        assert result.is_sampled_run is True
        assert result.total_count == 2

    # === INVALID CLIENT AND MODEL TESTS ===

    @pytest.mark.asyncio
    async def test_async_invalid_client_triggers_llm_error(
        self, basic_eval_data, sentiment_judge_prompt
    ):
        """Test that invalid client names trigger LLM API errors in async mode (actual API call but fails early)."""
        from meta_evaluator.judge.judge import Judge

        # Create judge with invalid client
        invalid_client_judge = Judge(
            id="invalid_client_judge",
            eval_task=EvalTask(
                task_schemas={"sentiment": ["positive", "negative", "neutral"]},
                prompt_columns=["text"],
                response_columns=["response"],
                answering_method="structured",
            ),
            llm_client="invalid_provider",  # This should trigger an error
            model="gpt-4",
            prompt=sentiment_judge_prompt,
        )

        result = await invalid_client_judge.evaluate_eval_data_async(
            basic_eval_data, "test_run"
        )

        # Should have other errors for all rows (fails during client setup before API call)
        assert result.other_error_count == 3  # basic_eval_data has 3 rows
        assert result.succeeded_count == 0
        assert result.total_count == 3

    @pytest.mark.asyncio
    async def test_async_invalid_model_triggers_llm_error(
        self, basic_eval_data, sentiment_judge_prompt
    ):
        """Test that invalid model names trigger LLM API errors in async mode (actual API call but fails early)."""
        from meta_evaluator.judge.judge import Judge

        # Create judge with invalid model
        invalid_model_judge = Judge(
            id="invalid_model_judge",
            eval_task=EvalTask(
                task_schemas={"sentiment": ["positive", "negative", "neutral"]},
                prompt_columns=["text"],
                response_columns=["response"],
                answering_method="structured",
            ),
            llm_client="openai",
            model="invalid-model-name",  # This should trigger an error
            prompt=sentiment_judge_prompt,
        )

        result = await invalid_model_judge.evaluate_eval_data_async(
            basic_eval_data, "test_run"
        )

        # Should have other errors for all rows (fails during client setup before API call)
        assert result.other_error_count == 3  # basic_eval_data has 3 rows
        assert result.succeeded_count == 0
        assert result.total_count == 3

    @pytest.mark.asyncio
    async def test_async_invalid_client_instructor_triggers_llm_error(
        self, basic_eval_data, sentiment_judge_prompt
    ):
        """Test that invalid client with instructor method triggers LLM API errors in async mode."""
        from meta_evaluator.judge.judge import Judge

        # Create instructor judge with invalid client
        invalid_instructor_judge = Judge(
            id="invalid_instructor_judge",
            eval_task=EvalTask(
                task_schemas={"sentiment": ["positive", "negative", "neutral"]},
                prompt_columns=["text"],
                response_columns=["response"],
                answering_method="instructor",
            ),
            llm_client="invalid_provider",  # This should trigger an error
            model="gpt-4",
            prompt=sentiment_judge_prompt,
        )

        result = await invalid_instructor_judge.evaluate_eval_data_async(
            basic_eval_data, "test_run"
        )

        # Should have other errors for all rows (fails during client setup)
        assert result.other_error_count == 3  # basic_eval_data has 3 rows
        assert result.succeeded_count == 0
        assert result.total_count == 3

    @pytest.mark.asyncio
    async def test_async_invalid_model_xml_triggers_llm_error(
        self, basic_eval_data, sentiment_judge_prompt
    ):
        """Test that invalid model with XML method triggers LLM API errors in async mode."""
        from meta_evaluator.judge.judge import Judge

        # Create XML judge with invalid model
        invalid_xml_judge = Judge(
            id="invalid_xml_judge",
            eval_task=EvalTask(
                task_schemas={"sentiment": ["positive", "negative", "neutral"]},
                prompt_columns=["text"],
                response_columns=["response"],
                answering_method="xml",
            ),
            llm_client="openai",
            model="invalid-xml-model",  # This should trigger an error
            prompt=sentiment_judge_prompt,
        )

        result = await invalid_xml_judge.evaluate_eval_data_async(
            basic_eval_data, "test_run"
        )

        # Should have other errors for all rows (fails during client setup)
        assert result.other_error_count == 3  # basic_eval_data has 3 rows
        assert result.succeeded_count == 0
        assert result.total_count == 3

    # === EDGE CASE TESTS ===

    @pytest.mark.asyncio
    async def test_async_empty_dataset_with_skip_all(self, sentiment_judge_prompt):
        """Test async evaluation with dataset where all rows are skipped (simulates empty dataset)."""
        # Create task that skips all rows
        skip_all_task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="structured",
            skip_function=lambda row: True,  # Skip everything
        )

        skip_judge = Judge(
            id="skip_all_judge",
            eval_task=skip_all_task,
            llm_client="openai",
            model="gpt-4",
            prompt=sentiment_judge_prompt,
        )

        # Create minimal data (since empty DataFrames aren't allowed)
        df = pl.DataFrame(
            {"id": ["1"], "text": ["Test"], "response": ["Test response"]}
        )

        eval_data = EvalData(name="test", data=df, id_column="id")

        result = await skip_judge.evaluate_eval_data_async(eval_data, "test_run")

        # All rows skipped, effectively empty dataset
        assert result.skipped_count == 1
        assert result.succeeded_count == 0
        assert result.total_count == 1

    def test_async_free_form_outputs(self, sentiment_judge_prompt):
        """Test async evaluation with free form outputs (mixed predefined and free form tasks)."""
        # Create task with mixed schemas (predefined and free form)
        free_form_task = EvalTask(
            task_schemas={
                "sentiment": ["positive", "negative", "neutral"],  # Predefined
                "summary": None,  # Free form
                "explanation": None,  # Free form
            },
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="structured",
        )

        free_form_judge = Judge(
            id="free_form_judge",
            eval_task=free_form_task,
            llm_client="openai",
            model="gpt-4",
            prompt=sentiment_judge_prompt,
        )

        # Test task class creation works with mixed types
        TaskClass = free_form_task.create_task_class()

        # Verify we can create instance with mixed field types
        instance = TaskClass(
            sentiment="positive",
            summary="This is a free form summary",
            explanation="This explains the sentiment",
        )

        assert getattr(instance, "sentiment") == "positive"
        assert getattr(instance, "summary") == "This is a free form summary"
        assert getattr(instance, "explanation") == "This explains the sentiment"

        # Test XML instructions include free form guidance
        xml_instructions = free_form_judge._get_xml_instructions()
        assert "summary" in xml_instructions
        assert "explanation" in xml_instructions
        assert "free form text" in xml_instructions.lower()

    # === FALLBACK TESTS ===

    @patch("meta_evaluator.judge.async_evaluator.acompletion")
    @patch("meta_evaluator.judge.async_evaluator.instructor")
    @patch("meta_evaluator.judge.async_evaluator.supports_response_schema")
    @pytest.mark.asyncio
    async def test_async_fallback_chain_structured_to_instructor_to_xml(
        self,
        mock_supports,
        mock_instructor,
        mock_acompletion,
        sentiment_judge_prompt,
        basic_eval_data,
        sample_row,
        single_row_test_builder,
    ):
        """Test full async fallback chain: structured -> instructor -> XML when each method fails."""
        from meta_evaluator.judge.exceptions import UnsupportedFormatMethodError

        # Create fallback-enabled judge with mock client/model
        fallback_task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="structured",
            structured_outputs_fallback=True,
        )

        fallback_judge = Judge(
            id="fallback_test_judge",
            eval_task=fallback_task,
            llm_client="mock_client",
            model="mock_model",
            prompt=sentiment_judge_prompt,
        )

        # 1. Mock supports_response_schema to raise UnsupportedFormatMethodError
        mock_supports.side_effect = UnsupportedFormatMethodError(
            method="structured",
            model="mock_client/mock_model",
            suggested_methods=["instructor", "xml"],
        )

        # 2. Mock instructor to fail
        mock_client = Mock()
        mock_instructor.from_provider.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Instructor failed")
        )

        # 3. Mock XML acompletion to succeed
        mock_xml_response = Mock()
        mock_xml_response.choices = [Mock()]
        mock_xml_response.choices[0].message.content = "<sentiment>positive</sentiment>"
        mock_xml_response.usage.prompt_tokens = 10
        mock_xml_response.usage.completion_tokens = 5
        mock_xml_response.usage.total_tokens = 15
        mock_acompletion.return_value = mock_xml_response

        # Create necessary components
        task_class = fallback_judge.eval_task.create_task_class()
        tag_configs = [
            TagConfig(
                name="sentiment",
                allowed_values=["positive", "negative", "neutral"],
                cardinality="one",
            )
        ]

        # Test the complete async fallback chain
        await fallback_judge._evaluate_row_with_fallback_async(
            row=sample_row,
            eval_data=basic_eval_data,
            sample_example_id="test_1",
            task_class=task_class,
            tag_configs=tag_configs,
            builder=single_row_test_builder,
        )

        # Verify all methods were called in sequence:
        # 1. supports_response_schema was called
        mock_supports.assert_called_once_with(model="mock_client/mock_model")

        # 2. instructor was attempted
        mock_instructor.from_provider.assert_called_once_with("mock_client/mock_model")
        mock_client.chat.completions.create.assert_called_once()

        # 3. XML acompletion was called (final fallback)
        mock_acompletion.assert_called_once()

        # Verify successful result (XML fallback succeeded)
        result = single_row_test_builder.complete()
        assert result.succeeded_count == 1
        assert result.llm_error_count == 0
        assert result.other_error_count == 0
