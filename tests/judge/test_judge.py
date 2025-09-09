"""Test suite for Judge class with comprehensive path coverage."""

from unittest.mock import patch

import pytest

from meta_evaluator.eval_task import EvalTask
from meta_evaluator.judge.enums import RoleEnum
from meta_evaluator.judge.exceptions import UnsupportedFormatMethodError
from meta_evaluator.judge.judge import Judge


class TestJudge:
    """Comprehensive test suite for Judge class achieving 100% path coverage."""

    # === Validation Tests ===

    def test_validate_id_valid(self, basic_eval_task, sample_prompt):
        """Test successful ID validation with valid characters."""
        judge = Judge(
            id="valid_judge_123",
            eval_task=basic_eval_task,
            llm_client="openai",
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
                llm_client="openai",
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
                llm_client="openai",
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
            llm_client="openai",
            model="gpt-4",
            prompt=sample_prompt,
        )
        instructions = judge._get_xml_instructions()

        assert "<sentiment>" in instructions
        assert "<toxicity>" in instructions
        assert "positive, negative, neutral" in instructions
        assert "toxic, non_toxic" in instructions

    def test_create_system_message_without_xml_no_row(self, basic_judge):
        """Test system message creation without XML instructions and no row data."""
        message = basic_judge._create_system_message(
            row=None, include_xml_instructions=False
        )

        assert message.role == RoleEnum.SYSTEM
        assert message.content == "Evaluate the sentiment of the given text."
        assert "<sentiment>" not in message.content

    def test_create_system_message_with_xml_no_row(self, basic_judge):
        """Test system message creation with XML instructions and no row data."""
        message = basic_judge._create_system_message(
            row=None, include_xml_instructions=True
        )

        assert message.role == RoleEnum.SYSTEM
        assert "Evaluate the sentiment of the given text." in message.content
        assert "<sentiment>" in message.content

    def test_create_system_message_with_template_substitution(self, sample_prompt):
        """Test system message creation with template variable substitution."""
        from meta_evaluator.common.models import Prompt

        # Create a prompt with template variables
        template_prompt = Prompt(
            id="template_test",
            prompt="Evaluate the sentiment of this text: {text}. The response was: {response}",
        )

        task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="structured",
        )
        judge = Judge(
            id="test_judge",
            eval_task=task,
            llm_client="openai",
            model="gpt-4",
            prompt=template_prompt,
        )

        row = {"text": "Good movie", "response": "I liked it"}
        message = judge._create_system_message(row=row, include_xml_instructions=False)

        assert message.role == RoleEnum.SYSTEM
        assert (
            "Evaluate the sentiment of this text: Good movie. The response was: I liked it"
            in message.content
        )
        assert "{text}" not in message.content  # Variables should be substituted
        assert "{response}" not in message.content

    def test_template_variable_validation_missing_variables(self, sample_prompt):
        """Test that missing template variables raise an error."""
        from meta_evaluator.common.models import Prompt
        from meta_evaluator.judge.exceptions import MissingTemplateVariablesError

        # Create a prompt missing required template variables
        incomplete_prompt = Prompt(
            id="incomplete_test",
            prompt="Evaluate the sentiment. Missing variables here!",
        )

        task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="structured",
        )
        judge = Judge(
            id="test_judge",
            eval_task=task,
            llm_client="openai",
            model="gpt-4",
            prompt=incomplete_prompt,
        )

        row = {"text": "Good movie", "response": "I liked it"}

        with pytest.raises(MissingTemplateVariablesError) as exc_info:
            judge._create_system_message(row=row, include_xml_instructions=False)

        error = exc_info.value
        assert "text" in error.missing_variables
        assert "response" in error.missing_variables

    def test_template_variable_validation_only_response_columns(self, sample_prompt):
        """Test template substitution when only response columns are defined."""
        from meta_evaluator.common.models import Prompt

        # Create a prompt with only response template variables
        response_only_prompt = Prompt(
            id="response_only_test", prompt="Evaluate this response: {response}"
        )

        task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative"]},
            prompt_columns=None,  # No prompt columns
            response_columns=["response"],
            answering_method="structured",
        )
        judge = Judge(
            id="test_judge_no_prompt",
            eval_task=task,
            llm_client="openai",
            model="gpt-4",
            prompt=response_only_prompt,
        )

        row = {"response": "bad movie"}
        message = judge._create_system_message(row=row, include_xml_instructions=False)

        assert message.role == RoleEnum.SYSTEM
        assert "Evaluate this response: bad movie" in message.content
        assert "{response}" not in message.content  # Variable should be substituted

    # === STRUCTURED OUTPUTS FALLBACK TESTS ===

    def test_eval_task_fallback_sequence_no_fallback(self, fallback_disabled_task):
        """Test that fallback sequence returns only original method when disabled."""
        sequence = fallback_disabled_task.get_fallback_sequence()
        assert sequence == ["structured"]

    def test_eval_task_fallback_sequence_with_fallback_structured(
        self, fallback_enabled_task
    ):
        """Test fallback sequence for structured method."""
        sequence = fallback_enabled_task.get_fallback_sequence()
        assert sequence == ["structured", "instructor", "xml"]

    def test_eval_task_fallback_sequence_with_fallback_instructor(self):
        """Test fallback sequence for instructor method."""
        task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="instructor",
            structured_outputs_fallback=True,
        )
        sequence = task.get_fallback_sequence()
        assert sequence == ["instructor", "structured", "xml"]

    def test_eval_task_fallback_sequence_with_fallback_xml(self):
        """Test fallback sequence for xml method."""
        task = EvalTask(
            task_schemas={"sentiment": ["positive", "negative", "neutral"]},
            prompt_columns=["text"],
            response_columns=["response"],
            answering_method="xml",
            structured_outputs_fallback=True,
        )
        sequence = task.get_fallback_sequence()
        assert sequence == ["xml"]  # XML doesn't need fallback

    @patch("meta_evaluator.judge.sync_evaluator.supports_response_schema")
    def test_structured_fallback_disabled_raises_error(
        self, mock_supports_schema, fallback_disabled_judge, basic_eval_data
    ):
        """Test that UnsupportedFormatMethodError is raised when fallback is disabled."""
        # Mock that the model doesn't support structured outputs
        mock_supports_schema.return_value = False

        # When fallback is disabled, the UnsupportedFormatMethodError should propagate
        # and create error rows instead of crashing the evaluation
        result = fallback_disabled_judge.evaluate_eval_data(basic_eval_data, "test_run")

        # The evaluation should complete but with error rows
        assert result.total_count > 0
        # All rows should have errors because structured outputs are not supported
        assert result.get_error_count() == result.total_count

    @patch("meta_evaluator.judge.sync_evaluator.supports_response_schema")
    @patch(
        "meta_evaluator.judge.sync_evaluator.SyncEvaluationMixin._evaluate_row_structured"
    )
    @patch(
        "meta_evaluator.judge.sync_evaluator.SyncEvaluationMixin._evaluate_row_instructor"
    )
    @patch("meta_evaluator.judge.sync_evaluator.SyncEvaluationMixin._evaluate_row_xml")
    def test_structured_fallback_enabled_tries_fallback(
        self,
        mock_evaluate_xml,
        mock_evaluate_instructor,
        mock_evaluate_structured,
        mock_supports_schema,
        fallback_enabled_judge,
        basic_eval_data,
    ):
        """Test that fallback methods are attempted when structured fails."""
        # Mock that structured outputs are not supported
        mock_supports_schema.return_value = False

        # Mock structured method to raise UnsupportedFormatMethodError
        mock_evaluate_structured.side_effect = UnsupportedFormatMethodError(
            method="structured",
            model="openai/gpt-4",
            suggested_methods=["instructor", "xml"],
        )

        # Mock instructor method to raise UnsupportedFormatMethodError (so it falls back to XML)
        mock_evaluate_instructor.side_effect = UnsupportedFormatMethodError(
            method="instructor", model="openai/gpt-4", suggested_methods=["xml"]
        )

        # Mock XML method to succeed by actually creating success rows
        def mock_xml_success(*args, **kwargs):
            # Extract builder from kwargs or args
            builder = kwargs.get("builder") or args[5]  # builder is 6th parameter
            sample_example_id = kwargs.get("sample_example_id") or args[2]
            row = kwargs.get("row") or args[0]
            eval_data = kwargs.get("eval_data") or args[1]

            builder.create_success_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                outcomes={"sentiment": "positive"},
                llm_raw_response_content="<sentiment>positive</sentiment>",
                llm_prompt_tokens=10,
                llm_completion_tokens=5,
                llm_total_tokens=15,
                llm_call_duration_seconds=1.0,
            )

        mock_evaluate_xml.side_effect = mock_xml_success

        # Should not raise an error - should fallback to XML
        result = fallback_enabled_judge.evaluate_eval_data(basic_eval_data, "test_run")

        # Verify that all methods were tried in the correct order
        assert mock_evaluate_structured.called
        assert mock_evaluate_instructor.called
        assert mock_evaluate_xml.called

        # Verify each method was called once per row
        assert mock_evaluate_structured.call_count == result.total_count
        assert mock_evaluate_instructor.call_count == result.total_count
        assert mock_evaluate_xml.call_count == result.total_count

    @patch("meta_evaluator.judge.sync_evaluator.supports_response_schema")
    @patch(
        "meta_evaluator.judge.sync_evaluator.SyncEvaluationMixin._evaluate_row_structured"
    )
    @patch(
        "meta_evaluator.judge.sync_evaluator.SyncEvaluationMixin._evaluate_row_instructor"
    )
    @patch("meta_evaluator.judge.sync_evaluator.SyncEvaluationMixin._evaluate_row_xml")
    def test_fallback_stops_when_method_succeeds(
        self,
        mock_evaluate_xml,
        mock_evaluate_instructor,
        mock_evaluate_structured,
        mock_supports_schema,
        fallback_enabled_judge,
        basic_eval_data,
    ):
        """Test that fallback stops trying methods once one succeeds."""
        # Mock that structured outputs are not supported
        mock_supports_schema.return_value = False

        # Mock structured method to raise UnsupportedFormatMethodError
        mock_evaluate_structured.side_effect = UnsupportedFormatMethodError(
            method="structured",
            model="openai/gpt-4",
            suggested_methods=["instructor", "xml"],
        )

        # Mock instructor method to succeed by creating success rows
        def mock_instructor_success(*args, **kwargs):
            # Extract builder from kwargs or args
            builder = kwargs.get("builder") or args[4]  # builder is 5th parameter
            sample_example_id = kwargs.get("sample_example_id") or args[2]
            row = kwargs.get("row") or args[0]
            eval_data = kwargs.get("eval_data") or args[1]

            builder.create_success_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                outcomes={"sentiment": "positive"},
                llm_raw_response_content='{"sentiment": "positive"}',
                llm_prompt_tokens=10,
                llm_completion_tokens=5,
                llm_total_tokens=15,
                llm_call_duration_seconds=1.0,
            )

        mock_evaluate_instructor.side_effect = mock_instructor_success

        # Mock XML method should NOT be called since instructor succeeds
        mock_evaluate_xml.return_value = None

        # Should not raise an error - should succeed with instructor
        result = fallback_enabled_judge.evaluate_eval_data(basic_eval_data, "test_run")

        # Verify structured and instructor were called
        assert mock_evaluate_structured.called
        assert mock_evaluate_instructor.called

        # Verify XML was NOT called since instructor succeeded
        assert not mock_evaluate_xml.called

        # Verify call counts
        assert mock_evaluate_structured.call_count == result.total_count
        assert mock_evaluate_instructor.call_count == result.total_count
        assert mock_evaluate_xml.call_count == 0

    def test_fallback_configuration_validation(
        self,
        fallback_enabled_judge,
    ):
        """Test that fallback is enabled and configured correctly."""
        # Verify that fallback is enabled for this judge
        assert fallback_enabled_judge.eval_task.structured_outputs_fallback is True

        # Verify fallback sequence includes instructor method
        sequence = fallback_enabled_judge.eval_task.get_fallback_sequence()
        assert "instructor" in sequence
        assert "xml" in sequence

        # Verify that the structured method is tried first
        assert sequence[0] == "structured"

    def test_fallback_disabled_does_not_call_other_methods(
        self, fallback_disabled_judge, basic_eval_data
    ):
        """Test that when fallback is disabled, only the specified method is used."""
        with patch(
            "meta_evaluator.judge.sync_evaluator.supports_response_schema"
        ) as mock_supports_schema:
            with patch(
                "meta_evaluator.judge.sync_evaluator.SyncEvaluationMixin._evaluate_row_structured"
            ) as mock_evaluate_structured:
                with patch(
                    "meta_evaluator.judge.sync_evaluator.SyncEvaluationMixin._evaluate_row_instructor"
                ) as mock_evaluate_instructor:
                    with patch(
                        "meta_evaluator.judge.sync_evaluator.SyncEvaluationMixin._evaluate_row_xml"
                    ) as mock_evaluate_xml:
                        # Mock that structured outputs are supported (so it tries structured)
                        mock_supports_schema.return_value = True

                        # Mock structured method to succeed by creating success rows
                        def mock_structured_success(*args, **kwargs):
                            # Extract builder from kwargs or args
                            builder = (
                                kwargs.get("builder") or args[4]
                            )  # builder is 5th parameter
                            sample_example_id = (
                                kwargs.get("sample_example_id") or args[2]
                            )
                            row = kwargs.get("row") or args[0]
                            eval_data = kwargs.get("eval_data") or args[1]

                            builder.create_success_row(
                                sample_example_id=sample_example_id,
                                original_id=row[eval_data.id_column],
                                outcomes={"sentiment": "positive"},
                                llm_raw_response_content='{"sentiment": "positive"}',
                                llm_prompt_tokens=10,
                                llm_completion_tokens=5,
                                llm_total_tokens=15,
                                llm_call_duration_seconds=1.0,
                            )

                        mock_evaluate_structured.side_effect = mock_structured_success

                        # Execute evaluation
                        fallback_disabled_judge.evaluate_eval_data(
                            basic_eval_data, "test_run"
                        )

                        # Only structured should be called
                        assert mock_evaluate_structured.called
                        assert not mock_evaluate_instructor.called
                        assert not mock_evaluate_xml.called

    # === Serialization Tests ===

    def test_judge_serialize(self, basic_judge):
        """Test that Judge can be serialized to JudgeState."""
        judge_state = basic_judge.serialize()

        # Check that all fields are preserved
        assert judge_state.id == basic_judge.id
        assert judge_state.llm_client == basic_judge.llm_client
        assert judge_state.model == basic_judge.model
        assert judge_state.prompt == basic_judge.prompt
        assert judge_state.eval_task == basic_judge.eval_task.serialize()

    def test_judge_deserialize(self, basic_judge):
        """Test that Judge can be deserialized from JudgeState."""
        # First serialize the judge
        judge_state = basic_judge.serialize()

        # Then deserialize it
        deserialized_judge = Judge.deserialize(judge_state)

        # Check that all fields match the original
        assert deserialized_judge.id == basic_judge.id
        assert deserialized_judge.llm_client == basic_judge.llm_client
        assert deserialized_judge.model == basic_judge.model
        assert deserialized_judge.prompt == basic_judge.prompt
        assert (
            deserialized_judge.eval_task.task_schemas
            == basic_judge.eval_task.task_schemas
        )
        assert (
            deserialized_judge.eval_task.prompt_columns
            == basic_judge.eval_task.prompt_columns
        )
        assert (
            deserialized_judge.eval_task.response_columns
            == basic_judge.eval_task.response_columns
        )
        assert (
            deserialized_judge.eval_task.answering_method
            == basic_judge.eval_task.answering_method
        )

        # Verify the deserialized judge is frozen (immutable)
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError, match="Instance is frozen"):
            deserialized_judge.id = "new_id"
