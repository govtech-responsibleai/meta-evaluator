"""Synchronous evaluation methods for Judge class."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, cast

import instructor
from litellm import completion
from litellm.types.utils import ModelResponse
from litellm.utils import supports_response_schema
from pydantic import BaseModel

from meta_evaluator.common.models import Prompt
from meta_evaluator.eval_task import EvalTask

from ..data import EvalData, SampleEvalData
from ..results import JudgeResults, JudgeResultsBuilder
from .enums import RoleEnum
from .exceptions import LLMAPIError, UnsupportedFormatMethodError
from .models import (
    LLMResponse,
    LLMUsage,
    Message,
    TagConfig,
)


class SyncEvaluationMixin(ABC):
    """Mixin providing synchronous evaluation methods for Judge class."""

    # Type hints for attributes that will be provided by Judge
    eval_task: EvalTask
    llm_client: str
    model: str
    prompt: Prompt
    id: str

    @property
    @abstractmethod
    def logger(self) -> logging.Logger:
        """Logger for this Judge instance."""
        pass

    def _evaluate_row_with_fallback(
        self,
        row: dict[str, Any],
        eval_data: EvalData,
        sample_example_id: str,
        task_class: type[BaseModel],
        tag_configs: list[TagConfig],
        builder: JudgeResultsBuilder,
    ) -> None:
        """Evaluate a single row with fallback support.

        Tries methods in the fallback sequence until one succeeds or all fail.
        If all methods fail, creates an error row instead of raising an exception.

        Args:
            row: Row data from EvalData
            eval_data: The EvalData instance
            sample_example_id: Unique identifier for this sample
            task_class: The dynamically created Pydantic model for responses
            tag_configs: List of TagConfigs for XML parsing
            builder: The JudgeResultsBuilder instance
        """
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )
        fallback_sequence = self.eval_task.get_fallback_sequence()
        last_error = None

        for attempt_idx, method in enumerate(fallback_sequence):
            try:
                if method == "structured":
                    self._evaluate_row_structured(
                        row=row,
                        eval_data=eval_data,
                        sample_example_id=sample_example_id,
                        task_class=task_class,
                        builder=builder,
                        fallback_enabled=True,
                    )
                    return  # Success, exit early
                elif method == "instructor":
                    # Create system message with template substitution for instructor method
                    system_message = self._create_system_message(  # type: ignore
                        row=row,
                        include_xml_instructions=False,
                    )
                    messages = [system_message]

                    self._evaluate_row_instructor(
                        row=row,
                        eval_data=eval_data,
                        sample_example_id=sample_example_id,
                        task_class=task_class,
                        builder=builder,
                        messages=messages,
                        fallback_enabled=True,
                    )
                    return  # Success, exit early
                elif method == "xml":
                    self._evaluate_row_xml(
                        row=row,
                        eval_data=eval_data,
                        sample_example_id=sample_example_id,
                        tag_configs=tag_configs,
                        builder=builder,
                        fallback_enabled=True,
                    )
                    return  # Success, exit early

            except Exception as e:
                # This method is not supported, try the next one
                last_error = e
                self.logger.warning(
                    f"Error: {e} | Method '{method}' not supported, trying next fallback method"
                )
                continue

        # All methods failed, create error row instead of raising exception
        if last_error:
            self.logger.error(
                f"All fallback methods failed for row {sample_example_id}. Last error: {last_error}"
            )
            builder.create_other_error_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                error=last_error,
            )
        else:
            # This shouldn't happen, but just in case
            full_model_name = f"{self.llm_client}/{self.model}"
            unsupported_error = UnsupportedFormatMethodError(
                method=self.eval_task.answering_method,
                model=full_model_name,
                suggested_methods=["structured", "instructor", "xml"],
            )
            self.logger.error(
                f"No fallback methods available for row {sample_example_id}. Error: {unsupported_error}"
            )
            builder.create_other_error_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                error=unsupported_error,
            )

    def _evaluate_row_structured(
        self,
        row: dict[str, Any],
        eval_data: EvalData,
        sample_example_id: str,
        task_class: type[BaseModel],
        builder: JudgeResultsBuilder,
        fallback_enabled: bool = False,
    ) -> None:
        """Evaluate a single row using structured response method.

        Args:
            row: Row data from EvalData
            eval_data: The EvalData instance
            sample_example_id: Unique identifier for this sample
            task_class: The dynamically created Pydantic model for responses
            builder: The JudgeResultsBuilder instance
            fallback_enabled: Whether fallback is enabled

        Raises:
            UnsupportedFormatMethodError: If the answering method is not supported.
            LLMAPIError: If the LLM API call fails.
        """
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        try:
            # Create system message with template substitution
            system_message = self._create_system_message(  # type: ignore
                row=row, include_xml_instructions=False
            )  # type: ignore
            messages = [system_message]

            # Call LLM with structured response
            start_time = time.time()

            full_model_name = f"{self.llm_client}/{self.model}"

            # Check if model supports structured output
            if not supports_response_schema(model=full_model_name):
                # Raise exception suggesting alternatives instead of auto-fallback
                raise UnsupportedFormatMethodError(
                    method="structured",
                    model=full_model_name,
                    suggested_methods=["instructor", "xml"],
                )

            # Model supports structured outputs
            try:
                # Convert messages to OpenAI format
                openai_messages = self._convert_messages_to_openai_format(messages)  # type: ignore
                response = cast(
                    ModelResponse,
                    completion(
                        model=full_model_name,
                        messages=openai_messages,
                        response_format=task_class,  # Pydantic model for output
                    ),
                )

                structured_response = response.choices[0].message.content  # type: ignore
                self.logger.info(f"Structured response: {structured_response}")

                usage = LLMUsage(
                    prompt_tokens=response.usage.prompt_tokens,  # type: ignore
                    completion_tokens=response.usage.completion_tokens,  # type: ignore
                    total_tokens=response.usage.total_tokens,  # type: ignore
                )

                new_message = Message(
                    role=RoleEnum.ASSISTANT,
                    content=structured_response
                    or "",  # structured_response is already a JSON string
                )
                new_message_list = messages + [new_message]

                llm_response = LLMResponse(
                    llm_client=self.llm_client,
                    model=self.model,
                    messages=new_message_list,
                    usage=usage,
                )
                self.logger.info(f"Latest response: {llm_response.latest_response}")
                self.logger.info(f"Output usage: {llm_response.usage}")

                call_duration = time.time() - start_time

                # Extract outcomes from JSON response
                outcomes, missing_tasks = self._extract_outcomes_from_json(  # type: ignore
                    structured_response
                )

                self._assign_outcomes(  # type: ignore
                    outcomes=outcomes,
                    missing_tasks=missing_tasks,
                    llm_response=llm_response,
                    call_duration=call_duration,
                    row=row,
                    eval_data=eval_data,
                    sample_example_id=sample_example_id,
                    builder=builder,
                )

            except Exception as e:
                raise LLMAPIError(
                    f"LLM API call failed for {self.llm_client}",
                    self.llm_client,
                    e,
                )

        except LLMAPIError as llm_error:
            if fallback_enabled:
                raise llm_error

            # LLM API error
            builder.create_llm_error_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                error=llm_error,
            )

        except Exception as unexpected_error:
            if fallback_enabled:
                raise unexpected_error

            # Any other unexpected error (e.g., message creation, network issues)
            builder.create_other_error_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                error=unexpected_error,
            )

    def _evaluate_row_instructor(
        self,
        row: dict[str, Any],
        eval_data: EvalData,
        sample_example_id: str,
        task_class: type[BaseModel],
        builder: JudgeResultsBuilder,
        messages: list[Message],
        fallback_enabled: bool = False,
    ) -> None:
        """Evaluate a single row using instructor for structured response (fallback method).

        Args:
            row: Row data from EvalData
            eval_data: The EvalData instance
            sample_example_id: Unique identifier for this sample
            task_class: The dynamically created Pydantic model for responses
            builder: The JudgeResultsBuilder instance
            messages: Pre-created messages for the LLM
            fallback_enabled: Whether fallback is enabled
        """
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        try:
            # Create instructor client using the provider format
            full_model_name = f"{self.llm_client}/{self.model}"
            client = instructor.from_provider(full_model_name)

            # Convert messages to OpenAI format
            openai_messages = self._convert_messages_to_openai_format(messages)  # type: ignore

            start_time = time.time()

            # Use instructor for structured response
            _resp = client.chat.completions.create(
                model=self.model,  # instructor handles the provider internally
                response_model=task_class,
                messages=openai_messages,
            )
            response = cast(BaseModel, _resp)  # <-- tell Pyright it's a Pydantic model

            # For instructor, response is the parsed Pydantic object directly
            # We need to simulate LLMUsage since instructor might not return completion details
            # TODO: Check if instructor provides usage information
            usage = LLMUsage(
                prompt_tokens=0,  # instructor may not provide this
                completion_tokens=0,
                total_tokens=0,
            )

            new_message = Message(
                role=RoleEnum.ASSISTANT,
                content=response.model_dump_json(),
            )
            new_message_list = messages + [new_message]

            llm_response = LLMResponse(
                llm_client=self.llm_client,
                model=self.model,
                messages=new_message_list,
                usage=usage,
            )

            call_duration = time.time() - start_time

            # Extract outcomes from Pydantic object (convert to JSON first)
            outcomes, missing_tasks = self._extract_outcomes_from_json(  # type: ignore
                response.model_dump_json()
            )

            self._assign_outcomes(  # type: ignore
                outcomes=outcomes,
                missing_tasks=missing_tasks,
                llm_response=llm_response,
                call_duration=call_duration,
                row=row,
                eval_data=eval_data,
                sample_example_id=sample_example_id,
                builder=builder,
            )

        except Exception as unexpected_error:
            if fallback_enabled:
                raise unexpected_error

            # Any unexpected error (LLM API error, instructor error, etc.)
            builder.create_other_error_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                error=unexpected_error,
            )

    def _evaluate_row_xml(
        self,
        row: dict[str, Any],
        eval_data: EvalData,
        sample_example_id: str,
        tag_configs: list[TagConfig],
        builder: JudgeResultsBuilder,
        fallback_enabled: bool = False,
    ) -> None:
        """Evaluate a single row using XML tags method.

        Args:
            row: Row data from EvalData
            eval_data: The EvalData instance
            sample_example_id: Unique identifier for this sample
            llm_client: The LLMClientEnum instance
            model: The model to use for the LLM
            tag_configs: List of TagConfigs for parsing XML response (one per task)
            builder: The JudgeResultsBuilder instance
            fallback_enabled: Whether fallback is enabled
        """
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        try:
            # Create system message with template substitution and XML instructions
            system_message = self._create_system_message(  # type: ignore
                row=row, include_xml_instructions=True
            )  # type: ignore
            messages = [system_message]

            start_time = time.time()

            # Call LLM with XML tags using litellm
            full_model_name = f"{self.llm_client}/{self.model}"

            # Convert messages to OpenAI format
            openai_messages = self._convert_messages_to_openai_format(messages)  # type: ignore

            response = cast(
                ModelResponse,
                completion(
                    model=full_model_name,
                    messages=openai_messages,
                ),
            )

            raw_response = response.choices[0].message.content  # type: ignore
            self.logger.info(f"XML response: {raw_response}")

            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,  # type: ignore
                completion_tokens=response.usage.completion_tokens,  # type: ignore
                total_tokens=response.usage.total_tokens,  # type: ignore
            )

            new_message = Message(
                role=RoleEnum.ASSISTANT,
                content=raw_response or "",
            )
            new_message_list = messages + [new_message]

            llm_response = LLMResponse(
                llm_client=self.llm_client,
                model=self.model,
                messages=new_message_list,
                usage=usage,
            )

            # Parse XML tags from the response
            parse_result = self._construct_xml_tag_parsing(tag_configs, raw_response)  # type: ignore
            call_duration = time.time() - start_time

            # Extract outcomes from XML parse result
            outcomes, missing_tasks = self._extract_outcomes_from_parse_result(  # type: ignore
                parse_result
            )

            self._assign_outcomes(  # type: ignore
                outcomes=outcomes,
                missing_tasks=missing_tasks,
                llm_response=llm_response,
                call_duration=call_duration,
                row=row,
                eval_data=eval_data,
                sample_example_id=sample_example_id,
                builder=builder,
            )

        except LLMAPIError as llm_error:
            if fallback_enabled:
                raise llm_error

            # LLM API error
            builder.create_llm_error_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                error=llm_error,
            )

        except Exception as unexpected_error:
            if fallback_enabled:
                raise unexpected_error

            # Any other unexpected error (e.g., message creation, network issues)
            builder.create_other_error_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                error=unexpected_error,
            )

    def evaluate_eval_data(
        self,
        eval_data: EvalData,
        run_id: str,
    ) -> JudgeResults:
        """Evaluate the input EvalData using the configured Judge.

        Args:
            eval_data (EvalData): The EvalData instance to be evaluated.
            run_id (str): A unique identifier for the evaluation run.

        Returns:
            JudgeResults: A JudgeResults instance containing the results of the evaluation.

        Raises:
            UnsupportedFormatMethodError: If the answering method is not supported.
        """
        self.logger.info(
            f"Evaluating {eval_data.name} with {self.llm_client} client and model {self.model}"
        )

        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        # No client validation needed since we use litellm internally

        # Create builder
        expected_ids = eval_data.data[eval_data.id_column].to_list()
        builder = JudgeResultsBuilder(
            run_id=run_id,
            judge_id=self.id,
            llm_client=self.llm_client,
            model_used=self.model,
            task_schemas=self.eval_task.task_schemas,
            expected_ids=expected_ids,
            required_tasks=self.eval_task.get_required_tasks(),
            is_sampled_run=isinstance(eval_data, SampleEvalData),
        )

        # Pre-create models/configs once to avoid recreating in loop
        task_class: Optional[type[BaseModel]] = None
        tag_configs: Optional[list[TagConfig]] = None

        task_class = self.eval_task.create_task_class()
        # If fallback is enabled, always create both task_class and tag_configs
        if (
            self.eval_task.structured_outputs_fallback
            or self.eval_task.answering_method == "xml"
        ):
            tag_configs = []
            for task_name, outcomes in self.eval_task.task_schemas.items():
                tag_configs.append(
                    TagConfig(
                        name=task_name,
                        allowed_values=outcomes,
                        cardinality="one",
                    )
                )

        if self.eval_task.structured_outputs_fallback:
            self.logger.info(
                f"Evaluating {eval_data.name} with structured outputs fallback enabled"
            )
        else:
            self.logger.info(
                f"Evaluating {eval_data.name} with structured outputs fallback disabled"
            )
        # Process each row in the evaluation data
        total_count = 0
        for row in self._get_dicts_as_generator(eval_data):  # type: ignore
            total_count += 1
            sample_example_id = f"{run_id}_{total_count}"

            try:
                # Check if this row should be skipped
                if self.eval_task.skip_function and self.eval_task.skip_function(row):
                    builder.create_skipped_row(
                        sample_example_id=sample_example_id,
                        original_id=row[eval_data.id_column],
                    )
                    continue

                # Evaluate the row based on answering method and fallback settings
                if self.eval_task.structured_outputs_fallback:
                    # Use fallback evaluation method
                    assert task_class is not None, (
                        "task_class should be set for fallback mode"
                    )
                    assert tag_configs is not None, (
                        "tag_configs should be set for fallback mode"
                    )
                    self._evaluate_row_with_fallback(
                        row=row,
                        eval_data=eval_data,
                        sample_example_id=sample_example_id,
                        task_class=task_class,
                        tag_configs=tag_configs,
                        builder=builder,
                    )
                else:
                    # Original evaluation logic (no fallback)
                    if self.eval_task.answering_method == "structured":
                        assert task_class is not None, (
                            "task_class was to be set previously"
                        )
                        self._evaluate_row_structured(
                            row=row,
                            eval_data=eval_data,
                            sample_example_id=sample_example_id,
                            task_class=task_class,
                            builder=builder,
                        )
                    elif self.eval_task.answering_method == "instructor":
                        assert task_class is not None, (
                            "task_class was to be set previously"
                        )
                        # Create system message with template substitution
                        system_message = self._create_system_message(  # type: ignore
                            row=row, include_xml_instructions=False
                        )
                        messages = [system_message]

                        self._evaluate_row_instructor(
                            row=row,
                            eval_data=eval_data,
                            sample_example_id=sample_example_id,
                            task_class=task_class,
                            builder=builder,
                            messages=messages,
                        )
                    elif self.eval_task.answering_method == "xml":
                        assert tag_configs is not None, (
                            "tag_configs was to be set previously"
                        )
                        self._evaluate_row_xml(
                            row=row,
                            eval_data=eval_data,
                            sample_example_id=sample_example_id,
                            tag_configs=tag_configs,
                            builder=builder,
                        )
                    else:
                        # Unknown answering method
                        full_model_name = f"{self.llm_client}/{self.model}"
                        raise UnsupportedFormatMethodError(
                            method=self.eval_task.answering_method,
                            model=full_model_name,
                            suggested_methods=["structured", "instructor", "xml"],
                        )

            except Exception as unexpected_error:
                # Handle errors that occur outside of the evaluation methods
                # (e.g., skip function errors, row processing errors)
                builder.create_other_error_row(
                    sample_example_id=sample_example_id,
                    original_id=row.get(
                        eval_data.id_column, "unknown"
                    ),  # Use .get() in case row access fails
                    error=unexpected_error,
                )

        # Complete the builder and return JudgeResults
        return builder.complete()
