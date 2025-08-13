"""Main class of judge module."""

import re
import time
from collections.abc import Generator
from typing import Any, Optional, cast

from pydantic import BaseModel, ConfigDict, model_validator

from ..common.models import Prompt
from ..data import EvalData, SampleEvalData
from ..eval_task import EvalTask
from ..llm_client import AsyncLLMClient, LLMClient
from ..llm_client.enums import AsyncLLMClientEnum, LLMClientEnum, RoleEnum
from ..llm_client.exceptions import LLMAPIError
from ..llm_client.models import Message, TagConfig
from ..results import JudgeResults, JudgeResultsBuilder
from .exceptions import IncorrectClientError


class Judge(BaseModel):
    """Represents a specific configuration for executing an evaluation task using an LLM.

    This class bundles all necessary parameters to define how a single evaluation
    process should be performed for a given task. It encapsulates the target evaluation
    criteria (what to evaluate), the AI model to use (which LLM provider and model),
    and the instruction or prompt that guides the LLM's evaluation process. Each Judge
    instance represents a unique setup for one evaluation run and is identified by a
    stable ID, which is critical for reproducibility and tracking results across runs.

    Attributes:
        id (str): A unique, stable identifier for this specific Judge configuration.
            This ID is used to reference this exact setup (task, model, prompt)
            in configurations, logs, and results. It must contain only alphanumeric
            characters and underscores to ensure compatibility with file paths
            and other system identifiers. This ID must be explicitly provided
            and is never auto-generated.
        eval_task (EvalTask): An instance of the EvalTask class
            defining the criteria and desired outcomes for the evaluation. This
            specifies *what* is being evaluated (e.g., toxicity, relevance) and
            the possible labels or scores the Judge is expected to produce.
        llm_client (LLMClientEnum): An enumeration value specifying the LLM provider
            to be used for this evaluation (e.g., OpenAI, Anthropic). This indicates
            which underlying client implementation should be selected by the
            MetaEvaluator.
        model (str): The specific name of the LLM model to be used from the
            selected provider (e.g., "gpt-4", "claude-3-opus-20240229"). This
            model will receive the prompt and perform the evaluation.
        prompt (Prompt): A Prompt object containing the instructions, few-shot examples,
            and structured output requirements (like XML tags or Pydantic models)
            that will be sent to the LLM. This dictates *how* the LLM should perform
            the evaluation based on the input data.

    model_config (ConfigDict): Pydantic configuration dictionary.
        - `frozen=True`: Makes the Judge instance immutable after creation,
          ensuring its configuration remains constant throughout its lifecycle.

    Validation:
        - The `id` attribute is validated to ensure it contains only alphanumeric
          characters and underscores, making it safe and consistent for use
          in various system contexts.
    """

    id: str
    model_config = ConfigDict(frozen=True)
    eval_task: EvalTask
    llm_client_enum: LLMClientEnum | AsyncLLMClientEnum
    model: str
    prompt: Prompt

    @model_validator(mode="after")
    def validate_id(self) -> "Judge":
        """Validate the id of the Judge.

        The id must only contain alphanumeric characters and underscores.

        Raises:
            ValueError: if the id contains invalid characters

        Returns:
            Judge: The instance of Judge with a valid id.
        """
        if not re.fullmatch(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self.id):
            raise ValueError(
                "id must only contain alphanumeric characters and underscores"
            )

        return self

    def _get_xml_instructions(self) -> str:
        """Get XML formatting instructions for the prompt.

        Returns:
            str: XML formatting instructions.
        """
        instructions = "\n\nPlease provide your evaluation results in XML format using the following tags:\n"

        for task_name, outcomes in self.eval_task.task_schemas.items():
            instructions += (
                f"<{task_name}>YOUR_ANSWER_FOR_{task_name.upper()}</{task_name}>\n"
            )
            if outcomes is None:
                instructions += (
                    f"For {task_name}, provide a free form text response.\n\n"
                )
            else:
                instructions += (
                    f"Valid values for {task_name} are: {', '.join(outcomes)}\n\n"
                )

        instructions += "For tasks with predefined values, you must choose exactly one value. For free form tasks, provide your response within the appropriate tags."
        return instructions

    def _create_system_message(self, include_xml_instructions: bool = False) -> Message:
        """Create the system message with evaluation instructions.

        Args:
            include_xml_instructions: Whether to include XML formatting instructions.

        Returns:
            Message: System message with evaluation context and instructions.
        """
        system_content = self.prompt.prompt
        if include_xml_instructions:
            system_content += self._get_xml_instructions()
        return Message(role=RoleEnum.SYSTEM, content=system_content)

    def _format_row_data(self, row: dict[str, Any]) -> str:
        """Format row data for the user message.

        Args:
            row: Dictionary containing the row data from EvalData.

        Returns:
            str: Formatted string with input and output data.
        """
        content = ""
        if self.eval_task.prompt_columns:
            content += f"The prompts to be evaluated are {', '.join(self.eval_task.prompt_columns)}."
            for column in self.eval_task.prompt_columns:
                content += f"\n{column}: {row[column]}"

        if self.eval_task.response_columns:
            prompt_prefix = (
                "The responses to be evaluated are"
                if self.eval_task.prompt_columns
                else "The texts to be evaluated are"
            )
            content += (
                f"\n\n{prompt_prefix} {', '.join(self.eval_task.response_columns)}."
            )
            for column in self.eval_task.response_columns:
                content += f"\n{column}: {row[column]}"

        return content

    def _get_dicts_as_generator(
        self, eval_data: EvalData
    ) -> Generator[dict[str, Any], None, None]:
        """Generate dictionaries from EvalData rows.

        Args:
            eval_data: The EvalData instance to iterate over.

        Yields:
            dict[str, Any]: Row data as dictionaries.
        """
        for row in eval_data.data.iter_rows(named=True):
            yield row

    def _evaluate_row_structured(
        self,
        row: dict[str, Any],
        eval_data: EvalData,
        sample_example_id: str,
        llm_client: LLMClient,
        task_class: type[BaseModel],
        builder: JudgeResultsBuilder,
    ) -> None:
        """Evaluate a single row using structured response method.

        Args:
            row: Row data from EvalData
            eval_data: The EvalData instance
            sample_example_id: Unique identifier for this sample
            llm_client: The LLMClient instance
            task_class: The dynamically created Pydantic model for responses
            builder: The JudgeResultsBuilder instance
        """
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        try:
            # Create messages
            system_message = self._create_system_message(include_xml_instructions=False)
            user_content = self._format_row_data(row)
            user_message = Message(role=RoleEnum.USER, content=user_content)
            messages = [system_message, user_message]

            # Call LLM with structured response
            start_time = time.time()
            structured_response, llm_response = (
                llm_client.prompt_with_structured_response(
                    messages=messages, response_model=task_class, model=self.model
                )
            )
            call_duration = time.time() - start_time

            # Extract outcomes from structured response for all tasks
            try:
                outcomes = {}
                missing_tasks = []

                for task_name in self.eval_task.task_schemas.keys():
                    try:
                        outcome = getattr(structured_response, task_name)
                        outcomes[task_name] = outcome
                    except AttributeError:
                        missing_tasks.append(task_name)

                if missing_tasks:
                    # Partial success - some tasks missing
                    if outcomes:  # At least some tasks succeeded
                        error_message = (
                            f"Structured response missing tasks: {missing_tasks}"
                        )
                        builder.create_partial_row(
                            sample_example_id=sample_example_id,
                            original_id=row[eval_data.id_column],
                            outcomes=outcomes,
                            error_message=error_message,
                            llm_raw_response_content=llm_response.content,
                            llm_prompt_tokens=llm_response.usage.prompt_tokens,
                            llm_completion_tokens=llm_response.usage.completion_tokens,
                            llm_total_tokens=llm_response.usage.total_tokens,
                            llm_call_duration_seconds=call_duration,
                        )
                    else:
                        # Complete failure - no tasks found
                        builder.create_parsing_error_row(
                            sample_example_id=sample_example_id,
                            original_id=row[eval_data.id_column],
                            error=AttributeError(
                                f"Structured response missing all tasks: {missing_tasks}"
                            ),
                            llm_raw_response_content=llm_response.content,
                            llm_prompt_tokens=llm_response.usage.prompt_tokens,
                            llm_completion_tokens=llm_response.usage.completion_tokens,
                            llm_total_tokens=llm_response.usage.total_tokens,
                            llm_call_duration_seconds=call_duration,
                        )
                else:
                    # Perfect success - all tasks found
                    builder.create_success_row(
                        sample_example_id=sample_example_id,
                        original_id=row[eval_data.id_column],
                        outcomes=outcomes,
                        llm_raw_response_content=llm_response.content,
                        llm_prompt_tokens=llm_response.usage.prompt_tokens,
                        llm_completion_tokens=llm_response.usage.completion_tokens,
                        llm_total_tokens=llm_response.usage.total_tokens,
                        llm_call_duration_seconds=call_duration,
                    )
                return

            except Exception as attr_error:
                # This is a parsing error - structured response doesn't have expected attributes
                builder.create_parsing_error_row(
                    sample_example_id=sample_example_id,
                    original_id=row[eval_data.id_column],
                    error=attr_error,
                    llm_raw_response_content=llm_response.content,
                    llm_prompt_tokens=llm_response.usage.prompt_tokens,
                    llm_completion_tokens=llm_response.usage.completion_tokens,
                    llm_total_tokens=llm_response.usage.total_tokens,
                    llm_call_duration_seconds=call_duration,
                )
                return

        except LLMAPIError as llm_error:
            # LLM API error
            builder.create_llm_error_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                error=llm_error,
            )

        except Exception as unexpected_error:
            # Any other unexpected error (e.g., message creation, network issues)
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
        llm_client: LLMClient,
        tag_configs: list[TagConfig],
        builder: JudgeResultsBuilder,
    ) -> None:
        """Evaluate a single row using XML tags method.

        Args:
            row: Row data from EvalData
            eval_data: The EvalData instance
            sample_example_id: Unique identifier for this sample
            llm_client: The LLMClient instance
            tag_configs: List of TagConfigs for parsing XML response (one per task)
            builder: The JudgeResultsBuilder instance
        """
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        try:
            # Create messages with XML instructions
            system_message = self._create_system_message(include_xml_instructions=True)
            user_content = self._format_row_data(row)
            user_message = Message(role=RoleEnum.USER, content=user_content)
            messages = [system_message, user_message]

            # Call LLM with XML tags
            start_time = time.time()
            parse_result, llm_response = llm_client.prompt_with_xml_tags(
                messages=messages, tag_configs=tag_configs, model=self.model
            )
            call_duration = time.time() - start_time

            # Check parsing results - we need all tasks to succeed for a success row
            task_names = set(self.eval_task.task_schemas.keys())
            parsed_tasks = set(parse_result.data.keys())

            if parse_result.success and parsed_tasks == task_names:
                # Perfect success - all tasks parsed successfully
                outcomes = {}
                for task_name in task_names:
                    outcome = parse_result.data[task_name]
                    # TagConfig cardinality="one" guarantees str, not list[str]
                    outcomes[task_name] = cast(str, outcome)

                builder.create_success_row(
                    sample_example_id=sample_example_id,
                    original_id=row[eval_data.id_column],
                    outcomes=outcomes,
                    llm_raw_response_content=llm_response.content,
                    llm_prompt_tokens=llm_response.usage.prompt_tokens,
                    llm_completion_tokens=llm_response.usage.completion_tokens,
                    llm_total_tokens=llm_response.usage.total_tokens,
                    llm_call_duration_seconds=call_duration,
                )
            elif (
                parse_result.partial_success
                and parsed_tasks.issubset(task_names)
                and len(parsed_tasks) > 0
            ):
                # Partial success - some tasks parsed successfully
                outcomes = {}
                for task_name in parsed_tasks:
                    outcome = parse_result.data[task_name]
                    outcomes[task_name] = cast(str, outcome)

                error_message = f"XML parsing partial success. Missing tasks: {task_names - parsed_tasks}. Errors: {[str(e) for e in parse_result.errors]}"
                builder.create_partial_row(
                    sample_example_id=sample_example_id,
                    original_id=row[eval_data.id_column],
                    outcomes=outcomes,
                    error_message=error_message,
                    llm_raw_response_content=llm_response.content,
                    llm_prompt_tokens=llm_response.usage.prompt_tokens,
                    llm_completion_tokens=llm_response.usage.completion_tokens,
                    llm_total_tokens=llm_response.usage.total_tokens,
                    llm_call_duration_seconds=call_duration,
                )
            else:
                # Complete parsing failure
                error_message = f"XML parsing failed completely: {[str(e) for e in parse_result.errors]}"
                builder.create_parsing_error_row(
                    sample_example_id=sample_example_id,
                    original_id=row[eval_data.id_column],
                    error=ValueError(error_message),
                    llm_raw_response_content=llm_response.content,
                    llm_prompt_tokens=llm_response.usage.prompt_tokens,
                    llm_completion_tokens=llm_response.usage.completion_tokens,
                    llm_total_tokens=llm_response.usage.total_tokens,
                    llm_call_duration_seconds=call_duration,
                )

        except LLMAPIError as llm_error:
            # LLM API error
            builder.create_llm_error_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                error=llm_error,
            )

        except Exception as unexpected_error:
            # Any other unexpected error (e.g., message creation, network issues)
            builder.create_other_error_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                error=unexpected_error,
            )

    def evaluate_eval_data(
        self, eval_data: EvalData, llm_client: LLMClient, run_id: str
    ) -> JudgeResults:
        """Evaluate the input EvalData using the configured Judge.

        Args:
            eval_data (EvalData): The EvalData instance to be evaluated.
            llm_client (LLMClient): The LLMClient instance to be used for evaluation.
            run_id (str): A unique identifier for the evaluation run.

        Returns:
            JudgeResults: A JudgeResults instance containing the results of the evaluation.

        Raises:
            IncorrectClientError: If the LLMClient is not equal to the configured LLMClient.
        """
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        # Validate LLM client matches configuration
        if llm_client.enum_value != self.llm_client_enum:
            raise IncorrectClientError(
                expected_client=self.llm_client_enum,
                actual_client=llm_client.enum_value,
            )

        # Create builder
        expected_ids = eval_data.data[eval_data.id_column].to_list()
        builder = JudgeResultsBuilder(
            run_id=run_id,
            judge_id=self.id,
            llm_client_enum=self.llm_client_enum,
            model_used=self.model,
            task_schemas=self.eval_task.task_schemas,
            expected_ids=expected_ids,
            is_sampled_run=isinstance(eval_data, SampleEvalData),
        )

        # Pre-create models/configs once to avoid recreating in loop
        task_class: Optional[type[BaseModel]] = None
        tag_configs: Optional[list[TagConfig]] = None

        if self.eval_task.answering_method == "structured":
            task_class = self.eval_task.create_task_class()
        elif self.eval_task.answering_method == "xml":
            tag_configs = []
            for task_name, outcomes in self.eval_task.task_schemas.items():
                tag_configs.append(
                    TagConfig(
                        name=task_name,
                        allowed_values=outcomes,
                        cardinality="one",
                    )
                )

        # Process each row in the evaluation data
        total_count = 0
        for row in self._get_dicts_as_generator(eval_data):
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

                # Evaluate the row based on answering method
                if self.eval_task.answering_method == "structured":
                    assert task_class is not None, "task_class was to be set previously"
                    self._evaluate_row_structured(
                        row=row,
                        eval_data=eval_data,
                        sample_example_id=sample_example_id,
                        llm_client=llm_client,
                        task_class=task_class,
                        builder=builder,
                    )
                else:  # xml
                    assert tag_configs is not None, (
                        "tag_configs was to be set previously"
                    )
                    self._evaluate_row_xml(
                        row=row,
                        eval_data=eval_data,
                        sample_example_id=sample_example_id,
                        llm_client=llm_client,
                        tag_configs=tag_configs,
                        builder=builder,
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

    #################
    # ASYNC METHODS #
    #################

    def _handle_batch_exception(
        self,
        exception: Exception,
        sample_example_id: str,
        row: dict[str, Any],
        eval_data: EvalData,
        builder: JudgeResultsBuilder,
    ) -> None:
        """Handle exceptions from batch processing."""
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        if isinstance(exception, LLMAPIError):
            builder.create_llm_error_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                error=exception,
            )
        else:
            builder.create_other_error_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                error=exception,
            )

    async def _process_batch_results_structured(
        self,
        results: list,
        rows_to_evaluate: list[tuple[dict[str, Any], str]],
        eval_data: EvalData,
        builder: JudgeResultsBuilder,
    ) -> None:
        """Process structured response batch results."""
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        start_time = time.time()
        total_duration = time.time() - start_time

        for i, result in enumerate(results):
            row, sample_example_id = rows_to_evaluate[i]

            # Handle exceptions from batch processing (return_exceptions=True can return exceptions directly)
            if isinstance(result, Exception):
                self._handle_batch_exception(
                    exception=result,
                    sample_example_id=sample_example_id,
                    row=row,
                    eval_data=eval_data,
                    builder=builder,
                )
                continue

            # Unpack successful result
            structured_response, llm_response = result

            call_duration = total_duration / len(results) if results else 0

            # Extract outcomes from structured response
            try:
                outcomes = {}
                missing_tasks = []

                for task_name in self.eval_task.task_schemas.keys():
                    try:
                        outcome = getattr(structured_response, task_name)
                        outcomes[task_name] = outcome
                    except AttributeError:
                        missing_tasks.append(task_name)

                if missing_tasks:
                    # Partial success - some tasks missing
                    if outcomes:  # At least some tasks succeeded
                        error_message = (
                            f"Structured response missing tasks: {missing_tasks}"
                        )
                        builder.create_partial_row(
                            sample_example_id=sample_example_id,
                            original_id=row[eval_data.id_column],
                            outcomes=outcomes,
                            error_message=error_message,
                            llm_raw_response_content=llm_response.content,
                            llm_prompt_tokens=llm_response.usage.prompt_tokens,
                            llm_completion_tokens=llm_response.usage.completion_tokens,
                            llm_total_tokens=llm_response.usage.total_tokens,
                            llm_call_duration_seconds=call_duration,
                        )
                    else:
                        # Complete failure - no tasks found
                        builder.create_parsing_error_row(
                            sample_example_id=sample_example_id,
                            original_id=row[eval_data.id_column],
                            error=AttributeError(
                                f"Structured response missing all tasks: {missing_tasks}"
                            ),
                            llm_raw_response_content=llm_response.content,
                            llm_prompt_tokens=llm_response.usage.prompt_tokens,
                            llm_completion_tokens=llm_response.usage.completion_tokens,
                            llm_total_tokens=llm_response.usage.total_tokens,
                            llm_call_duration_seconds=call_duration,
                        )
                else:
                    # Perfect success - all tasks found
                    builder.create_success_row(
                        sample_example_id=sample_example_id,
                        original_id=row[eval_data.id_column],
                        outcomes=outcomes,
                        llm_raw_response_content=llm_response.content,
                        llm_prompt_tokens=llm_response.usage.prompt_tokens,
                        llm_completion_tokens=llm_response.usage.completion_tokens,
                        llm_total_tokens=llm_response.usage.total_tokens,
                        llm_call_duration_seconds=call_duration,
                    )

            except Exception as attr_error:
                builder.create_parsing_error_row(
                    sample_example_id=sample_example_id,
                    original_id=row[eval_data.id_column],
                    error=attr_error,
                    llm_raw_response_content=llm_response.content,
                    llm_prompt_tokens=llm_response.usage.prompt_tokens,
                    llm_completion_tokens=llm_response.usage.completion_tokens,
                    llm_total_tokens=llm_response.usage.total_tokens,
                    llm_call_duration_seconds=call_duration,
                )

    async def _process_batch_results_xml(
        self,
        results: list,
        rows_to_evaluate: list[tuple[dict[str, Any], str]],
        eval_data: EvalData,
        builder: JudgeResultsBuilder,
    ) -> None:
        """Process XML tag parsing batch results."""
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        start_time = time.time()
        total_duration = time.time() - start_time

        for i, result in enumerate(results):
            row, sample_example_id = rows_to_evaluate[i]

            # Handle exceptions from batch processing (return_exceptions=True can return exceptions directly)
            if isinstance(result, Exception):
                self._handle_batch_exception(
                    exception=result,
                    sample_example_id=sample_example_id,
                    row=row,
                    eval_data=eval_data,
                    builder=builder,
                )
                continue

            # Unpack successful result
            parse_result, llm_response = result

            call_duration = total_duration / len(results) if results else 0

            # Process XML parsing results
            task_names = set(self.eval_task.task_schemas.keys())
            parsed_tasks = set(parse_result.data.keys())

            if parse_result.success and parsed_tasks == task_names:
                # Perfect success - all tasks parsed successfully
                outcomes = {}
                for task_name in task_names:
                    outcome = parse_result.data[task_name]
                    outcomes[task_name] = cast(str, outcome)

                builder.create_success_row(
                    sample_example_id=sample_example_id,
                    original_id=row[eval_data.id_column],
                    outcomes=outcomes,
                    llm_raw_response_content=llm_response.content,
                    llm_prompt_tokens=llm_response.usage.prompt_tokens,
                    llm_completion_tokens=llm_response.usage.completion_tokens,
                    llm_total_tokens=llm_response.usage.total_tokens,
                    llm_call_duration_seconds=call_duration,
                )
            elif (
                parse_result.partial_success
                and parsed_tasks.issubset(task_names)
                and len(parsed_tasks) > 0
            ):
                # Partial success - some tasks parsed successfully
                outcomes = {}
                for task_name in parsed_tasks:
                    outcome = parse_result.data[task_name]
                    outcomes[task_name] = cast(str, outcome)

                error_message = f"XML parsing partial success. Missing tasks: {task_names - parsed_tasks}. Errors: {[str(e) for e in parse_result.errors]}"
                builder.create_partial_row(
                    sample_example_id=sample_example_id,
                    original_id=row[eval_data.id_column],
                    outcomes=outcomes,
                    error_message=error_message,
                    llm_raw_response_content=llm_response.content,
                    llm_prompt_tokens=llm_response.usage.prompt_tokens,
                    llm_completion_tokens=llm_response.usage.completion_tokens,
                    llm_total_tokens=llm_response.usage.total_tokens,
                    llm_call_duration_seconds=call_duration,
                )
            else:
                # Complete parsing failure
                error_message = f"XML parsing failed completely: {[str(e) for e in parse_result.errors]}"
                builder.create_parsing_error_row(
                    sample_example_id=sample_example_id,
                    original_id=row[eval_data.id_column],
                    error=ValueError(error_message),
                    llm_raw_response_content=llm_response.content,
                    llm_prompt_tokens=llm_response.usage.prompt_tokens,
                    llm_completion_tokens=llm_response.usage.completion_tokens,
                    llm_total_tokens=llm_response.usage.total_tokens,
                    llm_call_duration_seconds=call_duration,
                )

    async def _evaluate_rows_batch_structured(
        self,
        rows_to_evaluate: list[tuple[dict[str, Any], str]],
        eval_data: EvalData,
        llm_client: AsyncLLMClient,
        builder: JudgeResultsBuilder,
        batch_size: int,
        max_concurrency: int,
    ) -> None:
        """Evaluate multiple rows using structured response method with batching."""
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        # Create the task class once for all requests
        task_class = self.eval_task.create_task_class()

        # Prepare batch items: (messages, response_model)
        batch_items = []
        for row, sample_example_id in rows_to_evaluate:
            system_message = self._create_system_message(include_xml_instructions=False)
            user_content = self._format_row_data(row)
            user_message = Message(role=RoleEnum.USER, content=user_content)
            messages = [system_message, user_message]
            batch_items.append((messages, task_class))

        # Execute batch and process results. Client will handle batching and concurrency.
        results = await llm_client.prompt_with_structured_response_batch(
            items=batch_items,
            model=self.model,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )

        # Process all results. Handle exceptions, extract outcomes, and create result rows.
        await self._process_batch_results_structured(
            results=results,
            rows_to_evaluate=rows_to_evaluate,
            eval_data=eval_data,
            builder=builder,
        )

    async def _evaluate_rows_batch_xml(
        self,
        rows_to_evaluate: list[tuple[dict[str, Any], str]],
        eval_data: EvalData,
        llm_client: AsyncLLMClient,
        builder: JudgeResultsBuilder,
        batch_size: int,
        max_concurrency: int,
    ) -> None:
        """Evaluate multiple rows using XML tags method with batching."""
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        # Create tag configs once for all requests
        tag_configs = []
        for task_name, outcomes in self.eval_task.task_schemas.items():
            tag_configs.append(
                TagConfig(
                    name=task_name,
                    allowed_values=outcomes,
                    cardinality="one",
                )
            )

        # Prepare batch items: (messages, tag_configs)
        batch_items = []
        for row, sample_example_id in rows_to_evaluate:
            system_message = self._create_system_message(include_xml_instructions=True)
            user_content = self._format_row_data(row)
            user_message = Message(role=RoleEnum.USER, content=user_content)
            messages = [system_message, user_message]
            batch_items.append((messages, tag_configs))

        # Execute batch and process results. Client will handle batching and concurrency.
        results = await llm_client.prompt_with_xml_tags_batch(
            items=batch_items,
            model=self.model,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )

        # Process all results. Handle exceptions, extract outcomes, and create result rows.
        await self._process_batch_results_xml(
            results=results,
            rows_to_evaluate=rows_to_evaluate,
            eval_data=eval_data,
            builder=builder,
        )

    async def evaluate_eval_data_async(
        self,
        eval_data: EvalData,
        llm_client: AsyncLLMClient,
        run_id: str,
        batch_size: int = 10,
        max_concurrency: int = 5,
    ) -> JudgeResults:
        """Evaluate the input EvalData using the configured Judge with async batching.

        This method provides the same functionality as evaluate_eval_data but uses
        async batching for better performance when processing large datasets.

        Args:
            eval_data (EvalData): The EvalData instance to be evaluated.
            llm_client (AsyncLLMClient): The AsyncLLMClient instance to be used for evaluation.
            run_id (str): A unique identifier for the evaluation run.
            batch_size (int): Number of requests to process in each batch. Defaults to 10.
            max_concurrency (int): Maximum number of concurrent requests. Defaults to 5.

        Returns:
            JudgeResults: A JudgeResults instance containing the results of the evaluation.

        Raises:
            IncorrectClientError: If the LLMClient is not equal to the configured LLMClient.
        """
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        # Validate LLM client matches configuration
        if llm_client.enum_value != self.llm_client_enum:
            raise IncorrectClientError(
                expected_client=self.llm_client_enum,
                actual_client=llm_client.enum_value,
            )

        # Create builder
        expected_ids = eval_data.data[eval_data.id_column].to_list()
        builder = JudgeResultsBuilder(
            run_id=run_id,
            judge_id=self.id,
            llm_client_enum=self.llm_client_enum,
            model_used=self.model,
            task_schemas=self.eval_task.task_schemas,
            expected_ids=expected_ids,
            is_sampled_run=isinstance(eval_data, SampleEvalData),
        )

        # Collect all rows that need evaluation
        rows_to_evaluate = []
        total_count = 0
        for row in self._get_dicts_as_generator(eval_data):
            total_count += 1
            sample_example_id = f"{run_id}_{total_count}"

            # Check if this row should be skipped
            if self.eval_task.skip_function and self.eval_task.skip_function(row):
                builder.create_skipped_row(
                    sample_example_id=sample_example_id,
                    original_id=row[eval_data.id_column],
                )
                continue

            rows_to_evaluate.append((row, sample_example_id))

        if not rows_to_evaluate:
            # All rows were skipped, return the completed builder
            return builder.complete()

        # Process based on answering method
        if self.eval_task.answering_method == "structured":
            await self._evaluate_rows_batch_structured(
                rows_to_evaluate=rows_to_evaluate,
                eval_data=eval_data,
                llm_client=llm_client,
                builder=builder,
                batch_size=batch_size,
                max_concurrency=max_concurrency,
            )
        else:  # xml
            await self._evaluate_rows_batch_xml(
                rows_to_evaluate=rows_to_evaluate,
                eval_data=eval_data,
                llm_client=llm_client,
                builder=builder,
                batch_size=batch_size,
                max_concurrency=max_concurrency,
            )

        # Complete the builder and return JudgeResults
        return builder.complete()
