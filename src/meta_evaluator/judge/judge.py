"""Main class of judge module."""

from typing import Any, Optional, cast
from ..eval_task import EvalTask
from ..llm_client import LLMClientEnum, LLMClient
from ..llm_client.models import Message, RoleEnum, TagConfig
from ..llm_client.exceptions import LLMAPIError
from collections.abc import Generator
from ..common.models import Prompt
from pydantic import BaseModel, ConfigDict, model_validator
import re
import time
from datetime import datetime
from .models import JudgeResults, JudgeResultsConfig, JudgeResultsBuilder
from ..data import EvalData, SampleEvalData
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
    llm_client_enum: LLMClientEnum
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
        if self.eval_task.input_columns:
            content += f"The inputs given to the LLM were {', '.join(self.eval_task.input_columns)}."
            for column in self.eval_task.input_columns:
                content += f"\n{column}: {row[column]}"

        if self.eval_task.output_columns:
            content += f"\n\nThe outputs given by the LLM were {', '.join(self.eval_task.output_columns)}."
            for column in self.eval_task.output_columns:
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
        assert (
            eval_data.id_column is not None
        ), f"EvalData {eval_data.name} has no ID column, but was expected to have one."

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
                            llm_raw_response=llm_response.content,
                            prompt_tokens=llm_response.usage.prompt_tokens,
                            completion_tokens=llm_response.usage.completion_tokens,
                            total_tokens=llm_response.usage.total_tokens,
                            call_duration=call_duration,
                        )
                    else:
                        # Complete failure - no tasks found
                        builder.create_parsing_error_row(
                            sample_example_id=sample_example_id,
                            original_id=row[eval_data.id_column],
                            error=AttributeError(
                                f"Structured response missing all tasks: {missing_tasks}"
                            ),
                            llm_raw_response=llm_response.content,
                            prompt_tokens=llm_response.usage.prompt_tokens,
                            completion_tokens=llm_response.usage.completion_tokens,
                            total_tokens=llm_response.usage.total_tokens,
                            call_duration=call_duration,
                        )
                else:
                    # Perfect success - all tasks found
                    builder.create_success_row(
                        sample_example_id=sample_example_id,
                        original_id=row[eval_data.id_column],
                        outcomes=outcomes,
                        llm_raw_response=llm_response.content,
                        prompt_tokens=llm_response.usage.prompt_tokens,
                        completion_tokens=llm_response.usage.completion_tokens,
                        total_tokens=llm_response.usage.total_tokens,
                        call_duration=call_duration,
                    )
                return

            except Exception as attr_error:
                # This is a parsing error - structured response doesn't have expected attributes
                builder.create_parsing_error_row(
                    sample_example_id=sample_example_id,
                    original_id=row[eval_data.id_column],
                    error=attr_error,
                    llm_raw_response=llm_response.content,
                    prompt_tokens=llm_response.usage.prompt_tokens,
                    completion_tokens=llm_response.usage.completion_tokens,
                    total_tokens=llm_response.usage.total_tokens,
                    call_duration=call_duration,
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
        assert (
            eval_data.id_column is not None
        ), f"EvalData {eval_data.name} has no ID column, but was expected to have one."

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
                    llm_raw_response=llm_response.content,
                    prompt_tokens=llm_response.usage.prompt_tokens,
                    completion_tokens=llm_response.usage.completion_tokens,
                    total_tokens=llm_response.usage.total_tokens,
                    call_duration=call_duration,
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
                    llm_raw_response=llm_response.content,
                    prompt_tokens=llm_response.usage.prompt_tokens,
                    completion_tokens=llm_response.usage.completion_tokens,
                    total_tokens=llm_response.usage.total_tokens,
                    call_duration=call_duration,
                )
            else:
                # Complete parsing failure
                error_message = f"XML parsing failed completely: {[str(e) for e in parse_result.errors]}"
                builder.create_parsing_error_row(
                    sample_example_id=sample_example_id,
                    original_id=row[eval_data.id_column],
                    error=ValueError(error_message),
                    llm_raw_response=llm_response.content,
                    prompt_tokens=llm_response.usage.prompt_tokens,
                    completion_tokens=llm_response.usage.completion_tokens,
                    total_tokens=llm_response.usage.total_tokens,
                    call_duration=call_duration,
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
        assert (
            eval_data.id_column is not None
        ), f"EvalData {eval_data.name} has no ID column, but was expected to have one."

        # Validate LLM client matches configuration
        if llm_client.enum_value != self.llm_client_enum:
            raise IncorrectClientError(
                expected_client=self.llm_client_enum.value,
                actual_client=llm_client.enum_value.value,
            )

        # Create builder configuration and builder
        expected_ids = eval_data.data[eval_data.id_column].to_list()
        config = JudgeResultsConfig(
            run_id=run_id,
            judge_id=self.id,
            task_schemas=self.eval_task.task_schemas,
            llm_client_enum=self.llm_client_enum,
            model_used=self.model,
            timestamp_local=datetime.now(),
            is_sampled_run=isinstance(eval_data, SampleEvalData),
            expected_ids=expected_ids,
        )
        builder = JudgeResultsBuilder(config)

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
                    assert (
                        tag_configs is not None
                    ), "tag_configs was to be set previously"
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
