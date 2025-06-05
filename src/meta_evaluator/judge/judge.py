"""Main class of judge module."""

from typing import Any, cast
from ..evaluation_task import EvaluationTask
from ..llm_client import LLMClientEnum, LLMClient
from ..llm_client.models import Message, RoleEnum, TagConfig
from ..llm_client.exceptions import LLMAPIError
from collections.abc import Generator
from ..common.models import Prompt
from pydantic import BaseModel, ConfigDict, model_validator
import re
import time
from datetime import datetime
import polars as pl
from .models import JudgeResults, EvaluationStatusEnum
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
        evaluation_task (EvaluationTask): An instance of the EvaluationTask class
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
    evaluation_task: EvaluationTask
    llm_client: LLMClientEnum
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
        return f"""

Please provide your evaluation result in XML format using the following tag:
<result_outcome>YOUR_ANSWER</result_outcome>

Valid values for YOUR_ANSWER are: {", ".join(self.evaluation_task.outcomes)}

You must choose exactly one of these values and place it within the result_outcome tags."""

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
        if self.evaluation_task.input_columns:
            content += f"The inputs given to the LLM were {', '.join(self.evaluation_task.input_columns)}."
            for column in self.evaluation_task.input_columns:
                content += f"\n{column}: {row[column]}"

        if self.evaluation_task.output_columns:
            content += f"\n\nThe outputs given by the LLM were {', '.join(self.evaluation_task.output_columns)}."
            for column in self.evaluation_task.output_columns:
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
        run_id: str,
        sample_example_id: str,
        llm_client: LLMClient,
        task_class: type[BaseModel],
    ) -> dict[str, Any]:
        """Evaluate a single row using structured response method.

        Args:
            row: Row data from EvalData
            eval_data: The EvalData instance
            run_id: Unique identifier for the evaluation run
            sample_example_id: Unique identifier for this sample
            llm_client: The LLMClient instance
            task_class: The dynamically created Pydantic model for responses

        Returns:
            dict[str, Any]: Result row dictionary ready for DataFrame
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

            # Extract outcome from structured response
            try:
                outcome = structured_response.result_outcome  # type: ignore
            except AttributeError as attr_error:
                # This is a parsing error - structured response doesn't have expected attribute
                return JudgeResults.create_parsing_error_row(
                    task_name=self.evaluation_task.task_name,
                    sample_example_id=sample_example_id,
                    original_id=row[eval_data.id_column],
                    run_id=run_id,
                    judge_id=self.id,
                    error=attr_error,
                    llm_raw_response=llm_response.content,
                    prompt_tokens=llm_response.usage.prompt_tokens,
                    completion_tokens=llm_response.usage.completion_tokens,
                    total_tokens=llm_response.usage.total_tokens,
                    call_duration=call_duration,
                )

            # Create success row
            return JudgeResults.create_success_row(
                task_name=self.evaluation_task.task_name,
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                run_id=run_id,
                judge_id=self.id,
                outcome=outcome,
                llm_raw_response=llm_response.content,
                prompt_tokens=llm_response.usage.prompt_tokens,
                completion_tokens=llm_response.usage.completion_tokens,
                total_tokens=llm_response.usage.total_tokens,
                call_duration=call_duration,
            )

        except LLMAPIError as llm_error:
            # LLM API error
            return JudgeResults.create_llm_error_row(
                task_name=self.evaluation_task.task_name,
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                run_id=run_id,
                judge_id=self.id,
                error=llm_error,
            )

        except Exception as unexpected_error:
            # Any other unexpected error (e.g., message creation, network issues)
            return JudgeResults.create_other_error_row(
                task_name=self.evaluation_task.task_name,
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                run_id=run_id,
                judge_id=self.id,
                error=unexpected_error,
            )

    def _evaluate_row_xml(
        self,
        row: dict[str, Any],
        eval_data: EvalData,
        run_id: str,
        sample_example_id: str,
        llm_client: LLMClient,
        tag_config: TagConfig,
    ) -> dict[str, Any]:
        """Evaluate a single row using XML tags method.

        Args:
            row: Row data from EvalData
            eval_data: The EvalData instance
            run_id: Unique identifier for the evaluation run
            sample_example_id: Unique identifier for this sample
            llm_client: The LLMClient instance
            tag_config: TagConfig for parsing XML response

        Returns:
            dict[str, Any]: Result row dictionary ready for DataFrame
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
                messages=messages, tag_configs=[tag_config], model=self.model
            )
            call_duration = time.time() - start_time

            # Check if parsing was successful
            if parse_result.success and "result_outcome" in parse_result.data:
                # Success case
                outcome = parse_result.data["result_outcome"]
                new_outcome = cast(
                    str, outcome
                )  # outcome should be a str at this point
                # TagConfig cardinality="one" guarantees str, not list[str]

                return JudgeResults.create_success_row(
                    task_name=self.evaluation_task.task_name,
                    sample_example_id=sample_example_id,
                    original_id=row[eval_data.id_column],
                    run_id=run_id,
                    judge_id=self.id,
                    outcome=new_outcome,
                    llm_raw_response=llm_response.content,
                    prompt_tokens=llm_response.usage.prompt_tokens,
                    completion_tokens=llm_response.usage.completion_tokens,
                    total_tokens=llm_response.usage.total_tokens,
                    call_duration=call_duration,
                )
            else:
                # Parsing failed
                error_message = (
                    f"XML parsing failed: {[str(e) for e in parse_result.errors]}"
                )
                return JudgeResults.create_parsing_error_row(
                    task_name=self.evaluation_task.task_name,
                    sample_example_id=sample_example_id,
                    original_id=row[eval_data.id_column],
                    run_id=run_id,
                    judge_id=self.id,
                    error=ValueError(error_message),
                    llm_raw_response=llm_response.content,
                    prompt_tokens=llm_response.usage.prompt_tokens,
                    completion_tokens=llm_response.usage.completion_tokens,
                    total_tokens=llm_response.usage.total_tokens,
                    call_duration=call_duration,
                )

        except LLMAPIError as llm_error:
            # LLM API error
            return JudgeResults.create_llm_error_row(
                task_name=self.evaluation_task.task_name,
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                run_id=run_id,
                judge_id=self.id,
                error=llm_error,
            )

        except Exception as unexpected_error:
            # Any other unexpected error (e.g., message creation, network issues)
            return JudgeResults.create_other_error_row(
                task_name=self.evaluation_task.task_name,
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                run_id=run_id,
                judge_id=self.id,
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
        if llm_client.enum_value != self.llm_client:
            raise IncorrectClientError(
                expected_client=self.llm_client.value,
                actual_client=llm_client.enum_value.value,
            )

        # Initialize tracking variables
        results = []
        total_count = 0
        skipped_count = 0
        success_count = 0
        llm_error_count = 0
        parsing_error_count = 0
        other_error_count = 0

        # Pre-create models/configs once to avoid recreating in loop
        task_class = None
        tag_config = None

        if self.evaluation_task.answering_method == "structured":
            task_class = self.evaluation_task.create_task_class()
        elif self.evaluation_task.answering_method == "xml":
            tag_config = TagConfig(
                name="result_outcome",
                allowed_values=self.evaluation_task.outcomes,
                cardinality="one",
            )

        # Process each row in the evaluation data
        for row in self._get_dicts_as_generator(eval_data):
            total_count += 1
            sample_example_id = f"{run_id}_{total_count}"

            try:
                # Check if this row should be skipped
                if (
                    self.evaluation_task.skip_function
                    and self.evaluation_task.skip_function(row)
                ):
                    skipped_count += 1
                    result_row = JudgeResults.create_skipped_row(
                        task_name=self.evaluation_task.task_name,
                        sample_example_id=sample_example_id,
                        original_id=row[eval_data.id_column],
                        run_id=run_id,
                        judge_id=self.id,
                    )
                    results.append(result_row)
                    continue

                # Evaluate the row based on answering method
                if self.evaluation_task.answering_method == "structured":
                    assert task_class is not None, "task_class was to be set previously"
                    result_row = self._evaluate_row_structured(
                        row=row,
                        eval_data=eval_data,
                        run_id=run_id,
                        sample_example_id=sample_example_id,
                        llm_client=llm_client,
                        task_class=task_class,
                    )
                else:  # xml
                    assert tag_config is not None, "tag_config was to be set previously"
                    result_row = self._evaluate_row_xml(
                        row=row,
                        eval_data=eval_data,
                        run_id=run_id,
                        sample_example_id=sample_example_id,
                        llm_client=llm_client,
                        tag_config=tag_config,
                    )

                # Update counters based on result status
                status = result_row["status"]
                if status == EvaluationStatusEnum.SUCCESS.value:
                    success_count += 1
                elif status == EvaluationStatusEnum.LLM_ERROR.value:
                    llm_error_count += 1
                elif status == EvaluationStatusEnum.PARSING_ERROR.value:
                    parsing_error_count += 1
                elif status == EvaluationStatusEnum.OTHER_ERROR.value:
                    other_error_count += 1

                results.append(result_row)

            except Exception as unexpected_error:
                # Handle errors that occur outside of the evaluation methods
                # (e.g., skip function errors, row processing errors)
                other_error_count += 1
                result_row = JudgeResults.create_other_error_row(
                    task_name=self.evaluation_task.task_name,
                    sample_example_id=sample_example_id,
                    original_id=row.get(
                        eval_data.id_column, "unknown"
                    ),  # Use .get() in case row access fails
                    run_id=run_id,
                    judge_id=self.id,
                    error=unexpected_error,
                )
                results.append(result_row)

        # Create results DataFrame
        results_df = pl.DataFrame(results)

        # Create and return JudgeResults
        return JudgeResults(
            run_id=run_id,
            judge_id=self.id,
            task_name=self.evaluation_task.task_name,
            task_outcomes=self.evaluation_task.outcomes,
            llm_client_enum=self.llm_client,
            model_used=self.model,
            timestamp_local=datetime.now(),
            total_examples_count=total_count,
            skipped_examples_count=skipped_count,
            succeeded_examples_count=success_count,
            llm_error_examples_count=llm_error_count,
            parsing_error_examples_count=parsing_error_count,
            other_error_examples_count=other_error_count,
            is_sampled_run=isinstance(eval_data, SampleEvalData),
            results_data=results_df,
        )
