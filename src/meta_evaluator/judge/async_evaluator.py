"""Asynchronous evaluation methods for Judge class."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, cast

import instructor
from litellm import acompletion
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


class AsyncEvaluationMixin(ABC):
    """Mixin providing asynchronous evaluation methods for Judge class."""

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

    async def _evaluate_row_with_fallback_async(
        self,
        row: dict[str, Any],
        eval_data: EvalData,
        sample_example_id: str,
        task_class: type[BaseModel],
        tag_configs: list[TagConfig],
        builder: JudgeResultsBuilder,
    ) -> None:
        """Evaluate a single row with fallback support (async version).

        Tries methods in the fallback sequence until one succeeds or all fail.

        Args:
            row: Row data from EvalData
            eval_data: The EvalData instance
            sample_example_id: Unique identifier for this sample
            task_class: The dynamically created Pydantic model for responses
            tag_configs: List of TagConfigs for XML parsing
            builder: The JudgeResultsBuilder instance

        Raises:
            UnsupportedFormatMethodError: If the answering method is not supported.
        """
        fallback_sequence = self.eval_task.get_fallback_sequence()
        last_error = None

        for attempt_idx, method in enumerate(fallback_sequence):
            try:
                if method == "structured":
                    await self._evaluate_row_structured_async(
                        row=row,
                        eval_data=eval_data,
                        sample_example_id=sample_example_id,
                        task_class=task_class,
                        builder=builder,
                    )
                    return  # Success, exit early
                elif method == "instructor":
                    # Create messages for instructor method
                    system_message = self._create_system_message(  # type: ignore
                        include_xml_instructions=False
                    )
                    user_content = self._format_row_data(row)  # type: ignore
                    user_message = Message(role=RoleEnum.USER, content=user_content)
                    messages = [system_message, user_message]

                    await self._evaluate_row_instructor_async(
                        row=row,
                        eval_data=eval_data,
                        sample_example_id=sample_example_id,
                        task_class=task_class,
                        builder=builder,
                        messages=messages,
                    )
                    return  # Success, exit early
                elif method == "xml":
                    await self._evaluate_row_xml_async(
                        row=row,
                        eval_data=eval_data,
                        sample_example_id=sample_example_id,
                        tag_configs=tag_configs,
                        builder=builder,
                    )
                    return  # Success, exit early

            except UnsupportedFormatMethodError as e:
                # This method is not supported, try the next one
                last_error = e
                self.logger.warning(
                    f"Method '{method}' not supported, trying next fallback method"
                )
                continue
            except Exception as e:
                # Other errors should not trigger fallback, they indicate actual failure
                raise e

        # All methods failed, raise the last UnsupportedFormatMethodError
        if last_error:
            raise last_error
        else:
            # This shouldn't happen, but just in case
            full_model_name = f"{self.llm_client}/{self.model}"
            raise UnsupportedFormatMethodError(
                method=self.eval_task.answering_method,
                model=full_model_name,
                suggested_methods=["structured", "instructor", "xml"],
            )

    async def _evaluate_row_structured_async(
        self,
        row: dict[str, Any],
        eval_data: EvalData,
        sample_example_id: str,
        task_class: type[BaseModel],
        builder: JudgeResultsBuilder,
    ) -> None:
        """Evaluate a single row using structured response method (async version).

        Args:
            row: Row data from EvalData
            eval_data: The EvalData instance
            sample_example_id: Unique identifier for this sample
            task_class: The dynamically created Pydantic model for responses
            builder: The JudgeResultsBuilder instance

        Raises:
            UnsupportedFormatMethodError: If the answering method is not supported.
            LLMAPIError: If the LLM API call fails.
        """
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        try:
            # Create messages
            system_message = self._create_system_message(include_xml_instructions=False)  # type: ignore
            user_content = self._format_row_data(row)  # type: ignore
            user_message = Message(role=RoleEnum.USER, content=user_content)
            messages = [system_message, user_message]

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

                # Cast to ModelResponse to avoid type errors
                response = cast(
                    ModelResponse,
                    await acompletion(
                        model=full_model_name,
                        messages=openai_messages,
                        response_format=task_class,
                        stream=False,
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
                    content=structured_response or "",  # Handle potential None
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

    async def _evaluate_row_instructor_async(
        self,
        row: dict[str, Any],
        eval_data: EvalData,
        sample_example_id: str,
        task_class: type[BaseModel],
        builder: JudgeResultsBuilder,
        messages: list[Message],
    ) -> None:
        """Evaluate a single row using instructor for structured response (async version).

        Args:
            row: Row data from EvalData
            eval_data: The EvalData instance
            sample_example_id: Unique identifier for this sample
            task_class: The dynamically created Pydantic model for responses
            builder: The JudgeResultsBuilder instance
            messages: Pre-created messages for the LLM
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

            # Use instructor for structured response (async)
            _resp = await client.chat.completions.create(
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
            # Any unexpected error (LLM API error, instructor error, etc.)
            builder.create_other_error_row(
                sample_example_id=sample_example_id,
                original_id=row[eval_data.id_column],
                error=unexpected_error,
            )

    async def _evaluate_row_xml_async(
        self,
        row: dict[str, Any],
        eval_data: EvalData,
        sample_example_id: str,
        tag_configs: list[TagConfig],
        builder: JudgeResultsBuilder,
    ) -> None:
        """Evaluate a single row using XML tags method (async version).

        Args:
            row: Row data from EvalData
            eval_data: The EvalData instance
            sample_example_id: Unique identifier for this sample
            tag_configs: List of TagConfigs for parsing XML response (one per task)
            builder: The JudgeResultsBuilder instance
        """
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        try:
            # Create messages with XML instructions
            system_message = self._create_system_message(include_xml_instructions=True)  # type: ignore
            user_content = self._format_row_data(row)  # type: ignore
            user_message = Message(role=RoleEnum.USER, content=user_content)
            messages = [system_message, user_message]

            start_time = time.time()

            # Call LLM with XML tags using litellm (async)
            full_model_name = f"{self.llm_client}/{self.model}"

            # Convert messages to OpenAI format
            openai_messages = self._convert_messages_to_openai_format(messages)  # type: ignore

            # Cast to ModelResponse to avoid type errors
            response = cast(
                ModelResponse,
                await acompletion(
                    model=full_model_name,
                    messages=openai_messages,
                    stream=False,
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

    #################
    # ASYNC METHODS #
    #################

    async def _evaluate_rows_batch_structured(
        self,
        rows_to_evaluate: list[tuple[dict[str, Any], str]],
        eval_data: EvalData,
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
            system_message = self._create_system_message(include_xml_instructions=False)  # type: ignore
            user_content = self._format_row_data(row)  # type: ignore
            user_message = Message(role=RoleEnum.USER, content=user_content)
            messages = [system_message, user_message]
            batch_items.append((messages, task_class))

        # Execute batch using concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_single_row(row_data, sample_id, task_cls):
            async with semaphore:
                # Use the existing async structured evaluation method
                await self._evaluate_row_structured_async(
                    row=row_data,
                    eval_data=eval_data,
                    sample_example_id=sample_id,
                    task_class=task_cls,
                    builder=builder,
                )

        # Create tasks for all rows
        tasks = []
        for (row, sample_example_id), (messages, task_cls) in zip(
            rows_to_evaluate, batch_items
        ):
            task = process_single_row(row, sample_example_id, task_cls)
            tasks.append(task)

        # Execute all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _evaluate_rows_batch_xml(
        self,
        rows_to_evaluate: list[tuple[dict[str, Any], str]],
        eval_data: EvalData,
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
            system_message = self._create_system_message(include_xml_instructions=True)  # type: ignore
            user_content = self._format_row_data(row)  # type: ignore
            user_message = Message(role=RoleEnum.USER, content=user_content)
            messages = [system_message, user_message]
            batch_items.append((messages, tag_configs))

        # Execute batch using concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_single_row(row_data, sample_id, tag_cfgs):
            async with semaphore:
                # Use the existing async XML evaluation method
                await self._evaluate_row_xml_async(
                    row=row_data,
                    eval_data=eval_data,
                    sample_example_id=sample_id,
                    tag_configs=tag_cfgs,
                    builder=builder,
                )

        # Create tasks for all rows
        tasks = []
        for (row, sample_example_id), (messages, tag_cfgs) in zip(
            rows_to_evaluate, batch_items
        ):
            task = process_single_row(row, sample_example_id, tag_cfgs)
            tasks.append(task)

        # Execute all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _evaluate_rows_batch_instructor(
        self,
        rows_to_evaluate: list[tuple[dict[str, Any], str]],
        eval_data: EvalData,
        builder: JudgeResultsBuilder,
        batch_size: int,
        max_concurrency: int,
    ) -> None:
        """Evaluate multiple rows using instructor method with batching."""
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )

        # Create the task class once for all requests
        task_class = self.eval_task.create_task_class()

        # Prepare batch items: (messages, task_class)
        batch_items = []
        for row, sample_example_id in rows_to_evaluate:
            system_message = self._create_system_message(include_xml_instructions=False)  # type: ignore
            user_content = self._format_row_data(row)  # type: ignore
            user_message = Message(role=RoleEnum.USER, content=user_content)
            messages = [system_message, user_message]
            batch_items.append((messages, task_class))

        # Execute batch using concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_single_row(row_data, sample_id, task_cls, msgs):
            async with semaphore:
                # Use the existing async instructor evaluation method
                await self._evaluate_row_instructor_async(
                    row=row_data,
                    eval_data=eval_data,
                    sample_example_id=sample_id,
                    task_class=task_cls,
                    builder=builder,
                    messages=msgs,
                )

        # Create tasks for all rows
        tasks = []
        for (row, sample_example_id), (messages, task_cls) in zip(
            rows_to_evaluate, batch_items
        ):
            task = process_single_row(row, sample_example_id, task_cls, messages)
            tasks.append(task)

        # Execute all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

    async def evaluate_eval_data_async(
        self,
        eval_data: EvalData,
        run_id: str,
        batch_size: int = 10,
        max_concurrency: int = 5,
    ) -> JudgeResults:
        """Evaluate the input EvalData using the configured Judge with async batching.

        This method provides the same functionality as evaluate_eval_data but uses
        async batching for better performance when processing large datasets.

        Args:
            eval_data (EvalData): The EvalData instance to be evaluated.
            run_id (str): A unique identifier for the evaluation run.
            batch_size (int): Number of requests to process in each batch. Defaults to 10.
            max_concurrency (int): Maximum number of concurrent requests. Defaults to 5.

        Returns:
            JudgeResults: A JudgeResults instance containing the results of the evaluation.
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
            is_sampled_run=isinstance(eval_data, SampleEvalData),
        )

        # Collect all rows that need evaluation
        rows_to_evaluate = []
        total_count = 0
        for row in self._get_dicts_as_generator(eval_data):  # type: ignore
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
                builder=builder,
                batch_size=batch_size,
                max_concurrency=max_concurrency,
            )
        elif self.eval_task.answering_method == "instructor":
            await self._evaluate_rows_batch_instructor(
                rows_to_evaluate=rows_to_evaluate,
                eval_data=eval_data,
                builder=builder,
                batch_size=batch_size,
                max_concurrency=max_concurrency,
            )
        else:  # xml
            await self._evaluate_rows_batch_xml(
                rows_to_evaluate=rows_to_evaluate,
                eval_data=eval_data,
                builder=builder,
                batch_size=batch_size,
                max_concurrency=max_concurrency,
            )

        # Complete the builder and return JudgeResults
        return builder.complete()
