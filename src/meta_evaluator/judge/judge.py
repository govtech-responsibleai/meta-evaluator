"""Main class of judge module."""

import json
import logging
import re
from collections.abc import Generator
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ConfigDict, model_validator

from ..common.models import Prompt
from ..data import EvalData
from ..eval_task import EvalTask
from ..results import JudgeResultsBuilder
from .async_evaluator import AsyncEvaluationMixin
from .enums import ErrorType, RoleEnum
from .exceptions import MissingTemplateVariablesError
from .models import (
    LLMResponse,
    Message,
    ParseError,
    ParseResult,
    TagConfig,
)
from .serialization import JudgeState
from .sync_evaluator import SyncEvaluationMixin


class Judge(AsyncEvaluationMixin, SyncEvaluationMixin, BaseModel):
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
        llm_client (str): The LLM client to be used for this evaluation (e.g., openai, azure).
        model (str): The specific name of the LLM model to be used from the
            selected provider (e.g., "gpt-4", "claude-3-opus-20240229"). This
            model will receive the prompt and perform the evaluation.
        prompt (Prompt): A Prompt object containing the instructions, few-shot examples,
            and structured output requirements (like XML tags or Pydantic models)
            that will be sent to the LLM. This dictates *how* the LLM should perform
            the evaluation based on the input data.
            and is never auto-generated.
        model_config (ConfigDict): Pydantic configuration dictionary.
            - `frozen=True`: Makes the Judge instance immutable after creation,
            ensuring its configuration remains constant throughout its lifecycle.
        eval_task (EvalTask): An instance of the EvalTask class
            defining the criteria and desired outcomes for the evaluation. This
            specifies *what* is being evaluated (e.g., toxicity, relevance) and
            the possible labels or scores the Judge is expected to produce.


    Validation:
        - The `id` attribute is validated to ensure it contains only alphanumeric
          characters and underscores, making it safe and consistent for use
          in various system contexts.
    """

    id: str
    llm_client: str
    model: str
    prompt: Prompt
    model_config = ConfigDict(frozen=True)
    eval_task: EvalTask

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this Judge instance."""
        return logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

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

    # def _convert_messages_to_openai_format(
    #     self, messages: list[Message]
    # ) -> list[dict[str, str]]:
    #     """Convert Message objects to OpenAI messages format.

    #     Returns:
    #         list[dict[str, str]]: List of OpenAI-formatted message parameters.
    #     """
    #     return [{"role": msg.role.value, "content": msg.content} for msg in messages]

    def _convert_messages_to_openai_format(
        self, messages: list[Message]
    ) -> list[ChatCompletionMessageParam]:
        """Convert Message objects to OpenAI ChatCompletionMessageParam format.

        Returns:
            list[ChatCompletionMessageParam]: List of OpenAI-formatted message parameters.
        """
        openai_messages: list[ChatCompletionMessageParam] = []

        for message in messages:
            if message.role == RoleEnum.USER:
                user_message: ChatCompletionUserMessageParam = {
                    "role": "user",
                    "content": message.content,
                }
                openai_messages.append(user_message)
            elif message.role == RoleEnum.SYSTEM:
                system_message: ChatCompletionSystemMessageParam = {
                    "role": "system",
                    "content": message.content,
                }
                openai_messages.append(system_message)
            elif message.role == RoleEnum.ASSISTANT:
                assistant_message: ChatCompletionAssistantMessageParam = {
                    "role": "assistant",
                    "content": message.content,
                }
                openai_messages.append(assistant_message)

        return openai_messages

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

    def _substitute_template_variables(self, template: str, row: dict[str, Any]) -> str:
        """Substitute template variables in curly brackets with row data.

        Args:
            template: The template string containing {variable} placeholders.
            row: Dictionary containing the row data for substitution.

        Returns:
            str: Template with variables substituted.
        """
        # Get all columns that should be available for substitution
        available_columns = set()
        if self.eval_task.prompt_columns:
            available_columns.update(self.eval_task.prompt_columns)
        if self.eval_task.response_columns:
            available_columns.update(self.eval_task.response_columns)

        # Perform substitution for available columns
        substituted = template
        for column in available_columns:
            if column in row:
                placeholder = "{" + column + "}"
                substituted = substituted.replace(placeholder, str(row[column]))

        return substituted

    def _validate_template_variables(self, template: str) -> None:
        """Validate that the template contains all required variable placeholders.

        Args:
            template: The template string to validate.

        Raises:
            MissingTemplateVariablesError: If required variables are missing from the template.
        """
        # Get all columns that should be available for substitution
        required_columns = set()
        if self.eval_task.prompt_columns:
            required_columns.update(self.eval_task.prompt_columns)
        if self.eval_task.response_columns:
            required_columns.update(self.eval_task.response_columns)

        # Find missing variables
        missing_variables = []
        for column in required_columns:
            placeholder = "{" + column + "}"
            if placeholder not in template:
                missing_variables.append(column)

        # Raise error if any variables are missing
        if missing_variables:
            raise MissingTemplateVariablesError(
                missing_variables=missing_variables,
                prompt_columns=self.eval_task.prompt_columns,
                response_columns=self.eval_task.response_columns,
            )

    def _create_system_message(
        self, row: dict[str, Any], include_xml_instructions: bool = False
    ) -> Message:
        """Create the system message with evaluation instructions.

        Args:
            include_xml_instructions: Whether to include XML formatting instructions.
            row: Optional row data for template variable substitution.

        Returns:
            Message: System message with evaluation context and instructions.

        """
        system_content = self.prompt.prompt

        # Perform template variable substitution if row data is provided
        if row is not None:
            # Validate that all required template variables are present in the prompt
            self._validate_template_variables(system_content)
            system_content = self._substitute_template_variables(system_content, row)

        if include_xml_instructions:
            system_content += self._get_xml_instructions()
        return Message(role=RoleEnum.SYSTEM, content=system_content)

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

    def _extract_outcomes_from_json(
        self, json_response: str
    ) -> tuple[dict[str, Any], list[str]]:
        """Extract outcomes from a JSON string response.

        Args:
            json_response: JSON string from structured response

        Returns:
            Tuple of (outcomes_dict, missing_tasks_list)
        """
        parsed_response = json.loads(json_response)
        outcomes = {}
        missing_tasks = []

        for task_name in self.eval_task.task_schemas.keys():
            if task_name in parsed_response:
                outcomes[task_name] = parsed_response[task_name]
            else:
                missing_tasks.append(task_name)

        return outcomes, missing_tasks

    def _extract_outcomes_from_parse_result(
        self, parse_result: ParseResult
    ) -> tuple[dict[str, Any], list[str]]:
        """Extract outcomes from XML ParseResult.

        Args:
            parse_result: ParseResult from XML parsing

        Returns:
            Tuple of (outcomes_dict, missing_tasks_list)
        """
        task_names = set(self.eval_task.task_schemas.keys())
        parsed_tasks = set(parse_result.data.keys())
        missing_tasks = list(task_names - parsed_tasks)

        # Convert ParseResult data to outcomes (cast to str for XML)
        outcomes = {}
        for task_name in parsed_tasks:
            outcome = parse_result.data[task_name]
            outcomes[task_name] = cast(
                str, outcome
            )  # TagConfig cardinality="one" guarantees str

        return outcomes, missing_tasks

    def _assign_outcomes(
        self,
        outcomes: dict[str, Any],
        missing_tasks: list[str],
        llm_response: LLMResponse,
        call_duration: float,
        row: dict[str, Any],
        eval_data: EvalData,
        sample_example_id: str,
        builder: JudgeResultsBuilder,
    ) -> None:
        """Assign outcomes to builder rows based on parsing results.

        Args:
            outcomes: Dictionary of successfully parsed task outcomes
            missing_tasks: List of task names that were missing from the response
            llm_response: The LLM response object
            call_duration: Duration of the LLM call
            row: Row data from EvalData
            eval_data: The EvalData instance
            sample_example_id: Unique identifier for this sample
            builder: The JudgeResultsBuilder instance
        """
        assert eval_data.id_column is not None, (
            f"EvalData {eval_data.name} has no ID column, but was expected to have one."
        )
        try:
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

    # ================================
    # Shared XML Parsing Methods
    # ================================

    def _extract_tag_values(self, tag_name: str, raw_text: str) -> list[str]:
        """Extract all content values from XML tags with the given name.

        Args:
            tag_name: The XML tag name to search for (e.g., "user_id", "status")
            raw_text: Raw XML text to search within

        Returns:
            List of string values found within the specified XML tags
        """
        # Pattern matches <tag_name>content</tag_name> or <tag_name attr="value">content</tag_name>
        pattern = rf"<{re.escape(tag_name)}(?:\s[^>]*)?>(.*?)</{re.escape(tag_name)}>"
        matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)

        # Strip whitespace from each match and filter out empty strings
        values = [match.strip() for match in matches if match.strip()]

        return values

    def _construct_xml_tag_parsing(
        self, tag_config_list: list[TagConfig], raw_text: str
    ) -> ParseResult:
        """Parse XML tags from raw text according to configuration rules.

        Extracts content from XML tags specified in the configuration list and validates
        the results according to value constraints, cardinality rules, and error handling
        preferences. Returns both successfully parsed data and detailed error information
        for any validation failures.

        The parsing process follows these steps for each configured tag:
        1. Extract all instances of the tag from raw text using regex matching
        2. If no instances found, record TAG_NOT_FOUND error and skip to next tag
        3. Validate extracted values against allowed_values if specified, recording
        INVALID_VALUE errors for non-matching values while keeping valid ones
        4. Apply cardinality constraints to valid values:
        - cardinality="one": Requires exactly 1 valid value
        - cardinality="many": Requires 1+ valid values (empty is an error)
        5. Handle multiple values for cardinality="one" according to multiple_handling rules
        6. Add successfully parsed values to data dict, failures only go to errors list

        This method uses a "clean separation" design where the data dict contains only
        successfully parsed values, while all failure information is recorded in the
        errors list. This avoids redundant None values and empty lists in the data.

        Args:
            tag_config_list: List of TagConfig objects defining parsing rules for each
                XML tag. Each config specifies the tag name, allowed values, cardinality
                expectations, and behavior when multiple values are found.
            raw_text: Raw XML text content to parse. Should contain the XML tags
                specified in the tag configurations.

        Returns:
            ParseResult containing:
                - data: Dictionary mapping tag names to successfully parsed values.
                Keys only exist for tags that were successfully parsed and satisfied
                all validation rules. Values are strings for cardinality="one" or
                lists of strings for cardinality="many".
                - errors: List of ParseError objects detailing any validation failures.
                Includes errors for missing tags, invalid values, cardinality mismatches,
                and multiple value conflicts.

        Note:
            Failed parsing attempts do not create entries in the data dictionary.
            Use `"tag_name" in result.data` or `result.data.get("tag_name")` to check
            for successful parsing. Use `result.get_errors_by_tag("tag_name")` to get
            detailed error information for failed tags.

            Tag extraction uses case-insensitive regex matching and handles both
            self-closing and standard XML tag formats. Whitespace is automatically
            trimmed from extracted values, and empty values are filtered out.

        Examples:
            Successful parsing with mixed results:
                >>> configs = [
                ...     TagConfig(name="status", allowed_values=["active"], cardinality="one"),
                ...     TagConfig(name="missing", cardinality="one")
                ... ]
                >>> result = self._construct_xml_tag_parsing(configs, "<status>active</status>")
                >>> result.data  # {"status": "active"} - no "missing" key
                >>> len(result.errors)  # 1 - TAG_NOT_FOUND for "missing"

            All parsing failures:
                >>> configs = [TagConfig(name="invalid", allowed_values=["valid"], cardinality="one")]
                >>> result = self._construct_xml_tag_parsing(configs, "<invalid>bad</invalid>")
                >>> result.data  # {} - empty dict, no successful parses
                >>> result.errors[0].error_type  # ErrorType.INVALID_VALUE
        """
        data = {}
        errors = []

        for config in tag_config_list:
            # Step 1: Extract all instances of this tag from the raw text
            raw_values = self._extract_tag_values(config.name, raw_text)

            # Step 2: Handle case where no tag instances were found
            if not raw_values:
                errors.append(
                    ParseError(
                        error_type=ErrorType.TAG_NOT_FOUND,
                        tag_name=config.name,
                        message=f"Required tag '{config.name}' not found in text",
                    )
                )
                # Skip to next config - no data entry for missing tags
                continue

            # Step 3: Validate extracted values against allowed_values constraint
            if config.allowed_values is not None:
                valid_values = []
                allowed_values_lower = [v.lower() for v in config.allowed_values]

                for value in raw_values:
                    if value in allowed_values_lower:
                        valid_values.append(value)
                    else:
                        # Record invalid value error but continue processing other values
                        errors.append(
                            ParseError(
                                error_type=ErrorType.INVALID_VALUE,
                                tag_name=config.name,
                                message=f"Invalid value '{value}' for tag '{config.name}'. "
                                f"Allowed values: {config.allowed_values}",
                                found_values=[value],
                                expected_values=config.allowed_values,
                            )
                        )
            else:
                # No value restrictions - all extracted values are considered valid
                valid_values = raw_values

            # Step 4: Check if we have any valid values after filtering
            if not valid_values:
                # All values were invalid - no data entry, errors already recorded above
                continue

            # Step 5: Apply cardinality constraints to valid values
            if config.cardinality == "one":
                # Expect exactly one valid value
                if len(valid_values) == 1:
                    # Perfect - exactly one valid value
                    data[config.name] = valid_values[0]
                else:
                    # Multiple valid values found - handle according to multiple_handling strategy

                    # Allow if all values are the same
                    if config.multiple_handling == "error_if_different":
                        # Check if all values are identical
                        unique_values = list(set(valid_values))
                        if len(unique_values) == 1:
                            # All values are the same - accept the common value
                            data[config.name] = unique_values[0]
                        else:
                            # Multiple different values - this is an error
                            errors.append(
                                ParseError(
                                    error_type=ErrorType.MULTIPLE_VALUES_CONFLICT,
                                    tag_name=config.name,
                                    message=f"Multiple different values for '{config.name}': "
                                    f"{valid_values}",
                                    found_values=valid_values,
                                )
                            )

                    # Allow all different values
                    elif config.multiple_handling == "allow_both":
                        # Accept multiple values as a list despite cardinality="one"
                        data[config.name] = valid_values

                    # Generate error if "error" specified or if undefined
                    else:
                        if config.multiple_handling != "error":
                            logging.warning(
                                f"Multiple handling {config.multiple_handling} not supported for {config.name}. "
                                "Assuming config.multiple_handling == 'error'."
                            )
                        errors.append(
                            ParseError(
                                error_type=ErrorType.CARDINALITY_MISMATCH,
                                tag_name=config.name,
                                message=f"Expected exactly 1 value for '{config.name}', "
                                f"found {len(valid_values)}",
                                found_values=valid_values,
                            )
                        )

            elif config.cardinality == "many":
                # Expect one or more valid values (empty list after filtering is an error)
                # Note: We already checked `not valid_values` above, so we have 1+ values here
                data[config.name] = valid_values

        return ParseResult(data=data, errors=errors)

    def serialize(self) -> JudgeState:
        """Serialize the Judge to metadata.

        Returns:
            JudgeState: Serialized state for Judge.
        """
        self.logger.info(f"Serializing Judge with id '{self.id}'")

        return JudgeState(
            id=self.id,
            llm_client=self.llm_client,
            model=self.model,
            prompt=self.prompt,
            eval_task=self.eval_task.serialize(),
        )

    @classmethod
    def deserialize(cls, state: JudgeState) -> "Judge":
        """Deserialize Judge from state.

        Args:
            state: Serialized state for Judge.

        Returns:
            Judge: Reconstructed Judge instance (frozen).
        """
        return cls(
            id=state.id,
            llm_client=state.llm_client,
            model=state.model,
            prompt=state.prompt,
            eval_task=EvalTask.deserialize(state.eval_task),
        )
