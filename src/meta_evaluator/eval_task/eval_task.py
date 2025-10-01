"""Main class for evaluation tasks."""

import logging
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field, create_model, model_validator

from .exceptions import TaskSchemaError
from .serialization import EvalTaskState


class EvalTask(BaseModel):
    """Central class used throughout evaluations to define what and how to evaluate.

    EvalTask configures evaluation tasks by specifying what should be evaluated
    (task_schemas), which data columns to use, and how responses should be parsed.
    It supports two main evaluation scenarios:

    1. **Judge LLM outputs**: When judges evaluate another LLM's responses
       - prompt_columns: contain the input to the evaluated LLM
       - response_columns: contain the evaluated LLM's outputs
       - task_schemas: define evaluation criteria.

    2. **Judge any text content**: When judges evaluate arbitrary text
       - response_columns: contain the text to evaluate
       - task_schemas: define what aspects to evaluate.

    Currently, this class handles
    - classification tasks (with predefined outcomes)
    - free-form text evaluation

    Attributes:
        task_schemas (dict[str, list[str] | None]): Maps task names to allowed outcomes.
            Use None for free-form text outputs, or list of strings for classification.
        required_tasks (Optional[list[str]]): List of task names that are required for
            valid annotations. If None, all non-null task_schemas are required.
        prompt_columns (Optional[list[str]]): Column names containing inputs to the
            evaluated LLM. Only used when judging LLM outputs, None when judging text.
        response_columns (list[str]): Column names containing text/outputs to evaluate.
            Required for all evaluation scenarios.
        skip_function (Callable): Function to determine if a data row should be skipped.
        answering_method (Literal["structured", "instructor", "xml"]): Output parsing method.
            "structured" uses Pydantic models, "instructor" uses instructor library,
            "xml" uses XML tag parsing.
        structured_outputs_fallback (bool): When True, automatically falls back to other
            answering methods if the specified method is unsupported by the model.
            When False, strictly uses the specified answering method, and raises
            UnsupportedFormatMethodError for unsupported methods.
            Only applies when answering_method is "structured" or "instructor".
        annotation_prompt (str): Static prompt text shown to human annotators in the
            annotation interface. Similar to judge prompts but simpler and one-off.
        logger (logging.Logger): Logger instance for this task.

    Raises:
        TaskSchemaError: If task_schemas is empty or any task has fewer than 2 outcomes.

    Examples:
        >>> # Evaluate LLM responses for toxicity and relevance
        >>> task = EvalTask(
        ...     task_schemas={"toxicity": ["toxic", "non_toxic"], "relevance": ["relevant", "irrelevant"]},
        ...     prompt_columns=["user_input"],  # Input to evaluated LLM
        ...     response_columns=["llm_response"],  # LLM output to judge
        ...     answering_method="structured",
        ...     structured_outputs_fallback=True  # Fallback to xml if structured not supported
        ... )
        >>>
        >>> # Evaluate arbitrary text summaries (free-form)
        >>> task = EvalTask(
        ...     task_schemas={"summary_quality": None},  # Free-form evaluation
        ...     response_columns=["summary_text"],  # No prompt_columns needed
        ...     answering_method="xml",
        ...     annotation_prompt="Please evaluate the quality of this summary."
        ... )
    """

    task_schemas: dict[str, list[str] | None] = Field(
        ...,
        description="Dictionary mapping task names to their allowed outcome values. Use None for free form text outputs.",
    )
    required_tasks: Optional[list[str]] = Field(
        default=None,
        description="List of task names that are required for valid annotations. If None, all non-null task_schemas are required.",
    )
    prompt_columns: Optional[list[str]] = Field(default=None)
    response_columns: list[str] = Field(..., min_length=1)
    skip_function: Callable[[dict[str, Any]], bool] = lambda x: False
    answering_method: Literal["structured", "instructor", "xml"]
    structured_outputs_fallback: bool = Field(
        default=False,
        description="When True, automatically falls back to other answering methods if the specified method is unsupported. When False, strictly uses the specified answering method, and raises an error for unsupported methods.",
    )
    annotation_prompt: str = Field(
        default="Please evaluate the following response:",
        description="If necessary, this is the prompt text shown to human annotators in the annotation interface.",
    )
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger(f"{__name__}.EvalTask")
    )

    model_config = {
        "arbitrary_types_allowed": True,  # Allow Logger
    }

    @model_validator(mode="after")
    def validate_task_configuration(self) -> "EvalTask":
        """Validate task schemas configuration.

        Returns:
            EvalTask: The validated instance

        Raises:
            TaskSchemaError: If task_schemas is empty or if any task has fewer than 2 outcomes
        """
        if not self.task_schemas:
            raise TaskSchemaError(
                "task_schema is empty. Please define your tasks and their allowed outcome values."
            )

        for task_name, outcomes in self.task_schemas.items():
            if outcomes is not None and len(outcomes) < 2:
                raise TaskSchemaError(
                    f"Please define at least 2 outcomes for task {task_name}."
                )

        # Validate required_tasks if provided
        if self.required_tasks is not None:
            for required_col in self.required_tasks:
                if required_col not in self.task_schemas:
                    raise TaskSchemaError(
                        f"Required column '{required_col}' not found in task_schemas. "
                        f"Available tasks: {list(self.task_schemas.keys())}"
                    )

        return self

    def get_task_names(self) -> list[str]:
        """Get list of task names.

        Returns:
            list[str]: List of task names
        """
        return list(self.task_schemas.keys())

    def get_required_tasks(self) -> list[str]:
        """Get list of required task names for valid annotations.

        Returns:
            list[str]: List of required task names. If required_tasks is None,
                returns all task names with non-null schemas.
        """
        if self.required_tasks is not None:
            return self.required_tasks
        else:
            # Default behavior: all non-null schemas are required
            return [
                task_name
                for task_name, schema in self.task_schemas.items()
                if schema is not None
            ]

    def get_all_outcomes(self) -> list[str]:
        """Get all possible outcomes across all tasks.

        Returns:
            list[str]: Flattened list of all possible outcomes from tasks with predefined outcomes
        """
        all_outcomes = []
        for outcomes in self.task_schemas.values():
            if outcomes is not None:
                all_outcomes.extend(outcomes)
        return list(set(all_outcomes))  # Remove duplicates

    def create_task_class(self) -> type[BaseModel]:
        """Create a new evaluation task class with Literal outcomes for predefined tasks and str for free form tasks.

        Returns:
            type[BaseModel]: A new evaluation task class with appropriate field types.
        """
        self.logger.info(
            f"Creating task class with {len(self.task_schemas)} tasks: {list(self.task_schemas.keys())}"
        )

        model_fields: dict[str, Any] = {}

        # Create one field per task
        for task_name, outcomes in self.task_schemas.items():
            if outcomes is None:
                # Free form text output
                model_fields[task_name] = (
                    str,
                    Field(
                        ...,
                        description=f"The free form text output for {task_name}",
                    ),
                )
            else:
                # Predefined outcomes using Literal
                outcomes_literal = Literal[tuple(outcomes)]
                model_fields[task_name] = (
                    outcomes_literal,
                    Field(
                        ...,
                        description=f"The outcome for {task_name}. Must be one of: {', '.join(outcomes)}",
                    ),
                )

        DynamicTaskOutcome = create_model(
            "MultiTaskOutcomeRecord", **model_fields, __base__=BaseModel
        )

        return DynamicTaskOutcome

    def get_fallback_sequence(self) -> list[str]:
        """Get the sequence of answering methods to try with fallback enabled.

        Returns the prioritized list of methods to attempt when fallback is enabled.
        The original method is tried first, followed by alternatives in order of preference.

        Returns:
            list[str]: Ordered list of answering methods to try
        """
        if not self.structured_outputs_fallback:
            return [self.answering_method]

        # Define fallback sequences for each method
        fallback_sequences = {
            "structured": ["structured", "instructor", "xml"],
            "instructor": ["instructor", "structured", "xml"],
            "xml": ["xml"],  # XML doesn't need fallback as it's most compatible
        }

        return fallback_sequences.get(self.answering_method, [self.answering_method])

    def serialize(self) -> EvalTaskState:
        """Serialize the EvalTask to metadata (excluding skip_function).

        Returns:
            EvalTaskState: Serialized state for EvalTask.
        """
        self.logger.info(f"Serializing EvalTask with {len(self.task_schemas)} tasks")

        return EvalTaskState(
            task_schemas=self.task_schemas,
            prompt_columns=self.prompt_columns,
            response_columns=self.response_columns,
            answering_method=self.answering_method,
            structured_outputs_fallback=self.structured_outputs_fallback,
            annotation_prompt=self.annotation_prompt,
        )

    @classmethod
    def deserialize(cls, state: EvalTaskState) -> "EvalTask":
        """Deserialize EvalTask from state.

        Args:
            state: Serialized state for EvalTask.

        Returns:
            EvalTask: Reconstructed EvalTask instance.
        """
        return cls(
            task_schemas=state.task_schemas,
            prompt_columns=state.prompt_columns,
            response_columns=state.response_columns,
            answering_method=state.answering_method,
            structured_outputs_fallback=getattr(
                state, "structured_outputs_fallback", False
            ),
            annotation_prompt=getattr(
                state, "annotation_prompt", "Please evaluate the following response:"
            ),
            # skip_function must be set manually or defaulted
        )
