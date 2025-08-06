"""Main class for evaluation tasks."""

import logging
from typing import Any, Callable, Literal, Optional
from pydantic import BaseModel, Field, create_model, model_validator
from .serialization import EvalTaskState
from .exceptions import TaskSchemaError


class EvalTask(BaseModel):
    """Main class for evaluation tasks."""

    task_schemas: dict[str, list[str] | None] = Field(
        ...,
        description="Dictionary mapping task names to their allowed outcome values. Use None for free form text outputs.",
    )
    prompt_columns: Optional[list[str]] = Field(default=None)
    response_columns: list[str] = Field(..., min_length=1)
    skip_function: Callable[[dict[str, Any]], bool] = lambda x: False
    answering_method: Literal["structured", "xml"]
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

        return self

    def get_task_names(self) -> list[str]:
        """Get list of task names.

        Returns:
            list[str]: List of task names
        """
        return list(self.task_schemas.keys())

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
            # skip_function must be set manually or defaulted
        )
