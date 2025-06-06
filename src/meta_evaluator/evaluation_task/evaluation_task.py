"""Main class for evaluation tasks."""

from typing import Any, Callable, Literal
from pydantic import BaseModel, Field, create_model, model_validator


class EvaluationTask(BaseModel):
    """Main class for evaluation tasks."""

    task_schemas: dict[str, list[str]] = Field(
        ..., description="Dictionary mapping task names to their allowed outcome values"
    )
    input_columns: list[str] = Field(..., min_length=1)
    output_columns: list[str] = Field(..., min_length=1)
    skip_function: Callable[[dict[str, Any]], bool] = lambda x: False
    answering_method: Literal["structured", "xml"]

    @model_validator(mode="after")
    def validate_task_configuration(self) -> "EvaluationTask":
        """Validate task schemas configuration.

        Returns:
            EvaluationTask: The validated instance

        Raises:
            ValueError: If task_schemas is empty or any task has fewer than 2 outcomes
        """
        if not self.task_schemas:
            raise ValueError("task_schemas cannot be empty")

        for task_name, outcomes in self.task_schemas.items():
            if len(outcomes) < 2:
                raise ValueError(f"Task '{task_name}' must have at least 2 outcomes")

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
            list[str]: Flattened list of all possible outcomes
        """
        all_outcomes = []
        for outcomes in self.task_schemas.values():
            all_outcomes.extend(outcomes)
        return list(set(all_outcomes))  # Remove duplicates

    def create_task_class(self) -> type[BaseModel]:
        """Create a new evaluation task class with Literal outcomes.

        Returns:
            type[BaseModel]: A new evaluation task class with Literal outcomes.
        """
        model_fields: dict[str, Any] = {}

        # Create one field per task
        for task_name, outcomes in self.task_schemas.items():
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
