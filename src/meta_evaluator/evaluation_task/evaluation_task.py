"""Main class for evaluation tasks."""

from typing import Any, Callable, Literal
from pydantic import BaseModel, Field, create_model


class EvaluationTask(BaseModel):
    """Main class for evaluation tasks."""

    task_name: str = Field(..., min_length=1)
    outcomes: list[str] = Field(..., min_length=2)
    input_columns: list[str] = Field(..., min_length=1)
    output_columns: list[str] = Field(..., min_length=1)
    skip_function: Callable[[dict[str, Any]], bool] = lambda x: False
    answering_method: Literal["structured", "xml"]

    def create_task_class(self) -> type[BaseModel]:
        """Create a new evaluation task class with Literal outcomes.

        Returns:
            type[BaseModel]: A new evaluation task class with Literal outcomes.
        """
        outcomes_literal = Literal[tuple(self.outcomes)]
        model_fields: dict[str, Any] = {}
        model_fields["result_outcome"] = (
            outcomes_literal,
            Field(
                ...,
                description=f"The actual outcome of this evaluation record. Must be one of: {', '.join(self.outcomes)}",
            ),
        )

        dynamic_model_name = f"{self.task_name.replace(' ', '')}OutcomeRecord"
        DynamicTaskOutcome = create_model(
            dynamic_model_name, **model_fields, __base__=BaseModel
        )

        return DynamicTaskOutcome
