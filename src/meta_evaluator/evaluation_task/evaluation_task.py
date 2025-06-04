"""Main class for evaluation tasks."""

from typing import Any, Callable
from pydantic import BaseModel, Field


class EvaluationTask(BaseModel):
    """Main class for evaluation tasks.

    Properties:
        task_name: str. A name for the task
        outcomes: list[str]. The outcomes of the task. Minimum of 2
        input_columns: list[str]. The input columns that were given to the LLM that are relevant to this task
        output_columns: list[str]. The output columns that show the LLM's responses that are relevant to this task
        skip_function: Callable[[dict[str, Any]], bool]. A function to skip the task. Takes in a dictionary of the row. Defaults to False. If the function returns true, then the task will be skipped for that row
    """

    task_name: str = Field(..., min_length=1)
    outcomes: list[str] = Field(..., min_length=2)
    input_columns: list[str] = Field(..., min_length=1)
    output_columns: list[str] = Field(..., min_length=1)
    skip_function: Callable[[dict[str, Any]], bool] = lambda x: False
