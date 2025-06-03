"""Main class for evaluation tasks."""

from pydantic import BaseModel, Field


class EvaluationTask(BaseModel):
    """Main class for evaluation tasks.

    Properties:
        task_name: str. A name for the task
        outcomes: list[str]. The outcomes of the task. Minimum of 2
    """

    task_name: str = Field(..., min_length=1)
    outcomes: list[str] = Field(..., min_length=2)
