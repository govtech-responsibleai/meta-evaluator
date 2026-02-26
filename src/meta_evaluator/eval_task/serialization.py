"""Pydantic models for Evaluation Task serialization."""

from typing import Literal

from pydantic import BaseModel


class EvalTaskState(BaseModel):
    """Serialized state for Evaluation Task.

    Contains all information needed to reconstruct an EvalTask object.
    """

    task_schemas: dict[str, list[str] | None]
    required_tasks: list[str] | None = None
    prompt_columns: list[str] | None
    response_columns: list[str]
    answering_method: Literal["structured", "instructor", "xml"]
    structured_outputs_fallback: bool = False
    annotation_prompt: str = "Please evaluate the following response:"
