"""Pydantic models for Evaluation Task serialization."""

from pydantic import BaseModel
from typing import Optional, Dict, List, Literal


class EvalTaskState(BaseModel):
    """Serialized state for Evaluation Task.

    Contains all information needed to reconstruct an EvalTask object.
    """

    task_schemas: Dict[str, Optional[List[str]]]
    prompt_columns: Optional[List[str]]
    response_columns: List[str]
    answering_method: Literal["structured", "xml"]
