"""Pydantic models for Evaluation Task serialization."""

from pydantic import BaseModel
from typing import Optional, Dict, List, Literal


class EvaluationTaskState(BaseModel):
    """Serialized state for Evaluation Task.

    Contains all information needed to reconstruct an EvaluationTask object.
    """

    task_schemas: Dict[str, Optional[List[str]]]
    input_columns: List[str]
    output_columns: List[str]
    answering_method: Literal["structured", "xml"]
