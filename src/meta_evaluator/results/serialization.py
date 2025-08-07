"""Pydantic models for Judge and Human Results serialization."""

from datetime import datetime
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel

from ..llm_client.enums import LLMClientEnum


class BaseResultsSerializedState(BaseModel):
    """Base serialized state for Results.

    Contains common fields needed by all result types.
    """

    run_id: str
    task_schemas: Dict[str, Optional[List[str]]]
    timestamp_local: datetime
    total_count: int
    succeeded_count: int
    is_sampled_run: bool
    data_file: str
    data_format: Literal["json", "csv", "parquet"]


class JudgeResultsSerializedState(BaseResultsSerializedState):
    """Serialized state for Judge Results.

    Contains all information needed to reconstruct a JudgeResults object
    from external data files.
    """

    judge_id: str
    llm_client_enum: LLMClientEnum
    model_used: str
    skipped_count: int
    partial_count: int
    llm_error_count: int
    parsing_error_count: int
    other_error_count: int


class HumanAnnotationResultsSerializedState(BaseResultsSerializedState):
    """Serialized state for Human Annotation Results.

    Contains all information needed to reconstruct a HumanAnnotationResults object
    from external data files.
    """

    annotator_id: str
    error_count: int
