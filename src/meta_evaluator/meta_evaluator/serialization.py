"""Pydantic models for MetaEvaluator serialization."""

from typing import Dict, Optional

from pydantic import BaseModel

from ..data.serialization import DataMetadata
from ..eval_task.serialization import EvalTaskState
from ..judge.serialization import JudgeState


class MetaEvaluatorState(BaseModel):
    """Complete serialized state for a MetaEvaluator instance.

    Contains all information needed to reconstruct a MetaEvaluator,
    including client configurations, data metadata, and judge registry.
    """

    version: str = "1.0"
    data: Optional[DataMetadata] = None
    eval_task: Optional[EvalTaskState] = None
    judge_registry: Dict[str, JudgeState] = {}
