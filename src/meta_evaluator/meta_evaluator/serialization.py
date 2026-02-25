"""Pydantic models for MetaEvaluator serialization."""

from pydantic import BaseModel

from ..data.serialization import DataMetadata
from ..eval_task.serialization import EvalTaskState
from ..judge.serialization import JudgeState


class MetaEvaluatorState(BaseModel):
    """Complete serialized state for a MetaEvaluator instance.

    Contains all information needed to reconstruct a MetaEvaluator,
    including client configurations, data metadata, judge registry, and metrics configuration.
    """

    version: str = "1.0"
    data: DataMetadata | None = None
    eval_task: EvalTaskState | None = None
    judge_registry: dict[str, JudgeState] = {}
