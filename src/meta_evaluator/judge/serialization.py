"""Pydantic models for Judge serialization."""

from pydantic import BaseModel

from ..common.models import Prompt
from ..eval_task.serialization import EvalTaskState


class JudgeState(BaseModel):
    """Serialized state for a Judge instance.

    Contains all information needed to reconstruct a Judge object,
    including the prompt, evaluation task, and LLM configuration.
    """

    id: str
    llm_client: str
    model: str
    prompt: Prompt
    eval_task: EvalTaskState
