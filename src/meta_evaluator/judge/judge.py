"""Main class of judge module."""

from ..evaluation_task import EvaluationTask
from ..llm_client import LLMClientEnum
from ..common.models import Prompt
from pydantic import BaseModel, ConfigDict


class Judge(BaseModel):
    """Main class of judge module."""

    model_config = ConfigDict(frozen=True)
    evaluation_task: EvaluationTask
    llm_client: LLMClientEnum
    model: str
    prompt: Prompt
