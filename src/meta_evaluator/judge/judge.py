"""Main class of judge module."""

from ..evaluation_task import EvaluationTask
from ..llm_client import LLMClientEnum
from ..common.models import Prompt
from pydantic import BaseModel, ConfigDict, model_validator
import re


class Judge(BaseModel):
    """Represents a specific configuration for executing an evaluation task using an LLM.

    This class bundles all necessary parameters to define how a single evaluation
    process should be performed for a given task. It encapsulates the target evaluation
    criteria (what to evaluate), the AI model to use (which LLM provider and model),
    and the instruction or prompt that guides the LLM's evaluation process. Each Judge
    instance represents a unique setup for one evaluation run and is identified by a
    stable ID, which is critical for reproducibility and tracking results across runs.

    Attributes:
        id (str): A unique, stable identifier for this specific Judge configuration.
            This ID is used to reference this exact setup (task, model, prompt)
            in configurations, logs, and results. It must contain only alphanumeric
            characters and underscores to ensure compatibility with file paths
            and other system identifiers. This ID must be explicitly provided
            and is never auto-generated.
        evaluation_task (EvaluationTask): An instance of the EvaluationTask class
            defining the criteria and desired outcomes for the evaluation. This
            specifies *what* is being evaluated (e.g., toxicity, relevance) and
            the possible labels or scores the Judge is expected to produce.
        llm_client (LLMClientEnum): An enumeration value specifying the LLM provider
            to be used for this evaluation (e.g., OpenAI, Anthropic). This indicates
            which underlying client implementation should be selected by the
            MetaEvaluator.
        model (str): The specific name of the LLM model to be used from the
            selected provider (e.g., "gpt-4", "claude-3-opus-20240229"). This
            model will receive the prompt and perform the evaluation.
        prompt (Prompt): A Prompt object containing the instructions, few-shot examples,
            and structured output requirements (like XML tags or Pydantic models)
            that will be sent to the LLM. This dictates *how* the LLM should perform
            the evaluation based on the input data.

    model_config (ConfigDict): Pydantic configuration dictionary.
        - `frozen=True`: Makes the Judge instance immutable after creation,
          ensuring its configuration remains constant throughout its lifecycle.

    Validation:
        - The `id` attribute is validated to ensure it contains only alphanumeric
          characters and underscores, making it safe and consistent for use
          in various system contexts.
    """

    id: str
    model_config = ConfigDict(frozen=True)
    evaluation_task: EvaluationTask
    llm_client: LLMClientEnum
    model: str
    prompt: Prompt

    @model_validator(mode="after")
    def validate_id(self) -> "Judge":
        """Validate the id of the Judge.

        The id must only contain alphanumeric characters and underscores.

        Raises:
            ValueError: if the id contains invalid characters

        Returns:
            Judge: The instance of Judge with a valid id.
        """
        if not re.fullmatch(r"^[a-zA-Z0-9_]+$", self.id):
            raise ValueError(
                "id must only contain alphanumeric characters and underscores"
            )

        return self
