"""File for all common models."""

from pydantic import BaseModel


class Prompt(BaseModel):
    """System prompt given to the LLM for evaluation tasks.

    This class represents the system prompt that provides instructions to the LLM
    judge on how to evaluate a task. The prompt should contain clear instructions
    on what fields to analyze, evaluation criteria, and the expected response format.

    Attributes:
        id (str): Unique identifier for the prompt.
        prompt (str): The system prompt text containing evaluation instructions,
            field specifications, and response format guidelines for the LLM judge.

    Examples:
        >>> prompt = Prompt(
        ...     id="eval_toxicity_prompt",
        ...     prompt="Evaluate the toxicity of the given text. Consider harmful language, threats, and offensive content. Respond with 'toxic' or 'non_toxic'."
        ... )
    """

    id: str
    prompt: str
