"""File for all common models."""

from pydantic import BaseModel


class Prompt(BaseModel):
    """Prompt model."""

    id: str
    prompt: str
