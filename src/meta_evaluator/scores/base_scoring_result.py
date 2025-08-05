"""Data models for scoring functionality."""

from typing import Any, Dict
from datetime import datetime

from pydantic import BaseModel, Field


class BaseScoringResult(BaseModel):
    """Base container for all scoring results."""

    scorer_name: str = Field(
        ..., description="Name of the scorer that produced this result"
    )
    task_name: str = Field(..., description="Name of the task that was scored")
    judge_id: str = Field(..., description="ID of the judge that was scored")
    score: float = Field(..., description="The computed score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the score"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When this score was computed"
    )

    def save_state(self, file_path: str) -> None:
        """Save result state to JSON file."""
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_state(cls, file_path: str) -> "BaseScoringResult":
        """Load result state from JSON file.

        Returns:
            BaseScoringResult: A BaseScoringResult instance.
        """
        with open(file_path, "r") as f:
            return cls.model_validate_json(f.read())
