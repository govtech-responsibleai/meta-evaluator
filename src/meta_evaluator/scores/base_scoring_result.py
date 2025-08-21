"""Data models for scoring functionality."""

import json
from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field, ValidationError

from ..common.error_constants import (
    INVALID_JSON_MSG,
    INVALID_JSON_STRUCTURE_MSG,
    STATE_FILE_NOT_FOUND_MSG,
)
from ..results.exceptions import InvalidFileError
from .enums import TaskAggregationMode


class BaseScoringResult(BaseModel):
    """Base container for all scoring results."""

    scorer_name: str = Field(
        ..., description="Name of the scorer that produced this result"
    )
    task_name: str = Field(..., description="Name of the task(s) that was scored")
    judge_id: str = Field(..., description="ID of the judge that was scored")
    scores: Dict[str, Any] = Field(
        ..., description="Dictionary of metric names to scores"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the score"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When this score was computed"
    )
    aggregation_mode: TaskAggregationMode = Field(
        ..., description="How the tasks were aggregated for this result"
    )
    num_comparisons: int = Field(
        ..., description="Number of judge-human pairs that were compared"
    )
    failed_comparisons: int = Field(
        default=0, description="Number of comparisons that failed or were skipped"
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

        Raises:
            InvalidFileError: If the file doesn't exist or the JSON structure is invalid.
        """
        try:
            with open(file_path, "r") as f:
                return cls.model_validate_json(f.read())
        except FileNotFoundError as e:
            raise InvalidFileError(f"{STATE_FILE_NOT_FOUND_MSG}: {file_path}", e)
        except ValidationError as e:
            raise InvalidFileError(INVALID_JSON_STRUCTURE_MSG, e)
        except json.JSONDecodeError as e:
            raise InvalidFileError(INVALID_JSON_MSG, e)
