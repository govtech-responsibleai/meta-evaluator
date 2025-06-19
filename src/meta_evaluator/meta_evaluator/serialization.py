"""Pydantic models for MetaEvaluator serialization."""

from typing import Any, Dict, Optional
from pydantic import BaseModel

from ..data.serialization import DataMetadata


class MetaEvaluatorState(BaseModel):
    """Complete serialized state for a MetaEvaluator instance.

    Contains all information needed to reconstruct a MetaEvaluator,
    including client configurations and data metadata.
    """

    version: str = "1.0"
    client_registry: Dict[str, Dict[str, Any]]
    data: Optional[DataMetadata] = None
    evaluation_task: Optional[Dict] = None
