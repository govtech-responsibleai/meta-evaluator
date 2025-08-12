"""Enumerations for the results domain."""

from enum import Enum


class BaseEvaluationStatusEnum(str, Enum):
    """Base enumeration for evaluation status types."""

    SUCCESS = "success"
    ERROR = "error"


class EvaluationStatusEnum(str, Enum):
    """Enumeration of possible evaluation outcomes for a single example."""

    SUCCESS = "success"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    LLM_ERROR = "llm_error"
    PARSING_ERROR = "parsing_error"
    OTHER_ERROR = "other_error"


class HumanAnnotationStatusEnum(str, Enum):
    """Enumeration of possible human annotation outcomes for a single example."""

    SUCCESS = "success"
    ERROR = "error"
