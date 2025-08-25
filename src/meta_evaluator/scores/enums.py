"""Task aggregation mode enumeration for scoring."""

from enum import Enum


class TaskAggregationMode(Enum):
    """Enumeration of task aggregation strategies for scoring.

    - SINGLE: Process a single task
    - MULTITASK: Process multiple tasks separately and average results
    - MULTILABEL: Combine multiple tasks into a single multilabel outcome
    """

    SINGLE = "single"
    MULTITASK = "multitask"
    MULTILABEL = "multilabel"
