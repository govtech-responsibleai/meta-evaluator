"""Configuration models for comparison metrics."""

from typing import List, Literal

from pydantic import BaseModel, computed_field, model_validator

from ..common.error_constants import (
    INVALID_MULTILABEL_MULTITASK_AGGREGATION_MSG,
    INVALID_SINGLE_AGGREGATION_MSG,
)
from .base_scorer import BaseScorer
from .enums import TaskAggregationMode
from .exceptions import InvalidAggregationModeError


class MetricConfig(BaseModel):
    """Configuration for a single metric comparison."""

    scorer: BaseScorer
    task_names: List[str]
    task_strategy: Literal["single", "multilabel", "multitask"]
    annotator_aggregation: Literal["individual_average", "majority_vote"] = (
        "individual_average"
    )
    display_name: str | None = None

    @model_validator(mode="after")
    def validate_task_strategy_constraints(self) -> "MetricConfig":
        """Validate that task strategy matches task names count constraints.

        - 'single' task strategy can only be used with exactly 1 task name
        - 'multilabel' and 'multitask' task strategies can only be used with more than 1 task name

        Returns:
            MetricConfig: The validated instance.

        Raises:
            InvalidAggregationModeError: If task strategy doesn't match task names count.
        """
        task_count = len(self.task_names)

        if self.task_strategy == "single" and task_count != 1:
            raise InvalidAggregationModeError(
                f"{INVALID_SINGLE_AGGREGATION_MSG}, but got {task_count} task names: {self.task_names}"
            )

        if self.task_strategy in ["multilabel", "multitask"] and task_count <= 1:
            raise InvalidAggregationModeError(
                f"{INVALID_MULTILABEL_MULTITASK_AGGREGATION_MSG}, but got {task_count} task names: {self.task_names}"
            )

        return self

    @computed_field
    @property
    def aggregation_mode(self) -> TaskAggregationMode:
        """Compute aggregation_mode from task_strategy.

        Raises:
            ValueError: If task_strategy is not recognized.
        """
        if self.task_strategy == "single":
            return TaskAggregationMode.SINGLE
        elif self.task_strategy == "multilabel":
            return TaskAggregationMode.MULTILABEL
        elif self.task_strategy == "multitask":
            return TaskAggregationMode.MULTITASK
        else:
            raise ValueError(f"Invalid task strategy: {self.task_strategy}")

    class Config:
        """Pydantic configuration for MetricConfig."""

        arbitrary_types_allowed = True

    def get_unique_name(self) -> str:
        """Get a unique name for a metric configuration.

        Returns:
            str: A unique identifier for this metric configuration.
                If display_name is set, returns that. Otherwise generates a hash-based name.
        """
        # If user provided display_name, use it
        if self.display_name:
            return self.display_name

        # Otherwise generate hash-based name
        import hashlib

        # Create a hash of the task names to avoid very long names
        task_names_sorted = sorted(self.task_names)
        task_hash = hashlib.md5("_".join(task_names_sorted).encode()).hexdigest()[:8]

        aggregation_str = (
            self.aggregation_mode.value
            if hasattr(self.aggregation_mode, "value")
            else str(self.aggregation_mode)
        )

        # Include task count for clarity
        task_count = len(self.task_names)
        return (
            f"{self.scorer.scorer_name}_{task_count}tasks_{task_hash}_{aggregation_str}"
        )


class MetricsConfig(BaseModel):
    """Configuration for a complete comparison run."""

    metrics: List[MetricConfig]

    class Config:
        """Pydantic configuration for MetricsConfig."""

        arbitrary_types_allowed = True

    def add_metric(
        self,
        scorer: "BaseScorer",
        task_names: List[str],
        task_strategy: Literal["single", "multilabel", "multitask"],
        annotator_aggregation: Literal[
            "individual_average", "majority_vote"
        ] = "individual_average",
        display_name: str | None = None,
    ) -> None:
        """Add a metric configuration."""
        self.metrics.append(
            MetricConfig(
                scorer=scorer,
                task_names=task_names,
                task_strategy=task_strategy,
                annotator_aggregation=annotator_aggregation,
                display_name=display_name,
            )
        )
