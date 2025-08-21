"""Configuration models for comparison metrics."""

from typing import List, Literal

from pydantic import BaseModel, computed_field

from .base_scorer import BaseScorer
from .enums import TaskAggregationMode


class MetricConfig(BaseModel):
    """Configuration for a single metric comparison."""

    scorer: BaseScorer
    task_names: List[str]
    aggregation_name: Literal["single", "multilabel", "multitask"]

    @computed_field
    @property
    def aggregation_mode(self) -> TaskAggregationMode:
        """Compute aggregation_mode from aggregation_name.

        Raises:
            ValueError: If aggregation_name is not recognized.
        """
        if self.aggregation_name == "single":
            return TaskAggregationMode.SINGLE
        elif self.aggregation_name == "multilabel":
            return TaskAggregationMode.MULTILABEL
        elif self.aggregation_name == "multitask":
            return TaskAggregationMode.MULTITASK
        else:
            raise ValueError(f"Invalid aggregation name: {self.aggregation_name}")

    class Config:
        """Pydantic configuration for MetricConfig."""

        arbitrary_types_allowed = True

    def get_unique_name(self) -> str:
        """Get a unique name for a metric configuration.

        Returns:
            str: A unique identifier for this metric configuration.
        """
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
        aggregation_name: Literal["single", "multilabel", "multitask"],
    ) -> None:
        """Add a metric configuration."""
        # Determine aggregation_name if not provided
        if aggregation_name is None:
            if len(task_names) == 1:
                aggregation_name = "single"
            else:
                aggregation_name = "multitask"

        self.metrics.append(
            MetricConfig(
                scorer=scorer,
                task_names=task_names,
                aggregation_name=aggregation_name,
            )
        )
