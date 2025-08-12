"""Configuration models for comparison metrics."""

from typing import List

from pydantic import BaseModel

from .base_scorer import BaseScorer


class MetricConfig(BaseModel):
    """Configuration for a single metric comparison."""

    scorer: BaseScorer
    task_names: List[str]

    class Config:
        """Pydantic configuration for MetricConfig."""

        arbitrary_types_allowed = True


class MetricsConfig(BaseModel):
    """Configuration for a complete comparison run."""

    metrics: List[MetricConfig]

    class Config:
        """Pydantic configuration for MetricsConfig."""

        arbitrary_types_allowed = True

    def add_metric(self, scorer: "BaseScorer", task_names: List[str]) -> None:
        """Add a metric configuration."""
        self.metrics.append(MetricConfig(scorer=scorer, task_names=task_names))
