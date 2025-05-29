"""Data ingestion and organization for LLM evaluation workflows.

This module provides tools for loading evaluation data from various sources
(CSV files, HuggingFace datasets, existing DataFrames) and organizing it into
structured containers with column categorization for evaluation pipelines.

Main Classes:
    DataLoader: Universal data ingestion from any source, returns structured EvalData
    EvalData: Immutable container with column categorization (inputs, outputs, metadata, labels)

"""

from .EvalData import EvalData

__all__ = ["EvalData"]
