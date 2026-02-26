"""Pydantic models for EvalData serialization."""

from typing import Literal

from pydantic import BaseModel


class DataMetadata(BaseModel):
    """Metadata for serialized evaluation data.

    Contains all information needed to reconstruct EvalData or SampleEvalData
    objects from external data files.
    """

    name: str
    id_column: str
    data_file: str
    data_format: Literal["json", "csv", "parquet"]
    type: Literal["EvalData", "SampleEvalData"]

    # Optional fields for SampleEvalData
    sample_name: str | None = None
    stratification_columns: list[str] | None = None
    sample_percentage: float | None = None
    seed: int | None = None
    sampling_method: str | None = None
