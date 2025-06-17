"""Pydantic models for EvalData serialization."""

from typing import List, Literal, Optional
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
    sample_name: Optional[str] = None
    stratification_columns: Optional[List[str]] = None
    sample_percentage: Optional[float] = None
    seed: Optional[int] = None
    sampling_method: Optional[str] = None
