"""Pydantic models for LLM client and MetaEvaluator serialization."""

from abc import ABC
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel
from .models import LLMClientEnum


class LLMClientSerializedState(BaseModel, ABC):
    """Abstract base class for serialized LLM client state.

    This defines the common structure that all LLM client serialized states
    must implement, ensuring consistency across different client types.
    """

    client_type: LLMClientEnum
    default_model: str
    default_embedding_model: str
    supports_structured_output: bool
    supports_logprobs: bool
    supports_instructor: bool


class OpenAISerializedState(LLMClientSerializedState):
    """Serialized state for OpenAI client configuration."""

    client_type: LLMClientEnum = LLMClientEnum.OPENAI


class AzureOpenAISerializedState(LLMClientSerializedState):
    """Serialized state for Azure OpenAI client configuration."""

    client_type: LLMClientEnum = LLMClientEnum.AZURE_OPENAI
    endpoint: str
    api_version: str


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


class MockLLMClientSerializedState(LLMClientSerializedState):
    """Serialized state for test client configuration."""

    client_type: LLMClientEnum = LLMClientEnum.TEST


class MetaEvaluatorState(BaseModel):
    """Complete serialized state for a MetaEvaluator instance.

    Contains all information needed to reconstruct a MetaEvaluator,
    including client configurations and data metadata.
    """

    version: str = "1.0"
    client_registry: Dict[str, Dict[str, Any]]
    data: Optional[DataMetadata] = None
