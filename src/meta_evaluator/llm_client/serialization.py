"""Pydantic models for LLM client serialization."""

from abc import ABC
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


class MockLLMClientSerializedState(LLMClientSerializedState):
    """Serialized state for test client configuration."""

    client_type: LLMClientEnum = LLMClientEnum.TEST
