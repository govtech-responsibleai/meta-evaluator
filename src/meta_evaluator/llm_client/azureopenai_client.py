"""Concrete implementation of LLMClient for Azure OpenAI."""

from .LLM_client import LLMClientConfig


class AzureOpenAIConfig(LLMClientConfig):
    """Configuration settings for Azure OpenAI LLMClient."""

    supports_structured_output: bool = True
    supports_logprobs: bool = True
    endpoint: str
    api_version: str

    def _prevent_instantiation(self) -> None:
        """Allow instantiation.

        This is a dummy method that must be implemented according to the
        abstract base class. It is not intended to be used, and calling it
        will have no effect.
        """
        pass
