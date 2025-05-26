import logging
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel
from .models import Message, LLMClientEnum, LLMResponse


class LLMClientConfig(ABC, BaseModel):
    logger: logging.Logger
    api_key: str
    supports_instructor: bool
    default_model: str
    default_embedding_model: str


class LLMClient(ABC):
    def __init__(self, config: LLMClientConfig):
        self.config = config
        self.logger = config.logger

    @abstractmethod
    def _prompt(self, model: str, messages: list[Message]) -> LLMResponse:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def enum_value(self) -> LLMClientEnum:
        raise NotImplementedError("Subclasses must implement this method")

    def prompt(
        self, messages: list[Message], model: Optional[str] = None
    ) -> LLMResponse:
        """Send a prompt to the underlying LLM client with comprehensive logging.

        This method handles model selection, logging, and delegation to the provider-specific
        implementation. If no model is specified, falls back to the client's configured
        default model.

        Args:
            messages (list[Message]): Complete conversation history including system, user,
                and assistant messages. The conversation context is preserved and sent to
                the underlying provider.
            model (Optional[str], optional): Model to use for this request. If None,
                uses self.config.default_model. Defaults to None.

        Returns:
            LLMResponse: Response object containing the full conversation (input + new
                assistant response), usage statistics, and metadata. Access the latest
                response via response.latest_response or response.content.

        Note:
            All requests are logged including:
            - Model selection (chosen vs default)
            - Complete input message payload
            - Assistant response content
            - Token usage statistics

        Example:
            >>> messages = [Message(role=RoleEnum.USER, content="Hello")]
            >>> response = client.prompt(messages, model="gpt-4")
            >>> print(response.content)  # Assistant's response
            >>> print(len(response.messages))  # Original + assistant response
        """
        model_used = model or self.config.default_model
        self.logger.info(f"Using model: {model_used}")
        self.logger.info(f"Input Payload: {messages}")
        output = self._prompt(model_used, messages)
        self.logger.info(f"Latest response: {output.latest_response}")
        self.logger.info(f"Output usage: {output.usage}")

        return output
