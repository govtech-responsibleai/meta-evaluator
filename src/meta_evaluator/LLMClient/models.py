"""Package meta_evaluator.LLMClient.models contains data models used for interacting with large language model (LLM) clients.

The package contains models for representing messages, responses, and usage statistics of LLM clients.

The package contains the following models:
*   Message: Represents a message in a conversation with an LLM client.
*   LLMResponse: Represents a response from an LLM client.
*   LLMUsage: Represents the usage statistics of an LLM client.

The models are used to provide a unified interface for interacting with LLM clients from different providers, such as OpenAI, Anthropic, and Gemini. The models are also used to provide type safety and to enable auto-completion when interacting with LLM clients in Python.
"""

from enum import Enum
import random
import time
from typing import Any
from pydantic import BaseModel


class RoleEnum(str, Enum):
    """Enum representing different roles in a conversation.

    This enum defines the roles supported in a conversation with an LLM client.

    The role enum is used when sending a prompt to the LLM client to specify the role of the message sender.

    Attributes:
        SYSTEM: Represents the system role, responsible for managing the conversation flow and providing context.
        USER: Represents the user role, typically the human participant interacting with the LLM.
        ASSISTANT: Represents the assistant role, which is the LLM itself providing responses to the user's queries.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Class representing a message in a conversation.

    This class represents a message sent by one of the participants in a conversation
    with an LLM client. It contains the message content and the role of the sender.

    The class is used to represent messages in the conversation history that is passed
    to the provider-specific LLM client implementations.

    Attributes:
        role: Enum value from RoleEnum indicating the role of the message sender.
        content: The text content of the message.
    """

    role: RoleEnum
    content: str

    def __str__(self):
        """Returns a string representation of the message.

        The string representation is a formatted string with the role and content of the message.
        The format is: "{role}: {content}".

        Returns:
            str: The string representation of the message.
        """
        return f"{self.role.value}: {self.content}"


class LLMClientEnum(Enum):
    """Enum class representing the supported LLM clients.

    This enum class is used to identify the provider-specific LLM client that is
    used to interact with the LLM service. The enum values are used in the
    LLMClient implementations as a unique identifier for the client.

    The enum values are:

    - OPENAI: OpenAI LLM client
    - AZURE_OPENAI: Azure OpenAI LLM client
    - GEMINI: Gemini LLM client
    - ANTHROPIC: Anthropic LLM client

    The values are used in the LLMClient implementations to identify the client
    that is used to interact with the LLM service.
    The enum values are also used in logging and other contexts to identify the
    client that is being used.
    """

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class LLMUsage(BaseModel):
    """Represents the usage statistics of a single interaction with an LLM client.

    The LLMUsage class is used to capture the usage statistics of a single interaction
    with an LLM client. The usage statistics include the number of prompt tokens,
    number of completion tokens, and the total number of tokens used in the interaction.

    The usage statistics are used by the LLMClient implementations to track the usage
    of the LLM clients and to provide detailed information about the usage to the
    user.

    Attributes:
        prompt_tokens: The number of prompt tokens used in the interaction.
        completion_tokens: The number of completion tokens used in the interaction.
        total_tokens: The total number of tokens used in the interaction.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMResponse(BaseModel):
    """Represents a response from a Large Language Model (LLM) client interaction.

    The `LLMResponse` class models the entire response from an LLM client after a prompt
    is sent. It encapsulates details such as the unique identifier for the response,
    the provider of the LLM service, the model used, the conversation messages, and the
    usage statistics.

    Attributes:
        id (str): A unique identifier for the response. If not provided, it is auto-generated
            based on the current timestamp, provider, model, and a random number.
        provider (LLMClientEnum): The provider of the LLM service (e.g., OpenAI, Azure).
        model (str): The specific model used for generating the response.
        messages (list[Message]): A list of messages representing the conversation, including
            both the prompt messages and the response from the LLM.
        usage (LLMUsage): An object containing the usage statistics for the interaction, such as
            the number of tokens used.

    Properties:
        latest_response (Message): Retrieves the most recent message from the conversation.
        content (str): Retrieves the content of the most recent message.

    Methods:
        model_post_init(__context: Any): Initializes the `id` attribute if it is not provided,
            using the current time, provider, model, and a random number.
    """

    id: str = ""
    provider: LLMClientEnum
    model: str
    messages: list[Message]
    usage: LLMUsage

    @property
    def latest_response(self) -> Message:
        """Returns the most recent message from the conversation.

        This property returns the latest message from the conversation, which is the
        last element in the list of messages. If the list is empty, returns None.

        Returns:
            Message: The most recent message from the conversation.
        """
        return self.messages[-1]

    @property
    def content(self) -> str:
        """Returns the content of the most recent message from the conversation.

        This property returns the content of the latest message from the conversation,
        which is the last element in the list of messages. If the list is empty, returns
        an empty string.

        Returns:
            str: The content of the most recent message from the conversation.
        """
        return self.latest_response.content

    def model_post_init(self, __context: Any) -> None:
        """Initializes the `id` attribute if it is not provided.

        This method is a post-init hook for Pydantic models. It is called after the
        model has been initialized with the provided data. If the `id` attribute is
        not provided, it is initialized with a unique identifier based on the current
        timestamp, provider, model, and a random number.

        The `id` attribute is used to identify the response in logging and other
        contexts.
        """
        if not self.id:
            timestamp = int(time.time())
            random_num = random.randint(1000, 9999)
            self.id = f"{timestamp}_{self.provider.value}_{self.model}_{random_num}"
