from enum import Enum
import random
import time
from typing import Any
from pydantic import BaseModel


class RoleEnum(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: RoleEnum
    content: str

    def __str__(self):
        return f"{self.role.value}: {self.content}"


class LLMClientEnum(Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class LLMUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMResponse(BaseModel):
    id: str = ""
    provider: LLMClientEnum
    model: str
    messages: list[Message]
    usage: LLMUsage

    @property
    def latest_response(self) -> Message:
        return self.messages[-1]

    @property
    def content(self) -> str:
        return self.latest_response.content

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            timestamp = int(time.time())
            random_num = random.randint(1000, 9999)
            self.id = f"{timestamp}_{self.provider.value}_{self.model}_{random_num}"
